/**
 * Web worker: loads the model with transformers.js and runs watermarked
 * generation + detection off the main thread.
 */
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  LogitsProcessor,
  LogitsProcessorList,
  InterruptableStoppingCriteria,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";
import { seedFromContext, isGreen, detect, keyToSeed, tournamentSample, detectTournament, sampleMultinomial, seededRng } from "./watermark.js?v=8";

let tokenizer = null;
let model = null;
let loadedModelId = null;
let device = null;

// Last generation, kept so the detector can run on it with (possibly edited) params.
let lastGen = null; // { ids: number[], promptLen: number }

const stoppingCriteria = new InterruptableStoppingCriteria();

/** Models whose q4 weights exceed what the 32-bit WASM backend can hold. */
const WEBGPU_ONLY = new Set(["onnx-community/Qwen3-1.7B-ONNX", "onnx-community/Qwen3-4B-ONNX"]);

const post = (msg) => self.postMessage(msg);

/**
 * Applies the Kirchenbauer watermark and temperature scaling to the raw
 * logits of each generation step. (Temperature is handled here because
 * transformers.js's multinomial sampler ignores generation_config.temperature.)
 */
class WatermarkProcessor extends LogitsProcessor {
  constructor({ mode, gamma, delta, h, m, forcedRed, temperature, topK, topP, keySeed, rng, sampleSelf }) {
    super();
    this.mode = mode;
    this.gamma = gamma;
    this.delta = delta;
    this.h = h;
    this.m = m;
    this.forcedRed = forcedRed;
    this.temperature = temperature;
    this.topK = topK;
    this.topP = topP;
    this.keySeed = keySeed;
    this.rng = rng; // uniform [0,1) source; seeded when the user sets a generation seed
    this.sampleSelf = sampleSelf; // draw the token here (seeded run) instead of in the library sampler
    this.probs = null; // scratch buffer
  }

  /**
   * Order of operations (matches the HF pipeline: watermark processor → warpers):
   *   1. green/red-list bias (soft) or mask (hard)
   *   2. temperature
   *   3. top-k / top-p truncation
   *   4. tournament sampling over whatever survives (tournament mode only)
   */
  _call(input_ids, logits) {
    const data = logits.data;
    const vocab = logits.dims.at(-1);

    let seed = 0;
    if (this.mode !== "none") {
      const ids = input_ids[0];
      const ctx = ids.slice(Math.max(0, ids.length - this.h)).map(Number);
      seed = seedFromContext(ctx, this.keySeed);
    }

    if (this.mode === "hard") {
      for (let t = 0; t < vocab; t++) {
        if (!isGreen(seed, t, this.gamma, this.forcedRed)) data[t] = -Infinity;
      }
    } else if (this.mode === "soft") {
      for (let t = 0; t < vocab; t++) {
        if (isGreen(seed, t, this.gamma, this.forcedRed)) data[t] += this.delta;
      }
    }

    if (this.temperature > 0 && this.temperature !== 1) {
      const inv = 1 / this.temperature;
      for (let i = 0; i < vocab; i++) data[i] *= inv; // -Inf stays -Inf
    }

    const probs = this.softmax(data, vocab);
    this.truncate(data, probs, vocab);

    if (this.mode === "tournament") {
      const winner = tournamentSample(probs, seed, this.m, this.forcedRed, this.rng);
      data.fill(-Infinity);
      data[winner] = 0; // downstream sampler/argmax can only pick the winner
    } else if (this.sampleSelf) {
      const winner = sampleMultinomial(probs, 1, this.rng)[0];
      data.fill(-Infinity);
      data[winner] = 0;
    }
    return logits;
  }

  /** Softmax of `data` into the scratch buffer (returned). */
  softmax(data, vocab) {
    if (!this.probs || this.probs.length !== vocab) this.probs = new Float64Array(vocab);
    const probs = this.probs;
    let max = -Infinity;
    for (let t = 0; t < vocab; t++) if (data[t] > max) max = data[t];
    let sum = 0;
    for (let t = 0; t < vocab; t++) {
      const e = data[t] === -Infinity ? 0 : Math.exp(data[t] - max);
      probs[t] = e;
      sum += e;
    }
    for (let t = 0; t < vocab; t++) probs[t] /= sum;
    return probs;
  }

  /**
   * Top-k then top-p (nucleus) truncation. Masks logits to -Inf and zeroes +
   * renormalises `probs` in place so both views stay consistent.
   */
  truncate(data, probs, vocab) {
    const k = this.topK;
    const p = this.topP;
    const useK = k > 0 && k < vocab;
    const useP = p > 0 && p < 1;
    if (!useK && !useP) return;

    const sorted = Float64Array.from(probs).sort(); // ascending, native & fast
    let threshold = 0;
    if (useK) threshold = sorted[vocab - k];
    if (useP) {
      let cum = 0;
      for (let i = vocab - 1; i >= 0; i--) {
        cum += sorted[i];
        if (cum >= p) {
          threshold = Math.max(threshold, sorted[i]);
          break;
        }
      }
    }

    let sum = 0;
    for (let t = 0; t < vocab; t++) {
      if (probs[t] < threshold || probs[t] === 0) {
        data[t] = -Infinity;
        probs[t] = 0;
      } else {
        sum += probs[t];
      }
    }
    if (sum > 0) for (let t = 0; t < vocab; t++) probs[t] /= sum;
  }
}

async function pickDevice(forced) {
  if (forced === "wasm" || forced === "webgpu") return forced;
  try {
    if (self.navigator?.gpu && (await self.navigator.gpu.requestAdapter())) return "webgpu";
  } catch { /* fall through */ }
  return "wasm";
}

async function loadModel(modelId, forcedDevice) {
  if (loadedModelId === modelId && model) {
    post({ type: "ready", device, modelId });
    return;
  }
  try {
    if (model) {
      try { await model.dispose(); } catch { /* best effort */ }
      model = null;
      tokenizer = null;
      lastGen = null;
    }
    device = await pickDevice(forcedDevice);
    if (device !== "webgpu" && WEBGPU_ONLY.has(modelId)) {
      throw new Error(
        `${modelId} needs WebGPU: its 4-bit weights are >2 GB on the WASM backend, beyond the 32-bit memory limit. ` +
        "Use a Chromium-based browser with WebGPU enabled, or pick a smaller model."
      );
    }
    const dtype = device === "webgpu" ? "q4f16" : "q4";
    post({ type: "status", text: `Loading ${modelId} (${device}, ${dtype})…` });

    tokenizer = await AutoTokenizer.from_pretrained(modelId);
    // Weight files may be split: model.onnx + model.onnx_data, model.onnx_data_1, …
    const isWeightFile = (f) => /\.onnx(_data(_\d+)?)?$/.test(f ?? "");
    const files = new Map(); // file -> { loaded, total }
    let downloading = false;
    model = await AutoModelForCausalLM.from_pretrained(modelId, {
      device,
      dtype,
      progress_callback: (p) => {
        if (!isWeightFile(p.file)) return;
        if (p.status === "progress") {
          downloading = true;
          files.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
          let loaded = 0, total = 0;
          for (const f of files.values()) { loaded += f.loaded; total += f.total; }
          post({ type: "progress", files: files.size, progress: total ? (100 * loaded) / total : 0, loaded, total });
        } else if (p.status === "done" && downloading) {
          post({ type: "status", text: `Downloaded ${p.file.split("/").pop()} — compiling model (can take a minute)…` });
        }
      },
    });
    loadedModelId = modelId;
    post({ type: "ready", device, modelId });
  } catch (err) {
    loadedModelId = null;
    model = null;
    post({ type: "error", message: `Failed to load model: ${err.message ?? err}` });
  }
}

/** Tokenize user red-list words in several surface forms into a set of token ids. */
function buildForcedRedSet(redWords) {
  const set = new Set();
  if (!tokenizer || !redWords) return set;
  const words = redWords.split(/[,\n]+/).map((w) => w.trim()).filter(Boolean);
  for (const w of words) {
    const lower = w.toLowerCase();
    const cap = lower.charAt(0).toUpperCase() + lower.slice(1);
    const forms = new Set([w, lower, cap, w.toUpperCase()]);
    for (const form of [...forms]) forms.add(" " + form);
    for (const form of forms) {
      for (const id of tokenizer.encode(form, { add_special_tokens: false })) {
        set.add(Number(id));
      }
    }
  }
  return set;
}

/** Drop trailing special tokens (e.g. EOS) so they don't show up as chips or skew detection. */
function stripTrailingSpecial(ids) {
  let end = ids.length;
  while (end > 0 && tokenizer.decode([ids[end - 1]], { skip_special_tokens: true }) === "") {
    end--;
  }
  return ids.slice(0, end);
}

async function generate(p) {
  if (!model || !tokenizer) {
    post({ type: "error", message: "Model not loaded yet." });
    return;
  }
  try {
    const forcedRed = buildForcedRedSet(p.redWords);

    // Trailing whitespace would become a standalone " " token, which BPE models almost never
    // see before a word (spaces are attached to the *following* token) and derails generation.
    const prompt = p.prompt.trim();

    let inputs;
    if (p.promptStyle === "chat") {
      inputs = tokenizer.apply_chat_template([{ role: "user", content: prompt }], {
        add_generation_prompt: true,
        return_dict: true,
        enable_thinking: false, // no-op for models without a thinking mode
      });
    } else {
      inputs = tokenizer(prompt);
    }
    const promptLen = inputs.input_ids.dims.at(-1);

    const greedy = p.temperature <= 0;
    const hasSeed = p.seed !== null && p.seed !== undefined;
    const rng = hasSeed ? seededRng(Math.round(p.seed) | 0) : Math.random;
    const processors = new LogitsProcessorList();
    processors.push(
      new WatermarkProcessor({
        mode: p.mode,
        gamma: p.gamma,
        delta: p.delta,
        h: p.h,
        m: p.m,
        forcedRed,
        temperature: greedy ? 1 : p.temperature,
        topK: p.topK,
        topP: p.topP,
        keySeed: keyToSeed(p.key),
        rng,
        sampleSelf: hasSeed && !greedy,
      })
    );

    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => post({ type: "stream", text }),
    });

    stoppingCriteria.reset();
    const output = await model.generate({
      ...inputs,
      generation_config: {
        max_new_tokens: p.maxNewTokens,
        do_sample: !greedy,
        top_k: 0, // sample from the full (watermarked) distribution
        temperature: 1.0,
      },
      logits_processor: processors,
      stopping_criteria: stoppingCriteria,
      streamer,
    });

    const allIds = stripTrailingSpecial(output.tolist()[0].map(Number));
    const genIds = allIds.slice(promptLen);
    lastGen = { ids: allIds, promptLen };

    post({
      type: "generated",
      promptText: tokenizer.decode(allIds.slice(0, promptLen), { skip_special_tokens: true }),
      tokens: genIds.map((id) => ({ id, text: tokenizer.decode([id], { skip_special_tokens: false }) })),
      interrupted: stoppingCriteria.interrupted,
    });
  } catch (err) {
    post({ type: "error", message: `Generation failed: ${err.message ?? err}` });
  }
}

function runDetect(p) {
  if (!lastGen) {
    post({ type: "error", message: "Nothing to detect yet. Generate some text first." });
    return;
  }
  const forcedRed = buildForcedRedSet(p.redWords);
  if (p.scheme === "tournament") {
    const r = detectTournament(lastGen.ids, lastGen.promptLen, {
      m: p.m,
      h: p.h,
      forcedRed,
      keySeed: keyToSeed(p.key),
    });
    post({
      type: "detected",
      scheme: "tournament",
      flags: r.flags,
      perTokenScore: r.perTokenScore,
      meanG: r.meanG,
      T: r.T,
      m: r.m,
      h: p.h,
      z: r.z,
      pValue: r.pValue,
    });
    return;
  }
  const result = detect(lastGen.ids, lastGen.promptLen, {
    gamma: p.gamma,
    h: p.h,
    forcedRed,
    keySeed: keyToSeed(p.key),
  });
  post({
    type: "detected",
    scheme: "greenlist",
    flags: result.flags,
    greenCount: result.greenCount,
    T: result.T,
    z: result.z,
    pValue: result.pValue,
    gamma: p.gamma,
    h: p.h,
  });
}

self.onmessage = async (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "load":
      await loadModel(msg.modelId, msg.device);
      break;
    case "generate":
      await generate(msg.params);
      break;
    case "detect":
      runDetect(msg.params);
      break;
    case "interrupt":
      stoppingCriteria.interrupt();
      break;
  }
};
