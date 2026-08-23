/** UI wiring for the watermarking playground. Inference runs in worker.js. */

const $ = (id) => document.getElementById(id);

const els = {
  model: $("model"),
  statusText: $("status-text"),
  backendBadge: $("backend-badge"),
  progress: $("progress"),
  progressFill: $("progress-fill"),
  prompt: $("prompt"),
  maxTokens: $("maxTokens"),
  temperature: $("temperature"),
  topK: $("topK"),
  topP: $("topP"),
  seed: $("seed"),
  gamma: $("gamma"),
  delta: $("delta"),
  ctxWidth: $("ctxWidth"),
  layers: $("layers"),
  wmKey: $("wmKey"),
  detectKey: $("detectKey"),
  redWords: $("redWords"),
  redWordsField: $("redwords-field"),
  legendHi: $("legend-hi"),
  legendLo: $("legend-lo"),
  statFirstLabel: $("stat-first-label"),
  statScheme: $("stat-scheme"),
  generateBtn: $("generate-btn"),
  stopBtn: $("stop-btn"),
  detectBtn: $("detect-btn"),
  output: $("output"),
  placeholder: $("output-placeholder"),
  promptEcho: $("prompt-echo"),
  tokens: $("tokens"),
  legend: $("legend"),
  verdict: $("verdict"),
  verdictLabel: $("verdict-label"),
  verdictConf: $("verdict-conf"),
  meterFill: $("meter-fill"),
  statGreen: $("stat-green"),
  statZ: $("stat-z"),
  statP: $("stat-p"),
  verdictNote: $("verdict-note"),
};

let modelReady = false;
let generating = false;
let haveGeneration = false;
let streamSpan = null;

const worker = new Worker("worker.js?v=8", { type: "module" });

// Optional backend override for debugging/perf comparison: ?device=wasm or ?device=webgpu
const forcedDevice = new URLSearchParams(location.search).get("device");
const loadMsg = () => ({ type: "load", modelId: els.model.value, device: forcedDevice });

/* ── helpers ── */

const clamp = (v, lo, hi, fallback) => {
  const n = parseFloat(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
};

const getMode = () => document.querySelector('input[name="wmMode"]:checked').value;

function readParams() {
  return {
    prompt: els.prompt.value,
    promptStyle: document.querySelector('input[name="promptStyle"]:checked').value,
    mode: getMode(),
    maxNewTokens: Math.round(clamp(els.maxTokens.value, 1, 512, 200)),
    temperature: clamp(els.temperature.value, 0, 2, 1),
    topK: Math.round(clamp(els.topK.value, 0, 1000, 40)),
    topP: clamp(els.topP.value, 0.05, 1, 0.9),
    seed: els.seed.value.trim() === "" ? null : Math.round(clamp(els.seed.value, 0, 2147483647, 0)),
    gamma: clamp(els.gamma.value, 0.05, 0.95, 0.5),
    delta: clamp(els.delta.value, 0, 15, 4),
    h: Math.round(clamp(els.ctxWidth.value, 1, 8, 1)),
    m: Math.round(clamp(els.layers.value, 1, 30, 15)),
    key: els.wmKey.value,
    redWords: els.redWords.value,
  };
}

function setStatus(text, isError = false) {
  els.statusText.textContent = text;
  els.statusText.style.color = isError ? "var(--red)" : "";
}

function updateButtons() {
  els.generateBtn.disabled = !modelReady || generating;
  els.detectBtn.disabled = !haveGeneration || generating;
  els.stopBtn.hidden = !generating;
  els.model.disabled = generating;
}

/** Which parameters each mode exposes (ids of .param wrappers / fields). */
const VISIBLE_PARAMS = {
  none: ["maxTokens", "temperature", "topK", "topP", "seed"],
  hard: ["maxTokens", "temperature", "topK", "topP", "seed", "wmSection", "gamma", "ctxWidth", "wmKey", "redWords"],
  soft: ["maxTokens", "temperature", "topK", "topP", "seed", "wmSection", "gamma", "delta", "ctxWidth", "wmKey", "redWords"],
  tournament: ["maxTokens", "temperature", "topK", "topP", "seed", "wmSection", "layers", "ctxWidth", "wmKey", "redWords"],
};

function syncParamVisibility() {
  const visible = new Set(VISIBLE_PARAMS[getMode()]);
  document.querySelectorAll("[data-param]").forEach((el) => {
    el.hidden = !visible.has(el.dataset.param);
  });
}

const MODE_LABEL = { hard: "hard watermark", soft: "soft watermark", tournament: "tournament sampling" };

function clearOutput() {
  els.placeholder.hidden = true;
  els.promptEcho.textContent = "";
  els.tokens.textContent = "";
  els.legend.hidden = true;
  els.verdict.hidden = true;
  haveGeneration = false;
}

/* ── worker messages ── */

worker.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "status":
      setStatus(msg.text);
      break;

    case "progress": {
      els.progress.hidden = false;
      const pct = Math.round(msg.progress ?? 0);
      els.progressFill.style.width = `${pct}%`;
      const mb = (b) => (b / 1024 / 1024).toFixed(0);
      const parts = msg.files > 1 ? `, ${msg.files} files` : "";
      setStatus(`Downloading model: ${pct}% (${mb(msg.loaded)} / ${mb(msg.total)} MB${parts})`);
      break;
    }

    case "ready":
      modelReady = true;
      els.progress.hidden = true;
      setStatus("Model ready.");
      els.backendBadge.textContent = msg.device === "webgpu" ? "WebGPU" : "WASM (slower)";
      els.backendBadge.hidden = false;
      updateButtons();
      break;

    case "stream":
      if (streamSpan) {
        streamSpan.textContent += msg.text;
        els.output.scrollTop = els.output.scrollHeight;
      }
      break;

    case "generated": {
      generating = false;
      streamSpan = null;
      els.tokens.textContent = "";
      const params = readParams();
      els.promptEcho.textContent = params.promptStyle === "raw" ? msg.promptText : "";
      for (const tok of msg.tokens) {
        const span = document.createElement("span");
        span.className = "tok";
        span.textContent = tok.text;
        span.title = `token id ${tok.id}`;
        els.tokens.appendChild(span);
      }
      haveGeneration = msg.tokens.length > 0;
      setStatus(msg.interrupted ? "Generation stopped." : "Done. Try the detector.");
      updateButtons();
      break;
    }

    case "detected": {
      const chips = els.tokens.children;
      const tournament = msg.scheme === "tournament";
      for (let i = 0; i < chips.length; i++) {
        chips[i].classList.toggle("green", !!msg.flags[i]);
        chips[i].classList.toggle("red", !msg.flags[i]);
        if (tournament) {
          chips[i].title = `${chips[i].title.split(" · ")[0]} · mean g = ${msg.perTokenScore[i].toFixed(2)}`;
        } else {
          chips[i].title = chips[i].title.split(" · ")[0];
        }
      }
      els.legendHi.textContent = tournament ? "g-value ≥ 0.5" : "green list";
      els.legendLo.textContent = tournament ? "g-value < 0.5" : "red list";
      els.legend.hidden = false;
      renderVerdict(msg);
      break;
    }

    case "error":
      generating = false;
      streamSpan = null;
      setStatus(msg.message, true);
      updateButtons();
      break;
  }
};

worker.onerror = (e) => {
  setStatus(`Worker error: ${e.message ?? "see console"}`, true);
  generating = false;
  updateButtons();
};

/* ── verdict rendering ── */

function renderVerdict({ scheme, z, pValue, greenCount, T, gamma, meanG, m, h }) {
  els.verdict.hidden = false;
  const tournament = scheme === "tournament";

  let label, cls, note;
  if (z > 4) {
    label = "Watermark detected";
    cls = "pos";
    note = tournament
      ? `With no watermark, a mean g-value this high would occur by chance with probability ${fmtP(pValue)}, far past the z > 4 threshold.`
      : `With no watermark, a run this green would occur by chance with probability ${fmtP(pValue)}, far past the paper's z > 4 threshold.`;
  } else if (z > 2) {
    label = "Weak evidence of a watermark";
    cls = "mid";
    note = tournament
      ? "Above chance, but below the z > 4 threshold. Longer outputs or more tournament layers (m) strengthen the signal."
      : "Above chance, but below the paper's z > 4 threshold. Longer outputs or a larger δ strengthen the signal.";
  } else {
    label = "No watermark detected";
    cls = "neg";
    note = tournament
      ? "Without a watermark every g-value is a fair coin flip, so the mean sits near 0.5; this text is consistent with that."
      : `About γ = ${gamma} of tokens land in the green list purely by chance; this text is consistent with that.`;
  }

  els.verdictLabel.textContent = label;
  els.verdictLabel.className = `verdict-label ${cls}`;

  const confidence = Math.max(0, Math.min(1, 1 - pValue));
  els.verdictConf.textContent = `confidence ${(confidence * 100).toFixed(confidence > 0.999 ? 3 : 1)}%`;
  els.meterFill.style.width = `${(confidence * 100).toFixed(2)}%`;

  els.statScheme.textContent = tournament ? `tournament (m=${m}, h=${h})` : `green list (γ=${gamma}, h=${h})`;
  if (tournament) {
    els.statFirstLabel.textContent = "mean g-value";
    els.statGreen.textContent = `${meanG.toFixed(3)} over ${T} × ${m}`;
  } else {
    els.statFirstLabel.textContent = "green tokens";
    els.statGreen.textContent = `${greenCount} / ${T} (${((greenCount / T) * 100).toFixed(0)}%)`;
  }
  els.statZ.textContent = z.toFixed(2);
  els.statP.textContent = fmtP(pValue);
  els.verdictNote.textContent = note;
}

function fmtP(p) {
  if (p < 1e-15) return "< 1e-15";
  if (p < 0.001) return p.toExponential(1);
  return p.toFixed(3);
}

/* ── user actions ── */

els.generateBtn.addEventListener("click", () => {
  const params = readParams();
  if (!params.prompt.trim()) {
    setStatus("Enter a prompt first.", true);
    return;
  }
  clearOutput();
  generating = true;
  updateButtons();
  setStatus(params.mode === "none" ? "Generating…" : `Generating with ${MODE_LABEL[params.mode]}…`);

  streamSpan = document.createElement("span");
  els.tokens.appendChild(streamSpan);
  els.promptEcho.textContent = params.promptStyle === "raw" ? params.prompt.trim() : "";

  worker.postMessage({ type: "generate", params });
});

els.stopBtn.addEventListener("click", () => worker.postMessage({ type: "interrupt" }));

els.detectBtn.addEventListener("click", () => {
  const { mode, gamma, h, m, redWords } = readParams();
  const scheme = mode === "tournament" ? "tournament" : "greenlist";
  worker.postMessage({ type: "detect", params: { scheme, gamma, h, m, redWords, key: els.detectKey.value } });
});

els.model.addEventListener("change", () => {
  modelReady = false;
  els.backendBadge.hidden = true;
  updateButtons();
  worker.postMessage(loadMsg());
});

document.querySelectorAll('input[name="wmMode"]').forEach((r) => r.addEventListener("change", syncParamVisibility));

els.prompt.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !els.generateBtn.disabled) {
    els.generateBtn.click();
  }
});

/* ── init ── */
syncParamVisibility();
updateButtons();
setStatus("Loading model…");
worker.postMessage(loadMsg());
