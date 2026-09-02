/**
 * UI wiring for the watermarking playground. Inference runs in worker.js.
 *
 * One worker per load attempt: switching models or failing a load terminates the
 * worker and spawns a fresh one (see worker.js for why). A WebGPU failure is retried
 * once on WASM; an out-of-memory failure ends with an explanation + diagnostics.
 */
import {
  APP_VERSION,
  MODELS,
  WEBGPU_ONLY,
  DTYPES,
  modelInfo,
  sizeMB,
  estimatedLoadMB,
  formatMB,
  classifyError,
  collectEnvironment,
  probeWasmHeapMB,
  createLog,
} from "./models.js?v=1";

const $ = (id) => document.getElementById(id);

const els = {
  model: $("model"),
  statusText: $("status-text"),
  statusNote: $("status-note"),
  backendBadge: $("backend-badge"),
  progress: $("progress"),
  progressFill: $("progress-fill"),
  diag: $("diagnostics"),
  diagText: $("diag-text"),
  copyDiag: $("copy-diag"),
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

/* ── URL overrides (debugging / perf comparison) ── */
const params = new URLSearchParams(location.search);
// ?device=wasm|webgpu pins the backend (no automatic WASM retry when pinned)
const forcedDevice = ["wasm", "webgpu"].includes(params.get("device")) ? params.get("device") : null;
// ?dtype=q4|q4f16|q8|fp16|fp32 pins the quantisation
const forcedDtype = DTYPES.includes(params.get("dtype")) ? params.get("dtype") : null;
// ?debug=1 always shows the diagnostics panel and mirrors worker messages to the console
const DEBUG = params.has("debug");

const LOAD_PHASES = new Set(["starting", "backend", "download", "compile"]);
const PHASE_TEXT = {
  starting: "starting the inference runtime",
  backend: "choosing a backend",
  download: "downloading",
  compile: "compiling the model (after the download finished)",
  generate: "generating",
  detect: "detecting",
};

/* ── worker lifecycle ── */

const log = createLog(80);
let active = null; // current attempt: { id, worker, modelId, device, dtype, retriedWasm, phase, … }
let attemptSeq = 0;
let lastBackend = null; // last { device, dtype, f16, note, adapter, … } reported by a worker
let crashReloads = 0;

function spawnWorker(meta) {
  const id = ++attemptSeq;
  const worker = new Worker("worker.js?v=9", { type: "module" });
  // Ignore events from a worker we already retired (a message can be queued before terminate()).
  worker.onmessage = (e) => {
    if (active?.id === id) handleMessage(e.data);
  };
  worker.onerror = (e) => {
    if (active?.id === id) handleWorkerError(e);
  };
  active = { id, worker, phase: "starting", startedAt: performance.now(), bytes: null, backend: null, hello: null, workerLog: null, ...meta };
  log.push("spawn", { id, modelId: meta.modelId, device: meta.device ?? "auto", dtype: meta.dtype ?? "auto", retriedWasm: !!meta.retriedWasm });
}

function retireWorker() {
  if (!active) return;
  log.push("retire", { id: active.id });
  active.worker.onmessage = null;
  active.worker.onerror = null;
  active.worker.terminate();
  active = null;
}

const send = (msg) => active?.worker.postMessage(msg);

/** Start a load attempt in a fresh worker. meta: { modelId, device, dtype, retriedWasm, keepOutput, firstFailure } */
function startLoad(meta) {
  retireWorker();
  modelReady = false;
  generating = false;
  streamSpan = null;
  haveGeneration = false; // the last generation lived in the old worker
  els.backendBadge.hidden = true;
  els.progress.hidden = true;
  els.progressFill.style.width = "0%";
  if (!meta.keepOutput) resetOutput();
  if (!meta.retriedWasm) {
    showNote("");
    hideDiagnostics();
  }
  updateButtons();
  setStatus("Starting inference runtime…");
  spawnWorker(meta);
  send({ type: "load", modelId: meta.modelId, device: meta.device ?? null, dtype: meta.dtype ?? null });
}

/* ── helpers ── */

const clamp = (v, lo, hi, fallback) => {
  const n = parseFloat(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
};

const shorten = (s, n = 160) => {
  s = String(s ?? "");
  return s.length > n ? `${s.slice(0, n)}…` : s;
};

const getMode = () => document.querySelector('input[name="wmMode"]:checked').value;

function readParams() {
  return {
    prompt: els.prompt.value,
    promptStyle: document.querySelector('input[name="promptStyle"]:checked').value,
    mode: getMode(),
    maxNewTokens: Math.round(clamp(els.maxTokens.value, 1, 512, 200)),
    temperature: clamp(els.temperature.value, 0, 2, 1),
    topK: Math.round(clamp(els.topK.value, 0, 1000, 200)),
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

/** level: "info" | "warn" | "error" */
function setStatus(text, level = "info") {
  els.statusText.textContent = text;
  els.statusText.classList.toggle("status-error", level === "error");
  els.statusText.classList.toggle("status-warn", level === "warn");
}

/** Secondary line under the status (advice, retry explanation). */
function showNote(text) {
  els.statusNote.textContent = text;
  els.statusNote.hidden = !text;
}

function showBackendBadge({ device, dtype }) {
  els.backendBadge.textContent = device === "webgpu" ? `WebGPU · ${dtype}` : `WASM · ${dtype} (slower)`;
  const b = lastBackend;
  els.backendBadge.title =
    [b?.adapter?.description || b?.adapter?.vendor, b?.note].filter(Boolean).join(" — ") ||
    (device === "webgpu" ? "WebGPU" : "WebAssembly, single-threaded");
  els.backendBadge.hidden = false;
}

function updateButtons() {
  els.generateBtn.disabled = !modelReady || generating;
  els.detectBtn.disabled = !haveGeneration || generating;
  els.stopBtn.hidden = !generating;
  els.model.disabled = generating;
}

/** Rewrite the dropdown labels with the sizes for the active backend/dtype. */
function refreshModelOptions(backend) {
  for (const opt of els.model.options) {
    const m = modelInfo(opt.value);
    const dtype = m.webgpuOnly ? "q4f16" : backend?.dtype;
    const size = dtype ? sizeMB(opt.value, dtype) : null;
    const parts = [];
    if (size) parts.push(`~${formatMB(size)}`);
    if (m.webgpuOnly) parts.push("WebGPU only");
    opt.textContent = parts.length ? `${m.name} (${parts.join(", ")})` : m.name;
    opt.disabled = !!m.webgpuOnly && !!backend && backend.device !== "webgpu";
  }
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

function resetOutput() {
  clearOutput();
  els.placeholder.hidden = false;
}

/* ── worker messages ── */

function handleMessage(msg) {
  if (DEBUG) console.debug("[worker]", msg);
  const a = active;
  switch (msg.type) {
    case "hello":
      a.hello = { app: msg.app, transformers: msg.transformers, webgpuApi: msg.webgpuApi };
      a.phase = "backend";
      log.push("hello", a.hello);
      break;

    case "phase":
      a.phase = msg.phase;
      break;

    case "backend":
      a.backend = msg;
      lastBackend = msg;
      log.push("backend", { device: msg.device, dtype: msg.dtype, note: msg.note });
      refreshModelOptions(msg);
      break;

    case "status":
      setStatus(msg.text);
      break;

    case "progress": {
      els.progress.hidden = false;
      const pct = Math.round(msg.progress ?? 0);
      els.progressFill.style.width = `${pct}%`;
      a.bytes = { loaded: msg.loaded, total: msg.total };
      const mb = (b) => (b / 1e6).toFixed(0); // decimal MB, like the dropdown labels
      const parts = msg.files > 1 ? `, ${msg.files} files` : "";
      setStatus(`Downloading model: ${pct}% (${mb(msg.loaded)} / ${mb(msg.total)} MB${parts})`);
      break;
    }

    case "ready": {
      modelReady = true;
      a.phase = "ready";
      a.workerLog = msg.log;
      els.progress.hidden = true;
      els.model.value = msg.modelId; // programmatic .value does not fire "change"
      showBackendBadge(msg);
      const secs = Math.round((performance.now() - a.startedAt) / 100) / 10;
      log.push("ready", { modelId: msg.modelId, device: msg.device, dtype: msg.dtype, secs });
      if (a.retriedWasm) {
        setStatus("Model ready (loaded on WASM after WebGPU failed — slower).", "warn");
      } else {
        setStatus("Model ready.");
      }
      if (DEBUG) buildReport(null).then(showDiagnostics, console.error);
      updateButtons();
      break;
    }

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
      const p = readParams();
      els.promptEcho.textContent = p.promptStyle === "raw" ? msg.promptText : "";
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
      handleFailure({ phase: msg.phase, kind: msg.kind, raw: msg.raw, name: msg.name, stack: msg.stack, workerLog: msg.log });
      break;
  }
}

/** Uncaught error in the worker, or the worker script itself failed to load. */
function handleWorkerError(e) {
  const raw = e.message || "the worker script could not be loaded";
  const c = classifyError(raw, active?.backend?.device);
  const where = e.filename ? `${e.filename}:${e.lineno}:${e.colno}` : "";
  handleFailure({ phase: active?.phase ?? "starting", kind: c.kind, raw, name: "WorkerErrorEvent", stack: where, workerLog: active?.workerLog });
}

/* ── failure handling: WebGPU → WASM retry, otherwise explain + diagnostics ── */

async function handleFailure(f) {
  const a = active;
  if (!a) return;
  const failure = {
    ...f,
    modelId: a.modelId,
    requestedDevice: a.device ?? "auto",
    device: a.backend?.device ?? null,
    dtype: a.backend?.dtype ?? null,
    retriedWasm: !!a.retriedWasm,
    firstFailure: a.firstFailure ?? null,
    elapsedMs: Math.round(performance.now() - a.startedAt),
    bytes: a.bytes,
    hello: a.hello,
    backend: a.backend,
    workerLog: f.workerLog ?? a.workerLog,
  };
  log.push("failure", { phase: failure.phase, kind: failure.kind, raw: shorten(failure.raw, 200) });

  if (LOAD_PHASES.has(failure.phase) || !modelReady) {
    const name = modelInfo(a.modelId).name;
    const canRetryOnWasm =
      failure.device === "webgpu" &&
      !forcedDevice &&
      !a.retriedWasm &&
      !WEBGPU_ONLY.has(a.modelId) &&
      !["network", "unsupported"].includes(failure.kind);
    if (canRetryOnWasm) {
      log.push("retry-wasm");
      setStatus(`WebGPU failed while ${PHASE_TEXT[failure.phase] ?? failure.phase} ${name} — retrying on WASM…`, "warn");
      showNote(`WebGPU error (${failure.kind}): ${shorten(failure.raw)}`);
      startLoad({
        modelId: a.modelId,
        device: "wasm",
        dtype: forcedDtype,
        retriedWasm: true,
        firstFailure: { phase: failure.phase, kind: failure.kind, raw: failure.raw, dtype: failure.dtype, elapsedMs: failure.elapsedMs, bytes: failure.bytes },
      });
      return;
    }

    retireWorker();
    els.progress.hidden = true;
    updateButtons();
    setStatus(`Failed to load ${name} — gathering diagnostics…`, "error");
    let report = null;
    try {
      report = await buildReport(failure);
    } catch (err) {
      console.error("[watermarking-playground] diagnostics failed", err);
    }
    setStatus(describeFailure(failure, report), "error");
    showNote(adviceFor(failure, report));
    if (report) showDiagnostics(report);
    console.error("[watermarking-playground] load failed", report ?? failure);
    return;
  }

  if (failure.phase === "generate") {
    generating = false;
    streamSpan = null;
    if (failure.kind === "oom" && crashReloads < 1) {
      // The ONNX Runtime instance is dead after an abort: reload the same model in a fresh worker.
      crashReloads++;
      const name = modelInfo(a.modelId).name;
      log.push("reload-after-crash");
      setStatus(`The inference runtime crashed during generation (${shorten(failure.raw)}). Reloading ${name} — try fewer tokens.`, "warn");
      startLoad({ modelId: a.modelId, device: a.backend?.device ?? a.device, dtype: forcedDtype, keepOutput: true });
      return;
    }
    setStatus(`Generation failed: ${failure.raw}`, "error");
    updateButtons();
    return;
  }

  setStatus(`${failure.phase === "detect" ? "Detection" : "Operation"} failed: ${failure.raw}`, "error");
}

function describeFailure(f, report) {
  const name = modelInfo(f.modelId).name;
  const where = PHASE_TEXT[f.phase] ?? f.phase;
  const file = sizeMB(f.modelId, f.dtype);
  const need = estimatedLoadMB(f.modelId, f.dtype);
  const heap = report?.heapProbeMB;
  const on = f.device ? ` on ${f.device === "webgpu" ? "WebGPU" : "WASM"} (${f.dtype}${file ? `, ${formatMB(file)} file` : ""})` : "";
  switch (f.kind) {
    case "oom":
      return (
        `${name} did not fit in your browser's memory while ${where}${on}. ` +
        `Loading needs ≈${formatMB(need)} of WebAssembly memory (the model is held twice while the graph is built)` +
        (heap
          ? `, but this browser could only grow the WebAssembly heap to ≈${formatMB(heap)}.`
          : "; WebAssembly is capped at 4 GB and phones/tablets allow far less.") +
        " Pick a smaller model from the dropdown."
      );
    case "network":
      return `Could not download ${name} while ${where}: ${shorten(f.raw)}. Check your connection and any ad-blocker or proxy (files come from huggingface.co and cdn.jsdelivr.net), then reload the page.`;
    case "unsupported":
      return f.raw;
    case "webgpu":
      return (
        `WebGPU failed while ${where} ${name}: ${shorten(f.raw)}.` +
        (forcedDevice === "webgpu" ? " Remove ?device=webgpu from the URL to let the demo fall back to WASM." : "")
      );
    default:
      if (f.phase === "starting") {
        return (
          `The inference runtime failed to start${f.raw ? ` (${shorten(f.raw)})` : ""}. ` +
          "This usually means cdn.jsdelivr.net is blocked by an ad-blocker or proxy, or the browser is too old (WebAssembly SIMD and ES modules in workers are required)."
        );
      }
      return `Failed to load ${name} while ${where}${on}: ${shorten(f.raw)}`;
  }
}

function adviceFor(f, report) {
  const parts = [];
  if (f.firstFailure) {
    parts.push(`WebGPU was tried first and failed too (${f.firstFailure.kind}: ${shorten(f.firstFailure.raw, 120)}).`);
  }
  if (f.kind === "oom") {
    const dtype = f.dtype ?? "q8";
    const heap = report?.heapProbeMB;
    const candidates = MODELS.filter((m) => !m.webgpuOnly && m.id !== f.modelId)
      .map((m) => ({ name: m.name, need: estimatedLoadMB(m.id, dtype) }))
      .filter((c) => c.need != null && (!heap || c.need <= heap));
    if (candidates.length) {
      parts.push(`Models that should fit here: ${candidates.map((c) => `${c.name} (≈${formatMB(c.need)})`).join(", ")}.`);
    } else {
      parts.push("Even the smallest model may not fit in this browser; try a desktop browser with WebGPU (Chrome or Edge).");
    }
  }
  return parts.join(" ");
}

/* ── diagnostics ── */

async function buildReport(failure) {
  const environment = await collectEnvironment();
  const heapProbeMB = failure?.kind === "oom" || DEBUG ? probeWasmHeapMB() : null;
  const a = active;
  const modelId = failure?.modelId ?? a?.modelId ?? null;
  const dtype = failure?.dtype ?? a?.backend?.dtype ?? null;
  return {
    app: APP_VERSION,
    time: new Date().toISOString(),
    url: location.href,
    result: failure ? "failure" : "success",
    failure: failure
      ? { phase: failure.phase, kind: failure.kind, raw: failure.raw, name: failure.name, stack: failure.stack, elapsedMs: failure.elapsedMs, bytes: failure.bytes, firstFailure: failure.firstFailure }
      : null,
    attempt: {
      modelId,
      requestedDevice: failure?.requestedDevice ?? a?.device ?? "auto",
      device: failure?.device ?? a?.backend?.device ?? null,
      dtype,
      fileMB: sizeMB(modelId, dtype),
      estimatedLoadMB: estimatedLoadMB(modelId, dtype),
      retriedWasm: failure?.retriedWasm ?? !!a?.retriedWasm,
      elapsedMs: failure?.elapsedMs ?? (a ? Math.round(performance.now() - a.startedAt) : null),
    },
    runtime: failure?.hello ?? a?.hello ?? null,
    backend: failure?.backend ?? a?.backend ?? lastBackend,
    heapProbeMB,
    environment,
    log: log.entries(),
    workerLog: failure?.workerLog ?? a?.workerLog ?? null,
  };
}

function showDiagnostics(report) {
  const f = report.failure;
  const at = report.attempt;
  const head = [
    `LLM Watermarking Playground — diagnostics (build ${report.app}, ${report.time})`,
    f ? `FAILURE while ${f.phase} · kind=${f.kind} · ${shorten(f.raw, 300)}` : "SUCCESS",
    `model=${at.modelId} device=${at.device ?? "?"} (requested ${at.requestedDevice}) dtype=${at.dtype ?? "?"} file≈${formatMB(at.fileMB)} needs≈${formatMB(at.estimatedLoadMB)} elapsed=${at.elapsedMs} ms`,
    report.heapProbeMB != null ? `WebAssembly heap could grow to ≈${formatMB(report.heapProbeMB)}` : null,
    "",
  ]
    .filter((l) => l !== null)
    .join("\n");
  els.diagText.textContent = `${head}${JSON.stringify(report, null, 2)}`;
  els.diag.hidden = false;
  els.diag.open = !!f;
}

function hideDiagnostics() {
  els.diag.hidden = true;
  els.diag.open = false;
  els.diagText.textContent = "";
}

els.copyDiag.addEventListener("click", async () => {
  const label = els.copyDiag.textContent;
  try {
    await navigator.clipboard.writeText(els.diagText.textContent);
    els.copyDiag.textContent = "Copied ✓";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(els.diagText);
    const sel = getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    els.copyDiag.textContent = "Selected — press Ctrl/⌘+C";
  }
  setTimeout(() => {
    els.copyDiag.textContent = label;
  }, 2000);
});

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
  const p = readParams();
  if (!p.prompt.trim()) {
    setStatus("Enter a prompt first.", "error");
    return;
  }
  clearOutput();
  generating = true;
  updateButtons();
  setStatus(p.mode === "none" ? "Generating…" : `Generating with ${MODE_LABEL[p.mode]}…`);

  streamSpan = document.createElement("span");
  els.tokens.appendChild(streamSpan);
  els.promptEcho.textContent = p.promptStyle === "raw" ? p.prompt.trim() : "";

  send({ type: "generate", params: p });
});

els.stopBtn.addEventListener("click", () => send({ type: "interrupt" }));

els.detectBtn.addEventListener("click", () => {
  const { mode, gamma, h, m, redWords } = readParams();
  const scheme = mode === "tournament" ? "tournament" : "greenlist";
  send({ type: "detect", params: { scheme, gamma, h, m, redWords, key: els.detectKey.value } });
});

els.model.addEventListener("change", () => {
  const modelId = els.model.value;
  crashReloads = 0;
  log.push("user-select", { modelId });
  if (WEBGPU_ONLY.has(modelId) && forcedDevice !== "webgpu" && lastBackend && lastBackend.device !== "webgpu") {
    retireWorker();
    modelReady = false;
    haveGeneration = false;
    els.backendBadge.hidden = true;
    updateButtons();
    showNote("");
    hideDiagnostics();
    setStatus(`${modelInfo(modelId).name} needs WebGPU, which is not available in this browser (running on WASM). Pick a smaller model.`, "error");
    return;
  }
  startLoad({ modelId, device: forcedDevice, dtype: forcedDtype });
});

document.querySelectorAll('input[name="wmMode"]').forEach((r) => r.addEventListener("change", syncParamVisibility));

els.prompt.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !els.generateBtn.disabled) {
    els.generateBtn.click();
  }
});

/* ── init ── */
syncParamVisibility();
refreshModelOptions(lastBackend);
els.statusText.title = `build ${APP_VERSION}`;
updateButtons();
startLoad({ modelId: els.model.value, device: forcedDevice, dtype: forcedDtype });
