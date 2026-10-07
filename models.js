/**
 * Shared, dependency-free helpers used by both main.js (page) and worker.js.
 *
 * The model catalogue, backend/dtype selection, error classification and the
 * diagnostics helpers live here so both threads agree on them.
 */

/** Build stamp shown in diagnostics, so a stale cached page is recognisable at once. */
export const APP_VERSION = "2026-09-03.1";

/**
 * Dropdown models, smallest first. `sizesMB` are the on-disk ONNX file sizes
 * per dtype (from the Hugging Face Hub), which is what gets downloaded.
 */
export const MODELS = [
  {
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    name: "SmolLM2-135M-Instruct",
    sizesMB: { q4f16: 118, q4: 182, q8: 137, fp16: 270, fp32: 540 },
  },
  {
    id: "HuggingFaceTB/SmolLM2-360M-Instruct",
    name: "SmolLM2-360M-Instruct",
    sizesMB: { q4f16: 273, q4: 388, q8: 365, fp16: 725, fp32: 1450 },
  },
  {
    id: "onnx-community/Qwen3-0.6B-ONNX",
    name: "Qwen3-0.6B",
    sizesMB: { q4f16: 570, q4: 919, q8: 618, fp16: 1203, fp32: 2405 },
  },
];

/** Models whose weights exceed what the 32-bit WASM backend can hold. */
export const WEBGPU_ONLY = new Set(MODELS.filter((m) => m.webgpuOnly).map((m) => m.id));

/** dtypes transformers.js understands (used to validate the `?dtype=` override). */
export const DTYPES = ["fp32", "fp16", "q8", "int8", "uint8", "q4", "bnb4", "q4f16"];

export function modelInfo(id) {
  return MODELS.find((m) => m.id === id) ?? { id, name: String(id).split("/").pop(), sizesMB: {} };
}

/**
 * Which quantisation to load for a backend.
 *   webgpu → q4f16 (4-bit weights, fp16 activations), or q4 (fp32 activations) when the
 *            adapter lacks the `shader-f16` feature (transformers.js does not check this for q4f16);
 *   wasm   → q8, transformers.js's own default for WASM: smaller than q4 for these models
 *            (q4 keeps the large embedding/lm_head in fp32) and faster on int8 CPU kernels.
 */
export function dtypeFor(device, f16, forced) {
  if (forced && DTYPES.includes(forced)) return forced;
  return device === "webgpu" ? (f16 ? "q4f16" : "q4") : "q8";
}

export function sizeMB(modelId, dtype) {
  return modelInfo(modelId).sizesMB[dtype] ?? null;
}

/**
 * Rough peak WASM-heap need while ONNX Runtime builds a session: the model bytes are
 * held twice (raw buffer + parsed graph) plus runtime overhead. This applies to WebGPU
 * too, because the model is parsed in WASM before the weights are uploaded to the GPU.
 */
export function estimatedLoadMB(modelId, dtype) {
  const s = sizeMB(modelId, dtype);
  return s == null ? null : 2 * s + 150;
}

export function formatMB(mb) {
  if (mb == null || !Number.isFinite(mb)) return "?";
  return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

/* ── error classification ── */

export function errorText(err) {
  if (err == null) return "unknown error";
  if (typeof err === "number" || typeof err === "string") return String(err);
  return err.message ?? String(err);
}

const NETWORK_RE = /failed to fetch|networkerror|network error|could not locate file|\b40[134]\b|unauthorized|load failed|ERR_(INTERNET|NETWORK|CONNECTION|NAME|PROXY)/i;
// "Aborted()" is Emscripten's abort inside ONNX Runtime's WASM binary — in practice the
// out-of-memory symptom of the 32-bit heap. A bare number is an uncaught C++ exception pointer.
const OOM_RE = /Aborted\(|RuntimeError|out of memory|offset is out of bounds|bad_alloc|Failed to allocate|memory access out of bounds|unreachable|Cannot enlarge memory|allocation failed/i;
const WEBGPU_RE = /webgpu|gpu adapter|shader|device lost|GPUDevice|GPUValidationError|no available backend|Unsupported device|requestDevice/i;

/**
 * Classify a load/generation failure so the UI can decide what to do.
 * kinds: "oom" | "webgpu" | "network" | "unsupported" | "other"
 */
export function classifyError(err, device) {
  const raw = errorText(err);
  let kind = typeof err === "object" && err?.kind ? err.kind : null;
  if (!kind) {
    if (NETWORK_RE.test(raw)) kind = "network";
    else if (typeof err === "number" || /^\d+$/.test(raw.trim()) || OOM_RE.test(raw)) kind = "oom";
    else if (WEBGPU_RE.test(raw) || device === "webgpu") kind = "webgpu";
    else kind = "other";
  }
  const stack = typeof err?.stack === "string" ? err.stack.split("\n").slice(0, 6).join("\n") : "";
  return { kind, raw, name: err?.name ?? typeof err, stack };
}

/* ── diagnostics helpers ── */

/** Small ring buffer of timestamped events (ms since the log was created). */
export function createLog(limit = 60) {
  const entries = [];
  const t0 = performance.now();
  return {
    push(event, data) {
      entries.push({ t: Math.round(performance.now() - t0), event, ...(data !== undefined ? { data } : {}) });
      if (entries.length > limit) entries.shift();
    },
    entries: () => entries.slice(),
  };
}

/** Serialisable summary of a WebGPU adapter (`adapter.info`; the old `adapter.isFallbackAdapter` is gone). */
export function describeAdapter(adapter) {
  const info = adapter.info ?? {};
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false,
    shaderF16: adapter.features.has("shader-f16"),
    subgroups: adapter.features.has("subgroups"),
    maxBufferSize: adapter.limits?.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize,
  };
}

// Minimal module using i8x16.splat/popcnt: validates only where WASM SIMD is supported
// (ONNX Runtime Web 1.22 ships SIMD-only binaries).
const SIMD_MODULE = Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

/** Snapshot of the browser environment, usable from the page and from a worker. */
export async function collectEnvironment() {
  const nav = globalThis.navigator ?? {};
  const uad = nav.userAgentData;
  let wasmSimd = false;
  try { wasmSimd = WebAssembly.validate(SIMD_MODULE); } catch { /* no WebAssembly at all */ }
  const out = {
    thread: typeof window === "undefined" ? "worker" : "page",
    userAgent: nav.userAgent,
    platform: uad?.platform ?? nav.platform,
    brands: uad?.brands?.map((b) => `${b.brand} ${b.version}`).join(", "),
    mobile: uad?.mobile,
    hardwareConcurrency: nav.hardwareConcurrency,
    deviceMemoryGB: nav.deviceMemory, // Chromium only, bucketed
    crossOriginIsolated: globalThis.crossOriginIsolated ?? false,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    wasmSimd,
    cacheApi: typeof caches !== "undefined",
    webgpuApi: !!nav.gpu,
  };
  if (nav.gpu) {
    try {
      const adapter = await nav.gpu.requestAdapter();
      out.gpuAdapter = adapter ? describeAdapter(adapter) : null;
    } catch (e) {
      out.gpuAdapter = { error: errorText(e) };
    }
  }
  return out;
}

/**
 * How far this browser lets a WebAssembly memory grow, in MB. Mirrors ONNX Runtime's own
 * memory (shared, 4 GB maximum, grown on demand) — growth refusal is exactly what turns into
 * "Aborted()" during model loading. Only run after a failure (or in debug mode): it briefly
 * reserves address space. Returns null if no memory could be created at all.
 */
export function probeWasmHeapMB() {
  const MAX_PAGES = 65536; // 4 GB of 64 KB pages
  const STEP = 4096; // 256 MB
  let mem = null;
  for (const shared of [true, false]) {
    try {
      mem = new WebAssembly.Memory({ initial: 256, maximum: MAX_PAGES, shared });
      break;
    } catch { /* try the other flavour */ }
  }
  if (!mem) return null;
  let pages = 256;
  let step = STEP; // 256 MB steps, halved down to 1 MB once growth is refused
  while (step >= 16 && pages < MAX_PAGES) {
    const delta = Math.min(step, MAX_PAGES - pages);
    try {
      mem.grow(delta);
      pages += delta;
    } catch {
      step = Math.floor(step / 2); // growth refused: narrow down the limit
    }
  }
  mem = null;
  return Math.round(pages / 16);
}
