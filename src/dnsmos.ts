/**
 * DNSMOS (Deep Noise Suppression Mean Opinion Score) — browser inference via ONNX Runtime Web.
 *
 * Uses a single ONNX model:
 *   sig_bak_ovr.onnx — input: raw audio [N, samples] → output [N, 3] (sig, bak, ovr)
 *
 * Zero-config: model and WASM files are loaded from CDN by default.
 */

import * as ort from "onnxruntime-web";

// ─── CDN defaults ────────────────────────────────────────────────────────────
const ORT_CDN_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/";
const MODEL_CDN_URL = "https://cdn.jsdelivr.net/npm/webmos/models/sig_bak_ovr.onnx";

// ─── Constants ───────────────────────────────────────────────────────────────
const SAMPLING_RATE = 16_000;

// ─── Polyfit coefficients (non-personalized) ─────────────────────────────────
function polyfitSig(x: number): number {
  return -0.08397278 * x * x + 1.22083953 * x + 0.0052439;
}
function polyfitBak(x: number): number {
  return -0.13166888 * x * x + 1.60915514 * x - 0.39604546;
}
function polyfitOvr(x: number): number {
  return -0.06766283 * x * x + 1.11546468 * x + 0.04602535;
}

// ─── Linear interpolation resampler ──────────────────────────────────────────
function resample(
  audio: Float32Array,
  fromSr: number,
  toSr: number,
): Float32Array {
  if (fromSr === toSr) return audio;
  const ratio = toSr / fromSr;
  const newLen = Math.round(audio.length * ratio);
  const out = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const srcIdx = i / ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, audio.length - 1);
    const frac = srcIdx - lo;
    out[i] = audio[lo] * (1 - frac) + audio[hi] * frac;
  }
  return out;
}

// ─── Session cache & init lock ───────────────────────────────────────────────
let session: ort.InferenceSession | null = null;
let initPromise: Promise<void> | null = null;

// ─── Configuration ───────────────────────────────────────────────────────────

/** Options for configuring DNSMOS inference. All fields are optional. */
export interface DNSMOSOptions {
  /**
   * URL to `sig_bak_ovr.onnx`.
   * @default CDN URL (jsdelivr)
   */
  modelUrl?: string;

  /**
   * ONNX Runtime WASM file paths prefix.
   * @default CDN URL (jsdelivr)
   */
  wasmPaths?: string;

  /**
   * ONNX Runtime log level.
   * @default "error"
   */
  ortLogLevel?: "verbose" | "info" | "warning" | "error" | "fatal";
}

/**
 * Initialise ONNX Runtime and pre-load the DNSMOS model session.
 *
 * This is called automatically on the first `runDNSMOS()` call if not
 * called explicitly. Call it ahead of time if you want to pre-warm the model.
 */
export async function initDNSMOS(options?: DNSMOSOptions): Promise<void> {
  if (session) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const opts = options ?? {};
    ort.env.wasm.wasmPaths = opts.wasmPaths ?? ORT_CDN_BASE;
    ort.env.logLevel = opts.ortLogLevel ?? "error";

    session = await ort.InferenceSession.create(
      opts.modelUrl ?? MODEL_CDN_URL,
      { executionProviders: ["wasm"] },
    );
  })();

  return initPromise;
}

// ─── Result type ─────────────────────────────────────────────────────────────

/** DNSMOS prediction scores. */
export interface DNSMOSResult {
  /** Signal quality (1–5) */
  mos_sig: number;
  /** Background noise quality (1–5) */
  mos_bak: number;
  /** Overall quality (1–5) */
  mos_ovr: number;
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Run DNSMOS inference on raw audio samples.
 *
 * Automatically initialises the model on first call (loads from CDN).
 * No setup required.
 *
 * @param audioData  PCM float32 samples (mono, any sample rate)
 * @param sampleRate Sample rate of the input audio
 * @returns DNSMOS quality scores
 *
 * @example
 * ```ts
 * import { runDNSMOS } from "webmos";
 *
 * const result = await runDNSMOS(audioFloat32, 16000);
 * console.log(result.mos_ovr); // e.g. 3.42
 * ```
 */
export async function runDNSMOS(
  audioData: Float32Array,
  sampleRate: number,
): Promise<DNSMOSResult> {
  // Auto-init if needed
  if (!session) {
    await initDNSMOS();
  }

  // Resample to 16 kHz if needed
  const audio =
    sampleRate !== SAMPLING_RATE
      ? resample(audioData, sampleRate, SAMPLING_RATE)
      : audioData;

  // Run inference on the full audio
  const input = new ort.Tensor("float32", audio, [1, audio.length]);
  const result = await session!.run({ input_1: input });
  const out =
    (result["Identity:0"]?.data as Float32Array) ??
    (result[Object.keys(result)[0]]?.data as Float32Array);

  // Apply polyfit
  return {
    mos_sig: polyfitSig(out[0]),
    mos_bak: polyfitBak(out[1]),
    mos_ovr: polyfitOvr(out[2]),
  };
}
