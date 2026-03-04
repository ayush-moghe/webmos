// src/dnsmos.ts
import * as ort from "onnxruntime-web";
var ORT_CDN_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/";
var MODEL_CDN_URL = "https://cdn.jsdelivr.net/npm/webmos/models/sig_bak_ovr.onnx";
var SAMPLING_RATE = 16e3;
function polyfitSig(x) {
  return -0.08397278 * x * x + 1.22083953 * x + 52439e-7;
}
function polyfitBak(x) {
  return -0.13166888 * x * x + 1.60915514 * x - 0.39604546;
}
function polyfitOvr(x) {
  return -0.06766283 * x * x + 1.11546468 * x + 0.04602535;
}
function resample(audio, fromSr, toSr) {
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
var session = null;
var initPromise = null;
async function initDNSMOS(options) {
  if (session) return;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const opts = options ?? {};
    ort.env.wasm.wasmPaths = opts.wasmPaths ?? ORT_CDN_BASE;
    ort.env.logLevel = opts.ortLogLevel ?? "error";
    session = await ort.InferenceSession.create(
      opts.modelUrl ?? MODEL_CDN_URL,
      { executionProviders: ["wasm"] }
    );
  })();
  return initPromise;
}
async function runDNSMOS(audioData, sampleRate) {
  if (!session) {
    await initDNSMOS();
  }
  const audio = sampleRate !== SAMPLING_RATE ? resample(audioData, sampleRate, SAMPLING_RATE) : audioData;
  const input = new ort.Tensor("float32", audio, [1, audio.length]);
  const result = await session.run({ input_1: input });
  const out = result["Identity:0"]?.data ?? result[Object.keys(result)[0]]?.data;
  return {
    mos_sig: polyfitSig(out[0]),
    mos_bak: polyfitBak(out[1]),
    mos_ovr: polyfitOvr(out[2])
  };
}
export {
  initDNSMOS,
  runDNSMOS
};
//# sourceMappingURL=index.js.map