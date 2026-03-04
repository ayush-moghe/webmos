"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/index.ts
var index_exports = {};
__export(index_exports, {
  initDNSMOS: () => initDNSMOS,
  runDNSMOS: () => runDNSMOS
});
module.exports = __toCommonJS(index_exports);

// src/dnsmos.ts
var ort = __toESM(require("onnxruntime-web"), 1);
var ORT_CDN_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/";
var MODEL_CDN_URL = "https://cdn.jsdelivr.net/npm/webmos/models/sig_bak_ovr.onnx";
var SAMPLING_RATE = 16e3;
var INPUT_LENGTH = 9.01;
var LEN_SAMPLES = Math.round(INPUT_LENGTH * SAMPLING_RATE);
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
  let audio = sampleRate !== SAMPLING_RATE ? resample(audioData, sampleRate, SAMPLING_RATE) : audioData;
  while (audio.length < LEN_SAMPLES) {
    const combined = new Float32Array(audio.length * 2);
    combined.set(audio);
    combined.set(audio, audio.length);
    audio = combined;
  }
  const numWindows = Math.max(1, Math.floor(audio.length / LEN_SAMPLES));
  const avg = [0, 0, 0];
  for (let idx = 0; idx < numWindows; idx++) {
    const start = idx * LEN_SAMPLES;
    const end = start + LEN_SAMPLES;
    if (end > audio.length) break;
    const segment = audio.slice(start, end);
    const input = new ort.Tensor("float32", segment, [1, LEN_SAMPLES]);
    const result = await session.run({ input_1: input });
    const out = result["Identity:0"]?.data ?? result[Object.keys(result)[0]]?.data;
    avg[0] += out[0];
    avg[1] += out[1];
    avg[2] += out[2];
  }
  avg[0] /= numWindows;
  avg[1] /= numWindows;
  avg[2] /= numWindows;
  return {
    mos_sig: polyfitSig(avg[0]),
    mos_bak: polyfitBak(avg[1]),
    mos_ovr: polyfitOvr(avg[2])
  };
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  initDNSMOS,
  runDNSMOS
});
//# sourceMappingURL=index.cjs.map