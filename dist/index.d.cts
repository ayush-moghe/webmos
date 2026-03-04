/**
 * DNSMOS (Deep Noise Suppression Mean Opinion Score) — browser inference via ONNX Runtime Web.
 *
 * Uses a single ONNX model:
 *   sig_bak_ovr.onnx — input: raw audio [N, samples] → output [N, 3] (sig, bak, ovr)
 *
 * Zero-config: model and WASM files are loaded from CDN by default.
 */
/** Options for configuring DNSMOS inference. All fields are optional. */
interface DNSMOSOptions {
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
declare function initDNSMOS(options?: DNSMOSOptions): Promise<void>;
/** DNSMOS prediction scores. */
interface DNSMOSResult {
    /** Signal quality (1–5) */
    mos_sig: number;
    /** Background noise quality (1–5) */
    mos_bak: number;
    /** Overall quality (1–5) */
    mos_ovr: number;
}
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
declare function runDNSMOS(audioData: Float32Array, sampleRate: number): Promise<DNSMOSResult>;

export { type DNSMOSOptions, type DNSMOSResult, initDNSMOS, runDNSMOS };
