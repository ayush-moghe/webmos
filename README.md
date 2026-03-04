# webmos

Run [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) (Mean Opinion Score) entirely in the browser using [ONNX Runtime Web](https://onnxruntime.ai/).

No server, no setup, no file copying — just import and go. Model and WASM files load from CDN automatically.

See it in action: [webmos-demo](https://github.com/ayush-moghe/webmos-demo)

## Install

```bash
npm install webmos
```

## Usage

```ts
import { runDNSMOS } from "webmos";

const result = await runDNSMOS(audioFloat32Array, sampleRate);

console.log(result);
// {
//   mos_sig:  3.82,   — Signal quality (1–5)
//   mos_bak:  4.12,   — Background noise quality (1–5)
//   mos_ovr:  3.42,   — Overall quality (1–5)
// }
```

That's it. The model loads automatically on the first call.

## Pre-warming (optional)

If you want to load the model ahead of time (e.g. while the user is selecting a file):

```ts
import { initDNSMOS } from "webmos";

await initDNSMOS(); // pre-loads model from CDN
```

## API

### `runDNSMOS(audioData: Float32Array, sampleRate: number): Promise<DNSMOSResult>`

Run DNSMOS inference on raw PCM audio. Auto-initialises on first call.

- **audioData** — Mono float32 samples (any sample rate; resampled to 16 kHz internally)
- **sampleRate** — Sample rate of the input audio

Returns:

```ts
interface DNSMOSResult {
  mos_sig: number;  // Signal quality (1–5)
  mos_bak: number;  // Background noise quality (1–5)
  mos_ovr: number;  // Overall quality (1–5)
}
```

### `initDNSMOS(options?: DNSMOSOptions): Promise<void>`

Optional. Pre-load the model. Called automatically by `runDNSMOS()` if not called explicitly.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelUrl` | `string` | jsdelivr CDN | URL to `sig_bak_ovr.onnx` |
| `wasmPaths` | `string` | jsdelivr CDN | ONNX Runtime WASM file URL prefix |
| `ortLogLevel` | `string` | `"error"` | Log level: `"verbose"`, `"info"`, `"warning"`, `"error"`, `"fatal"` |

## How It Works

1. Audio is resampled to 16 kHz if needed
2. Raw audio is fed directly to `sig_bak_ovr.onnx` running in WebAssembly
3. Model outputs are calibrated via polynomial fitting to produce MOS scores

No DSP preprocessing — the model takes raw audio directly.

## License

ISC
