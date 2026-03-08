# Graviton UI

Chat interface for the [Graviton](https://github.com/opengraviton/graviton) engine — the open-source project that breaks AI free from cloud monopolies.

Pick a model up to 500B+ parameters, choose your compression settings, and start chatting — locally, on your own hardware. For large models, a progress bar shows real-time download and layer-by-layer loading progress.

## Features

- **One command** — `pip install graviton-ui && graviton-ui`
- **Suggested models** — from TinyLlama 1.1B to Qwen 72B and Llama 70B, with RAM requirements shown
- **Real-time streaming** — tokens appear as they're generated with a live tok/s counter
- **Loading progress bar** — percentage, layer count, elapsed time, and ETA for large models
- **Cancel loading** — stop a model load at any time
- **Quantization settings** — FP16, INT8, INT4, Ternary, Mixed-Precision
- **Speculative decoding** — toggle faster generation
- **Conversation history** — multi-turn chat with system prompt support
- **Dark theme** — easy on the eyes

## Install

```bash
pip install graviton-ui && graviton-ui
```

Your browser opens at `http://localhost:7860`. Pick a model, choose quantization, start chatting.

### Options

```
graviton-ui --help
  --port PORT       Port to serve on (default: 7860)
  --host HOST       Host to bind to (default: 127.0.0.1)
  --no-browser      Don't auto-open the browser
```

### For AI Agents

Agents don't need a UI. The headless API server lives in the engine package:

```bash
pip install "graviton-ai[api]" && graviton-api
```

See the [Graviton engine README](https://github.com/opengraviton/graviton#for-ai-agents) for API documentation.

## How It Works

```
Browser                    FastAPI Server              Graviton Engine
  │                            │                            │
  │── POST /api/models/load ──▶│── GravitonEngine() ──────▶│
  │◀── { status: loading } ────│   .progress_callback ──▶ load_stage
  │                            │   .load_model()            │
  │── GET /api/models/status ─▶│                            │
  │◀── { progress: 0.56 } ────│   "Loading layer 42/80"   │
  │                            │                            │
  │── POST /api/models/cancel ▶│── cancel_requested ───────▶│ (stops)
  │                            │                            │
  │── POST /api/chat ─────────▶│── engine.generate() ─────▶│
  │◀── SSE: token, tps ────────│◀── yield chunk ───────────│
```

1. **Load**: You pick a model and settings. The backend downloads it, streams and compresses each layer, and reports progress. You can cancel at any time.

2. **Chat**: Messages are sent with conversation history. The engine generates tokens and streams them back in real-time.

## Requirements

- Python >= 3.9
- Graviton (`graviton-ai` package)
- A HuggingFace account + token (for gated models)

## License

Apache-2.0
