# Graviton UI

Beautiful chat interface for the [Graviton](https://github.com/opengraviton/graviton) AI inference engine.

Enter a HuggingFace model ID, pick your quantization settings, and start chatting — all from a sleek dark-themed web UI running on your local machine. For large models (70B+), the UI shows real-time layer-by-layer loading progress as Graviton streams and quantizes each transformer layer.

## Features

- **One-click model loading** — paste a HuggingFace model ID and token, hit Load
- **Real-time streaming** — tokens stream in via SSE with a live tok/s counter
- **Layer-by-layer loading progress** — watch "Loading layer 42/80..." as Graviton streams large models
- **Full quantization control** — FP16, INT8, INT4, Ternary, Mixed-Precision
- **Speculative decoding toggle** — enable layer-skip draft acceleration
- **Conversation history** — multi-turn chat with system prompt support
- **Dark theme** — easy on the eyes, purple Graviton accent

## One-Command Install

### For Humans

```bash
pip install graviton-ui && graviton-ui
```

One command. Installs the engine, quantization stack, HuggingFace integration, and the chat UI. Browser opens at `http://localhost:7860`.

### For AI Agents

Agents don't need a UI. The headless API server lives in the engine package:

```bash
pip install "graviton-ai[api]" && graviton-api
```

See the [Graviton engine README](https://github.com/opengraviton/graviton#for-ai-agents) for full API documentation.

### Options

```
graviton-ui --help
  --port PORT       Port to serve on (default: 7860)
  --host HOST       Host to bind to (default: 127.0.0.1)
  --no-browser      Don't auto-open the browser
```

## How It Works

```
Browser (index.html)          FastAPI (server.py)           Graviton Engine
       │                            │                            │
       │── POST /api/models/load ──▶│── GravitonEngine() ──────▶│
       │◀── { status: loading } ────│   .progress_callback ──▶ state.load_stage
       │                            │   .load_model()            │
       │── GET /api/models/status ─▶│                            │
       │◀── { load_stage: "..." } ──│   "Loading layer 42/80"   │
       │                            │                            │
       │── GET /api/models/status ─▶│                            │
       │◀── { loaded: true } ───────│                            │
       │                            │                            │
       │── POST /api/chat ─────────▶│── engine.generate() ─────▶│
       │◀── SSE: token, tps ────────│◀── yield chunk ───────────│
       │◀── SSE: done, stats ───────│                            │
```

1. **Load**: The UI posts your model ID + settings to `/api/models/load`. The backend creates a `GravitonEngine` with a progress callback wired to `state.load_stage`, downloads weights from HuggingFace, and — for large models — streams each transformer layer from safetensors shards, quantizes in-flight, and frees the FP16 originals. Status is polled via `/api/models/status`.

2. **Chat**: Each message is posted to `/api/chat` with conversation history. The backend formats a prompt using the ChatML template, calls `engine.generate(stream=True)`, and streams tokens back as Server-Sent Events.

3. **Stream**: The frontend reads the SSE stream, renders markdown in real-time with [marked.js](https://marked.js.org/), and displays a live tokens/second counter. Broken pipe errors from client disconnects are handled gracefully.

## Tech Stack

| Layer     | Technology                  |
|-----------|-----------------------------|
| Backend   | FastAPI + Uvicorn           |
| Frontend  | Vanilla JS + Tailwind CSS   |
| Markdown  | marked.js                   |
| Streaming | Server-Sent Events (SSE)    |
| Engine    | Graviton (QuantizedLinear)  |

## Requirements

- Python >= 3.9
- Graviton (`graviton-ai` package)
- A HuggingFace account + token (for gated models)

## License

Apache-2.0
