"""Graviton UI — FastAPI backend."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("graviton-ui")

app = FastAPI(title="Graviton UI", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"


# ── Engine state ────────────────────────────────────────────────────

class _EngineState:
    def __init__(self):
        self.engine = None
        self.model_id: Optional[str] = None
        self.loading: bool = False
        self.load_stage: str = ""
        self.error: Optional[str] = None
        self.gen_lock = threading.Lock()
        self.config_summary: dict = {}

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def reset(self):
        self.engine = None
        self.model_id = None
        self.loading = False
        self.load_stage = ""
        self.error = None
        self.config_summary = {}


state = _EngineState()


# ── Request / response models ──────────────────────────────────────

class LoadRequest(BaseModel):
    model_id: str
    hf_token: str = ""
    bits: float = 4.0
    no_quantize: bool = False
    no_mixed: bool = False
    speculative: bool = False
    spec_tokens: int = 4


class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 256
    system_prompt: str = ""
    history: list = []


# ── Routes ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/models/load")
async def load_model(req: LoadRequest):
    if state.loading:
        raise HTTPException(409, "A model is already loading")

    if state.loaded:
        state.reset()

    state.loading = True
    state.error = None
    state.load_stage = "Initializing..."

    def _load():
        try:
            from graviton.core.config import GravitonConfig, QuantMode
            from graviton.core.engine import GravitonEngine

            if req.hf_token:
                os.environ["HF_TOKEN"] = req.hf_token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = req.hf_token

            state.load_stage = "Building config..."
            config = GravitonConfig(
                model_path=req.model_id,
                quant_bits=req.bits,
                memory=GravitonConfig().memory,
                decoding=GravitonConfig().decoding,
                use_speculative=req.speculative,
            )

            if req.no_quantize:
                config.quantization.mode = QuantMode.NONE
            if req.no_mixed:
                config.quantization.use_mixed_precision = False
            if req.speculative:
                config.decoding.num_speculative_tokens = req.spec_tokens

            state.load_stage = "Creating engine..."
            engine = GravitonEngine(config=config)

            state.load_stage = "Downloading & loading weights..."
            engine.load_model()

            if req.no_quantize:
                qlabel = "FP16"
            elif req.no_mixed:
                qlabel = f"INT{int(req.bits)}"
            else:
                qlabel = f"Mixed (critical=8bit, other={int(req.bits)}bit)"

            state.engine = engine
            state.model_id = req.model_id
            state.config_summary = {
                "quantization": qlabel,
                "speculative": req.speculative,
                "bits": req.bits,
            }
        except Exception as exc:
            logger.exception("Model load failed")
            state.error = str(exc)
        finally:
            state.loading = False
            state.load_stage = ""

    threading.Thread(target=_load, daemon=True).start()
    return {"status": "loading", "model_id": req.model_id}


@app.get("/api/models/status")
async def model_status():
    return {
        "loading": state.loading,
        "load_stage": state.load_stage,
        "loaded": state.loaded,
        "model_id": state.model_id,
        "error": state.error,
        "config": state.config_summary,
    }


@app.post("/api/models/unload")
async def unload_model():
    state.reset()
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not state.loaded:
        raise HTTPException(400, "No model loaded")

    prompt = _format_prompt(req.system_prompt, req.history, req.message)

    engine = state.engine
    engine.config.decoding.temperature = req.temperature
    engine.config.decoding.max_tokens = req.max_tokens

    def generate():
        token_count = 0
        start = time.time()
        acquired = state.gen_lock.acquire(timeout=30)
        if not acquired:
            yield f"data: {json.dumps({'error': 'Another generation is in progress'})}\n\n"
            return
        try:
            for chunk in engine.generate(prompt, stream=True):
                token_count += 1
                elapsed = time.time() - start
                tps = token_count / max(elapsed, 0.001)
                yield f"data: {json.dumps({'token': chunk, 'tps': round(tps, 1)})}\n\n"

            elapsed = time.time() - start
            tps = token_count / max(elapsed, 0.001)
            yield f"data: {json.dumps({'done': True, 'total_tokens': token_count, 'elapsed': round(elapsed, 2), 'tps': round(tps, 1)})}\n\n"
        except GeneratorExit:
            pass
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            state.gen_lock.release()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Helpers ─────────────────────────────────────────────────────────

def _format_prompt(system: str, history: list, message: str) -> str:
    parts = []
    if system:
        parts.append(f"System: {system}\n")
    for msg in history:
        role = msg.get("role", "user").capitalize()
        parts.append(f"{role}: {msg['content']}")
    parts.append(f"User: {message}")
    parts.append("Assistant:")
    return "\n".join(parts)
