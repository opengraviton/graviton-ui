"""Graviton UI — FastAPI backend."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("graviton-ui")

app = FastAPI(title="Graviton UI", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"

app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")


# ── Engine state ────────────────────────────────────────────────────

class _CancelledError(Exception):
    pass


class _EngineState:
    def __init__(self):
        self.engine = None
        self.model_id: Optional[str] = None
        self.loading: bool = False
        self.load_stage: str = ""
        self.load_progress: float = 0.0
        self.load_current_layer: int = 0
        self.load_total_layers: int = 0
        self.error: Optional[str] = None
        self.gen_lock = threading.Lock()
        self.config_summary: dict = {}
        self._load_start_time: float = 0.0
        self._cancel_requested: bool = False

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def reset(self):
        self.engine = None
        self.model_id = None
        self.loading = False
        self.load_stage = ""
        self.load_progress = 0.0
        self.load_current_layer = 0
        self.load_total_layers = 0
        self.error = None
        self.config_summary = {}
        self._load_start_time = 0.0
        self._cancel_requested = False


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
    state._cancel_requested = False
    state.load_stage = "Initializing..."
    state.load_progress = 0.0
    state.load_current_layer = 0
    state.load_total_layers = 0
    state._load_start_time = time.time()

    _layer_re = re.compile(r"Loading layer (\d+)/(\d+)")
    _dl_re = re.compile(r"(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*GB")

    def _on_progress(msg: str):
        if state._cancel_requested:
            raise _CancelledError("Loading cancelled by user")
        state.load_stage = msg
        m = _layer_re.search(msg)
        dm = _dl_re.search(msg)
        if m:
            current, total = int(m.group(1)), int(m.group(2))
            state.load_current_layer = current
            state.load_total_layers = total
            state.load_progress = 0.05 + (current / total) * 0.90
        elif dm:
            done_gb, total_gb = float(dm.group(1)), float(dm.group(2))
            if total_gb > 0:
                state.load_progress = 0.01 + (done_gb / total_gb) * 0.04
        elif "Downloading" in msg:
            state.load_progress = max(state.load_progress, 0.01)
        elif "Building model skeleton" in msg or "Building inference" in msg:
            state.load_progress = 0.05
        elif "Loading embeddings" in msg:
            state.load_progress = 0.06
        elif "Moving model to device" in msg:
            state.load_progress = 0.70
        elif "Applying" in msg and "quantization" in msg:
            state.load_progress = 0.80
        elif "Model ready" in msg:
            state.load_progress = 1.0

    def _check_cancel():
        if state._cancel_requested:
            raise _CancelledError("Cancelled")

    def _load():
        try:
            from graviton.core.config import GravitonConfig, QuantMode
            from graviton.core.engine import GravitonEngine

            if req.hf_token:
                os.environ["HF_TOKEN"] = req.hf_token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = req.hf_token

            _check_cancel()
            state.load_stage = "Building config..."
            state.load_progress = 0.01
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

            _check_cancel()
            state.load_stage = "Creating engine..."
            state.load_progress = 0.02
            engine = GravitonEngine(config=config)
            engine.progress_callback = _on_progress

            _check_cancel()
            state.load_stage = "Downloading model files..."
            state.load_progress = 0.03
            engine.load_model()

            _check_cancel()

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
            state.loading = False
            state.load_stage = ""
        except _CancelledError:
            logger.info("Model loading cancelled by user")
        except Exception as exc:
            if not state._cancel_requested:
                logger.exception("Model load failed")
                state.error = str(exc)
                state.loading = False
                state.load_stage = ""

    threading.Thread(target=_load, daemon=True).start()
    return {"status": "loading", "model_id": req.model_id}


@app.get("/api/models/status")
async def model_status():
    elapsed = 0.0
    if state.loading and state._load_start_time:
        elapsed = time.time() - state._load_start_time
    return {
        "loading": state.loading,
        "load_stage": state.load_stage,
        "load_progress": round(state.load_progress, 4),
        "load_current_layer": state.load_current_layer,
        "load_total_layers": state.load_total_layers,
        "load_elapsed": round(elapsed, 1),
        "loaded": state.loaded,
        "model_id": state.model_id,
        "error": state.error,
        "config": state.config_summary,
    }


@app.post("/api/models/cancel")
async def cancel_loading():
    if not state.loading:
        return {"status": "not_loading"}
    state._cancel_requested = True
    state.loading = False
    state.load_stage = ""
    return {"status": "cancelled"}


@app.post("/api/models/unload")
async def unload_model():
    if state.loading:
        state._cancel_requested = True
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
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("Client disconnected during streaming")
        except Exception as exc:
            try:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        finally:
            state.gen_lock.release()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Helpers ─────────────────────────────────────────────────────────

def _format_prompt(system: str, history: list, message: str) -> str:
    """Format using ChatML template (TinyLlama-Chat, Mistral-Instruct, etc.)."""
    parts = []
    sys_text = system or "You are a friendly and helpful assistant."
    parts.append(f"<|system|>\n{sys_text}</s>")
    for msg in history:
        role = msg.get("role", "user")
        parts.append(f"<|{role}|>\n{msg['content']}</s>")
    parts.append(f"<|user|>\n{message}</s>")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)
