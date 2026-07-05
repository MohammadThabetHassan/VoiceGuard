#!/usr/bin/env python3
"""Persistent *warm* voice-cloning server — loads the model once, serves many.

Runs inside an engine's isolated venv (like ``clone_worker.py``) so it imports
only that engine's library plus stdlib. The API process POSTs
``{"text", "ref", "out", "language"}`` to ``/generate``; because the model stays
resident, only the first request pays the multi-GB load cost — every subsequent
clone is just inference (seconds on GPU). ``GET /health`` reports readiness.

Started as a systemd service per engine (see deploy/); the API falls back to the
one-shot ``clone_worker.py`` subprocess if this server is not running.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

_MODEL = None
_ENGINE = ""
_LOCK = Lock()  # one generation at a time — the model is not re-entrant


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _load(engine: str, weights: str):
    if engine == "xtts":
        from TTS.api import TTS  # type: ignore

        model = (
            weights
            if weights and os.path.isdir(weights)
            else "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        return TTS(model).to("cuda" if _cuda() else "cpu")
    if engine == "indextts2":
        from indextts.infer_v2 import IndexTTS2  # type: ignore

        return IndexTTS2(
            cfg_path=f"{weights}/config.yaml", model_dir=weights, use_cuda_kernel=False
        )
    raise ValueError(f"unknown engine {engine!r}")


def _generate(text: str, ref: str, out: str, language: str) -> None:
    if _ENGINE == "xtts":
        _MODEL.tts_to_file(text=text, speaker_wav=ref, language=language, file_path=out)
    elif _ENGINE == "indextts2":
        _MODEL.infer(spk_audio_prompt=ref, text=text, output_path=out)
    else:
        raise ValueError(f"unknown engine {_ENGINE!r}")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):  # silence default request logging
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "engine": _ENGINE, "ready": _MODEL is not None})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            with _LOCK:
                _generate(req["text"], req["ref"], req["out"], req.get("language", "en"))
            self._json(200, {"status": "ok"})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": str(exc)[:300]})


def main() -> None:
    global _MODEL, _ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    _ENGINE = args.engine
    print(f"[{args.engine}] loading model (cuda={_cuda()}) ...", flush=True)
    _MODEL = _load(args.engine, args.weights)
    print(f"[{args.engine}] ready on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
