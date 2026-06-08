#!/usr/bin/env python3
"""Out-of-process voice-cloning worker.

Run by a cloning engine's **isolated venv** (see clone_engine.py), NOT by the API
process — so it must stay self-contained: it imports only its engine's library
(present only in that venv) plus stdlib. It writes a raw wav to ``--out``; the
API process reads it back and applies the watermark.

Usage:
    python clone_worker.py --engine {indextts2,xtts} --text TEXT \
        --ref REF.wav --out OUT.wav --weights WEIGHTS_DIR [--language en]
"""

from __future__ import annotations

import argparse
import sys


def _clone_indextts2(text: str, ref: str, out: str, weights: str, language: str) -> None:
    from indextts.infer_v2 import IndexTTS2  # type: ignore

    cfg = f"{weights}/config.yaml"
    tts = IndexTTS2(cfg_path=cfg, model_dir=weights, use_cuda_kernel=False)
    tts.infer(spk_audio_prompt=ref, text=text, output_path=out)


def _clone_xtts(text: str, ref: str, out: str, weights: str, language: str) -> None:
    from TTS.api import TTS  # type: ignore

    # weights dir may host a local XTTS v2; fall back to the hub id.
    model = (
        weights
        if weights and __import__("os").path.isdir(weights)
        else "tts_models/multilingual/multi-dataset/xtts_v2"
    )
    tts = TTS(model).to("cuda" if _cuda() else "cpu")
    tts.tts_to_file(text=text, speaker_wav=ref, language=language, file_path=out)


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


_ENGINES = {"indextts2": _clone_indextts2, "xtts": _clone_xtts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(_ENGINES))
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--language", default="en")
    a = ap.parse_args()
    try:
        _ENGINES[a.engine](a.text, a.ref, a.out, a.weights, a.language)
    except Exception as e:  # noqa: BLE001
        print(f"clone_worker error: {e!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
