"""Zero-shot voice-cloning engines, run out-of-process.

Cloning models (IndexTTS2, XTTS) pin dependency versions that conflict with the
API process (e.g. transformers 4.52 vs 5.x). Each therefore lives in its **own
durable venv** under ``$VG_SYNTH_HOME`` (default ``~/.voiceguard/synth`` — never
``/tmp``, which is wiped on reboot) and is invoked via ``clone_worker.py`` as a
subprocess. The engine writes a raw wav; the API layer watermarks it.

An engine is only offered (``available=True``) when both its venv interpreter and
its weights directory exist, so a missing install degrades gracefully instead of
breaking the API or CI.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceguard.synthesis.base import SynthEngine

SYNTH_HOME = Path(os.environ.get("VG_SYNTH_HOME", str(Path.home() / ".voiceguard" / "synth")))
_WORKER = Path(__file__).with_name("clone_worker.py")
_MIN_REF_SEC = 3.0


class CloneEngine(SynthEngine):
    """Generic subprocess-isolated cloning engine."""

    requires_reference = True
    worker_key: str = ""  # passed to clone_worker --engine

    def _venv_python(self) -> Path:
        return SYNTH_HOME / self.name / "venv" / "bin" / "python"

    def _weights_dir(self) -> Path:
        env = os.environ.get(f"VG_{self.name.upper()}_WEIGHTS")
        return Path(env) if env else SYNTH_HOME / self.name / "weights"

    def is_available(self) -> bool:
        return self._venv_python().exists() and self._weights_dir().exists()

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        reference_wav: str | Path | None = None,
        language: str = "en",
    ) -> tuple[np.ndarray, int]:
        if reference_wav is None:
            raise ValueError(f"{self.label} requires a reference audio clip")
        ref = Path(reference_wav)
        dur = sf.info(str(ref)).duration
        if dur < _MIN_REF_SEC:
            raise ValueError(f"Reference audio too short ({dur:.1f}s); need >= {_MIN_REF_SEC}s")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out_path = Path(tf.name)
        try:
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = "/tmp/nvml_fix:" + env.get("LD_LIBRARY_PATH", "")  # noqa: S108  # nosec B108
            cmd = [
                str(self._venv_python()),
                str(_WORKER),
                "--engine",
                self.worker_key,
                "--text",
                text,
                "--ref",
                str(ref),
                "--out",
                str(out_path),
                "--weights",
                str(self._weights_dir()),
                "--language",
                language,
            ]
            proc = subprocess.run(  # noqa: S603
                cmd, env=env, capture_output=True, text=True, timeout=300, check=False
            )
            if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
                raise RuntimeError(
                    f"{self.label} worker failed (rc={proc.returncode}): {proc.stderr[-500:]}"
                )
            audio, sr = sf.read(str(out_path), dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) == 2:
                audio = audio.mean(axis=1)
            return np.asarray(audio, dtype=np.float32), int(sr)
        finally:
            out_path.unlink(missing_ok=True)


class IndexTTS2Engine(CloneEngine):
    name = "indextts2"
    label = "IndexTTS-2 (zero-shot voice cloning)"
    worker_key = "indextts2"
    languages = ["en"]
    description = "High-quality zero-shot cloning from a short reference clip."


class XTTSEngine(CloneEngine):
    name = "xtts"
    label = "Coqui XTTS v2 (zero-shot voice cloning)"
    worker_key = "xtts"
    languages = ["en", "es", "fr", "de", "it", "pt", "ar", "zh", "ja"]
    description = "Multilingual zero-shot cloning from a short reference clip."
