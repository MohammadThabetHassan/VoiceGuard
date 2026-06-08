# Synthesis Engines (Generate)

VoiceGuard's **Generate** tab supports multiple synthesis engines, discovered at
runtime via `GET /synthesis/engines` and selected per request on `POST /synthesize`.
Every output is spectrally watermarked as AI-generated.

| Engine | Type | Reference needed | Runs |
|--------|------|:----------------:|------|
| `kokoro` | Kokoro-82M preset voices | no | in-process |
| `xtts` | Coqui XTTS v2 zero-shot cloning | yes | isolated venv (subprocess) |
| `indextts2` | IndexTTS-2 zero-shot cloning | yes | isolated venv (subprocess) |

## Why cloning engines run out-of-process

Cloning models pin dependency versions that conflict with the API process
(e.g. XTTS needs `transformers` **4.x** — it imports `isin_mps_friendly`, removed
in transformers 5.x — while the API runs transformers 5.x). Each cloning engine
therefore lives in its **own durable venv** under `$VG_SYNTH_HOME`
(default `~/.voiceguard/synth`, never `/tmp`) and is invoked via
`clone_worker.py` as a subprocess. The API never imports the cloning stack.

An engine reports `available: true` only when its venv interpreter **and** its
weights directory exist, so an absent or unverified install degrades gracefully
(the engine is listed but disabled; CI and fresh deploys are unaffected).

## Enabling XTTS (Coqui XTTS v2)

```bash
export VG_SYNTH_HOME="$HOME/.voiceguard/synth"      # default
python3 -m venv "$VG_SYNTH_HOME/xtts/venv"
VPY="$VG_SYNTH_HOME/xtts/venv/bin/python"
curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VPY" -    # ensurepip is disabled here
"$VPY" -m pip install coqui-tts "transformers>=4.57,<5"       # 4.x — XTTS needs isin_mps_friendly
"$VPY" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# (or a CUDA torch wheel matching the host driver for GPU inference)

# Verify one clone, then enable the engine:
COQUI_TOS_AGREED=1 "$VPY" path/to/src/voiceguard/synthesis/clone_worker.py \
  --engine xtts --text "Hello from VoiceGuard." --ref reference.wav --out /tmp/clone.wav --weights /nonexistent
mkdir -p "$VG_SYNTH_HOME/xtts/weights"   # marks the engine verified/available
```

> **Licence:** XTTS v2 weights are **CPML (non-commercial)** — fine for research,
> not for commercial use.

## Enabling IndexTTS-2

`indextts` is not on PyPI — install from the official repo into its own venv:

```bash
SYNTH="$VG_SYNTH_HOME/indextts2"
python3 -m venv "$SYNTH/venv"; VPY="$SYNTH/venv/bin/python"
curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VPY" -
"$VPY" -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
git clone --depth 1 https://github.com/index-tts/index-tts.git "$SYNTH/src"
# The repo pins numba==0.58.1 (no py3.12 wheel) — relax it:
sed -i 's/numba==0.58.1/numba==0.61.2/' "$SYNTH/src/pyproject.toml"
"$VPY" -m pip install "$SYNTH/src" "transformers>=4.40,<5"   # transformers 4.52.x
# Weights (~5.5GB) -> $SYNTH/weights (must contain config.yaml):
"$VPY" -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', local_dir='$SYNTH/weights')"
```

IndexTTS-2 is the best choice for the **Test against detector** demo (the hardened
detector flags its clones confidently). Note: CPU inference is slow (~25s for a
short clip, RTF ~9); a CUDA torch wheel makes it real-time.

## Responsible use

Voice cloning here is for **detector evaluation and red-teaming**, not
impersonation. Every clip is watermarked and flagged AI-generated, the endpoint
requires authentication, and the UI gates cloning behind an authorization
acknowledgement. Reference uploads are auto-deleted (PDPL).
