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

Same pattern in `$VG_SYNTH_HOME/indextts2/venv` (`pip install indextts` + its
transformers 4.52 pin), with weights downloaded from `IndexTeam/IndexTTS-2` to
`$VG_SYNTH_HOME/indextts2/weights` (or set `VG_INDEXTTS2_WEIGHTS`). IndexTTS-2 is
the better choice for the **Test against detector** demo — the production detector
flags its clones at ~100%.

## Responsible use

Voice cloning here is for **detector evaluation and red-teaming**, not
impersonation. Every clip is watermarked and flagged AI-generated, the endpoint
requires authentication, and the UI gates cloning behind an authorization
acknowledgement. Reference uploads are auto-deleted (PDPL).
