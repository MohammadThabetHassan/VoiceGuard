# Known Limitations & Detection-vs-Cloning Status

_Last updated 2026-06-08._

## Detector vs. modern voice cloning (Generate → Detect loop)

The production detector is XLS-R + AASIST, hardened against out-of-distribution
TTS. Status per synthesis engine, scored as fake-probability (>0.5 ⇒ flagged fake):

| Engine | Caught by deployed model (v3)? | Notes |
|--------|:------------------------------:|-------|
| Kokoro-82M | ✅ ~0.93 | hardened (was 0% pre-hardening) |
| XTTS v2 (cloning) | ✅ ~0.97 | hardened (was ~0.99 *real* before) |
| IndexTTS-2 (cloning) | ⚠️ **not reliably** | known-good clips caught; **fresh clones still read ~0.2 (real)** |

### Why IndexTTS-2 is hard
IndexTTS-2 uses a BigVGAN vocoder and is near-indistinguishable from real speech.
A **frozen-backbone, head-only** fine-tune cannot separate it from genuine voice:
when pushed harder (the v4 attempt, trained on 90 IndexTTS-2 clones) it nudged a
fresh clone from 0.06→0.21 fake-prob but **never crossed 0.5**, and it began
**false-flagging real speech** (clean real-pass dropped to ~78%). So v4 was **not
deployed** — the deployed model remains **v3** (XTTS/Kokoro caught, good real-pass).
Cracking IndexTTS-2 requires **unfreezing the XLS-R backbone** (deeper features),
which is the next experiment.

## ASVspoof EER measurement is currently blocked
The ASVspoof 2019 (train) and 2021 LA (eval) tensors are **no longer on disk**
(`voiceguard-checkpoints/tensors/` is empty; raw LA data absent). Consequences:
- The headline **2.61% eval EER** figure stands as previously measured (see
  [RESULTS](RESULTS.md)) but **cannot be re-measured or improved** until the
  ASVspoof 2021 LA dataset is re-acquired (registration-gated ~25 GB + prep).
- Any new fine-tune can be gated on the **real-world harness** and held-out clone
  clips (which we have), but **not** on ASVspoof EER — so EER-impact of new models
  is currently unverifiable. The 2.61% Kokoro-hardened checkpoint is preserved as
  the EER reference.

## Infrastructure notes
- Cloning engines (XTTS, IndexTTS-2) now run on **GPU** (`torch cu128`, RTX 5090):
  IndexTTS-2 ~25s→~4s per clip (RTF 0.86). Engines live in isolated venvs under
  `~/.voiceguard/synth` (durable). See [SYNTHESIS_ENGINES](SYNTHESIS_ENGINES.md).
