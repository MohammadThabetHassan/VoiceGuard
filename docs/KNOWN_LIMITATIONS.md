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
**false-flagging real speech** (clean real-pass dropped to ~78%).

A **backbone fine-tune** was then tried (v5: top-6 XLS-R layers unfrozen, more
reals, low LR) — it was **worse**: a fresh IndexTTS-2 clone scored **0.001**
(more confidently *real*), and real-pass + overall fake-detection both regressed
(3 of 4 real clips flagged fake; noisy fake-detect fell to 54%). With only the
narrow on-disk clone corpus and **no ASVspoof anchor data**, unfreezing the
backbone overfits/destabilises rather than learning a generalisable IndexTTS-2
feature.

**Conclusion:** neither head-only nor partial-backbone fine-tuning cracks *fresh*
IndexTTS-2 with the currently-available data. The deployed model remains **v3**
(XTTS/Kokoro caught, good real-pass). Properly solving IndexTTS-2 + pushing EER
needs: (1) the ASVspoof 2021 LA dataset back on disk (to anchor training and
*measure* EER), and (2) a far larger, more diverse multi-source cloning-fake
corpus (many TTS systems, voices, texts, channels) — a substantial data effort,
not another quick fine-tune. IndexTTS-2 may also sit near the detectability limit
for this frozen front-end.

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

## 2026-06-08 — trustworthy held-out eval + unified model (v6, DEPLOYED)
A proper **speaker/text-disjoint** eval was built (20 eval LibriSpeech speakers
held out from 20 train; 50 real + 30 XTTS + 30 IndexTTS-2 + 8 Kokoro held-out
clones, plus a 600/600 ASVspoof-balanced test slice). This **corrected earlier
noisy conclusions** (the old 16-clip eval included ALSA test tones).

A **unified anchored fine-tune** (`train_unified.py`): from v3, top-6 XLS-R layers
unfrozen, trained on ASVspoof-balanced (2500 real + 2500 fake) + diverse train
clones (120 IndexTTS-2 w/ emotion + 80 XTTS + 20 Kokoro) + 700 LibriSpeech reals.

| held-out metric | v3 (floor) | **v6 (deployed)** |
|-----------------|:----------:|:-----------------:|
| real-pass | 98% | **100%** |
| Kokoro detect | 100% | 100% |
| XTTS detect | 90% | 90% |
| **IndexTTS-2 detect** | 60% | **63%** |
| ASVspoof-balanced EER | 29.3% | 29.0% |

**Findings (honest):**
- **IndexTTS-2 is near the detectability ceiling (~60%)** for this XLS-R front-end:
  even 120 diverse clones + backbone unfreeze + anchor moved it only 60→63%
  (within eval noise). Cracking it likely needs a **different/stronger anti-spoof
  front-end** specialised for near-real (BigVGAN) vocoders, not more data here.
- **EER < 2% was NOT achieved and is not currently achievable:** the official
  ASVspoof 2021 LA eval (where 2.61% was measured) is off-disk; on the obtainable
  *balanced* mirror the whole robustness-tuned lineage sits at ~29% EER (a
  different, harder ruler — not comparable to 2.61%). A genuine low-EER model
  needs the **official ASVspoof protocol + a from-scratch SSL fine-tune**, not a
  head/top-layer tweak of a robustness derivative.
- **v6 is a marginal, gated improvement** (better real-pass, slightly better
  IndexTTS-2, no regression) → deployed as the new production checkpoint.
  Kokoro/XTTS clones are caught reliably; IndexTTS-2 remains a partial (~60%) catch.
