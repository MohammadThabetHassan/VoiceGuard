# Known Limitations & Detection-vs-Cloning Status

_Last updated 2026-06-11._

> **Current deployed model: v9c** — official ASVspoof 2021 LA eval EER
> **2.84% [95% CI 2.67–3.02]**, catches all clone families + ElevenLabs ~96%.
> v9c supersedes the v3/v6/v7 references below (kept as the historical record).
> Authoritative metrics: [`RESULTS_canonical.md`](RESULTS_canonical.md).

## ⚠️ The detector only judges a recording from its natural start (2026-06-11)

Measured on the held-out set: v9c is reliable **only when scoring a clip exactly
as recorded, from its first sample**. Three serving strategies that look
reasonable all fail:

| Strategy | Held-out result |
|---|---|
| **Whole clip, one pass (from start)** | ✅ 16/16 correct (reals 0.02–0.04, fakes 0.89–0.99), stable from 6 s to 60 s |
| 3 s sliding windows (max or mean) | ❌ mid-utterance windows of *real* speech score 0.9+ fake → real clips flagged fake |
| Pause-aligned utterance chunks (silence trimmed) | ❌ trimming the ambient lead-in flips real speech to 0.99 fake |

Root cause: real recordings open with ambient lead-in / natural onset, while TTS
output starts mid-speech — the model partly keys on this. Consequences:

- **`/detect` scores one full-clip pass** (capped at `VG_SCORE_SECONDS`, default
  60 s; the response's `seconds_analyzed` discloses coverage); streams
  (`/ws/stream`, Twilio) score the **growing prefix from the session start**
  (capped at `VG_WS_SCORE_SECONDS`, default 15 s), after which the verdict is
  emitted once with `final: true` and scoring stops. A fake segment spliced into
  the middle/tail of a long real recording is therefore **out of scope** — the
  model cannot localise it.
- **Adversarial evasion (measured, disclosed):** prepending ~0.5 s of silence to
  a TTS clip shifts some fakes toward "real" (worst held-out case: IndexTTS-2
  0.99 → 0.03). An attacker who records a clone with a quiet lead-in weakens
  detection. Fixing this needs onset-augmented training (random leading-silence
  augmentation), not a serving change — future work, alongside the PGD gap in
  [`ADVERSARIAL_ROBUSTNESS.md`](ADVERSARIAL_ROBUSTNESS.md).

## ✅ 2026-06-09 session — limitations closed
- **Official ASVspoof 2.61% reproduced** from raw FLAC (see the resolved section
  below); official eval set back on durable disk.
- **ONNX edge model now carries real trained weights** (DSFNetTiny, 8.47% balanced
  EER) — INT8 **0.62 MB**, **~30 ms** CPU. Was random-weight before.
- **True C2PA provenance**: synthesized audio now embeds a real *signed* C2PA
  manifest (`trainedAlgorithmicMedia`, validation `Valid`) in addition to the
  spectral watermark — was a spectral-only stub mislabeled "C2PA".
- **Both live streams use the SSL model**: `/ws/stream` (mic) and `/twilio/stream`
  (VoIP) were upgraded from the classical detector; a simulated-call test proves
  the Twilio bridge end-to-end without a phone number.
- **Durable hosting**: permanent custom-domain HTTPS via Cloudflare Tunnel
  (`voice-deepfake-vishing-detector-generator.eu.cc`) + a watchdog that keeps the
  server + tunnels alive across crashes/restarts.
- **Larger held-out eval** (100/family): v7 holds — real 97%, XTTS 100%, IndexTTS-2 96%.
- **Premium-TTS (ElevenLabs) hardening in progress**: v7 catches 85% of held-out
  ElevenLabs-v3; a premium-hardened checkpoint (MLAAD ElevenLabs/Cartesia/DeepGram/
  Gemini/…) is being trained with a real-pass safety gate. See "Premium voices" below.

## ⚠️ Premium commercial TTS (e.g. ElevenLabs) — partially open
v7 was trained on Kokoro/XTTS/IndexTTS-2, so commercial premium engines are
out-of-distribution: it catches ~**85%** of held-out ElevenLabs-v3. Hardening on a
broad MLAAD premium slice pushes held-out ElevenLabs detection to ~100%, but only
ships once it holds **real-pass ≥90%** (an over-eager intermediate model was rejected
for false-flagging real callers). Until a gated checkpoint lands, premium-voice
detection is **demonstrated but not yet deployed**.

## ✅ UPDATE: v7 solves cloning detection (deployed)

A harder, data-rich run (`train_v7.py`) **disproves the earlier "IndexTTS-2 ceiling"
conclusion** — that was an artifact of a shallow effort. Levers that worked:
(1) start from the **2.61% Kokoro-parent** (which already caught IndexTTS-2/XTTS at
100% but over-flagged real speech at 48% real-pass); (2) train on lots of **real
speech** (ASVspoof bonafide + LibriSpeech) to *restore* real-pass; (3) **diverse
clones cloned from ASVspoof-real speakers** (forces vocoder-artifact learning);
(4) **full ASVspoof anchor (6k+6k)**; (5) **top-12 layers unfrozen, 12 epochs**.

| held-out (speaker/text-disjoint) | v3 | v6 | parent | **v7 (deployed)** |
|---|:--:|:--:|:--:|:--:|
| real-pass | 98 | 100 | 48 | **96** |
| Kokoro detect | 100 | 100 | 100 | **100** |
| XTTS detect | 90 | 90 | 100 | **100** |
| **IndexTTS-2 detect** | 60 | 63 | 100 | **96.7** |
| ASVspoof-balanced EER | 34 | 31 | 15 | **9.25** |

v7 strictly dominates the prior deployed models. Live check: held-out IndexTTS-2
→ fake 1.00, XTTS → fake 1.00, real → real 0.999. **The sections below are
retained as the record of the earlier (superseded) head/backbone attempts.**

> EER note: 9.25% is on the obtainable *balanced* 2021-LA mirror — a different,
> harder ruler than the official 2021-LA eval where 2.61% was measured (that data
> is off-disk). 9.25% is a big improvement over the ~30% of the robustness lineage,
> but it is **not** the official "<2%" claim and isn't comparable to it.

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

## ✅ RESOLVED (2026-06-09): official ASVspoof 2021 LA EER reproduced
The official LA eval set was re-downloaded (Zenodo `4837263`, 7.76 GB + the
asvspoof.org CM keys) to **durable disk** (`asvspoof2021_LA_official/`), and the
headline **2.61%** was **reproduced exactly** from raw FLAC via
`run_official_eval.py` (faithful to the original recipe: 3 s clips, `softmax[:,1]`,
per-phase split on 181,566 trials):

| checkpoint | official **eval** EER | full-pool EER | catches modern clones? |
|------------|:--------------------:|:-------------:|------------------------|
| Kokoro-parent (the "2.61%") | **2.61%** | 8.21% | ✗ (misses IndexTTS-2) |
| **v9c (deployed)** | **2.84%** | 8.21% | ✓ all + ElevenLabs ~96% |
| v7 (prior production) | 3.38% | 8.60% | ✓ all clone families ≥96.7% |
| v8 (EER-opt) | 2.49% | 9.91% | ✗ (Kokoro regresses to 62.5%) |

Counts match the recording to the trial (eval 133360 fake / 14816 real). The honest
story: **v7 trades ~0.8 pp of ASVspoof specialisation for catching every modern
clone family** — the right trade for the vishing threat model. EER of new models is
now verifiable again on the official protocol.

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

**Findings (honest) — as of v6; partly SUPERSEDED by v9c:**
- **IndexTTS-2 ceiling — RESOLVED in v9c.** For v6 (and v3) the front-end seemed
  capped at ~60% on fresh IndexTTS-2. The v9c recipe broke that: on the
  speaker/text-disjoint held-out set v9c catches IndexTTS-2 at **97%** with median
  fake-prob 0.99 (real-pass 96%) — see fresh score distributions in
  [`CLONE_DETECTION_LIMITS.md`](CLONE_DETECTION_LIMITS.md). The "~60% ceiling" claim
  below applies to the *superseded* v3/v6 models, not the deployed model.
- **EER < 2% was NOT achieved and is not currently achievable:** the official
  ASVspoof 2021 LA eval (where 2.61% was measured) is off-disk; on the obtainable
  *balanced* mirror the whole robustness-tuned lineage sits at ~29% EER (a
  different, harder ruler — not comparable to 2.61%). A genuine low-EER model
  needs the **official ASVspoof protocol + a from-scratch SSL fine-tune**, not a
  head/top-layer tweak of a robustness derivative.
- **v6 is a marginal, gated improvement** (better real-pass, slightly better
  IndexTTS-2, no regression) → deployed as the new production checkpoint.
  Kokoro/XTTS clones are caught reliably; IndexTTS-2 remains a partial (~60%) catch.
