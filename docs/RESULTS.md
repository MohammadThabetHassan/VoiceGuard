# VoiceGuard — Evaluation Results

> Canonical results live in the [README](../README.md#-results) and
> [CHANGELOG](../CHANGELOG.md). This file is a concise summary.

## Detection (official ASVspoof 2021 LA, full 181,566-trial eval)

> **Single source of truth:** [`RESULTS_canonical.md`](RESULTS_canonical.md),
> auto-generated from the official-eval JSONs (with 95% bootstrap CIs and the
> corrected minDCF). The table below is a human-readable summary kept in sync with it.

_Reproduced from the official FLAC via `run_official_eval.py`. The deployed **v9c**
eval EER was re-verified to four decimals on 2026-06-10: **2.84% [95% CI 2.67–3.02]**._

| Model | EER (eval) | EER (full-pool) | Catches modern clones | Notes |
|-------|:----------:|:---------------:|:---------------------:|-------|
| **XLS-R + AASIST — v9c (DEPLOYED)** | **2.84%** [2.67–3.02] | 8.21% | ✓ all + ElevenLabs 96% | production: low EER *and* clones |
| XLS-R + AASIST — v7 (prior production) | 3.38% | 8.60% | ✓ all families ≥96.7% | robustness-first, superseded by v9c |
| XLS-R + AASIST (Kokoro-parent) | 2.61% | 8.21% | ✗ misses IndexTTS-2 | EER-only headline checkpoint |
| XLS-R + AASIST — v8 (EER-opt) | 2.49% | 9.91% | ✗ Kokoro→62.5% | lowest official EER, weaker clones |
| Wav2Vec2-large | 3.09% | 7.07% | — | baseline |
| WavLM-base-plus | 8.11% | — | — | baseline |
| DSFNet-V2 / DSFNetTiny (edge) | — | 12.67% / 8.47%* | — | own dual-stream; *balanced-mirror EER |

Metrics: F1 ≈ 0.96, ROC-AUC ≈ 0.97. minDCF uses the corrected estimator
(`p_target=0.05`; an earlier inverted cost model that pinned it ≈1.0 was fixed — see
[`RESEARCH_RIGOR_PLAN.md`](RESEARCH_RIGOR_PLAN.md) Phase 0.3); v9c minDCF = 0.229.
**v9c is deployed**: it recovers most of v7's EER gap *and* catches every modern clone
family plus premium TTS.

## Held-out clone detection (speaker/text-disjoint, 100/family)

| Family | v9c (deployed) | v7 (prior) |
|--------|:--------------:|:----------:|
| real-pass | 96% | 97% |
| Kokoro-82M | 100% | 100% |
| XTTS v2 | 100% | 100% |
| IndexTTS-2 | 97% | 96% |
| ElevenLabs-v3 (OOD premium) | 96% | 85% |

v9c supersedes v7: same clone coverage, lower official EER, and it now also catches
premium commercial TTS (ElevenLabs) that v7 only partly flagged.

## SM2026 classical baseline

Enhanced+XGBoost on `osr_features.csv` (474 samples, 5-fold CV): **F1 = 0.9500**
(published, IEEE SM2026).

## Out-of-distribution & real-world robustness

| Metric | Value |
|--------|:-----:|
| IndexTTS2 detection | 100% |
| Kokoro-82M (flow-matching) detection | 93.3% |
| Genuine-voice pass-rate | 90% |
| Real-world harness (avg) | real-pass 87.5% / fake-detect 90.3% |

## Edge

DSFNetTiny (554K params) → ONNX INT8 **0.62 MB**, CPU p50 **30 ms** (size/latency
validated; accuracy pending a trained tiny checkpoint).

## Adversarial robustness (negative result)

The deployed model is fragile to PGD (ε=0.01): clean acc 90.7% → PGD acc 0%. A
frozen-backbone head-only adversarial fine-tune did **not** confer PGD robustness
and regressed real-world detection, so it was not promoted. True robustness needs
backbone adversarial fine-tuning — future work.

> **Honest note.** The deployed checkpoint is a real-world-robustness fine-tune of
> the 2.61% Kokoro-hardened model; its ASVspoof EER was not separately benchmarked.
