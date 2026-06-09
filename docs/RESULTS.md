# VoiceGuard — Evaluation Results

> Canonical results live in the [README](../README.md#-results) and
> [CHANGELOG](../CHANGELOG.md). This file is a concise summary.

## Detection (official ASVspoof 2021 LA, full 181,566-trial eval)

_Reproduced 2026-06-09 from the official FLAC via `run_official_eval.py` — the 2.61%
headline was re-measured exactly (eval 2.612 / progress 2.148 / full-pool 8.213)._

| Model | EER (eval) | EER (full-pool) | Catches modern clones | Notes |
|-------|:----------:|:---------------:|:---------------------:|-------|
| XLS-R + AASIST (Kokoro-parent) | **2.61%** | 8.21% | ✗ misses IndexTTS-2 | the headline checkpoint |
| **XLS-R + AASIST — v7 (DEPLOYED)** | **3.38%** | 8.60% | ✓ all families ≥96.7% | robustness-first production model |
| XLS-R + AASIST — v8 (EER-opt) | 2.49% | 9.91% | ✗ Kokoro→62.5% | lowest official EER, weaker clones |
| Wav2Vec2-large | 3.09% | 7.07% | — | baseline |
| WavLM-base-plus | 8.11% | — | — | baseline |
| DSFNet-V2 / DSFNetTiny (edge) | — | 12.67% / 8.47%* | — | own dual-stream; *balanced-mirror EER |

Metrics: F1 ≈ 0.96, ROC-AUC ≈ 0.98, minDCF (p_target=0.01). **v7 is deployed**: it
trades ~0.8 pp of ASVspoof specialisation for catching every modern clone family.

## Held-out clone detection (speaker/text-disjoint, 100/family — 2026-06-09)

| Family | v7 detection |
|--------|:------------:|
| real-pass | 97% |
| Kokoro-82M | 100% |
| XTTS v2 | 100% |
| IndexTTS-2 | 96% |
| ElevenLabs-v3 (OOD premium) | 85% (premium hardening in progress) |

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
