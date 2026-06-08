# VoiceGuard — Evaluation Results

> Canonical results live in the [README](../README.md#-results) and
> [CHANGELOG](../CHANGELOG.md). This file is a concise summary.

## Detection (ASVspoof 2021 LA, full 181,566-trial eval)

| Model | EER (eval) | EER (full-pool) | Notes |
|-------|:----------:|:---------------:|-------|
| **XLS-R + AASIST** (Kokoro-hardened) | **2.61%** | 8.21% | production headline |
| Wav2Vec2-large | 3.09% | 7.07% | lowest-EER baseline |
| WavLM-base-plus | 8.11% | — | baseline |
| AASIST | 10.90% | — | baseline |
| DSFNet-V2 | — | 12.67% | own dual-stream architecture |

Metrics: F1 ≈ 0.96, ROC-AUC ≈ 0.98, minDCF (p_target=0.01).

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
