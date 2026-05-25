# VoiceGuard — Evaluation Results

> Results will be updated after GPU training runs complete.
> See `docs/GPU_TRAINING_PLAN.md` for the training schedule.

## SM2026 Baseline (Classical ML)

Evaluated on `osr_features.csv` (474 samples, 237 real/237 fake),
5-fold stratified cross-validation.

| Model | Accuracy | F1 | Precision | Recall | EER |
|-------|----------|----|-----------|--------|-----|
| Enhanced+XGBoost | 0.9979 | 0.9979 | 0.9958 | 1.0000 | — |

> Published baseline target: F1 ≥ 0.9500 ✓

## DSFNet (Deep Learning)

Evaluated on ASVspoof 2021 LA evaluation set.
Results to be populated after training run.

| Model | Accuracy | F1 | EER | minDCF | Latency p95 (ms) |
|-------|----------|----|-----|--------|-----------------|
| DSFNet (epoch 40) | — | — | — | — | — |
| Wav2Vec2-base FT | — | — | — | — | — |

**Target:** EER < 0.5% (fallback: ≤ 1.0%)

## Adversarial Robustness

PGD adversarial attack (ε=0.01), 3 cycles.
Results to be populated after adversarial training run.

| Attack | EER (no defence) | EER (with defence) |
|--------|------------------|--------------------|
| PGD ε=0.01 | — | — |

**Target:** EER ≤ 5% under attack.

## Real-Time Performance

Latency measured on GPU (g5.xlarge, A10G) for 3-second audio windows.

| Model | GPU Latency (ms) | CPU Latency (ms) |
|-------|-----------------|-----------------|
| DSFNet | — | — |
| Wav2Vec2-base FT | — | — |

**Target:** ≤ 200 ms per 3-second window on GPU.
