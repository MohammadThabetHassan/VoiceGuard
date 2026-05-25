# GPU Training Plan

Six sequential runs on a g5.xlarge Spot instance (ap-southeast-1).
Total estimated cost: ~$27–35 USD.
Budget cap: $200 USD.

Launch scripts live in `scripts/aws/`.
**NEVER launch without explicit user confirmation.**

---

## Run 1 — Data Preparation

**Script:** `scripts/aws/prepare_asvspoof_data.sh`
**Instance:** g5.xlarge Spot (~$1.01/hr)
**Estimated time:** 2 hours
**Estimated cost:** ~$2.02

**Steps:**
1. Download ASVspoof 2021 LA FLAC files from Zenodo (requires `ZENODO_TOKEN`)
2. Resample all FLAC → 16kHz mono WAV
3. Convert WAV → `.pt` tensors (float32, normalised to [-1, 1])
4. Upload to `s3://voiceguard-mt-2026/asvspoof2021_la/`
5. Upload partition manifest JSON

**Output:** `s3://voiceguard-mt-2026/asvspoof2021_la/{train,dev,eval}/{real,fake}/*.pt`

---

## Run 2 — DSFNet Baseline Training

**Script:** `scripts/aws/launch_dsfnet_training.sh`
**Instance:** g5.xlarge Spot (~$1.01/hr)
**Estimated time:** 8–10 hours
**Estimated cost:** ~$8–10

**Config:**
- Epochs: 50 (EarlyStopping patience=5 on dev EER)
- Batch: 32, LR: 1e-4, CosineAnnealingLR
- fp16 AMP, AdamW (wd=0.01)
- Target: EER <1.0% on dev set

**Output:**
- `s3://voiceguard-mt-2026/checkpoints/dsfnet_best.pt`
- `s3://voiceguard-mt-2026/checkpoints/dsfnet_epoch_{n}.pt` (every 5 epochs)

---

## Run 3 — DSFNet Adversarial Hardening

**Script:** `scripts/aws/launch_adversarial_training.sh`
**Instance:** g5.xlarge Spot (~$1.01/hr)
**Estimated time:** 6 hours
**Estimated cost:** ~$6

**Config:**
- Warm-start from `dsfnet_best.pt`
- PGD ε=0.01, α=0.001, 5 steps per batch
- Epochs: 10, LR: 5e-5
- Target: <5% accuracy drop vs. PGD adversarial examples

**Output:** `s3://voiceguard-mt-2026/checkpoints/adversarial/dsfnet_adversarial_best.pt`

---

## Run 4 — Wav2Vec2 Fine-Tuning

**Script:** `scripts/aws/launch_wav2vec2_training.sh` *(to be written in Phase 7)*
**Instance:** g5.xlarge Spot (~$1.01/hr)
**Estimated time:** 4 hours
**Estimated cost:** ~$4

**Config:**
- Backbone: facebook/wav2vec2-base (frozen feature encoder for first 2 epochs)
- Head: Dropout(0.1) → Linear(768,256) → ReLU → Linear(256,2)
- Epochs: 10, LR: 5e-5, fp16 AMP
- Target: EER <2.0% on dev set

**Output:** `s3://voiceguard-mt-2026/checkpoints/wav2vec2_best.pt`

---

## Run 5 — Evaluation on ASVspoof 2021 LA Eval Set

**Script:** Inline Python via harness.py after model download
**Instance:** t3.medium (no GPU needed)
**Estimated time:** 30 minutes
**Estimated cost:** ~$0.05

**Models evaluated:**
| Model | Expected EER | minDCF |
|---|---|---|
| ClassicalDetector (XGBoost) | ~8–15% | ~0.4 |
| DSFNet baseline | <1.0% | <0.05 |
| DSFNet adversarial | <1.2% | <0.06 |
| Wav2Vec2-FT | <2.0% | <0.1 |

**Output:** `docs/RESULTS.md` updated with final numbers

---

## Run 6 — ONNX INT8 Export (Phase 7)

**Script:** `scripts/export_onnx.sh` *(to be written in Phase 7)*
**Instance:** t3.medium
**Estimated time:** 15 minutes
**Estimated cost:** <$0.01

**Steps:**
1. Load `dsfnet_adversarial_best.pt`
2. Export to `dsfnet.onnx` (dynamic batch, opset 17)
3. Quantize to INT8 with `onnxruntime.quantization`
4. Validate: ONNX output matches PyTorch within 1e-4
5. Upload to `s3://voiceguard-mt-2026/onnx/`

**Latency target:** ≤200 ms for 3-second window on CPU (t3.medium)

---

## Execution Checklist

- [ ] Run 1 — Data preparation complete, manifest uploaded
- [ ] Run 2 — DSFNet trained, dev EER verified <1.0%
- [ ] Run 3 — Adversarial hardening complete, robustness verified
- [ ] Run 4 — Wav2Vec2 fine-tuned, dev EER verified <2.0%
- [ ] Run 5 — Eval set results recorded in RESULTS.md
- [ ] Run 6 — ONNX INT8 model exported and latency verified

## Cost Tracking

| Run | Actual Cost |
|---|---|
| 1 Data prep | — |
| 2 DSFNet train | — |
| 3 Adversarial | — |
| 4 Wav2Vec2 | — |
| 5 Evaluation | — |
| 6 ONNX export | — |
| **Total** | **—** |
