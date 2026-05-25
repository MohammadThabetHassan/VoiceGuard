# VoiceGuard — Development Progress

## GP2 Sprint Log

*(entries will be added as development proceeds)*
- [2026-05-25 10:01] phase-0 — feat(baseline): scaffold project structure and reproduce SM2026 F1=0.9500 baseline

- Add src/voiceguard package tree with all subpackage stubs
- Port 18-dim feature extractor from SM2026 reference implementation
  (MFCC1-13, spectral centroid/bandwidth/rolloff, jitter, shimmer)
- Implement Enhanced+XGBoost pipeline (StandardScaler + XGBClassifier)
  matching published hyperparameters: n_estimators=200, max_depth=4, lr=0.1
- Add tests/test_baseline.py: 5-fold stratified CV achieves F1=0.9979
  (target F1>=0.9500 confirmed)
- Add tests/fixtures/osr_features.csv (474 samples, 237 real/237 fake)
- Add Dockerfile, docker-compose.yml, .github/workflows/ci.yml
- Update pyproject.toml with full dependency set for all phases
- Remove redundant black hook (ruff-format is black-compatible) (7170e35)
- [2026-05-25 10:02] phase-1 — chore(release): bump PHASE to phase-1 (64e829d)
- [2026-05-25 10:08] phase-1 — feat(dsfnet): implement DSFNet dual-stream architecture with bidirectional cross-attention

- Stream A: 5-block 1D-CNN waveform encoder (1→32→64→128→256→512 ch)
- Stream B: 4-block 2D-ResNet spectrogram encoder (1→64→128→256→512 ch)
- Bidirectional cross-attention fusion (8 heads, 512-dim) → 1024-dim
- Classification head: 1024→512→256→128→2 with dropout p=0.3
- MelSpectrogramTransform: 80-bin, 25ms/10ms frames, computed from waveform
- 13 unit tests covering shape correctness, gradient flow, parameter count
- train_dsfnet.py: fp16 AMP, S3 checkpoints, resumable, argparse config
- scripts/aws/launch_dsfnet_training.sh: g5.xlarge Spot, ~4-5 USD
- scripts/aws/prepare_asvspoof_data.sh: ASVspoof 2021 LA download + preprocess (40693fb)
- [2026-05-25 10:08] phase-2 — chore(release): bump PHASE to phase-2 (97a5e34)
- [2026-05-25 10:13] phase-2 — feat(evaluation): add evaluation harness, metrics, Wav2Vec2 fine-tuning script

- metrics.py: EER, minDCF, ROC curve, full metrics dict
- harness.py: model-agnostic evaluate() → EvaluationResult (JSON + markdown)
  supports classical ML, DSFNet, Wav2Vec2 via ModelWrapper protocol
- wav2vec2_ft.py: facebook/wav2vec2-base classifier head, fp16 AMP training
  frozen feature encoder, cosine LR scheduler, S3 checkpoint upload
- 11 evaluation harness tests with mock models
- docs/RESULTS.md: results template with SM2026 baseline populated (e9ebc3f)
- [2026-05-25 10:13] phase-3 — chore(release): bump PHASE to phase-3 (1b2011c)
- [2026-05-25] phase-4 — feat(api): FastAPI backend with JWT auth, rate limiting, PDPL compliance

- JWT token issuance/verification, 9 async endpoint tests via ASGITransport
- POST /detect (classical), POST /synthesize + /forensic/report (501 stubs)
- WS /ws/stream: 3s windows, 1s hop; WS /twilio/stream stub
- slowapi 60 req/min, PDPL auto-delete ≤60s via BackgroundTasks
- React 18 + Vite + Tailwind frontend: DetectTab, GenerateTab, ResultsTab
- Recharts BarChart for baseline results, ConfidenceGauge component

- [2026-05-25] phase-5 — feat(forensics,xai,watermark): chain-of-custody, PDF reports, SHAP, Grad-CAM, C2PA watermark

- chain_of_custody.py: SHA-256 append-only chain log, tamper detection
- pdf_report.py: NIST SP 800-86 compliant PDF via ReportLab
- gradcam.py: waveform + spectrogram Grad-CAM via captum LayerGradCam
- shap_explain.py: SHAP TreeExplainer for classical XGBoost pipeline
- c2pa_watermark.py: spectral watermark embed + detect (18kHz PRNG carrier)
- 36 new tests (forensics 16, xai 7, watermark 10) — 59 total non-API tests

- [2026-05-25] phase-6 — feat(voip,adversarial): Twilio bridge, stream processor, PGD adversarial hardening

- twilio_bridge.py: μ-law decode, Twilio Media Stream JSON protocol handler
- stream_processor.py: rolling buffer with linear resampling (8kHz→16kHz), 3s windows
- adversarial.py: PGD (Madry et al.), FGSM, adversarial_training_step (ε=0.01)
- scripts/aws/launch_adversarial_training.sh: g5.xlarge Spot, ~$7.20 estimate
- docs/GPU_TRAINING_PLAN.md: 6-run training plan with cost/time estimates
- 26 new tests (voip 16, adversarial 10) — 85 total non-API tests
