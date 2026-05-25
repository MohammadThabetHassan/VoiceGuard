---
name: research-old-repo
description: Inspect old VoiceGuard repo for reference implementations. Use when porting baseline code or verifying SM2026 behavior.
context: fork
agent: Explore
argument-hint: "<file or feature to find>"
---

## Instructions

Explore `/tmp/voiceguard-old` for: `$ARGUMENTS`

The old repo is READ-ONLY reference. Do not modify it.

1. `ls /tmp/voiceguard-old/` to orient.
2. Grep for `$ARGUMENTS` across the repo.
3. Read relevant files with full context.
4. Return code with annotations:
   - **Port as-is**: code that can be copied directly
   - **Adapt**: code that needs changes for the new architecture
   - **Skip**: code superseded by DSFNet or new approach
5. Pay special attention to:
   - `backend/app.py` — FastAPI endpoints
   - `train_detector.py` — SM2026 baseline training
   - `backend/requirements.txt` — dependencies
   - `features.csv`, `osr_features.csv` — training data structure
   - `models/` — saved model artifacts
   - `backend/tests/` — existing test patterns
