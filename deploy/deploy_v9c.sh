#!/bin/bash
# Deploy v9c as the live production detector. Run with:  ! bash deploy/deploy_v9c.sh
# Stages v9c into models/, restarts the :8000 server (watchdog respawns with v9c),
# and verifies. v7 is backed up to models/xls_r_aasist_v7.pt for rollback.
set -u
REPO=/srv/thabet/VoiceGuard
SRC=/srv/thabet/voiceguard-checkpoints/runs/xlsr_aasist_v9c
echo "backing up current model -> models/xls_r_aasist_v7.pt"
cp "$REPO/models/xls_r_aasist.pt" "$REPO/models/xls_r_aasist_v7.pt"
echo "staging v9c -> models/"
cp "$SRC/model_best.pt" "$REPO/models/xls_r_aasist.pt"
cp "$SRC/config.json" "$REPO/models/config.json"
echo "restarting server (kill :8000; watchdog respawns with v9c from models/)"
for p in $(pgrep -f "uvicorn vg_serve"); do kill "$p" 2>/dev/null; done
echo "waiting for watchdog to bring it back up ..."
for i in $(seq 1 20); do
  sleep 3
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/health 2>/dev/null)
  [ "$code" = "200" ] && break
done
echo "local health: $(curl -s http://127.0.0.1:8000/api/health 2>/dev/null | head -c 200)"
echo "public: $(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://voice-deepfake-vishing-detector-generator.eu.cc/api/health)"
echo "DONE — to roll back: cp models/xls_r_aasist_v7.pt models/xls_r_aasist.pt && pkill -f 'uvicorn vg_serve'"
