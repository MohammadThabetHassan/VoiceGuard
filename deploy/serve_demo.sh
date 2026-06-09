#!/bin/bash
# VoiceGuard demo watchdog — keeps the combined server (frontend + /api) and the
# ngrok tunnel alive. Survives crashes and reboots (run via @reboot cron — no sudo
# needed). The combined app is deploy/vg_serve.py (durable, not /tmp).
#
#   bash deploy/serve_demo.sh        # run the watchdog (loops forever)
#   crontab:  @reboot /bin/bash /srv/thabet/VoiceGuard/deploy/serve_demo.sh >/dev/null 2>&1
set -u
REPO=/srv/thabet/VoiceGuard
CKPT="${XLS_R_AASIST_PATH:-$REPO/models/xls_r_aasist.pt}"
NGROK=/srv/thabet/bin/ngrok
HOME_VG="$HOME/.voiceguard"
SECRET_FILE="$HOME_VG/demo_secret"
LOG="$HOME_VG/serve_demo.log"
mkdir -p "$HOME_VG"
[ -f "$SECRET_FILE" ] || openssl rand -hex 32 > "$SECRET_FILE"   # stable JWT secret across restarts

# Rebuild the NVML stub if missing (lost on reboot; source is persistent).
mkdir -p /tmp/nvml_fix
[ -f /tmp/nvml_fix/libnvidia-ml.so.1 ] || \
  gcc -shared -fPIC -o /tmp/nvml_fix/libnvidia-ml.so.1 \
      /srv/thabet/voiceguard-checkpoints/nvml_stub.c 2>/dev/null || true

export LD_LIBRARY_PATH="/tmp/nvml_fix:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/src"
export XLS_R_AASIST_PATH="$CKPT"
export SECRET_KEY="$(cat "$SECRET_FILE")"
export VG_SYNTH_HOME="$HOME_VG/synth"
export COQUI_TOS_AGREED=1
export VOICEGUARD_DOMAIN="voice-deepfake-vishing-detector-generator.eu.cc"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "watchdog started (ckpt=$CKPT)"

while true; do
  if ! curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    log "API down -> starting combined server (deploy/vg_serve.py)"
    setsid python3 -m uvicorn vg_serve:root --app-dir "$REPO/deploy" \
      --host 127.0.0.1 --port 8000 --workers 1 --log-level warning >> "$LOG" 2>&1 &
    sleep 20
  fi
  if ! pgrep -f "ngrok http" >/dev/null 2>&1; then
    log "tunnel down -> starting ngrok (pinned static domain -> stable URL)"
    # Pin the account's free static domain so the public URL never changes across
    # restarts/reboots. Override with NGROK_DOMAIN; empty falls back to ephemeral.
    NGROK_DOMAIN="${NGROK_DOMAIN:-cradle-geography-zit.ngrok-free.dev}"
    if [ -n "$NGROK_DOMAIN" ]; then
      setsid "$NGROK" http 8000 --domain "$NGROK_DOMAIN" --log stdout --log-format logfmt >> "$LOG" 2>&1 &
    else
      setsid "$NGROK" http 8000 --log stdout --log-format logfmt >> "$LOG" 2>&1 &
    fi
    sleep 5
  fi
  sleep 30
done
