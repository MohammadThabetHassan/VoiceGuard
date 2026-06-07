#!/bin/bash
# Pull the latest code, rebuild the frontend, and restart the backend.
# Run after every push to main (or wire it to CI — see .github/workflows/deploy.yml).
set -euo pipefail

REPO_DIR="/home/$USER/voiceguard"
DIST_DIR="/var/www/voiceguard/dist"

echo "=== Updating VoiceGuard ==="

cd "$REPO_DIR"
git pull --ff-only origin main

# Backend deps may have changed (pyproject.toml)
# shellcheck disable=SC1091
source "$REPO_DIR/venv/bin/activate"
pip install -e "$REPO_DIR"

# Rebuild frontend against the same-origin /api path
cd "$REPO_DIR/frontend"
npm ci
VITE_API_BASE="/api" npm run build
sudo rm -rf "${DIST_DIR:?}/"*
sudo cp -r dist/* "$DIST_DIR/"
sudo chown -R www-data:www-data /var/www/voiceguard

# Restart backend
sudo systemctl restart voiceguard

echo "=== Update complete ==="
sudo systemctl status voiceguard --no-pager | head -n 5
