#!/bin/bash
# VoiceGuard one-time server setup (Linux). Idempotent — safe to re-run.
#
# Before running:
#   1. Point YOUR_DOMAIN's A records at this server's public IP (see DNS_SETUP.md)
#   2. Edit DOMAIN and EMAIL below
#   3. Open ports 80 and 443 (router port-forward if on a home connection)
set -euo pipefail

DOMAIN="voice-deepfake-vishing-detector-generator.eu.cc"
EMAIL="mohammad_thabet@hotmail.com"   # Let's Encrypt expiry notices; change if you prefer
REPO_URL="https://github.com/MohammadThabetHassan/VoiceGuard.git"
REPO_DIR="$HOME/voiceguard"
DIST_DIR="/var/www/voiceguard/dist"
# Production detector checkpoint (XLS-R + AASIST, ~1.2GB; NOT in the repo).
# Copy it onto this host and set MODEL_CKPT to its path before running.
# Override by exporting MODEL_CKPT before invoking this script.
MODEL_CKPT="${MODEL_CKPT:-$REPO_DIR/models/xls_r_aasist.pt}"

echo "=== VoiceGuard Server Setup (domain: $DOMAIN) ==="

# 1. System packages
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx python3-venv python3-pip nodejs npm git

# 2. Clone repo (pull if it already exists)
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only origin main
fi

# 3. Python venv + backend (installed as an editable package via pyproject.toml)
if [ ! -d "$REPO_DIR/venv" ]; then
    python3 -m venv "$REPO_DIR/venv"
fi
# shellcheck disable=SC1091
source "$REPO_DIR/venv/bin/activate"
pip install --upgrade pip
pip install -e "$REPO_DIR"

# 4. Build the React frontend against the same-origin /api path
cd "$REPO_DIR/frontend"
npm ci
VITE_API_BASE="/api" npm run build

# 5. Publish the build to the web root
sudo mkdir -p "$DIST_DIR"
sudo rm -rf "${DIST_DIR:?}/"*
sudo cp -r dist/* "$DIST_DIR/"
sudo chown -R www-data:www-data /var/www/voiceguard

# 6. Nginx site config (substitute the domain)
sudo cp "$REPO_DIR/deploy/nginx/voiceguard.conf" /etc/nginx/sites-available/voiceguard
sudo sed -i "s/YOUR_DOMAIN/$DOMAIN/g" /etc/nginx/sites-available/voiceguard
sudo ln -sf /etc/nginx/sites-available/voiceguard /etc/nginx/sites-enabled/voiceguard
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

# 7. SSL via Certbot (requires DNS already pointing at this host)
sudo certbot --nginx -d "$DOMAIN" -d "www.$DOMAIN" --non-interactive --agree-tos -m "$EMAIL"

# 8. systemd service for the FastAPI backend
#    Substitute the username + domain, and generate a strong JWT secret plus
#    random admin/analyst passwords (the demo logins are disabled in production).
SECRET="$(openssl rand -hex 32)"
ADMIN_PW="$(openssl rand -hex 16)"
ANALYST_PW="$(openssl rand -hex 16)"
sudo cp "$REPO_DIR/deploy/systemd/voiceguard.service" /etc/systemd/system/voiceguard.service
# Replace the full base path first (handles $HOME != /home/$USER), then the
# bare username (for User=) and domain.
sudo sed -i \
    -e "s|/home/YOUR_USERNAME/voiceguard|$REPO_DIR|g" \
    -e "s/YOUR_USERNAME/$USER/g" \
    -e "s/YOUR_DOMAIN/$DOMAIN/g" \
    -e "s|SECRET_KEY=change-me-openssl-rand-hex-32|SECRET_KEY=$SECRET|" \
    -e "s|VG_ADMIN_PASSWORD=change-me-admin-password|VG_ADMIN_PASSWORD=$ADMIN_PW|" \
    -e "s|VG_ANALYST_PASSWORD=change-me-analyst-password|VG_ANALYST_PASSWORD=$ANALYST_PW|" \
    -e "s|XLS_R_AASIST_PATH=.*|XLS_R_AASIST_PATH=$MODEL_CKPT\"|" \
    /etc/systemd/system/voiceguard.service
echo "Generated API logins — admin password: $ADMIN_PW   analyst password: $ANALYST_PW"
echo "(stored in /etc/systemd/system/voiceguard.service; save them now)"
if [ ! -f "$MODEL_CKPT" ]; then
    echo "WARNING: model checkpoint not found at $MODEL_CKPT — /detect will 503 until it exists."
fi
sudo systemctl daemon-reload
sudo systemctl enable voiceguard
sudo systemctl restart voiceguard

# 9. Firewall
if command -v ufw >/dev/null 2>&1; then
    sudo ufw allow 'Nginx Full'
    sudo ufw allow 22
fi

echo ""
echo "=== DONE ==="
echo "Site live at:   https://$DOMAIN"
echo "API (internal): http://127.0.0.1:8000  (public via https://$DOMAIN/api/)"
echo "Backend status: sudo systemctl status voiceguard"
echo "Nginx status:   sudo systemctl status nginx"
echo "Demo login:     admin / voiceguard2026   (change in src/voiceguard/api/auth.py)"
