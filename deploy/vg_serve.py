"""Combined server for the ngrok demo: serves the built React frontend at /
and mounts the VoiceGuard FastAPI app under /api (matching the frontend's
same-origin /api base and the production Nginx layout). One origin = one tunnel.
"""

import sys

sys.path.insert(0, "/srv/thabet/VoiceGuard/src")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from voiceguard.api.main import app as api_app

DIST = "/srv/thabet/VoiceGuard/frontend/dist"

root = FastAPI(title="VoiceGuard (demo)")
# /api/* -> the FastAPI app's bare routes (/detect, /token, /health, ...)
root.mount("/api", api_app)
# everything else -> the static SPA build
root.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
