@echo off
REM VoiceGuard backend launcher (Windows). Assumes the repo is at C:\voiceguard
REM with a venv created by `python -m venv venv` and `pip install -e .` run from
REM the repo root. The app is the module voiceguard.api.main:app.

cd /d C:\voiceguard
call C:\voiceguard\venv\Scripts\activate
set PYTHONPATH=C:\voiceguard\src
REM Set a real secret in production: powershell -c "[guid]::NewGuid()" or openssl
if "%SECRET_KEY%"=="" set SECRET_KEY=change-me-in-production
uvicorn voiceguard.api.main:app --host 127.0.0.1 --port 8000 --workers 1 --log-level info --access-log
