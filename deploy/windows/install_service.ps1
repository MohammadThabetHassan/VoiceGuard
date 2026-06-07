# Install VoiceGuard as a Windows service using NSSM.
# Run this script from an elevated (Administrator) PowerShell prompt.
# Requires NSSM (Non-Sucking Service Manager): https://nssm.cc
#
# Assumes:
#   Repo at        C:\voiceguard   (git clone + `pip install -e .` from repo root)
#   venv at        C:\voiceguard\venv
#   NSSM at        C:\nssm\nssm.exe
#
# The backend is launched as the package module voiceguard.api.main:app with
# PYTHONPATH pointing at the src layout.

$ErrorActionPreference = "Stop"

$nssm    = "C:\nssm\nssm.exe"
$service = "VoiceGuard"
$python  = "C:\voiceguard\venv\Scripts\python.exe"
$appArgs = "-m uvicorn voiceguard.api.main:app --host 127.0.0.1 --port 8000 --workers 1 --log-level info --access-log"
$workdir = "C:\voiceguard"

New-Item -ItemType Directory -Force -Path "C:\voiceguard\logs" | Out-Null

& $nssm install $service $python $appArgs
& $nssm set $service AppDirectory $workdir
# src layout + a real JWT secret. Replace the secret before going to production.
& $nssm set $service AppEnvironmentExtra "PYTHONPATH=C:\voiceguard\src" "SECRET_KEY=change-me-in-production"
& $nssm set $service AppStdout "C:\voiceguard\logs\voiceguard.log"
& $nssm set $service AppStderr "C:\voiceguard\logs\voiceguard_err.log"
& $nssm set $service Start SERVICE_AUTO_START
& $nssm start $service

Write-Host "VoiceGuard service installed and started."
Write-Host "Manage with: nssm restart VoiceGuard | nssm stop VoiceGuard | nssm remove VoiceGuard confirm"
