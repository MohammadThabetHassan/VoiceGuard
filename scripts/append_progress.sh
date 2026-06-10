#!/usr/bin/env bash
# Usage: append_progress.sh <sha> <description>
# Appends a timestamped line to docs/PROGRESS.md

set -euo pipefail

SHA="${1:-}"
DESC="${2:-}"

[[ -z "$DESC" ]] && { echo "Usage: append_progress.sh <sha> <description>"; exit 1; }

# The repo-root PHASE file holds a single token (e.g. "phase-6") naming the
# current development phase. It is the only consumer of that file — each
# progress line is tagged with it. Bump it by editing PHASE when a phase starts.
PHASE=$(cat "$(git rev-parse --show-toplevel)/PHASE" 2>/dev/null || echo "phase-0")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
PROGRESS="$(git rev-parse --show-toplevel)/docs/PROGRESS.md"

LINE="- [$TIMESTAMP] $PHASE — $DESC${SHA:+ ($SHA)}"
echo "$LINE" >> "$PROGRESS"
echo "Progress logged: $LINE"
