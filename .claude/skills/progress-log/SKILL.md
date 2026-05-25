---
name: progress-log
description: Append a timestamped entry to docs/PROGRESS.md after a milestone.
disable-model-invocation: true
argument-hint: "<description> [commit-sha]"
---

## Instructions

Append to `docs/PROGRESS.md` in this format:

```bash
PHASE=$(cat PHASE 2>/dev/null || echo "phase-0")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
SHA=$(echo "$ARGUMENTS" | grep -oE '[a-f0-9]{7,}' | head -1 || echo "")
DESC=$(echo "$ARGUMENTS" | sed 's/[a-f0-9]\{7,\}//g' | xargs)
echo "- [$TIMESTAMP] $PHASE — $DESC${SHA:+ ($SHA)}" >> docs/PROGRESS.md
```

Show the line that was appended.
