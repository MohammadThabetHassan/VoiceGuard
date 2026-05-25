---
name: research-docs
description: Fetch official documentation for PyTorch, FastAPI, Coqui XTTS, captum, SHAP, Twilio, or other project libraries.
context: fork
agent: Explore
argument-hint: "<library> <topic>"
---

## Instructions

Fetch and summarize documentation for: `$ARGUMENTS`

1. Look up the canonical URL in `.claude/skills/research-docs/reference.md`.
2. Fetch the relevant documentation page.
3. Return a concise summary with:
   - Key API signatures / configuration options
   - Code example relevant to VoiceGuard's use case
   - Any gotchas or version-specific notes
4. Cite the source URL.

See `reference.md` for canonical URLs.
