---
name: load-midterm-context
description: Load a specific midterm spec section into main context. Use sparingly — only when implementing a feature requiring the exact spec.
argument-hint: "<section number or topic>"
---

## Instructions

1. Find files in `docs/midterm/` matching `$ARGUMENTS`:
   ```bash
   find docs/midterm/ -type f | xargs grep -l "$ARGUMENTS" 2>/dev/null
   ```
2. Read the matching files.
3. Strip LaTeX markup (remove `\begin{}`, `\end{}`, `\\`, `\textbf{}`, etc.) for readability.
4. Return the clean prose section with the original file/line reference.
5. If docs/midterm/ does not exist: "Midterm docs not yet committed. Ask user to `git add docs/midterm/` and commit."

**Use sparingly** — this loads content into main context and consumes tokens.
Prefer `/research-midterm` for lookup without context pollution.
