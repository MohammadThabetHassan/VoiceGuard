---
name: research-midterm
description: Search midterm report for a specific topic. Returns relevant sections without polluting main context.
context: fork
agent: Explore
argument-hint: "<topic or section>"
---

## Instructions

Search `docs/midterm/` for the topic: `$ARGUMENTS`

1. Run: `find docs/midterm/ -type f | head -20` to see what's available.
2. Grep for `$ARGUMENTS` across all files in docs/midterm/.
3. Read the most relevant files at the matching lines (±30 lines of context).
4. Return relevant sections with **file path and line numbers**.
5. Strip LaTeX markup for readability where possible.
6. If docs/midterm/ does not exist, report: "Midterm docs not yet committed — ask user to commit them."
