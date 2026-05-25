---
name: research-github
description: Search GitHub for prior art, reference implementations, issues, or solutions.
context: fork
agent: Explore
allowed-tools:
  - Bash(gh search *)
  - Bash(gh api *)
  - Bash(gh repo view *)
argument-hint: "<search query>"
---

## Instructions

Search GitHub for: `$ARGUMENTS`

Use available gh commands:
```bash
gh search code "$ARGUMENTS" --limit 10
gh search repos "$ARGUMENTS" --limit 10 --sort stars
gh search issues "$ARGUMENTS" --limit 10
```

Return:
- Repository names and URLs
- Relevant code snippets with file paths
- Issue titles/links if relevant
- Assessment: "port as-is" / "adapt" / "reference only"

Note: `gh` may require auth. If auth fails, report the error and suggest the user run
`echo $GITHUB_TOKEN | gh auth login --with-token`.
