---
name: commit-f
description: Commit staged changes as Fahad Sadek (feature extraction, classical ML, Wav2Vec2, evaluation).
disable-model-invocation: true
allowed-tools:
  - Bash(git *)
argument-hint: "<conventional commit message>"
---

## Instructions

1. Verify staged changes exist: `git diff --cached --name-only`
2. Commit with Fahad's identity:
   ```bash
   git -c user.name="Fahad Sadek" \
       -c user.email="20220001790@students.cud.ac.ae" \
       commit -m "$ARGUMENTS"
   ```
3. Print the resulting SHA: `git rev-parse --short HEAD`
4. Append to docs/PROGRESS.md:
   ```bash
   bash scripts/append_progress.sh "$(git rev-parse --short HEAD)" "$ARGUMENTS"
   ```

Reject if `$ARGUMENTS` does not match conventional commit format:
`^(feat|fix|docs|chore|test|refactor|ci)(\(.+\))?: .+`
