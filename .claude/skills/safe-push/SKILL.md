---
name: safe-push
description: Run /gate then push to GitHub. Use when ready to publish.
disable-model-invocation: true
allowed-tools:
  - Bash(git *)
  - Bash(ruff *)
  - Bash(black *)
  - Bash(pytest *)
  - Bash(bandit *)
---

## Instructions

1. Run the full gate (ruff → black → pytest → bandit). Stop if any fails.
2. If gate passes, push via token — never echo the token:
   ```bash
   source ~/.voiceguard-env
   git push "https://${GITHUB_TOKEN}@github.com/MohammadThabetHassan/VoiceGuard.git" main
   ```
3. Report the push result and current HEAD SHA.
4. If pushing a tag: `git push "https://${GITHUB_TOKEN}@github.com/MohammadThabetHassan/VoiceGuard.git" <tagname>`

**Never echo `$GITHUB_TOKEN` in output. Never commit `~/.voiceguard-env`.**
