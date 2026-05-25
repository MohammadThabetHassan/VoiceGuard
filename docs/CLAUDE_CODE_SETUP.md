# VoiceGuard — Claude Code Operator's Manual

This document explains every skill: what it does, when to use it,
and why its frontmatter is configured the way it is.

---

## Workflow Skills

### `/plan`
**What:** Reads `docs/PROGRESS.md`, `PHASE`, and recent git log to propose 3 concrete
next tasks for the current build phase.
**When:** Start of every session. Always run this first.
**Frontmatter:** Dynamic context loads progress + git state so the plan is
grounded in actual repo state, not stale memory.

---

### `/gate`
**What:** Runs `ruff` → `black` → `pytest` → `bandit` in sequence. Stops on first failure.
**When:** Before any commit. Automatically called by `/safe-push`.
**Frontmatter:** `allowed-tools` restricts to only lint/test commands — prevents
accidental file modification during a quality check.

---

### `/commit-m`, `/commit-f`, `/commit-a`
**What:** Commit staged changes under a specific team member's identity.
- `/commit-m` → Mohammad Thabet (DSFNet, FastAPI, CI/CD, Docker, tests)
- `/commit-f` → Fahad Sadek (features, classical ML, Wav2Vec2, evaluation)
- `/commit-a` → Ahmed Alameri (React, XTTS, watermarking, Twilio, forensics, XAI)

**When:** After `git add <files>`, call the skill matching the topic owner.
**Argument:** A conventional commit message, e.g.:
```
/commit-m feat(dsfnet): add dual-stream feature extraction module
```
**Frontmatter:** `disable-model-invocation: true` — git commit should never
trigger LLM reasoning. The commit message comes entirely from the argument.
Validates conventional commit format before committing.

---

### `/safe-push`
**What:** Runs `/gate`, then pushes to GitHub via `$GITHUB_TOKEN`. Never echoes the token.
**When:** When a feature is ready to publish.
**Frontmatter:** `disable-model-invocation: true` — push is a mechanical operation.
`allowed-tools` restricted to git + lint commands only.

---

### `/progress-log`
**What:** Appends a timestamped milestone entry to `docs/PROGRESS.md`.
**When:** After completing a significant task (not every commit — use for milestones).
**Argument:** `<description> [commit-sha]`
**Frontmatter:** `disable-model-invocation: true` — pure string formatting.

---

### `/session-summary`
**What:** Writes a 4-bullet session summary (done / next / blockers / cost) and
appends it to `docs/SESSIONS.md`.
**When:** Before ending any work session.
**Frontmatter:** Dynamic context loads recent git log and PROGRESS tail so the
summary reflects actual work, not recalled state.

---

## AWS Skills

### `/aws-status`
**What:** Lists running EC2 instances in ap-southeast-1 and reports month-to-date spend.
Alerts if GPU instances are running or budget thresholds ($50/$100/$150) are exceeded.
**When:** Session start and end. Run it before any AWS work.
**Frontmatter:** Dynamic context fetches live AWS state directly.

---

### `/aws-launch-gpu`
**What:** Generates a GPU launch script at `scripts/aws/launch_<task>.sh`.
**When:** When ready to start a training job (DSFNet, Wav2Vec2, etc.).
**IMPORTANT:** This skill generates the script but does NOT execute it.
Review the script, fill in AMI_ID + KEY_NAME + SECURITY_GROUP, confirm the
cost estimate, then run it manually.
**Argument:** `<task-name> <estimated-hours>`
**Frontmatter:** `disable-model-invocation: true` — script generation is templated,
not something that should involve LLM reasoning about AWS state.

---

### `/aws-teardown`
**What:** Lists running EC2 instances and shows terminate commands.
**When:** When a GPU training job completes or you suspect forgotten instances.
**IMPORTANT:** Requires explicit "yes go" confirmation before terminating anything.
**Frontmatter:** `disable-model-invocation: true` — termination is destructive;
no reasoning should happen between "show list" and "terminate".

---

## Research Skills

All research skills use `context: fork` (where supported) to avoid polluting the
main context with raw documentation or code dumps.

### `/research-midterm`
**What:** Searches `docs/midterm/` for a topic and returns relevant sections
with file/line references.
**When:** When implementing a feature — check the spec first.
**Argument:** `<topic or section>`

---

### `/research-old-repo`
**What:** Explores `/tmp/voiceguard-old` (read-only) for reference implementations.
Returns code annotated as "port as-is", "adapt", or "skip".
**When:** When porting the SM2026 baseline or verifying existing behavior.
**Argument:** `<file or feature>`
**IMPORTANT:** Never modify `/tmp/voiceguard-old`.

---

### `/research-docs`
**What:** Fetches and summarizes official documentation for project libraries
(PyTorch, FastAPI, XTTS, captum, SHAP, Twilio, etc.).
**When:** When integrating a new library or checking API signatures.
**Argument:** `<library> <topic>`
**Reference:** `.claude/skills/research-docs/reference.md` has canonical URLs.

---

### `/research-github`
**What:** Searches GitHub for prior art using `gh search`.
**When:** Before building a novel component — check if good reference implementations exist.
**Argument:** `<search query>`
**Note:** Requires `gh auth login` if GitHub CLI authentication fails.

---

### `/load-midterm-context`
**What:** Loads a midterm spec section directly into main context (cleaned of LaTeX).
**When:** Only when actively implementing a component and you need the exact spec inline.
**Argument:** `<section number or topic>`
**Use sparingly** — this consumes main context tokens. Prefer `/research-midterm`
for lookup without context pollution.

---

## Scripts

### `scripts/make_commit.sh`
CLI alternative to commit skills. Validates conventional commits, supports
`--co` flag for co-author trailers.
```
bash scripts/make_commit.sh --m -m "feat(dsfnet): add cross-attention"
bash scripts/make_commit.sh --f -m "feat(features): add pitch extraction" --co m
```

### `scripts/append_progress.sh`
Appends a timestamped line to `docs/PROGRESS.md`. Called automatically by commit scripts.

### `scripts/doctor.sh`
Full environment health check. Run at session start if something seems off.
Reports ✓/✗ for: Python, Node, git, docker, aws, gh, ruff, black, pytest,
bandit, pre-commit, npm, GITHUB_TOKEN, S3, pre-commit hooks, skill files, CLAUDE.md.

### `scripts/aws/teardown.sh`
Interactive EC2 termination for ap-southeast-1. Interactive — requires "yes go".

---

## Conventional Commits

```
feat(scope): new feature
fix(scope): bug fix
docs(scope): documentation
chore(scope): maintenance (deps, config, tooling)
test(scope): tests
refactor(scope): restructure without behaviour change
ci(scope): CI/CD changes
```

Common scopes: `dsfnet`, `features`, `api`, `frontend`, `docker`, `ci`,
`forensics`, `xai`, `watermark`, `voip`, `eval`, `baseline`

## Guardrails Reminder

- Never commit `.env`, `~/.voiceguard-env`, `*.pt`, `*.wav`, `data/`
- Always run `/gate` before pushing
- Always ask before any AWS resource launch
- Never modify `docs/midterm/`
- Never touch `/tmp/voiceguard-old`
