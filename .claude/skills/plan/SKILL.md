---
name: plan
description: Read PROGRESS.md and propose next 3 tasks for the current build phase.
---

## Dynamic Context

!`cat docs/PROGRESS.md 2>/dev/null | tail -30`
!`cat PHASE 2>/dev/null || echo "phase-0"`
!`git log --oneline -10 2>/dev/null`
!`git status --short 2>/dev/null`

## Instructions

1. Read the dynamic context above to understand the current phase and recent work.
2. Identify the current build phase from the PHASE file.
3. Reference CLAUDE.md for the phase's scope and acceptance criteria.
4. Propose exactly 3 tasks: the smallest concrete next steps that advance the current phase.
   - Each task must be specific and completable in one session.
   - Order by dependency (unblocked first).
   - Flag any blockers.
5. **WAIT for user "go" before starting any task.**

Format:
```
Phase N — <phase name>

Task 1: <title>
  What: <one sentence>
  Files: <files to create/modify>
  Done when: <acceptance check>

Task 2: ...

Task 3: ...

Blockers: <none | description>
```
