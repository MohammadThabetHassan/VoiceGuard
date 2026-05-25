---
name: session-summary
description: Summarize session accomplishments. Use before exiting.
---

## Dynamic Context

!`git log --oneline --since="8 hours ago" 2>/dev/null`
!`tail -20 docs/PROGRESS.md 2>/dev/null`
!`git status --short 2>/dev/null`

## Instructions

Write a 4-bullet summary and append it to `docs/SESSIONS.md`:

```
## Session — YYYY-MM-DD HH:MM

- **Done:** <what was completed>
- **Next:** <immediate next step>
- **Blockers:** <none | description>
- **Cost:** <AWS spend this session, or "none">
```

Then show the summary in the response.
