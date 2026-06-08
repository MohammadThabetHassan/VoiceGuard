<!-- Thanks for contributing to VoiceGuard! -->

## What & why

<!-- What does this PR change, and why? Link any related issue (e.g. "Closes #12"). -->

## Type of change

- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 📝 Docs
- [ ] ♻️ Refactor / chore
- [ ] ✅ Tests
- [ ] 🔧 CI / build

## Checklist

- [ ] `ruff check src/ tests/` and `ruff format --check src/ tests/` pass
- [ ] `pytest -q` passes (CPU-only — no GPU/model needed)
- [ ] `bandit -r src/ -ll --exclude src/voiceguard/training` is clean
- [ ] Frontend `npm run lint` passes (if frontend touched)
- [ ] Tests added/updated for the change
- [ ] Docs / CHANGELOG updated if user-facing
- [ ] Commits follow Conventional Commits (`type(scope): summary`)

## Notes for reviewers

<!-- Anything reviewers should focus on, screenshots, trade-offs, follow-ups. -->
