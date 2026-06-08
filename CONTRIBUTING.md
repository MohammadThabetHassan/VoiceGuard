# Contributing to VoiceGuard

Thanks for your interest in VoiceGuard! This is a graduation research project
(GP2) on voice deepfake detection, synthesis watermarking, and vishing defence.
Contributions — bug reports, fixes, docs, and ideas — are welcome.

## Ways to contribute

- 🐛 **Report a bug** — open a [bug report](.github/ISSUE_TEMPLATE/bug_report.yml).
- 💡 **Request a feature** — open a [feature request](.github/ISSUE_TEMPLATE/feature_request.yml).
- 🔒 **Report a vulnerability** — see [SECURITY.md](SECURITY.md) (please do **not** open a public issue).
- 📝 **Improve docs**, ✅ **add tests**, or 🔧 **fix a bug** via a pull request.

## Development setup

```bash
git clone https://github.com/MohammadThabetHassan/VoiceGuard.git
cd VoiceGuard

# Backend (Python 3.12). The package is import-run via PYTHONPATH; on
# externally-managed environments use a venv.
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Frontend
cd frontend && npm ci
```

Run the API locally (the production detector needs a checkpoint; for most work
the lightweight `classical` model is enough):

```bash
PYTHONPATH=src SECRET_KEY="$(openssl rand -hex 32)" \
  uvicorn voiceguard.api.main:app --host 127.0.0.1 --port 8000
```

## Before you open a PR

Run the same gate CI runs — all four must pass:

```bash
ruff check src/ tests/          # lint
ruff format --check src/ tests/ # formatting
pytest -q                       # tests (CPU-only; no GPU/model needed)
bandit -r src/ -ll --exclude src/voiceguard/training
cd frontend && npm run lint     # frontend
```

Please add or update tests for any behaviour change, and keep coverage healthy.

## Commit & PR conventions

- **Conventional Commits**: `type(scope): summary` — e.g. `fix(api): …`,
  `feat(eval): …`, `docs(readme): …`. Types: `feat, fix, docs, chore, test,
  refactor, ci, build, perf`.
- Keep PRs focused; describe **what** and **why**, and link any issue.
- Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
- By contributing you agree your work is licensed under the project's
  [Apache 2.0](LICENSE) license.

## Code style

- Python: [ruff](https://docs.astral.sh/ruff/) (lint + format), type hints,
  module-level docstrings.
- Tests: `pytest` + `pytest-asyncio`; API tests use `httpx.ASGITransport`
  (no network) and monkeypatch heavy ML deps so they run CPU-only.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you agree to uphold it.
