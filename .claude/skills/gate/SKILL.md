---
name: gate
description: Run full quality gate — ruff, black, pytest, bandit. Stop on first failure.
allowed-tools:
  - Bash(ruff *)
  - Bash(black *)
  - Bash(pytest *)
  - Bash(bandit *)
---

## Instructions

Run each check in sequence. Stop and report failure immediately if any step fails.

```bash
TOOLS=/home/ubuntu/.voiceguard-tools/bin
SRC=src

echo "=== ruff ==="
$TOOLS/ruff check $SRC 2>/dev/null || { echo "FAIL: ruff"; exit 1; }
echo "PASS"

echo "=== black ==="
$TOOLS/black --check $SRC 2>/dev/null || { echo "FAIL: black"; exit 1; }
echo "PASS"

echo "=== pytest ==="
$TOOLS/pytest tests/ -q 2>/dev/null || { echo "FAIL: pytest"; exit 1; }
echo "PASS"

echo "=== bandit ==="
$TOOLS/bandit -r $SRC -ll -q 2>/dev/null || { echo "FAIL: bandit"; exit 1; }
echo "PASS"

echo "=== GATE PASSED ==="
```

If `src/` does not exist yet, report "no src/ to lint — gate passes vacuously" and exit 0.
