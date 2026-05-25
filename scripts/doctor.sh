#!/usr/bin/env bash
# VoiceGuard environment health check
# Run: bash scripts/doctor.sh

set -uo pipefail

PASS=0
FAIL=0
WARN=0

ok()   { echo "  ✓ $1"; ((PASS++)); }
fail() { echo "  ✗ $1"; ((FAIL++)); }
warn() { echo "  ⚠ $1"; ((WARN++)); }

check_cmd() {
  local label="$1"; local cmd="$2"
  if command -v "$cmd" &>/dev/null; then
    ok "$label: $(${cmd} --version 2>&1 | head -1)"
  else
    fail "$label: not found"
  fi
}

check_version() {
  local label="$1"; local cmd="$2"; local min_major="$3"; local min_minor="${4:-0}"
  if ! command -v "$cmd" &>/dev/null; then
    fail "$label: not found"
    return
  fi
  version=$(${cmd} --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
  major=$(echo "$version" | cut -d. -f1)
  minor=$(echo "$version" | cut -d. -f2)
  if [[ "$major" -gt "$min_major" ]] || { [[ "$major" -eq "$min_major" ]] && [[ "$minor" -ge "$min_minor" ]]; }; then
    ok "$label: $version"
  else
    fail "$label: $version (need >=$min_major.$min_minor)"
  fi
}

echo ""
echo "═══════════════════════════════════════"
echo "  VoiceGuard Environment Doctor"
echo "═══════════════════════════════════════"
echo ""

# ── Language runtimes ───────────────────────────────────────────
echo "── Language runtimes ──"
check_version "Python" "python3" 3 10
check_version "Node" "node" 18 0

# ── Core tools ──────────────────────────────────────────────────
echo ""
echo "── Core tools ──"
check_cmd "git" "git"
check_cmd "docker" "docker"
check_cmd "aws cli" "aws"
check_cmd "gh" "gh"

# ── Python dev tools ────────────────────────────────────────────
echo ""
echo "── Python dev tools ──"
TOOLS=/home/ubuntu/.voiceguard-tools/bin
for tool in ruff black pytest bandit pre-commit; do
  if [[ -x "$TOOLS/$tool" ]]; then
    ok "$tool: $($TOOLS/$tool --version 2>&1 | head -1)"
  else
    fail "$tool: not in $TOOLS"
  fi
done

# ── npm ─────────────────────────────────────────────────────────
echo ""
echo "── npm ──"
check_cmd "npm" "npm"

# ── GitHub token ────────────────────────────────────────────────
echo ""
echo "── GitHub ──"
if [[ -f ~/.voiceguard-env ]]; then
  source ~/.voiceguard-env 2>/dev/null
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    # Test token without echoing it
    if git ls-remote "https://${GITHUB_TOKEN}@github.com/MohammadThabetHassan/VoiceGuard.git" &>/dev/null; then
      ok "GITHUB_TOKEN: valid (git access confirmed)"
    else
      fail "GITHUB_TOKEN: set but git access failed"
    fi
  else
    fail "GITHUB_TOKEN: ~/.voiceguard-env exists but token not set"
  fi
else
  fail "GITHUB_TOKEN: ~/.voiceguard-env not found"
fi

# ── AWS ─────────────────────────────────────────────────────────
echo ""
echo "── AWS ──"
if command -v aws &>/dev/null; then
  if aws sts get-caller-identity &>/dev/null; then
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
    ok "AWS auth: account $ACCOUNT"
  else
    warn "AWS auth: not configured (attach IAM role or run 'aws configure')"
  fi
  if aws s3 ls "s3://voiceguard-mt-2026/" &>/dev/null; then
    ok "S3 bucket: voiceguard-mt-2026 accessible"
  else
    warn "S3 bucket: voiceguard-mt-2026 not accessible (expected if no IAM role)"
  fi
else
  fail "AWS CLI: not installed"
fi

# ── Pre-commit hooks ────────────────────────────────────────────
echo ""
echo "── Pre-commit hooks ──"
REPO=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
if [[ -f "$REPO/.pre-commit-config.yaml" ]]; then
  ok ".pre-commit-config.yaml: found"
else
  fail ".pre-commit-config.yaml: missing"
fi
if [[ -f "$REPO/.git/hooks/pre-commit" ]]; then
  ok "pre-commit hook: installed"
else
  warn "pre-commit hook: not installed (run: pre-commit install)"
fi
if git config core.hooksPath | grep -q ".githooks"; then
  ok "commit-msg hook: .githooks configured"
else
  warn "commit-msg hook: not configured (run: git config core.hooksPath .githooks)"
fi

# ── Skills ──────────────────────────────────────────────────────
echo ""
echo "── Claude Code skills ──"
SKILLS_DIR="$REPO/.claude/skills"
if [[ -d "$SKILLS_DIR" ]]; then
  FAIL_SKILLS=0
  for skill_dir in "$SKILLS_DIR"/*/; do
    skill=$(basename "$skill_dir")
    skill_file="$skill_dir/SKILL.md"
    if [[ ! -f "$skill_file" ]]; then
      fail "skill/$skill: SKILL.md missing"
      FAIL_SKILLS=1
    elif ! grep -q "^---$" "$skill_file"; then
      fail "skill/$skill: missing YAML frontmatter (--- markers)"
      FAIL_SKILLS=1
    elif ! grep -q "^name:" "$skill_file"; then
      fail "skill/$skill: missing 'name:' in frontmatter"
      FAIL_SKILLS=1
    else
      ok "skill/$skill: valid"
    fi
  done
  [[ $FAIL_SKILLS -eq 0 ]] && true || true
else
  fail ".claude/skills/: directory missing"
fi

# ── CLAUDE.md ───────────────────────────────────────────────────
echo ""
echo "── Project files ──"
[[ -f "$REPO/CLAUDE.md" ]] && ok "CLAUDE.md: found" || fail "CLAUDE.md: missing"
[[ -f "$REPO/PHASE" ]] && ok "PHASE: $(cat $REPO/PHASE)" || fail "PHASE: missing"
[[ -f "$REPO/docs/PROGRESS.md" ]] && ok "docs/PROGRESS.md: found" || fail "docs/PROGRESS.md: missing"

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════"
echo "  Results: ${PASS} passed | ${FAIL} failed | ${WARN} warnings"
echo "═══════════════════════════════════════"
echo ""

if [[ $FAIL -gt 0 ]]; then
  echo "Fix failing checks before proceeding."
  exit 1
else
  echo "Environment is ready."
  exit 0
fi
