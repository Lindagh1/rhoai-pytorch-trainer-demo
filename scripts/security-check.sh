#!/usr/bin/env bash
# Scans the TRACKED working tree (never .gitignore'd files) for the secret patterns this
# demo must never commit: OpenShift/Kubernetes tokens, GitHub tokens, AWS/S3 keys,
# kubeconfigs, passwords, MinIO/MLflow credentials, and routes/URLs with embedded
# credentials. Run this before every push (`make security-check`, also wired into
# `make validate`).
#
# This is a defense-in-depth check, not a replacement for judgment: it only catches
# patterns that LOOK like real secrets, so still read `git diff` yourself before pushing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh

FOUND=0

log_section "Scanning git-tracked files for likely secrets"

# Scan tracked files AND untracked-but-not-ignored files (so new files not yet committed
# are caught before their first `git add`), but never anything .gitignore already excludes
# (build artifacts, .venv, notebooks/.ipynb_checkpoints, etc).
FILES=$(git ls-files --cached --others --exclude-standard)

check_pattern() {
  local description="$1" pattern="$2"
  local matches
  matches=$(echo "$FILES" | xargs -I{} grep -liE "$pattern" {} 2>/dev/null || true)
  if [ -n "$matches" ]; then
    log_fail "${description}:"
    echo "$matches" | sed 's/^/    /'
    FOUND=1
  fi
}

check_pattern "OpenShift/Kubernetes bearer token (sha256~...)"   'sha256~[A-Za-z0-9_-]{20,}'
check_pattern "GitHub personal access token (ghp_/gho_/ghs_...)" '(ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{20,}'
check_pattern "AWS access key ID (AKIA...)"                       'AKIA[0-9A-Z]{16}'
check_pattern "hardcoded AWS_SECRET_ACCESS_KEY assignment"        'AWS_SECRET_ACCESS_KEY[[:space:]]*[:=][[:space:]]*[A-Za-z0-9/+=]{30,}'
check_pattern "embedded kubeconfig 'current-context:' block"      '^current-context:'
check_pattern "private key block (BEGIN ... PRIVATE KEY)"         'BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY'
check_pattern "URL with embedded basic-auth credentials"          'https?://[^/[:space:]]+:[^/[:space:]@]{6,}@'
check_pattern "hardcoded password= assignment (non-template)"     "password[[:space:]]*[:=][[:space:]]*['\"][^'\"\$]{6,}['\"]"

echo ""
if [ "$FOUND" -ne 0 ]; then
  log_fail "Potential secret(s) found in tracked files -- see above. Do NOT push until resolved."
  echo "If this is a false positive (e.g. a template placeholder), adjust the pattern in scripts/security-check.sh"
  echo "or add the file to .gitignore if it should never be tracked at all."
  exit 1
fi

log_pass "No obvious secrets found in git-tracked files."
exit 0
