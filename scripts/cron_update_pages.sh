#!/usr/bin/env bash
# Refresh docs/index.html from latest Kaggle LB + auto-commit/push.
#
# Run periodically (cron / /loop) to keep the GitHub Pages dashboard live.
# Idempotent: no diff => no commit. Safe to run any frequency.
#
# Usage:
#   scripts/cron_update_pages.sh
#   scripts/cron_update_pages.sh --no-fetch    # skip kaggle CLI, only refresh timestamps

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Pass through --no-fetch etc.
"$ROOT/scripts/run.sh" python3 "$ROOT/scripts/update_pages.py" "$@"

# Only commit when docs/index.html actually changed.
if git diff --quiet -- docs/index.html; then
    echo "no changes to docs/index.html — skipping commit"
    exit 0
fi

git add docs/index.html
git commit -m "Auto-update GitHub Pages dashboard from kaggle LB"
git push origin main
echo "GitHub Pages dashboard updated and pushed"
