#!/usr/bin/env bash
# Run a command inside the working Python env. See scripts/env.sh for details.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/env.sh"
exec "$@"
