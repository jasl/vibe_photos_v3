#!/usr/bin/env bash
# Phase 1 bootstrap helper. Validates prerequisites and mirrors quick-start automation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python3"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: python3 is not available on PATH. Install Python 3.12 or update PYTHON_BIN." >&2
  exit 1
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" blueprints/phase1/quick_start.py
