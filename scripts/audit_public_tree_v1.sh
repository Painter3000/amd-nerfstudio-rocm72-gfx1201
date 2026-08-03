#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" "$REPO_ROOT/tools/audit_public_tree_v1.py" --repo "$REPO_ROOT" "$@"
