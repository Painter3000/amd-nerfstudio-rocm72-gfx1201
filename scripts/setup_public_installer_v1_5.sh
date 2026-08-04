#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_LAUNCHER="${NERFSTUDIO_RDNA4_INSTALLER_PYTHON:-python3.12}"

exec "$PYTHON_LAUNCHER" "$REPO_ROOT/amd_nerfstudio_setup.py" "$@"
