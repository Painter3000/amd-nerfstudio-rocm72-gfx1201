#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_LAUNCHER="${NERFSTUDIO_RDNA4_ADAPTIVE_LAUNCHER_PYTHON:-python3}"
PAUSE_ON_ERROR="${NERFSTUDIO_RDNA4_ADAPTIVE_PAUSE_ON_ERROR:-1}"

on_error() {
  local rc=$?
  set +e
  trap - ERR
  echo >&2
  echo "PUBLIC_RDNA4_ADAPTIVE_ENV_WRAPPER: FAIL" >&2
  echo "RETURN_CODE=$rc" >&2
  if [[ "$PAUSE_ON_ERROR" == "1" && -t 0 && -t 1 ]]; then
    echo
    read -r -p "Fehler sichtbar. Mit Enter beenden ..." _unused
  fi
  exit "$rc"
}
trap on_error ERR

"$PYTHON_LAUNCHER" "$REPO_ROOT/tools/setup_public_adaptive_env_v1.py" "$@"
echo "PUBLIC_RDNA4_ADAPTIVE_ENV_WRAPPER: PASS"
