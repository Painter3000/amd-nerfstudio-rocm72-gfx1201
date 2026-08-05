#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${NERFSTUDIO_RDNA4_PUBLIC_PYTHON:-}"
NERFSTUDIO="${NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE:-}"
TCNN_RUNTIME="${NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME:-}"
DATASET_ARCHIVE="${NERFSTUDIO_RDNA4_PUBLIC_DATASET_ARCHIVE:-}"
DATASET="${NERFSTUDIO_RDNA4_PUBLIC_DATASET:-}"
OUTPUT_ROOT="${NERFSTUDIO_RDNA4_PUBLIC_OUTPUT_ROOT:-}"

required=(PYTHON NERFSTUDIO TCNN_RUNTIME DATASET_ARCHIVE DATASET OUTPUT_ROOT)
for name in "${required[@]}"; do
    if [[ -z "${!name}" ]]; then
        echo "Missing required environment variable backing $name" >&2
        exit 2
    fi
done

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

exec "$PYTHON" "$ROOT/tools/run_public_dev5_p0_p1_v1.py" \
    --python "$PYTHON" \
    --nerfstudio-worktree "$NERFSTUDIO" \
    --tcnn-runtime "$TCNN_RUNTIME" \
    --dataset-archive "$DATASET_ARCHIVE" \
    --dataset "$DATASET" \
    --output-root "$OUTPUT_ROOT" \
    "$@"
