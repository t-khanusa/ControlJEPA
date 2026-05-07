#!/usr/bin/env bash
# Runs a command inside the project's Conda environment.
#
# Always use this (or activate manually) before training or validation —
# deps (torch/transformers/peft) are expected on env `controlJEPA`.
#
#   conda activate controlJEPA           # interactive
#   bash scripts/run_under_control_jepa.sh python scripts/validate_preserving_diversity.py
#
# Override env name:
#   CONTROLJEPA_CONDA_ENV=my_other_env bash scripts/run_under_control_jepa.sh ...

set -euo pipefail
ENV_NAME="${CONTROLJEPA_CONDA_ENV:-controlJEPA}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not on PATH. Install conda or run:" >&2
  echo "       conda activate ${ENV_NAME} && ./scripts/<...>" >&2
  exit 127
fi

exec conda run --no-capture-output -n "$ENV_NAME" -- "$@"
