#!/usr/bin/env bash
# Reproduce every public experiment behind main_gap_paper.tex through the
# unified runner (main.py). All outputs land in the centralized artifacts/ tree
# and a machine-readable summary is written to artifacts/run_summary.json.
#
# Usage:
#   bash scripts/reproduce_all.sh                  # CPU public set on the real corpora
#   bash scripts/reproduce_all.sh --with-gpu       # also run the GPU deep blocks (E14, S3)
#   bash scripts/reproduce_all.sh --smoke          # offline synthetic smoke (no downloads)
#   bash scripts/reproduce_all.sh --industrial-csv data/flows.csv   # include N1 on real data
#
# Prereqs (see README.md):
#   pip install -e ".[api,dev,notebooks]"
#
# The TelecomTS corpus downloads from HuggingFace on first run; SpotLight and the
# operator industrial CSV are external (see DATA_AVAILABILITY in the README).
set -euo pipefail
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${REPO_ROOT}"

PRESET="paper"
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --smoke)    PRESET="smoke" ;;
    --with-gpu) EXTRA_ARGS+=("--with-gpu") ;;
    *)          EXTRA_ARGS+=("$arg") ;;   # pass through e.g. --industrial-csv PATH
  esac
done

PYTHON="${PYTHON:-python3}"

echo "[reproduce] preset=${PRESET}  extra='${EXTRA_ARGS[*]:-}'"
echo "[reproduce] start at $(date)"

# --all runs every registered experiment; GPU/external-data ones self-skip
# unless --with-gpu / --industrial-csv are supplied.
"${PYTHON}" main.py --all --preset "${PRESET}" "${EXTRA_ARGS[@]:-}"

echo "[reproduce] done. See artifacts/run_summary.json for the per-experiment status."
