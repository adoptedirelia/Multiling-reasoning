#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" -m src.evaluation.translation_audit.build_translation_quality_analysis
