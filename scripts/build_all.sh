#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"
PYTHONPATH="$ROOT/core" "$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m compileall -q core tests scripts
"$PYTHON" -m pip check
"$ROOT/scripts/build_android.sh"
"$ROOT/scripts/build_linux.sh"
echo "Linux and Android artifacts were built locally. Windows and macOS builds run on the appropriate runners in .github/workflows/build.yml."
