#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.10}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing $PYTHON_BIN. Install Python 3.10 with Homebrew first: brew install python@3.10" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import tkinter" >/dev/null 2>&1; then
  echo "Missing tkinter for Python 3.10. Installing python-tk@3.10 with Homebrew..."
  brew install python-tk@3.10
fi

"$PYTHON_BIN" -m venv --clear .venv-native
.venv-native/bin/python -m pip install --upgrade pip
.venv-native/bin/python -m pip install -r requirements.txt

cat <<'MSG'
Native desktop environment is ready.

Run:
  source .venv-native/bin/activate
  python -m electricity_forecast.app
MSG
