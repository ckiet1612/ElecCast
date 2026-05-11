#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x ".venv-native/bin/python" ]; then
  bash scripts/setup_native_macos.sh
fi

ELECTRICITY_FORECAST_UI=tk .venv-native/bin/python -m PyInstaller \
  --name ElectricityForecast \
  --windowed \
  --onefile \
  run_app.py
