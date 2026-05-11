# Electricity Price Forecast

Desktop Python app for forecasting future hourly electricity consumption (`kWh`) from SCADA-style CSV exports.

## Data Inputs

The app expects these CSV files:

- `data_kwh.csv`: hourly target values with `name,time,hour,value`
- `energy_log.csv`: cumulative KWH readings with `time,name,original_value_float`
- `data_pf.csv`: power factor readings
- `data_current.csv`: current readings
- `data_2026.csv`: wide electrical telemetry source including `P`, `PF`, current, and cumulative `KWH`

Raw timestamped CSV files are treated as UTC and converted to `Asia/Ho_Chi_Minh`. `data_kwh.csv` is treated as already local because it has separate `time` and `hour` columns.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Native macOS desktop app:

```bash
bash scripts/setup_native_macos.sh
source .venv-native/bin/activate
python -m electricity_forecast.app
```

This native path uses Homebrew Python 3.10 plus `python-tk@3.10`. The system Python in `/usr/bin/python3` is not suitable for this app because its native GUI bindings fail on this macOS setup.

Fallback local web UI:

```bash
source .venv/bin/activate
ELECTRICITY_FORECAST_UI=web python -m electricity_forecast.app
```

## Weather Temperature

The Forecast tab uses Open-Meteo to fetch the average monthly temperature from a selected location and month. The default location is `Hòn Thơm, Phú Quốc`; you can choose another preset location or type a place name / `lat,long`. Internet access is required for this API lookup.

## Anomaly Detection

The Anomaly tab uses Isolation Forest to flag unusual hourly readings. It prioritizes hourly kWh deltas calculated from cumulative `KWH` in `data_2026.csv`, then falls back to `data_kwh.csv` when telemetry deltas are missing or invalid. Results include anomaly score, severity, and a short reason such as low PF, high current, kWh spike, or telemetry reset/outlier.

## CLI Smoke Workflow

```bash
python -m electricity_forecast.cli \
  --kwh /Users/macbook/Downloads/data_kwh.csv \
  --pf /Users/macbook/Downloads/data_pf.csv \
  --current /Users/macbook/Downloads/data_current.csv \
  --telemetry /Users/macbook/Downloads/data_2026.csv \
  --weather-location "Hòn Thơm, Phú Quốc" \
  --weather-month 2026-01 \
  --horizon 168 \
  --output exports/forecast_168h.csv
```

CLI anomaly run:

```bash
python -m electricity_forecast.cli \
  --kwh /Users/macbook/Downloads/data_kwh.csv \
  --pf /Users/macbook/Downloads/data_pf.csv \
  --current /Users/macbook/Downloads/data_current.csv \
  --telemetry /Users/macbook/Downloads/data_2026.csv \
  --detect-anomalies \
  --anomaly-contamination 0.05 \
  --anomaly-output exports/anomalies.csv
```

## Package macOS App

```bash
bash scripts/build_macos_app.sh
```

For large datasets, keep CSV files outside the app bundle and select them from the Data tab.

## Package Windows Native App

Build this on a Windows machine. PyInstaller cannot create a Windows `.exe` from macOS.

```powershell
scripts\build_windows.ps1
```

The output is:

```text
dist\ElectricityForecast\ElectricityForecast.exe
```

The Windows build intentionally uses `onedir` instead of `onefile` so startup is fast. Distribute the whole `dist\ElectricityForecast` folder, not only the `.exe`.
