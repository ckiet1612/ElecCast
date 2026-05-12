from __future__ import annotations

import html
import socket
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from .anomaly import detect_anomalies as run_anomaly_detection
from .data import summarize_paths
from .features import build_feature_table, feature_summary
from .models import forecast_dataframe, train_models
from .types import AnomalyRequest, DataPaths, ForecastRequest
from .weather import (
    default_weather_month,
    default_weather_location_label,
    month_bounds,
    month_options,
    monthly_average_temperature,
)


@dataclass
class WebState:
    paths: dict[str, str] = field(default_factory=dict)
    feature_table: object | None = None
    trained_models: dict = field(default_factory=dict)
    metrics_df: object | None = None
    forecast_df: object | None = None
    anomaly_df: object | None = None
    anomaly_only: bool = True
    weather_result: object | None = None
    message: str = ""
    error: str = ""
    csv_summary: list[dict[str, object]] = field(default_factory=list)
    feature_summary: dict[str, object] = field(default_factory=dict)


def run_app(
    host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True
) -> int:
    state = WebState(paths=_default_paths())
    server = ThreadingHTTPServer((host, _available_port(host, port)), _handler(state))
    url = f"http://{host}:{server.server_port}"
    print(f"Electricity Forecast running at {url}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Electricity Forecast.")
    finally:
        server.server_close()
    return 0


def _handler(state: WebState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/download/forecast"):
                self._download_df(state.forecast_df, "forecast.csv")
                return
            if self.path.startswith("/download/metrics"):
                self._download_df(state.metrics_df, "metrics.csv")
                return
            if self.path.startswith("/download/anomalies"):
                self._download_df(state.anomaly_df, "anomalies.csv")
                return
            self._send_html(_render(state))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            action = self.path.strip("/")
            try:
                if action == "import":
                    _import_action(state, form)
                elif action == "train":
                    _train_action(state, form)
                elif action == "forecast":
                    _forecast_action(state, form)
                elif action == "anomaly":
                    _anomaly_action(state, form)
                else:
                    state.error = f"Unknown action: {action}"
            except Exception as exc:  # pragma: no cover - UI safety net
                state.error = str(exc)
            self._redirect("/")

        def log_message(self, format, *args):  # noqa: A002
            return

        def _send_html(self, content: str):
            body = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, location: str):
            self.send_response(303)
            self.send_header("Location", location)
            self.end_headers()

        def _download_df(self, df, filename: str):
            if df is None or df.empty:
                self.send_error(404, "No data to download")
                return
            body = df.to_csv(index=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _import_action(state: WebState, form: dict[str, list[str]]) -> None:
    state.error = ""
    state.paths = {key: _first(form, key) for key in _path_keys()}
    paths = _data_paths(state.paths)
    state.csv_summary = summarize_paths(paths)
    state.feature_table = build_feature_table(paths)
    state.feature_summary = feature_summary(state.feature_table)
    state.trained_models = {}
    state.metrics_df = None
    state.forecast_df = None
    state.anomaly_df = None
    state.weather_result = None
    state.message = "Imported data and built feature table."


def _train_action(state: WebState, form: dict[str, list[str]]) -> None:
    state.error = ""
    if state.feature_table is None:
        raise ValueError("Import data before training.")
    meter = _first(form, "train_meter")
    meters = None if not meter or meter == "All meters" else [meter]
    state.trained_models, state.metrics_df = train_models(
        state.feature_table, meters=meters
    )
    state.message = "Training and backtest completed."


def _forecast_action(state: WebState, form: dict[str, list[str]]) -> None:
    state.error = ""
    if state.feature_table is None or not state.trained_models:
        raise ValueError("Import data and train models before forecasting.")
    meter = _first(form, "forecast_meter")
    meters = (
        list(state.trained_models) if not meter or meter == "All meters" else [meter]
    )
    weather = monthly_average_temperature(
        _first(form, "weather_location") or default_weather_location_label(),
        _first(form, "weather_month") or default_weather_month(),
    )
    request = ForecastRequest(
        meters=meters,
        horizon_hours=int(_first(form, "horizon_hours") or 168),
        temperature_c=weather.average_c,
        weather_location=weather.location_label,
        weather_month=weather.month,
    )
    state.forecast_df = forecast_dataframe(
        state.trained_models, state.feature_table, request
    )
    state.weather_result = weather
    state.message = "Forecast completed."


def _anomaly_action(state: WebState, form: dict[str, list[str]]) -> None:
    state.error = ""
    if state.feature_table is None:
        raise ValueError("Import data before detecting anomalies.")
    meter = _first(form, "anomaly_meter")
    meters = None if not meter or meter == "All meters" else [meter]
    state.anomaly_only = _first(form, "anomaly_only") == "on"
    state.anomaly_df = run_anomaly_detection(
        state.feature_table,
        AnomalyRequest(
            meters=meters,
            contamination=float(_first(form, "anomaly_contamination") or 0.05),
        ),
    )
    state.message = "Anomaly detection completed."


def _render(state: WebState) -> str:
    meters = _meters(state)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Electricity Forecast</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; color: #17202a; }}
    header {{ padding: 16px 24px; border-bottom: 1px solid #d8dee4; background: #f6f8fa; }}
    main {{ padding: 18px 24px 40px; }}
    section {{ margin-bottom: 22px; }}
    h1 {{ font-size: 22px; margin: 0; }}
    h2 {{ font-size: 17px; margin: 0 0 10px; }}
    label {{ display: block; font-size: 13px; font-weight: 600; margin: 8px 0 4px; }}
    input, select {{ width: 100%; box-sizing: border-box; padding: 7px 9px; border: 1px solid #c9d1d9; border-radius: 6px; }}
    button, a.button {{ display: inline-block; padding: 8px 12px; border: 1px solid #1f6feb; background: #1f6feb; color: white; border-radius: 6px; text-decoration: none; cursor: pointer; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 18px; }}
    .row {{ display: flex; gap: 10px; align-items: end; flex-wrap: wrap; }}
    .row > div {{ min-width: 180px; }}
    .notice {{ padding: 10px 12px; border-radius: 6px; margin-bottom: 14px; background: #e7f5ff; }}
    .error {{ padding: 10px 12px; border-radius: 6px; margin-bottom: 14px; background: #ffebe9; color: #82071e; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 6px 8px; text-align: left; white-space: nowrap; }}
    th {{ background: #f6f8fa; }}
    .table-wrap {{ overflow: auto; max-height: 360px; border: 1px solid #d8dee4; }}
    svg {{ width: 100%; height: 280px; border: 1px solid #d8dee4; background: white; }}
    .muted {{ color: #57606a; font-size: 13px; }}
  </style>
</head>
<body>
<header><h1>Electricity Forecast</h1><div class="muted">Local desktop web app</div></header>
<main>
  {_message(state)}
  <section>
    <h2>Data</h2>
    <p class="muted">Import uses only data_2026.csv. The app derives hourly kWh, P, PF, and current from this file.</p>
    <form method="post" action="/import">
      <div class="grid">{_path_inputs(state.paths)}</div>
      <p><button type="submit">Import / Build Features</button></p>
    </form>
    {_summaries(state)}
  </section>
  <section>
    <h2>Training</h2>
    <form method="post" action="/train" class="row">
      <div><label>Meter</label>{_meter_select("train_meter", meters)}</div>
      <button type="submit">Train / Backtest</button>
    </form>
    {_df_table(state.metrics_df)}
  </section>
  <section>
    <h2>Forecast</h2>
    <form method="post" action="/forecast" class="row">
      <div><label>Meter</label>{_meter_select("forecast_meter", list(state.trained_models) if state.trained_models else meters)}</div>
      <div><label>Horizon</label><select name="horizon_hours"><option value="24">24 hours</option><option value="48">48 hours</option><option value="168" selected>168 hours</option><option value="720">30 days</option></select></div>
      <div><label>Location</label><input name="weather_location" value="{html.escape(_weather_location_value(state))}"></div>
      <div><label>Month</label>{_month_select(_weather_month_value(state))}</div>
      <div><label>Avg Temp</label><input value="{html.escape(_weather_temperature_value(state))}" disabled></div>
      <button type="submit">Forecast</button>
    </form>
    {_forecast_svg(state.forecast_df)}
    {_df_table(state.forecast_df.head(500) if state.forecast_df is not None else None)}
  </section>
  <section>
    <h2>Anomaly</h2>
    <form method="post" action="/anomaly" class="row">
      <div><label>Meter</label>{_meter_select("anomaly_meter", meters)}</div>
      <div><label>Contamination</label><input name="anomaly_contamination" value="0.05"></div>
      <div><label>Only anomalies</label><input type="checkbox" name="anomaly_only" {_checked(state.anomaly_only)}></div>
      <button type="submit">Detect Anomalies</button>
    </form>
    {_anomaly_svg(state.anomaly_df)}
    {_df_table(_anomaly_table_df(state))}
  </section>
  <section>
    <h2>Export</h2>
    <a class="button" href="/download/forecast">Download Forecast CSV</a>
    <a class="button" href="/download/metrics">Download Metrics CSV</a>
    <a class="button" href="/download/anomalies">Download Anomaly CSV</a>
  </section>
</main>
</body>
</html>"""


def _path_inputs(paths: dict[str, str]) -> str:
    labels = {
        "telemetry_csv": "data_2026.csv",
    }
    return "".join(
        f'<div><label>{label}</label><input name="{key}" value="{html.escape(paths.get(key, ""))}"></div>'
        for key, label in labels.items()
    )


def _weather_location_value(state: WebState) -> str:
    if state.weather_result is not None:
        return state.weather_result.location_label
    return default_weather_location_label()


def _weather_month_value(state: WebState) -> str:
    if state.weather_result is not None:
        return state.weather_result.month
    if state.feature_table is not None:
        max_time = state.feature_table["timestamp_local"].max()
        if hasattr(max_time, "strftime"):
            return max_time.strftime("%Y-%m")
    return default_weather_month()


def _weather_temperature_value(state: WebState) -> str:
    if state.weather_result is None:
        return "API on forecast"
    return f"{state.weather_result.average_c:.1f} C"


def _month_select(selected: str) -> str:
    try:
        start_date, _ = month_bounds(selected)
        options = month_options(start_date)
    except Exception:
        options = month_options()
    if selected not in options:
        options = [selected, *options]
    html_options = []
    for value in options:
        selected_attr = " selected" if value == selected else ""
        html_options.append(
            f'<option value="{html.escape(value)}"{selected_attr}>{html.escape(value)}</option>'
        )
    return f'<select name="weather_month">{"".join(html_options)}</select>'


def _message(state: WebState) -> str:
    if state.error:
        return f'<div class="error">{html.escape(state.error)}</div>'
    if state.message:
        return f'<div class="notice">{html.escape(state.message)}</div>'
    return ""


def _summaries(state: WebState) -> str:
    if not state.feature_summary:
        return ""
    lines = [
        "<div class='table-wrap'><table><tr><th>File</th><th>Rows</th><th>Meters</th><th>Size MB</th><th>Range</th></tr>"
    ]
    for item in state.csv_summary:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(item['label']))}</td>"
            f"<td>{item['rows']}</td><td>{item['meters']}</td><td>{item['size_mb']}</td>"
            f"<td>{html.escape(str(item['min_time']))} -> {html.escape(str(item['max_time']))}</td>"
            "</tr>"
        )
    lines.append("</table></div>")
    fs = state.feature_summary
    lines.append(
        f"<p class='muted'>Feature rows: {fs.get('rows')} | meters: {fs.get('meters')} | "
        f"areas: {fs.get('areas')} | range: {html.escape(str(fs.get('min_time')))} -> {html.escape(str(fs.get('max_time')))}</p>"
    )
    return "".join(lines)


def _df_table(df) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No data yet.</p>"
    columns = list(df.columns)
    lines = ["<div class='table-wrap'><table><tr>"]
    lines.extend(f"<th>{html.escape(str(col))}</th>" for col in columns)
    lines.append("</tr>")
    for _, row in df.iterrows():
        lines.append("<tr>")
        lines.extend(f"<td>{html.escape(str(row[col]))}</td>" for col in columns)
        lines.append("</tr>")
    lines.append("</table></div>")
    return "".join(lines)


def _forecast_svg(df) -> str:
    if df is None or df.empty:
        return ""
    subset = df.copy()
    meters = list(subset["meter"].unique())[:8]
    subset = subset[subset["meter"].isin(meters)]
    values = subset["predicted_kwh"].astype(float)
    min_y, max_y = float(values.min()), float(values.max())
    span = max(max_y - min_y, 1.0)
    width, height = 1000, 260
    colors = [
        "#0969da",
        "#1a7f37",
        "#d1242f",
        "#8250df",
        "#9a6700",
        "#bf3989",
        "#0550ae",
        "#57606a",
    ]
    lines = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    lines.append('<line x1="40" y1="220" x2="980" y2="220" stroke="#d8dee4"/>')
    lines.append('<line x1="40" y1="20" x2="40" y2="220" stroke="#d8dee4"/>')
    for idx, meter in enumerate(meters):
        group = subset[subset["meter"].eq(meter)].reset_index(drop=True)
        if len(group) < 2:
            continue
        points = []
        for row_idx, row in group.iterrows():
            x = 40 + (row_idx / max(len(group) - 1, 1)) * 940
            y = 220 - ((float(row["predicted_kwh"]) - min_y) / span) * 190
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[idx % len(colors)]
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{50 + idx * 115}" y="16" fill="{color}" font-size="12">{html.escape(meter)}</text>'
        )
    lines.append("</svg>")
    return "".join(lines)


def _anomaly_svg(df) -> str:
    if df is None or df.empty:
        return ""
    subset = df.copy()
    meters = list(subset["meter"].unique())[:8]
    subset = subset[subset["meter"].isin(meters)].sort_values(
        ["meter", "timestamp_local"]
    )
    values = subset["kwh"].astype(float)
    min_y, max_y = float(values.min()), float(values.max())
    span = max(max_y - min_y, 1.0)
    width, height = 1000, 260
    colors = [
        "#0969da",
        "#1a7f37",
        "#8250df",
        "#9a6700",
        "#0550ae",
        "#57606a",
        "#bf3989",
        "#116329",
    ]
    lines = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    lines.append('<line x1="40" y1="220" x2="980" y2="220" stroke="#d8dee4"/>')
    lines.append('<line x1="40" y1="20" x2="40" y2="220" stroke="#d8dee4"/>')
    for idx, meter in enumerate(meters):
        group = subset[subset["meter"].eq(meter)].reset_index(drop=True)
        if len(group) < 2:
            continue
        points = []
        for row_idx, row in group.iterrows():
            x = 40 + (row_idx / max(len(group) - 1, 1)) * 940
            y = 220 - ((float(row["kwh"]) - min_y) / span) * 190
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[idx % len(colors)]
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.65"/>'
        )
        anomalies = group[group["is_anomaly"]]
        for row_idx, row in anomalies.iterrows():
            x = 40 + (row_idx / max(len(group) - 1, 1)) * 940
            y = 220 - ((float(row["kwh"]) - min_y) / span) * 190
            fill = "#d1242f" if row["severity"] == "High" else "#fb8500"
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{fill}"/>')
        lines.append(
            f'<text x="{50 + idx * 115}" y="16" fill="{color}" font-size="12">{html.escape(meter)}</text>'
        )
    lines.append("</svg>")
    return "".join(lines)


def _anomaly_table_df(state: WebState):
    if state.anomaly_df is None:
        return None
    data = state.anomaly_df
    if state.anomaly_only:
        data = data[data["is_anomaly"]]
    return data.head(500)


def _checked(value: bool) -> str:
    return "checked" if value else ""


def _meter_select(name: str, meters: list[str]) -> str:
    options = ['<option value="All meters">All meters</option>']
    options.extend(
        f'<option value="{html.escape(meter)}">{html.escape(meter)}</option>'
        for meter in meters
    )
    return f'<select name="{name}">{"".join(options)}</select>'


def _meters(state: WebState) -> list[str]:
    if state.feature_table is None:
        return []
    return sorted(state.feature_table["meter"].dropna().unique())


def _first(form: dict[str, list[str]], key: str) -> str:
    return form.get(key, [""])[0].strip()


def _path_keys() -> list[str]:
    return ["telemetry_csv"]


def _data_paths(paths: dict[str, str]) -> DataPaths:
    if not paths.get("telemetry_csv"):
        raise ValueError("data_2026.csv is required.")
    return DataPaths(telemetry_csv=Path(paths["telemetry_csv"]))


def _default_paths() -> dict[str, str]:
    downloads = Path.home() / "Downloads"
    telemetry_path = downloads / "data_2026.csv"
    paths = {}
    if telemetry_path.exists():
        paths["telemetry_csv"] = str(telemetry_path)
    return paths


def _available_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])
