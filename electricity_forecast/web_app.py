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
from .models import backtest_predictions_dataframe, forecast_dataframe, train_models
from .optimizer import optimize_consumption
from .types import AnomalyRequest, DataPaths, ForecastRequest, OptimizationRequest
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
    backtest_df: object | None = None
    forecast_df: object | None = None
    anomaly_df: object | None = None
    anomaly_only: bool = True
    weather_result: object | None = None
    optimization_schedule_df: object | None = None
    optimization_summary_df: object | None = None
    optimization_results: list = field(default_factory=list)
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
            if self.path.startswith("/download/backtest"):
                self._download_df(state.backtest_df, "backtest_actual_vs_predicted.csv")
                return
            if self.path.startswith("/download/anomalies"):
                self._download_df(state.anomaly_df, "anomalies.csv")
                return
            if self.path.startswith("/download/optimization"):
                self._download_df(state.optimization_schedule_df, "optimization_schedule.csv")
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
                elif action == "optimize":
                    _optimize_action(state, form)
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
    state.backtest_df = None
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
    state.backtest_df = backtest_predictions_dataframe(state.trained_models)
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


def _optimize_action(state: WebState, form: dict[str, list[str]]) -> None:
    state.error = ""
    if state.feature_table is None or not state.trained_models:
        raise ValueError("Import data and train models before optimizing.")
    meter = _first(form, "optimize_meter")
    meters = (
        list(state.trained_models) if not meter or meter == "All meters" else [meter]
    )
    request = OptimizationRequest(
        meters=meters,
        horizon_hours=int(_first(form, "opt_horizon_hours") or 24),
        temp_min=float(_first(form, "opt_temp_min") or 22.0),
        temp_max=float(_first(form, "opt_temp_max") or 30.0),
        learning_rate=float(_first(form, "opt_learning_rate") or 0.01),
        max_iterations=int(_first(form, "opt_max_iterations") or 500),
    )
    schedule_df, summary_df, results = optimize_consumption(
        state.trained_models, state.feature_table, request
    )
    state.optimization_schedule_df = schedule_df
    state.optimization_summary_df = summary_df
    state.optimization_results = results
    total_saved = sum(r.savings_kwh for r in results)
    state.message = f"Optimization completed. Total savings: {total_saved:.2f} kWh."


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
    <p class="muted">Electrical telemetry uses data_2026.csv. Customer list CSV is optional and fills guest_count for forecasting.</p>
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
    {_backtest_svg(state.backtest_df)}
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
    <h2>Optimization (Gradient Descent)</h2>
    <p class="muted">Tìm bộ tham số tối ưu (nhiệt độ, lượng khách) để giảm thiểu tiêu thụ điện bằng thuật toán Gradient Descent.</p>
    <form method="post" action="/optimize" class="row">
      <div><label>Meter</label>{_meter_select("optimize_meter", list(state.trained_models) if state.trained_models else meters)}</div>
      <div><label>Horizon</label><select name="opt_horizon_hours"><option value="12">12 hours</option><option value="24" selected>24 hours</option><option value="48">48 hours</option><option value="168">168 hours</option></select></div>
      <div><label>Temp min (°C)</label><input name="opt_temp_min" value="22.0"></div>
      <div><label>Temp max (°C)</label><input name="opt_temp_max" value="30.0"></div>
      <div><label>Learning rate</label><input name="opt_learning_rate" value="0.01"></div>
      <div><label>Max iterations</label><input name="opt_max_iterations" value="500"></div>
      <button type="submit">Optimize</button>
    </form>
    {_optimization_convergence_svg(state.optimization_results)}
    {_optimization_comparison_svg(state.optimization_schedule_df)}
    {_df_table(state.optimization_summary_df)}
    {_df_table(state.optimization_schedule_df.head(500) if state.optimization_schedule_df is not None and not state.optimization_schedule_df.empty else None)}
  </section>
  <section>
    <h2>Export</h2>
    <a class="button" href="/download/forecast">Download Forecast CSV</a>
    <a class="button" href="/download/metrics">Download Metrics CSV</a>
    <a class="button" href="/download/backtest">Download Backtest CSV</a>
    <a class="button" href="/download/anomalies">Download Anomaly CSV</a>
    <a class="button" href="/download/optimization">Download Optimization CSV</a>
  </section>
</main>
</body>
</html>"""


def _path_inputs(paths: dict[str, str]) -> str:
    labels = {
        "telemetry_csv": "data_2026.csv",
        "guests_csv": "Danh sách khách/khách hàng CSV",
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


def _backtest_svg(df) -> str:
    if df is None or df.empty:
        return ""
    import math

    areas = sorted(df["area"].dropna().unique())[:12]
    if not areas:
        return ""
    cols = min(3, len(areas))
    rows = math.ceil(len(areas) / cols)
    cell_w, cell_h = 330, 260
    width, height = cell_w * cols, cell_h * rows
    lines = [
        f'<svg viewBox="0 0 {width} {height}" style="height:{height}px" role="img">'
    ]
    for idx, area in enumerate(areas):
        group = df[df["area"].eq(area)]
        actual = group["actual_kwh"].astype(float)
        predicted = group["predicted_kwh"].astype(float)
        min_value = min(float(actual.min()), float(predicted.min()))
        max_value = max(float(actual.max()), float(predicted.max()))
        span = max(max_value - min_value, 1.0)
        pad = span * 0.08
        min_value -= pad
        max_value += pad
        span = max(max_value - min_value, 1.0)
        col = idx % cols
        row = idx // cols
        ox = col * cell_w
        oy = row * cell_h
        left, top = ox + 48, oy + 34
        plot_w, plot_h = cell_w - 72, cell_h - 72
        lines.append(
            f'<text x="{left}" y="{oy + 18}" font-size="13" font-weight="600">{html.escape(str(area))}: Actual vs Predicted</text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#d8dee4"/>'
        )
        lines.append(
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#d8dee4"/>'
        )
        lines.append(
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#d55e5e" stroke-dasharray="5 4"/>'
        )
        for _, point in group.iterrows():
            x = left + ((float(point["actual_kwh"]) - min_value) / span) * plot_w
            y = (
                top
                + plot_h
                - ((float(point["predicted_kwh"]) - min_value) / span) * plot_h
            )
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#3158d4" opacity="0.65"/>'
            )
        lines.append(
            f'<text x="{left + plot_w / 2 - 34}" y="{oy + cell_h - 12}" font-size="11">Actual kWh</text>'
        )
        lines.append(
            f'<text x="{ox + 4}" y="{top + plot_h / 2}" font-size="11" transform="rotate(-90 {ox + 4},{top + plot_h / 2})">Predicted kWh</text>'
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


def _optimization_convergence_svg(results: list) -> str:
    """SVG chart showing cost function convergence over GD iterations."""
    if not results:
        return ""
    width, height = 1000, 260
    colors = [
        "#0969da", "#1a7f37", "#d1242f", "#8250df",
        "#9a6700", "#bf3989", "#0550ae", "#57606a",
    ]
    lines = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    lines.append(f'<text x="40" y="16" font-size="13" font-weight="600">Cost Function Convergence (Gradient Descent)</text>')
    lines.append('<line x1="40" y1="220" x2="980" y2="220" stroke="#d8dee4"/>')
    lines.append('<line x1="40" y1="30" x2="40" y2="220" stroke="#d8dee4"/>')

    for idx, result in enumerate(results[:8]):
        cost_h = result.cost_history
        if len(cost_h) < 2:
            continue
        min_c = min(cost_h)
        max_c = max(cost_h)
        span = max(max_c - min_c, 1e-9)
        points = []
        for i, c in enumerate(cost_h):
            x = 40 + (i / max(len(cost_h) - 1, 1)) * 940
            y = 220 - ((c - min_c) / span) * 180
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[idx % len(colors)]
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        label = html.escape(result.meter)
        conv = "✓" if result.converged else f"×{result.iterations}"
        lines.append(
            f'<text x="{50 + idx * 120}" y="{height - 8}" fill="{color}" font-size="11">{label} ({conv})</text>'
        )

    lines.append(
        '<text x="4" y="130" font-size="11" transform="rotate(-90 4,130)">Cost (kWh)</text>'
    )
    lines.append(
        f'<text x="{width // 2 - 30}" y="{height - 24}" font-size="11">Iteration</text>'
    )
    lines.append("</svg>")
    return "".join(lines)


def _optimization_comparison_svg(schedule_df) -> str:
    """SVG chart comparing kWh before vs after optimization per hour."""
    if schedule_df is None or schedule_df.empty:
        return ""
    meters = list(schedule_df["meter"].unique())[:4]
    if not meters:
        return ""

    import math

    cols = min(2, len(meters))
    rows = math.ceil(len(meters) / cols)
    cell_w, cell_h = 500, 260
    width, height = cell_w * cols, cell_h * rows

    lines = [f'<svg viewBox="0 0 {width} {height}" style="height:{height}px" role="img">']

    for idx, meter in enumerate(meters):
        group = schedule_df[schedule_df["meter"].eq(meter)].reset_index(drop=True)
        before = group["kwh_before"].astype(float)
        after = group["kwh_after"].astype(float)
        all_vals = list(before) + list(after)
        min_v = min(all_vals) if all_vals else 0
        max_v = max(all_vals) if all_vals else 1
        span = max(max_v - min_v, 1e-9)
        pad = span * 0.08
        min_v -= pad
        span = max(max_v + pad - min_v, 1e-9)

        col = idx % cols
        row = idx // cols
        ox = col * cell_w
        oy = row * cell_h
        left, top = ox + 48, oy + 34
        plot_w, plot_h = cell_w - 72, cell_h - 72

        lines.append(
            f'<text x="{left}" y="{oy + 18}" font-size="13" font-weight="600">{html.escape(meter)}: Before vs After</text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#d8dee4"/>'
        )
        lines.append(
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#d8dee4"/>'
        )

        n = len(group)
        for i in range(n):
            x = left + (i / max(n - 1, 1)) * plot_w
            y_b = top + plot_h - ((float(before.iloc[i]) - min_v) / span) * plot_h
            y_a = top + plot_h - ((float(after.iloc[i]) - min_v) / span) * plot_h
            lines.append(f'<circle cx="{x:.1f}" cy="{y_b:.1f}" r="2.5" fill="#d1242f" opacity="0.6"/>')
            lines.append(f'<circle cx="{x:.1f}" cy="{y_a:.1f}" r="2.5" fill="#1a7f37" opacity="0.6"/>')

        if n >= 2:
            pts_b = []
            pts_a = []
            for i in range(n):
                x = left + (i / max(n - 1, 1)) * plot_w
                y_b = top + plot_h - ((float(before.iloc[i]) - min_v) / span) * plot_h
                y_a = top + plot_h - ((float(after.iloc[i]) - min_v) / span) * plot_h
                pts_b.append(f"{x:.1f},{y_b:.1f}")
                pts_a.append(f"{x:.1f},{y_a:.1f}")
            lines.append(
                f'<polyline points="{" ".join(pts_b)}" fill="none" stroke="#d1242f" stroke-width="1.5" opacity="0.7"/>'
            )
            lines.append(
                f'<polyline points="{" ".join(pts_a)}" fill="none" stroke="#1a7f37" stroke-width="1.5" opacity="0.7"/>'
            )

        legend_x = left + plot_w - 140
        lines.append(f'<rect x="{legend_x}" y="{oy + 6}" width="8" height="8" fill="#d1242f"/>')
        lines.append(f'<text x="{legend_x + 12}" y="{oy + 14}" font-size="10">Before</text>')
        lines.append(f'<rect x="{legend_x + 60}" y="{oy + 6}" width="8" height="8" fill="#1a7f37"/>')
        lines.append(f'<text x="{legend_x + 74}" y="{oy + 14}" font-size="10">After</text>')

        lines.append(
            f'<text x="{left + plot_w / 2 - 15}" y="{oy + cell_h - 12}" font-size="11">Hour</text>'
        )
        lines.append(
            f'<text x="{ox + 4}" y="{top + plot_h / 2}" font-size="11" transform="rotate(-90 {ox + 4},{top + plot_h / 2})">kWh</text>'
        )

    lines.append("</svg>")
    return "".join(lines)


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
    return ["telemetry_csv", "guests_csv"]


def _data_paths(paths: dict[str, str]) -> DataPaths:
    if not paths.get("telemetry_csv"):
        raise ValueError("data_2026.csv is required.")
    guests = paths.get("guests_csv")
    return DataPaths(
        telemetry_csv=Path(paths["telemetry_csv"]),
        guests_csv=Path(guests) if guests else None,
    )


def _default_paths() -> dict[str, str]:
    downloads = Path.home() / "Downloads"
    telemetry_path = downloads / "data_2026.csv"
    guests_path = downloads / "sunworld_honthom_hourly_jan2026.csv"
    paths = {}
    if telemetry_path.exists():
        paths["telemetry_csv"] = str(telemetry_path)
    if guests_path.exists():
        paths["guests_csv"] = str(guests_path)
    return paths


def _available_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])
