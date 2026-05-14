from __future__ import annotations

import os
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .anomaly import detect_anomalies as run_anomaly_detection
from .data import summarize_paths
from .features import build_feature_table, feature_summary
from .models import backtest_predictions_dataframe, forecast_dataframe, train_models
from .plots import actual_vs_predicted_figure
from .types import AnomalyRequest, DataPaths, ForecastRequest
from .weather import (
    default_weather_month,
    default_weather_location_label,
    month_options,
    monthly_average_temperature,
    weather_location_labels,
)


def run_app() -> int:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "electricity_forecast_mpl")
    )
    root = tk.Tk()
    ElectricityForecastTk(root)
    root.mainloop()
    return 0


class ElectricityForecastTk:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Electricity Forecast")
        self.root.geometry("1280x820")

        self.feature_table = None
        self.trained_models = {}
        self.metrics_df = None
        self.backtest_df = None
        self.forecast_df = None
        self.anomaly_df = None

        self.paths: dict[str, tk.StringVar] = {}
        self.train_meter = tk.StringVar(value="All meters")
        self.forecast_meter = tk.StringVar(value="All meters")
        self.anomaly_meter = tk.StringVar(value="All meters")
        self.anomaly_contamination = tk.DoubleVar(value=0.05)
        self.anomaly_only = tk.BooleanVar(value=True)
        self.horizon = tk.StringVar(value="168 hours")
        self.temperature = tk.DoubleVar(value=28.0)
        self.weather_location = tk.StringVar(value=default_weather_location_label())
        self.weather_month = tk.StringVar(value=default_weather_month())
        self.weather_result = None

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self._build_data_tab(notebook)
        self._build_training_tab(notebook)
        self._build_forecast_tab(notebook)
        self._build_anomaly_tab(notebook)
        self._build_export_tab(notebook)

    def _build_data_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Data")
        defaults = _default_paths()
        rows = [
            ("telemetry_csv", "data_2026.csv"),
            ("guests_csv", "Danh sách khách/khách hàng CSV"),
        ]
        for row, (key, label) in enumerate(rows):
            ttk.Label(frame, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=4
            )
            value = tk.StringVar(value=defaults.get(key, ""))
            self.paths[key] = value
            ttk.Entry(frame, textvariable=value).grid(
                row=row, column=1, sticky="ew", pady=4
            )
            ttk.Button(
                frame,
                text="Browse",
                command=lambda target=value: self._pick_csv(target),
            ).grid(row=row, column=2, padx=(8, 0), pady=4)
        frame.columnconfigure(1, weight=1)
        actions = ttk.Frame(frame)
        actions.grid(row=len(rows), column=0, columnspan=3, sticky="ew", pady=(10, 8))
        self.import_button = ttk.Button(
            actions, text="Import / Build Features", command=self.import_data
        )
        self.import_button.pack(side="left")
        self.data_status = ttk.Label(actions, text="No data loaded")
        self.data_status.pack(side="left", padx=12)
        ttk.Label(
            frame,
            text="Electrical telemetry uses data_2026.csv. Customer list CSV is optional and fills guest_count for forecasting.",
        ).grid(row=len(rows) + 1, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.data_summary = tk.Text(frame, height=24, wrap="word")
        self.data_summary.grid(row=len(rows) + 2, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(len(rows) + 2, weight=1)

    def _build_training_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Training")
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Meter").pack(side="left")
        self.train_meter_combo = ttk.Combobox(
            controls, textvariable=self.train_meter, values=["All meters"], width=28
        )
        self.train_meter_combo.pack(side="left", padx=8)
        self.train_button = ttk.Button(
            controls, text="Train / Backtest", command=self.train
        )
        self.train_button.pack(side="left")
        self.backtest_chart_frame = ttk.Frame(frame)
        self.backtest_chart_frame.pack(fill="both", expand=True)
        self.metrics_tree = _tree(frame)
        self.metrics_tree.pack(fill="both", expand=True, pady=(8, 0))

    def _build_forecast_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Forecast")
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Meter").pack(side="left")
        self.forecast_meter_combo = ttk.Combobox(
            controls, textvariable=self.forecast_meter, values=["All meters"], width=28
        )
        self.forecast_meter_combo.pack(side="left", padx=8)
        ttk.Label(controls, text="Horizon").pack(side="left")
        ttk.Combobox(
            controls,
            textvariable=self.horizon,
            values=["24 hours", "48 hours", "168 hours", "30 days"],
            width=12,
            state="readonly",
        ).pack(side="left", padx=8)
        ttk.Label(controls, text="Location").pack(side="left")
        self.weather_location_combo = ttk.Combobox(
            controls,
            textvariable=self.weather_location,
            values=weather_location_labels(),
            width=24,
        )
        self.weather_location_combo.pack(side="left", padx=8)
        self.weather_location_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_temperature()
        )
        self.weather_location_combo.bind(
            "<Return>", lambda _event: self.update_temperature()
        )
        ttk.Label(controls, text="Month").pack(side="left")
        self.weather_month_combo = ttk.Combobox(
            controls,
            textvariable=self.weather_month,
            values=month_options(),
            width=9,
            state="readonly",
        )
        self.weather_month_combo.pack(side="left", padx=8)
        self.weather_month_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_temperature()
        )
        self.weather_button = ttk.Button(
            controls, text="Get Temp", command=self.update_temperature
        )
        self.weather_button.pack(side="left", padx=(0, 8))
        self.temperature_status = ttk.Label(
            controls, text=f"Avg Temp: {self.temperature.get():.1f} C"
        )
        self.temperature_status.pack(side="left", padx=(0, 8))
        self.forecast_button = ttk.Button(
            controls, text="Forecast", command=self.forecast
        )
        self.forecast_button.pack(side="left")

        self.chart_frame = ttk.Frame(frame)
        self.chart_frame.pack(fill="both", expand=True)
        self.forecast_tree = _tree(frame)
        self.forecast_tree.pack(fill="both", expand=True, pady=(8, 0))

    def _build_anomaly_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Anomaly")
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Meter").pack(side="left")
        self.anomaly_meter_combo = ttk.Combobox(
            controls, textvariable=self.anomaly_meter, values=["All meters"], width=28
        )
        self.anomaly_meter_combo.pack(side="left", padx=8)
        ttk.Label(controls, text="Contamination").pack(side="left")
        ttk.Spinbox(
            controls,
            from_=0.01,
            to=0.20,
            increment=0.01,
            textvariable=self.anomaly_contamination,
            width=6,
        ).pack(side="left", padx=8)
        ttk.Checkbutton(
            controls,
            text="Only anomalies",
            variable=self.anomaly_only,
            command=self._refresh_anomaly_table,
        ).pack(side="left", padx=8)
        self.anomaly_button = ttk.Button(
            controls, text="Detect Anomalies", command=self.detect_anomalies
        )
        self.anomaly_button.pack(side="left")

        self.anomaly_chart_frame = ttk.Frame(frame)
        self.anomaly_chart_frame.pack(fill="both", expand=True)
        self.anomaly_tree = _tree(frame)
        self.anomaly_tree.pack(fill="both", expand=True, pady=(8, 0))

    def _build_export_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Export")
        ttk.Button(frame, text="Save Forecast CSV", command=self.save_forecast).pack(
            anchor="w", pady=4
        )
        ttk.Button(frame, text="Save Metrics CSV", command=self.save_metrics).pack(
            anchor="w", pady=4
        )
        ttk.Button(frame, text="Save Backtest CSV", command=self.save_backtest).pack(
            anchor="w", pady=4
        )
        ttk.Button(frame, text="Save Anomaly CSV", command=self.save_anomalies).pack(
            anchor="w", pady=4
        )
        self.export_status = ttk.Label(frame, text="")
        self.export_status.pack(anchor="w", pady=12)

    def import_data(self) -> None:
        self.import_button.configure(state="disabled")
        self.data_status.configure(text="Importing...")

        def task():
            paths = self._paths()
            summaries = summarize_paths(paths)
            features = build_feature_table(paths)
            return summaries, features, feature_summary(features)

        self._run_background(task, self._on_imported, self._on_import_failed)

    def train(self) -> None:
        if self.feature_table is None:
            messagebox.showerror("Error", "Import data before training.")
            return
        self.train_button.configure(state="disabled")

        def task():
            meter = self.train_meter.get()
            meters = None if meter == "All meters" else [meter]
            return train_models(self.feature_table, meters=meters)

        self._run_background(task, self._on_trained, self._on_train_failed)

    def update_temperature(self) -> None:
        self.weather_button.configure(state="disabled")
        self.temperature_status.configure(text="Fetching weather...")
        self._run_background(
            self._fetch_weather_temperature,
            self._on_weather_loaded,
            self._on_weather_failed,
        )

    def forecast(self) -> None:
        if self.feature_table is None or not self.trained_models:
            messagebox.showerror(
                "Error", "Import data and train models before forecasting."
            )
            return
        self.forecast_button.configure(state="disabled")

        def task():
            meter = self.forecast_meter.get()
            meters = list(self.trained_models) if meter == "All meters" else [meter]
            weather = self._fetch_weather_temperature()
            request = ForecastRequest(
                meters=meters,
                horizon_hours=_horizon_hours(self.horizon.get()),
                temperature_c=weather.average_c,
                weather_location=weather.location_label,
                weather_month=weather.month,
            )
            forecast = forecast_dataframe(
                self.trained_models, self.feature_table, request
            )
            return weather, forecast

        self._run_background(task, self._on_forecasted, self._on_forecast_failed)

    def detect_anomalies(self) -> None:
        if self.feature_table is None:
            messagebox.showerror("Error", "Import data before detecting anomalies.")
            return
        self.anomaly_button.configure(state="disabled")

        def task():
            meter = self.anomaly_meter.get()
            meters = None if meter == "All meters" else [meter]
            request = AnomalyRequest(
                meters=meters,
                contamination=float(self.anomaly_contamination.get()),
            )
            return run_anomaly_detection(self.feature_table, request)

        self._run_background(task, self._on_anomalies_detected, self._on_anomaly_failed)

    def save_forecast(self) -> None:
        self._save_df(self.forecast_df, "forecast.csv")

    def save_metrics(self) -> None:
        self._save_df(self.metrics_df, "metrics.csv")

    def save_backtest(self) -> None:
        self._save_df(self.backtest_df, "backtest_actual_vs_predicted.csv")

    def save_anomalies(self) -> None:
        self._save_df(self.anomaly_df, "anomalies.csv")

    def _on_imported(self, result) -> None:
        summaries, features, summary = result
        self.feature_table = features
        self.data_status.configure(text="Loaded")
        self.import_button.configure(state="normal")
        self.anomaly_df = None
        self.data_summary.delete("1.0", "end")
        self.data_summary.insert("1.0", _format_summary(summaries, summary))
        self._refresh_meters()

    def _on_trained(self, result) -> None:
        self.trained_models, self.metrics_df = result
        self.backtest_df = backtest_predictions_dataframe(self.trained_models)
        self.train_button.configure(state="normal")
        _fill_tree(self.metrics_tree, self.metrics_df)
        self._plot_backtest(self.backtest_df)

    def _on_weather_loaded(self, result) -> None:
        self.weather_button.configure(state="normal")
        self._set_weather_temperature(result)

    def _on_forecasted(self, result) -> None:
        weather, result = result
        self._set_weather_temperature(weather)
        self.forecast_df = result
        self.forecast_button.configure(state="normal")
        _fill_tree(self.forecast_tree, result.head(500))
        self._plot_forecast(result)

    def _on_anomalies_detected(self, result) -> None:
        self.anomaly_df = result
        self.anomaly_button.configure(state="normal")
        self._refresh_anomaly_table()
        self._plot_anomalies(result)

    def _on_import_failed(self, message: str) -> None:
        self.import_button.configure(state="normal")
        self.data_status.configure(text="Error")
        messagebox.showerror("Import failed", message)

    def _on_train_failed(self, message: str) -> None:
        self.train_button.configure(state="normal")
        messagebox.showerror("Training failed", message)

    def _on_weather_failed(self, message: str) -> None:
        self.weather_button.configure(state="normal")
        self.temperature_status.configure(
            text=f"Avg Temp: {self.temperature.get():.1f} C"
        )
        messagebox.showerror("Weather API failed", message)

    def _on_forecast_failed(self, message: str) -> None:
        self.forecast_button.configure(state="normal")
        messagebox.showerror("Forecast failed", message)

    def _on_anomaly_failed(self, message: str) -> None:
        self.anomaly_button.configure(state="normal")
        messagebox.showerror("Anomaly detection failed", message)

    def _plot_forecast(self, df) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        for child in self.chart_frame.winfo_children():
            child.destroy()
        fig = Figure(figsize=(9, 3.8), tight_layout=True)
        ax = fig.add_subplot(111)
        if not df.empty:
            for meter, group in df.groupby("meter"):
                ax.plot(group["timestamp_local"], group["predicted_kwh"], label=meter)
            ax.set_xlabel("Time")
            ax.set_ylabel("kWh")
            ax.legend(loc="upper left", fontsize="small", ncols=2)
            fig.autofmt_xdate()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_backtest(self, df) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        for child in self.backtest_chart_frame.winfo_children():
            child.destroy()
        fig = actual_vs_predicted_figure(df)
        canvas = FigureCanvasTkAgg(fig, master=self.backtest_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_anomalies(self, df) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        for child in self.anomaly_chart_frame.winfo_children():
            child.destroy()
        fig = Figure(figsize=(9, 3.8), tight_layout=True)
        ax = fig.add_subplot(111)
        if df is not None and not df.empty:
            plot_df = df.sort_values(["meter", "timestamp_local"])
            for meter, group in plot_df.groupby("meter"):
                ax.plot(group["timestamp_local"], group["kwh"], label=meter, alpha=0.65)
            anomalies = plot_df[plot_df["is_anomaly"]]
            if not anomalies.empty:
                colors = (
                    anomalies["severity"].map({"High": "#d1242f"}).fillna("#fb8500")
                )
                ax.scatter(
                    anomalies["timestamp_local"],
                    anomalies["kwh"],
                    c=colors,
                    s=32,
                    zorder=3,
                )
            ax.set_xlabel("Time")
            ax.set_ylabel("kWh")
            ax.legend(loc="upper left", fontsize="small", ncols=2)
            fig.autofmt_xdate()
        canvas = FigureCanvasTkAgg(fig, master=self.anomaly_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _refresh_meters(self) -> None:
        meters = sorted(self.feature_table["meter"].dropna().unique())
        values = ["All meters", *meters]
        self.train_meter_combo.configure(values=values)
        self.forecast_meter_combo.configure(values=values)
        self.anomaly_meter_combo.configure(values=values)
        self._sync_weather_month_to_data()

    def _refresh_anomaly_table(self) -> None:
        if self.anomaly_df is None:
            return
        data = self.anomaly_df
        if self.anomaly_only.get():
            data = data[data["is_anomaly"]]
        _fill_tree(self.anomaly_tree, data.head(500))

    def _fetch_weather_temperature(self):
        return monthly_average_temperature(
            self.weather_location.get(),
            self.weather_month.get(),
        )

    def _set_weather_temperature(self, weather) -> None:
        self.weather_result = weather
        self.weather_location.set(weather.location_label)
        self.weather_month.set(weather.month)
        self.temperature.set(weather.average_c)
        self.temperature_status.configure(text=f"Avg Temp: {weather.average_c:.1f} C")

    def _sync_weather_month_to_data(self) -> None:
        max_time = self.feature_table["timestamp_local"].max()
        if hasattr(max_time, "strftime"):
            self.weather_month.set(max_time.strftime("%Y-%m"))
            self.weather_month_combo.configure(values=month_options(max_time))

    def _paths(self) -> DataPaths:
        telemetry = self.paths["telemetry_csv"].get().strip()
        if not telemetry:
            raise ValueError("data_2026.csv is required.")
        guests = self.paths["guests_csv"].get().strip()
        return DataPaths(
            telemetry_csv=Path(telemetry),
            guests_csv=Path(guests) if guests else None,
        )

    def _pick_csv(self, target: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            target.set(path)

    def _save_df(self, df, default_name: str) -> None:
        if df is None or df.empty:
            messagebox.showerror("Error", "Nothing to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=default_name
        )
        if path:
            df.to_csv(path, index=False)
            self.export_status.configure(text=f"Saved {path}")

    def _run_background(self, task, on_success, on_error) -> None:
        def runner():
            try:
                result = task()
            except Exception as exc:  # pragma: no cover - UI safety net
                message = str(exc)
                self.root.after(0, lambda: on_error(message))
            else:
                self.root.after(0, lambda: on_success(result))

        threading.Thread(target=runner, daemon=True).start()


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


def _horizon_hours(label: str) -> int:
    return {"24 hours": 24, "48 hours": 48, "168 hours": 168, "30 days": 720}[label]


def _tree(parent) -> ttk.Treeview:
    tree = ttk.Treeview(parent, show="headings")
    yscroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=yscroll.set)
    return tree


def _fill_tree(tree: ttk.Treeview, df) -> None:
    tree.delete(*tree.get_children())
    tree["columns"] = list(df.columns)
    for column in df.columns:
        tree.heading(column, text=str(column))
        tree.column(column, width=max(110, min(240, len(str(column)) * 12)), anchor="w")
    for _, row in df.iterrows():
        tree.insert("", "end", values=[str(value) for value in row])


def _format_summary(csv_summaries, feature_summary_data) -> str:
    lines = ["CSV summaries:"]
    for item in csv_summaries:
        lines.append(
            f"- {item['label']}: {item['rows']} rows, {item['meters']} meters, "
            f"{item['size_mb']} MB, {item['min_time']} -> {item['max_time']}"
        )
    lines.append("")
    lines.append("Feature table:")
    for key, value in feature_summary_data.items():
        if key == "columns":
            lines.append(f"- columns: {', '.join(value)}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)
