from __future__ import annotations

import os
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .data import summarize_paths
from .features import build_feature_table, feature_summary
from .models import forecast_dataframe, train_models
from .paths import default_guests_csv
from .types import DataPaths, ForecastRequest


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
        self.forecast_df = None

        self.paths: dict[str, tk.StringVar] = {}
        self.train_meter = tk.StringVar(value="All meters")
        self.forecast_meter = tk.StringVar(value="All meters")
        self.horizon = tk.StringVar(value="168 hours")
        self.temperature = tk.DoubleVar(value=28.0)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self._build_data_tab(notebook)
        self._build_training_tab(notebook)
        self._build_forecast_tab(notebook)
        self._build_export_tab(notebook)

    def _build_data_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Data")
        defaults = _default_paths()
        rows = [
            ("kwh_csv", "data_kwh.csv"),
            ("guests_csv", "sunworld_honthom_hourly_jan2026.csv (khách mặc định)"),
            ("energy_log_csv", "energy_log.csv (optional)"),
            ("pf_csv", "data_pf.csv (optional, slower)"),
            ("current_csv", "data_current.csv (optional, slower)"),
            ("telemetry_csv", "data_2026.csv (optional, very slow)"),
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
            text="Quick import uses only data_kwh.csv. Add optional CSV files only when you need P/PF/current features.",
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
        self.metrics_tree = _tree(frame)
        self.metrics_tree.pack(fill="both", expand=True)

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
        ttk.Label(controls, text="Temp C").pack(side="left")
        ttk.Spinbox(
            controls, from_=0, to=50, textvariable=self.temperature, width=6
        ).pack(side="left", padx=8)
        self.forecast_button = ttk.Button(
            controls, text="Forecast", command=self.forecast
        )
        self.forecast_button.pack(side="left")

        self.chart_frame = ttk.Frame(frame)
        self.chart_frame.pack(fill="both", expand=True)
        self.forecast_tree = _tree(frame)
        self.forecast_tree.pack(fill="both", expand=True, pady=(8, 0))

    def _build_export_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Export")
        ttk.Button(frame, text="Save Forecast CSV", command=self.save_forecast).pack(
            anchor="w", pady=4
        )
        ttk.Button(frame, text="Save Metrics CSV", command=self.save_metrics).pack(
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
            request = ForecastRequest(
                meters=meters,
                horizon_hours=_horizon_hours(self.horizon.get()),
                temperature_c=float(self.temperature.get()),
            )
            return forecast_dataframe(self.trained_models, self.feature_table, request)

        self._run_background(task, self._on_forecasted, self._on_forecast_failed)

    def save_forecast(self) -> None:
        self._save_df(self.forecast_df, "forecast.csv")

    def save_metrics(self) -> None:
        self._save_df(self.metrics_df, "metrics.csv")

    def _on_imported(self, result) -> None:
        summaries, features, summary = result
        self.feature_table = features
        self.data_status.configure(text="Loaded")
        self.import_button.configure(state="normal")
        self.data_summary.delete("1.0", "end")
        self.data_summary.insert("1.0", _format_summary(summaries, summary))
        self._refresh_meters()

    def _on_trained(self, result) -> None:
        self.trained_models, self.metrics_df = result
        self.train_button.configure(state="normal")
        _fill_tree(self.metrics_tree, self.metrics_df)

    def _on_forecasted(self, result) -> None:
        self.forecast_df = result
        self.forecast_button.configure(state="normal")
        _fill_tree(self.forecast_tree, result.head(500))
        self._plot_forecast(result)

    def _on_import_failed(self, message: str) -> None:
        self.import_button.configure(state="normal")
        self.data_status.configure(text="Error")
        messagebox.showerror("Import failed", message)

    def _on_train_failed(self, message: str) -> None:
        self.train_button.configure(state="normal")
        messagebox.showerror("Training failed", message)

    def _on_forecast_failed(self, message: str) -> None:
        self.forecast_button.configure(state="normal")
        messagebox.showerror("Forecast failed", message)

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

    def _refresh_meters(self) -> None:
        meters = sorted(self.feature_table["meter"].dropna().unique())
        values = ["All meters", *meters]
        self.train_meter_combo.configure(values=values)
        self.forecast_meter_combo.configure(values=values)

    def _paths(self) -> DataPaths:
        kwh = self.paths["kwh_csv"].get().strip()
        if not kwh:
            raise ValueError("data_kwh.csv is required.")
        return DataPaths(
            kwh_csv=Path(kwh),
            guests_csv=_optional_path(self.paths["guests_csv"].get()),
            energy_log_csv=_optional_path(self.paths["energy_log_csv"].get()),
            pf_csv=_optional_path(self.paths["pf_csv"].get()),
            current_csv=_optional_path(self.paths["current_csv"].get()),
            telemetry_csv=_optional_path(self.paths["telemetry_csv"].get()),
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
    kwh_path = downloads / "data_kwh.csv"
    guests_path = default_guests_csv()
    paths = {}
    if kwh_path.exists():
        paths["kwh_csv"] = str(kwh_path)
    if guests_path.exists():
        paths["guests_csv"] = str(guests_path)
    return paths


def _optional_path(value: str) -> Path | None:
    text = value.strip()
    return Path(text) if text else None


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
