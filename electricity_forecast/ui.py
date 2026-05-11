from __future__ import annotations

from pathlib import Path
from typing import Callable

from .data import summarize_paths
from .features import build_feature_table, feature_summary
from .models import forecast_dataframe, train_models
from .paths import default_guests_csv
from .types import DataPaths, ForecastRequest


class WorkerMixin:
    def run_task(self, task: Callable, on_success: Callable, on_error: Callable | None = None):
        from .qt_compat import QObject, QThread, Signal

        class Worker(QObject):
            finished = Signal(object)
            failed = Signal(str)

            def run(self):
                try:
                    self.finished.emit(task())
                except Exception as exc:  # pragma: no cover - UI safety net
                    self.failed.emit(str(exc))

        thread = QThread(self.qt_window)
        worker = Worker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_success)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(on_error or self.show_error)
        worker.failed.connect(thread.quit)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._threads.append(thread)
        thread.start()


class MainWindow(WorkerMixin):
    def __init__(self):
        from .qt_compat import (
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QPushButton,
            QTabWidget,
            QTableWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
            Qt,
        )

        self.qt_window = QMainWindow()
        self.qt_window.setWindowTitle("Electricity Forecast")
        self.qt_window.resize(1280, 820)
        self._threads = []
        self.feature_table = None
        self.trained_models = {}
        self.metrics_df = None
        self.forecast_df = None

        root = QWidget()
        self.tabs = QTabWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self.tabs)
        self.qt_window.setCentralWidget(root)

        self.path_inputs = {}
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        path_grid = QGridLayout()
        defaults = _default_paths()
        labels = [
            ("kwh_csv", "data_kwh.csv"),
            ("guests_csv", "sunworld_honthom_hourly_jan2026.csv (khách mặc định)"),
            ("energy_log_csv", "energy_log.csv (optional)"),
            ("pf_csv", "data_pf.csv (optional, slower)"),
            ("current_csv", "data_current.csv (optional, slower)"),
            ("telemetry_csv", "data_2026.csv (optional, very slow)"),
        ]
        for row, (key, label) in enumerate(labels):
            edit = QLineEdit(defaults.get(key, ""))
            browse = QPushButton("Browse")
            browse.clicked.connect(lambda _checked=False, target=edit: self.pick_file(target))
            path_grid.addWidget(QLabel(label), row, 0)
            path_grid.addWidget(edit, row, 1)
            path_grid.addWidget(browse, row, 2)
            self.path_inputs[key] = edit
        data_layout.addLayout(path_grid)
        data_buttons = QHBoxLayout()
        self.import_button = QPushButton("Import / Build Features")
        self.import_button.clicked.connect(self.import_data)
        self.data_status = QLabel("No data loaded")
        data_buttons.addWidget(self.import_button)
        data_buttons.addWidget(self.data_status)
        data_buttons.addStretch()
        data_layout.addLayout(data_buttons)
        self.data_summary = QTextEdit()
        self.data_summary.setReadOnly(True)
        data_layout.addWidget(self.data_summary)
        self.tabs.addTab(data_tab, "Data")

        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        train_controls = QHBoxLayout()
        self.train_meter_combo = QComboBox()
        self.train_meter_combo.addItem("All meters", "")
        self.train_button = QPushButton("Train / Backtest")
        self.train_button.clicked.connect(self.train)
        train_controls.addWidget(QLabel("Meter"))
        train_controls.addWidget(self.train_meter_combo)
        train_controls.addWidget(self.train_button)
        train_controls.addStretch()
        training_layout.addLayout(train_controls)
        self.metrics_table = QTableWidget()
        training_layout.addWidget(self.metrics_table)
        self.tabs.addTab(training_tab, "Training")

        forecast_tab = QWidget()
        forecast_layout = QVBoxLayout(forecast_tab)
        forecast_controls = QHBoxLayout()
        self.forecast_meter_combo = QComboBox()
        self.forecast_meter_combo.addItem("All meters", "")
        self.horizon_combo = QComboBox()
        for label, hours in [("24 hours", 24), ("48 hours", 48), ("168 hours", 168), ("30 days", 720)]:
            self.horizon_combo.addItem(label, hours)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0, 50)
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setValue(28.0)
        self.guest_spin = QDoubleSpinBox()
        self.guest_spin.setRange(0, 10000)
        self.guest_spin.setDecimals(0)
        self.guest_spin.setSpecialValueText("Auto")
        self.forecast_button = QPushButton("Forecast")
        self.forecast_button.clicked.connect(self.forecast)
        for widget in [
            QLabel("Meter"),
            self.forecast_meter_combo,
            QLabel("Horizon"),
            self.horizon_combo,
            QLabel("Temp C"),
            self.temperature_spin,
            QLabel("Guests"),
            self.guest_spin,
            self.forecast_button,
        ]:
            forecast_controls.addWidget(widget)
        forecast_controls.addStretch()
        forecast_layout.addLayout(forecast_controls)
        self.figure, self.canvas = _make_canvas()
        forecast_layout.addWidget(self.canvas, stretch=2)
        self.forecast_table = QTableWidget()
        forecast_layout.addWidget(self.forecast_table, stretch=1)
        self.tabs.addTab(forecast_tab, "Forecast")

        export_tab = QWidget()
        export_layout = QFormLayout(export_tab)
        self.export_forecast_button = QPushButton("Save Forecast CSV")
        self.export_forecast_button.clicked.connect(self.save_forecast)
        self.export_metrics_button = QPushButton("Save Metrics CSV")
        self.export_metrics_button.clicked.connect(self.save_metrics)
        self.export_status = QLabel("")
        export_layout.addRow(self.export_forecast_button)
        export_layout.addRow(self.export_metrics_button)
        export_layout.addRow(self.export_status)
        self.tabs.addTab(export_tab, "Export")
        self.qt_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

    def show(self):
        self.qt_window.show()

    def show_error(self, message: str):
        from .qt_compat import QMessageBox

        QMessageBox.critical(self.qt_window, "Error", message)
        self.data_status.setText("Error")

    def pick_file(self, target):
        from .qt_compat import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self.qt_window, "Select CSV", str(Path.home()), "CSV Files (*.csv)")
        if path:
            target.setText(path)

    def paths(self) -> DataPaths:
        def optional(key):
            text = self.path_inputs[key].text().strip()
            return Path(text) if text else None

        kwh = optional("kwh_csv")
        if not kwh:
            raise ValueError("data_kwh.csv is required.")
        return DataPaths(
            kwh_csv=kwh,
            guests_csv=optional("guests_csv"),
            energy_log_csv=optional("energy_log_csv"),
            pf_csv=optional("pf_csv"),
            current_csv=optional("current_csv"),
            telemetry_csv=optional("telemetry_csv"),
        )

    def import_data(self):
        self.import_button.setEnabled(False)
        self.data_status.setText("Importing...")

        def task():
            paths = self.paths()
            summaries = summarize_paths(paths)
            features = build_feature_table(paths)
            return summaries, features, feature_summary(features)

        def success(result):
            summaries, features, summary = result
            self.feature_table = features
            self.data_status.setText("Loaded")
            self.import_button.setEnabled(True)
            self.data_summary.setPlainText(_format_summary(summaries, summary))
            self._refresh_meters()

        self.run_task(task, success, self._import_error)

    def _import_error(self, message):
        self.import_button.setEnabled(True)
        self.show_error(message)

    def _refresh_meters(self):
        if self.feature_table is None:
            return
        meters = sorted(self.feature_table["meter"].dropna().unique())
        for combo in [self.train_meter_combo, self.forecast_meter_combo]:
            combo.clear()
            combo.addItem("All meters", "")
            for meter in meters:
                combo.addItem(meter, meter)

    def train(self):
        if self.feature_table is None:
            self.show_error("Import data before training.")
            return
        self.train_button.setEnabled(False)

        def task():
            meter = self.train_meter_combo.currentData()
            meters = [meter] if meter else None
            return train_models(self.feature_table, meters=meters)

        def success(result):
            self.trained_models, self.metrics_df = result
            self.train_button.setEnabled(True)
            _fill_table(self.metrics_table, self.metrics_df)

        self.run_task(task, success, self._train_error)

    def _train_error(self, message):
        self.train_button.setEnabled(True)
        self.show_error(message)

    def forecast(self):
        if self.feature_table is None or not self.trained_models:
            self.show_error("Import data and train models before forecasting.")
            return
        self.forecast_button.setEnabled(False)

        def task():
            meter = self.forecast_meter_combo.currentData()
            meters = [meter] if meter else list(self.trained_models.keys())
            guest_value = self.guest_spin.value() if self.guest_spin.value() > 0 else None
            request = ForecastRequest(
                meters=meters,
                horizon_hours=int(self.horizon_combo.currentData()),
                temperature_c=float(self.temperature_spin.value()),
                guest_count=guest_value,
            )
            return forecast_dataframe(self.trained_models, self.feature_table, request)

        def success(result):
            self.forecast_df = result
            self.forecast_button.setEnabled(True)
            _fill_table(self.forecast_table, result.head(500))
            self._plot_forecast(result)

        self.run_task(task, success, self._forecast_error)

    def _forecast_error(self, message):
        self.forecast_button.setEnabled(True)
        self.show_error(message)

    def _plot_forecast(self, df):
        axes = self.figure.subplots()
        axes.clear()
        if not df.empty:
            for meter, group in df.groupby("meter"):
                axes.plot(group["timestamp_local"], group["predicted_kwh"], label=meter)
            axes.set_ylabel("kWh")
            axes.set_xlabel("Time")
            axes.legend(loc="upper left", fontsize="small", ncols=2)
            self.figure.autofmt_xdate()
        self.canvas.draw()

    def save_forecast(self):
        self._save_dataframe(self.forecast_df, "forecast.csv")

    def save_metrics(self):
        self._save_dataframe(self.metrics_df, "metrics.csv")

    def _save_dataframe(self, df, default_name: str):
        from .qt_compat import QFileDialog

        if df is None or df.empty:
            self.show_error("Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self.qt_window, "Save CSV", default_name, "CSV Files (*.csv)")
        if path:
            df.to_csv(path, index=False)
            self.export_status.setText(f"Saved {path}")


def run_app(argv: list[str]) -> int:
    from .qt_runtime import configure_qt_runtime

    configure_qt_runtime()
    from .qt_compat import QApplication, add_library_path

    add_library_path()

    app = QApplication(argv)
    window = MainWindow()
    window.show()
    return app.exec()


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


def _make_canvas():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4), tight_layout=True)
    return figure, FigureCanvasQTAgg(figure)


def _fill_table(table, df):
    from .qt_compat import QTableWidgetItem

    if df is None:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)
        return
    table.setColumnCount(len(df.columns))
    table.setRowCount(len(df))
    table.setHorizontalHeaderLabels([str(col) for col in df.columns])
    for row_idx, (_, row) in enumerate(df.iterrows()):
        for col_idx, value in enumerate(row):
            table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
    table.resizeColumnsToContents()


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
