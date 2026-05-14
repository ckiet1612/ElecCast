from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    ui = os.environ.get("ELECTRICITY_FORECAST_UI", _default_ui()).lower()
    if ui == "native":
        ui = "tk"
    if "--qt" in sys.argv:
        ui = "qt"
    if "--tk" in sys.argv:
        ui = "tk"
    if ui == "qt":
        from .qt_runtime import configure_qt_runtime

        configure_qt_runtime()
        try:
            from .ui import run_app
        except ModuleNotFoundError as exc:
            return _missing_dependency(exc)
        return run_app(sys.argv)
    if ui == "tk":
        try:
            from .tk_app import run_app
        except ModuleNotFoundError as exc:
            return _missing_dependency(exc)
        return run_app()

    try:
        from .web_app import run_app
    except ModuleNotFoundError as exc:
        return _missing_dependency(exc)
    return run_app()


def _missing_dependency(exc: ModuleNotFoundError) -> int:
    missing = exc.name or "dependency"
    print(
        f"Missing dependency: {missing}. Install dependencies with "
        "'python -m pip install -r requirements.txt'.",
        file=sys.stderr,
    )
    return 1


def _default_ui() -> str:
    executable_root = Path(sys.executable).parent.parent.name
    if executable_root in {".venv-native", ".venv"}:
        return "tk"
    return "web"


if __name__ == "__main__":
    raise SystemExit(main())
