from __future__ import annotations

import os
import importlib
from pathlib import Path


def configure_qt_runtime() -> dict[str, str]:
    """Point Qt at bundled plugins on macOS virtualenv installs."""
    package_dir, qt_dir_name = _find_qt_package()
    if package_dir is None:
        return {}

    plugins_dir = package_dir / qt_dir_name / "plugins"
    platforms_dir = plugins_dir / "platforms"
    configured: dict[str, str] = {}

    if plugins_dir.exists():
        _prepend_env_path("QT_PLUGIN_PATH", plugins_dir)
        configured["QT_PLUGIN_PATH"] = os.environ["QT_PLUGIN_PATH"]
    if platforms_dir.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
        configured["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)

    return configured


def _find_qt_package() -> tuple[Path | None, str]:
    for package_name, qt_dir_name in (("PyQt6", "Qt6"), ("PySide6", "Qt")):
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            continue
        return Path(package.__file__).resolve().parent, qt_dir_name
    return None, ""


def _prepend_env_path(name: str, path: Path) -> None:
    path_text = str(path)
    current = os.environ.get(name)
    if not current:
        os.environ[name] = path_text
        return
    parts = [item for item in current.split(os.pathsep) if item]
    if path_text not in parts:
        os.environ[name] = os.pathsep.join([path_text, *parts])
