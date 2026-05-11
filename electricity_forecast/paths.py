from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def default_guests_csv() -> Path:
    return project_root() / "data" / "defaults" / "sunworld_honthom_hourly_jan2026.csv"
