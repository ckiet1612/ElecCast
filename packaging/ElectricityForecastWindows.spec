# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


try:
    SPEC_DIR = Path(SPECPATH).resolve()
except NameError:
    SPEC_DIR = Path(__file__).resolve().parent

ROOT = SPEC_DIR.parent
ENTRYPOINT = ROOT / "desktop_app.py"


a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "matplotlib.backends.backend_tkagg",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._typedefs",
        "sklearn.neighbors._quad_tree",
        "sklearn.tree._utils",
        "statsmodels.tsa.arima.model",
        "unittest",
        "unittest.mock",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt6",
        "PySide6",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "tkinter.test",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ElectricityForecast",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ElectricityForecast",
)
