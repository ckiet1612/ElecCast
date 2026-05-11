$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Venv = Join-Path $Root ".venv-win"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.10 -m venv $Venv
    }
    else {
        python -m venv $Venv
    }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements-windows.txt
& $Python -m PyInstaller --noconfirm --clean packaging\ElectricityForecastWindows.spec

Write-Host ""
Write-Host "Built Windows native app:"
Write-Host "  dist\ElectricityForecast\ElectricityForecast.exe"
Write-Host ""
Write-Host "Use the whole dist\ElectricityForecast folder. This is intentional: onedir starts faster than onefile."
