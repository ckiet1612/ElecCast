$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Venv = Join-Path $Root ".venv-win"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.10 -m venv $Venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv $Venv
    }
    else {
        Write-Error "Python is not installed. Install it with: winget install --id Python.Python.3.10 -e --source winget"
    }
}

if (-not (Test-Path $Python)) {
    Write-Host ""
    Write-Host "Could not create .venv-win because Python 3.10 is not installed." -ForegroundColor Red
    Write-Host "Run this in PowerShell, then close and reopen PowerShell:" -ForegroundColor Yellow
    Write-Host "  winget install --id Python.Python.3.10 -e --source winget"
    Write-Host ""
    Write-Host "Verify:"
    Write-Host "  py -3.10 --version"
    Write-Host ""
    Write-Host "Then run:"
    Write-Host "  .\scripts\build_windows.ps1"
    exit 1
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m PyInstaller --noconfirm --clean packaging\ElectricityForecastWindows.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Exe = Join-Path $Root "dist\ElectricityForecast\ElectricityForecast.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Build finished but $Exe was not created."
}

$PythonDll = Join-Path $Root "dist\ElectricityForecast\_internal\python310.dll"
if (-not (Test-Path $PythonDll)) {
    Write-Error "Build finished but $PythonDll was not created. The app folder is incomplete."
}

$Zip = Join-Path $Root "dist\ElectricityForecast-windows.zip"
if (Test-Path $Zip) {
    Remove-Item $Zip -Force
}
Compress-Archive -Path (Join-Path $Root "dist\ElectricityForecast") -DestinationPath $Zip

Write-Host ""
Write-Host "Built Windows native app:"
Write-Host "  dist\ElectricityForecast\ElectricityForecast.exe"
Write-Host "  dist\ElectricityForecast-windows.zip"
Write-Host ""
Write-Host "Use the whole dist\ElectricityForecast folder, or unzip ElectricityForecast-windows.zip."
Write-Host "Do not copy only ElectricityForecast.exe; it needs the _internal folder beside it."
