# Build in 64-bit Windows PowerShell. The output is
# dist\windows\polaris_client_cont2.exe. Requires the
# Python 3.9 launcher; set $Python to another Python 3.9 executable (and
# $PythonArgs to @()) if needed.
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot

$Python = 'py'
$PythonArgs = @('-3.9')
$buildVenv = '.polaris-build-venv'
$env:PIP_CACHE_DIR = (Join-Path $repoRoot '.polaris-pip-cache')
$env:PYINSTALLER_CONFIG_DIR = (Join-Path $repoRoot '.polaris-pyinstaller-cache')

& $Python @PythonArgs -m venv $buildVenv
& "$buildVenv\Scripts\python.exe" -m pip install --disable-pip-version-check --only-binary=:all: `
    'tensorflow==2.13.1' 'keras==2.13.1' 'h5py>=3.8' `
    'numpy>=1.24,<1.25' 'pandas>=2.0' 'scipy>=1.10' `
    'matplotlib>=3.7' 'seaborn>=0.13' 'pyserial==3.5' `
    'pyinstaller>=6.5,<7'
& "$buildVenv\Scripts\python.exe" -m PyInstaller `
    --noconfirm --clean `
    --distpath dist/windows `
    --workpath build/windows `
    packaging/polaris_client_cont2.spec
