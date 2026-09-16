#!/usr/bin/env bash
# Build on x86_64 Linux. The output is dist/linux/polaris_client_cont2.
# Set PYTHON to another Python 3.9 executable if needed.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

build_python="${PYTHON:-python3.9}"
build_venv=".polaris-build-venv"
export PIP_CACHE_DIR="$repo_root/.polaris-pip-cache"
export PYINSTALLER_CONFIG_DIR="$repo_root/.polaris-pyinstaller-cache"

"$build_python" -m venv "$build_venv"
"$build_venv/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: \
  'tensorflow==2.13.1' 'keras==2.13.1' 'h5py>=3.8' \
  'numpy>=1.24,<1.25' 'pandas>=2.0' 'scipy>=1.10' \
  'matplotlib>=3.7' 'seaborn>=0.13' 'pyserial==3.5' \
  'pyinstaller>=6.5,<7'
"$build_venv/bin/python" -m PyInstaller \
  --noconfirm --clean \
  --distpath dist/linux \
  --workpath build/linux \
  packaging/polaris_client_cont2.spec
