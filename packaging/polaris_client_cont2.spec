# Build this spec on the target operating system; PyInstaller does not
# cross-compile Python extension modules or TensorFlow binaries.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


ROOT = Path(SPECPATH).parent
MODEL_FILES = [
    "results/final_datasets3/models_2024-03-26T112415/mlp_1_62pct_3L_70.keras",
    "results/final_datasets3/models_2024-04-09T130207/mlp_1_62pct_2L_80.keras",
    "results/final_datasets_comp/models_2024-06-07T102829/mlp_1_87pct_4L_75.keras",
    "results/final_datasets_comp/models_2024-06-06T121224/mlp_1_87pct_4L_70.keras",
    "results/final_datasets_rnn/models_2024-04-03T155223/rnn_100pct_2L_70.keras",
    "results/final_datasets_rnn_comp/models_2024-06-11T204342/rnn_25pct_2L_70.keras",
]
RESOURCE_FILES = [
    "data_norm_w-diff.npz",
    "data_norm_w-diff_comp.npz",
    "polaris/8700449.rom",
    *MODEL_FILES,
]

missing = [path for path in RESOURCE_FILES if not (ROOT / path).is_file()]
if missing:
    raise FileNotFoundError("Cannot package missing resource(s):\n" + "\n".join(missing))

# Preserve each resource's repository-relative path.  runtime_paths.bundled_path
# uses precisely this layout in both source and frozen execution.
datas = [(str(ROOT / path), str(Path(path).parent)) for path in RESOURCE_FILES]
binaries, extra_datas, hiddenimports = [], [], ["inv_kin_val"]
# Keras is imported directly by keras_compat.py and PyInstaller's normal module
# analysis includes it. Recursively importing every Keras submodule can crash
# PyInstaller's isolated scanner under TensorFlow 2.13.
for package in ("tensorflow", "h5py"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    extra_datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    [str(ROOT / "polaris_client_cont2.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas + extra_datas,
    hiddenimports=hiddenimports,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="polaris_client_cont2",
    console=True,
)
