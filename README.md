# Comparison of machine learning architectures for inverse kinematics accuracy in soft continuum robots driven by flexible rods

Research code and experimental data accompanying the work on the mechanical behaviour and
inverse kinematics of a flexible-rod-driven soft continuum robot. The study compares a
constant-curvature (PCC) baseline with feed-forward neural networks and recurrent neural
networks for predicting actuator commands from desired end-effector positions.

Work by André Raposo, Prof. João Reis and Prof. António Campos @ Instituto Superior Técnico,
ULisboa

## Repository guide

- `data/` contains the experimental datasets used for mechanical characterisation and
  model training. A standard record comprises three actuator commands, the Polaris
  reference-tool transform, and the tracked marker position. `data_functions.py` parses
  the files and transforms tracked points into the robot base frame.
- `read_datafile.py` is the main post-processing entry point. Select the desired
  files/folders in its configuration block to produce mechanical-test plots and static or
  continuous-trajectory validation plots.
- `final_ml_kinematics_ik*.ipynb` contains the final training and architecture-comparison
  experiments; supporting model and kinematics utilities are in `nn_builder.py`,
  `ml_kinematics_ik.py`, and `kinematics_functions.py`.
- `val/` stores static (point-to-point) inverse-kinematics validation runs, while
  `cont_val/` stores continuous trajectory runs and their logs.
- `polaris_client.py` records experimental datasets using the NDI Polaris optical tracker.
  `polaris_client_cont2.py` executes and records continuous trajectory tests.
  `polaris/8700449.rom` is the tracker tool definition used by the acquisition scripts.
- `results/` and the model-result archives preserve training outputs and selected trained
  models (untar the `minimal_model_training_results.tar.gz` to access part of the results,
  contact andreraposo91@gmail.com for the rest).

## Use

This code is provided as a research resource, with experiment-specific paths, constants,
serial ports, and run selections retained in the scripts. Install the Python dependencies
listed in `pyproject.toml`, then start with `read_datafile.py` for analysis or open one of
the final training notebooks for the model experiments. Re-running data acquisition
additionally requires the original robot hardware, motor-control setup, NDI Polaris
tracker, ROM file, and the relevant tracker/serial Python packages.

The file names encode the acquisition or test type and timestamp; commented examples in
`read_datafile.py` and `polaris_client.py` document the available experimental
configurations.

# Packaging the Polaris client

Build the application on the operating system where it will run. The recipes
do not use `uv`: each creates a repository-local `.polaris-build-venv`, installs
the build dependencies there, and produces one executable. The executable
contains the Python runtime, so its user does not need Python or a virtual
environment.

Linux (x86_64):

```bash
bash scripts/build-polaris-linux.sh
./dist/linux/polaris_client_cont2
```

Windows (64-bit PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-polaris-windows.ps1
.\dist\windows\polaris_client_cont2.exe
```

Each package includes the six default inverse-kinematics models selected by
`polaris_client_cont2.py`, both normalization files, and the Polaris ROM file.
On start, it prompts for a model and square/circle/coil trajectory, then offers
to use that trajectory's default arguments before asking for individual geometry
values. It then asks for the output location, preview, and whether to send robot
commands. Sending defaults to **No**, so a
trajectory can be checked without opening a serial port. `--model`,
`--trajectory`, `--default-trajectory`, `--no-plot`, `--send`, `--robot-port`, and `--output-dir` are
also available for partially scripted invocation.

The `.keras` model archives are copied as-is; `keras_compat.py` loads their
Windows-style weight layout at inference time. TensorFlow/PyInstaller binaries
are platform-specific, so a Windows executable must be built on Windows and a
Linux executable on Linux. Building requires an installed 64-bit Python 3.9;
running the resulting executable does not. A single-file PyInstaller executable
extracts its bundled TensorFlow runtime and models into a temporary directory
while it runs; it does not alter the original model archives.
