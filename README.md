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
