# SCM Neural-Network Inverse Kinematics

For headless or Android testing, install only the core numerical dependencies:

```sh
uv sync
```

Optional dependency groups can be added when needed:

```sh
uv sync --extra plots      # matplotlib, seaborn, plotly
uv sync --extra ml         # TensorFlow/Keras and scikit-learn
uv sync --extra notebooks  # Jupyter
uv sync --extra hardware   # serial and NDI tracker support
```

`read_datafile.py` has `ENABLE_PLOTS = False` by default, so it can run without
the plotting group. Set it to `True` only when the `plots` extra is installed.
