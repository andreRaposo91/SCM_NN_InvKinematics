"""Compatibility loader for the repository's Windows-saved Keras 2 archives.

Some of the saved ``.keras`` files contain literal backslashes in the HDF5
weight-object names.  Keras on Linux consequently cannot locate their
weights through ``keras.models.load_model``.  This module reads those archives
without modifying them and assigns their stored tensors to the reconstructed
layers.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any

import h5py
from keras.models import load_model as _load_model
from keras.models import model_from_json


_DEPENDENCY_PREFIX = "_layer_checkpoint_dependencies\\"


def _natural_key(value: str) -> list[object]:
    """Sort ``dense_10`` after ``dense_2`` rather than before it."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _windows_weight_groups(weights_file: h5py.File) -> list[tuple[str, list[Any]]]:
    """Return non-empty layer tensor groups from the affected HDF5 layout."""
    groups = []
    for name in sorted(weights_file, key=_natural_key):
        if not name.startswith(_DEPENDENCY_PREFIX):
            continue
        variables = weights_file[name]["vars"]
        if len(variables):
            groups.append(
                (name, [variables[str(index)][()] for index in range(len(variables))])
            )
    return groups


def _load_windows_archive(path: str | Path):
    """Load a Keras 2 archive whose HDF5 internal paths contain ``\\``."""
    with zipfile.ZipFile(path) as archive:
        model = model_from_json(archive.read("config.json"))
        weights_bytes = archive.read("model.weights.h5")

    with h5py.File(io.BytesIO(weights_bytes), "r") as weights_file:
        candidates = _windows_weight_groups(weights_file)

    for layer in model.layers:
        expected_shapes = [tuple(variable.shape) for variable in layer.weights]
        if not expected_shapes:
            continue

        matches = [
            (name, tensors)
            for name, tensors in candidates
            if [tensor.shape for tensor in tensors] == expected_shapes
        ]
        if not matches:
            available = {
                name: [tensor.shape for tensor in tensors]
                for name, tensors in candidates
            }
            raise ValueError(
                f"Could not match stored weights for layer {layer.name!r} in {path}: "
                f"expected {expected_shapes}, available {available}"
            )

        _, tensors = matches[0]
        layer.set_weights(tensors)
        candidates.remove(matches[0])

    if candidates:
        raise ValueError(
            f"Unused stored layer weights in {path}: {[name for name, _ in candidates]}"
        )
    return model


def load_model(path: str | Path, *, compile: bool = False, **kwargs: Any):
    """Load a repository model without changing its archive on disk.

    Normal Keras archives use the regular loader.  The repository's affected
    Windows-saved archives are detected from their internal HDF5 names and
    loaded through the read-only compatibility path instead.
    """
    with zipfile.ZipFile(path) as archive:
        weights_bytes = archive.read("model.weights.h5")

    with h5py.File(io.BytesIO(weights_bytes), "r") as weights_file:
        is_windows_layout = any(
            name.startswith(_DEPENDENCY_PREFIX) for name in weights_file
        )

    if is_windows_layout:
        if kwargs:
            raise TypeError(
                "The Windows-archive compatibility loader supports only the "
                "'compile' argument."
            )
        return _load_windows_archive(path)

    return _load_model(path, compile=compile, **kwargs)
