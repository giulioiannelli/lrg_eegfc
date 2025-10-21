"""Helpers for loading the SEEG datasets shipped with the project."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .constants import PARAMETER_KEYS, PHASE_LABELS

__all__ = [
    "PatientRecording",
    "load_mat_file",
    "load_timeseries",
    "load_patient_metadata",
    "load_patient_dataset",
    "load_dataset",
]


@dataclass
class PatientRecording:
    """Container holding one patient's time series and optional metadata.

    Attributes
    ----------
    timeseries:
        The SEEG signal arranged as ``(n_channels, n_samples)``.
    parameters:
        Mapping of acquisition parameters parsed from the ``.mat`` file.  When
        a value is not present in the source file it is simply omitted.
    channel_metadata:
        Per-channel metadata (typically anatomical coordinates) as loaded from
        ``channel_labels.csv`` joined with ``Implant_pat_XX.csv``.  This value is
        optional because some datasets only provide raw signals.
    """

    timeseries: np.ndarray
    parameters: Mapping[str, float]
    channel_metadata: Optional[pd.DataFrame] = None


def load_mat_file(
    patient: str,
    phase: str,
    root_path: Path,
    *,
    fields: Optional[Iterable[str]] = None,
) -> Mapping[str, object]:
    """Load a ``.mat`` file using :func:`scipy.io.loadmat` with an ``h5py`` fallback.

    Parameters
    ----------
    patient, phase, root_path:
        Identify the ``.mat`` file to read.
    fields:
        Optional iterable of variable names to load from the ``.mat`` file.  Supplying
        this limits the amount of data parsed which can dramatically reduce IO for
        large recordings.  When omitted the full file content is returned, mimicking
        :func:`scipy.io.loadmat`.
    """

    mat_file = root_path / patient / f"{phase}.mat"
    fields_tuple = tuple(fields) if fields is not None else None
    load_kwargs = {"variable_names": fields_tuple} if fields_tuple is not None else {}

    try:
        return loadmat(mat_file, **load_kwargs)
    except NotImplementedError:
        logging.debug("Falling back to h5py for %s", mat_file)
        with h5py.File(mat_file, "r") as handle:
            keys: Iterable[str]
            if fields_tuple is None:
                keys = handle.keys()
            else:
                keys = (name for name in fields_tuple if name in handle)
            return {key: np.array(handle[key]) for key in keys}


def _extract_timeseries(
    mat: Mapping[str, object],
    patient: str,
    phase: str,
) -> np.ndarray:
    """Return the canonical ``(channels, samples)`` arrangement from ``mat``."""

    if "Data" not in mat:
        raise KeyError(f"'Data' entry not found in MAT file for {patient} {phase}.")

    data = np.asarray(mat["Data"])
    if data.ndim != 2:
        raise ValueError(
            f"Expected 2-D SEEG matrix for {patient} {phase}; received shape {data.shape}."
        )
    if data.shape[0] > data.shape[1]:
        data = data.T
    return data


def load_timeseries(patient: str, phase: str, root_path: Path) -> np.ndarray:
    """Load and normalise the SEEG time series for ``patient`` and ``phase``.

    The raw data is stored in the ``Data`` field of the ``.mat`` files.  Some
    files store the samples as ``(samples, channels)`` so the function always
    returns the canonical ``(channels, samples)`` layout.
    """

    mat = load_mat_file(patient, phase, root_path, fields=("Data",))
    return _extract_timeseries(mat, patient, phase)


def _load_parameters(mat: Mapping[str, object]) -> Mapping[str, float]:
    """Extract acquisition parameters from ``mat`` if available."""

    parameters: Dict[str, float] = {}
    params = mat.get("Parameters")
    if params is None:
        return parameters

    try:
        names = params.dtype.names or ()
        for name in names:
            if name in PARAMETER_KEYS:
                parameters[name] = float(params[name][0][0][0][0])
    except Exception:  # pragma: no cover - defensive fallback
        logging.debug("Unable to parse parameters for entry; ignoring", exc_info=True)
    return parameters


def load_patient_metadata(patient: str, root_path: Path) -> Optional[pd.DataFrame]:
    """Load the metadata CSV files that accompany a patient directory."""

    patient_path = root_path / patient
    implant_csv = patient_path / f"Implant_pat_{int(patient.split('_')[-1]):02d}.csv"
    channel_labels_csv = patient_path / "channel_labels.csv"

    if not implant_csv.exists() or not channel_labels_csv.exists():
        return None

    implant = pd.read_csv(implant_csv)
    labels = pd.read_csv(channel_labels_csv)
    metadata = labels.merge(implant, on="label", how="left")

    numeric_columns = ["x", "y", "z"]
    for column in numeric_columns:
        if column in metadata.columns:
            metadata[column] = (
                metadata[column]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
    return metadata


def load_patient_dataset(
    patient: str,
    root_path: Path,
    phases: Iterable[str] = PHASE_LABELS,
) -> Dict[str, PatientRecording]:
    """Load all requested ``phases`` for ``patient``."""

    dataset: Dict[str, PatientRecording] = {}
    metadata = load_patient_metadata(patient, root_path)

    for phase in phases:
        mat = load_mat_file(
            patient,
            phase,
            root_path,
            fields=("Data", "Parameters"),
        )
        dataset[phase] = PatientRecording(
            timeseries=_extract_timeseries(mat, patient, phase),
            parameters=_load_parameters(mat),
            channel_metadata=metadata,
        )
    return dataset


def load_dataset(
    root_path: Path,
    patients: Iterable[str],
    phases: Iterable[str] = PHASE_LABELS,
) -> Dict[str, Dict[str, PatientRecording]]:
    """Load a nested ``dict`` of ``patient -> phase -> recording``."""

    return {
        patient: load_patient_dataset(patient, root_path, phases)
        for patient in patients
    }
