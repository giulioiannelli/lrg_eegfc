"""High-level patient dataset loaders (moved from :mod:`lrg_eegfc.io`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from ...config.const import PARAMETER_KEYS, PHASE_LABELS
from .loaders import load_mat_pat_data


__all__ = [
    "PatientRecording",
    "load_timeseries",
    "load_patient_metadata",
    "load_patient_dataset",
    "load_dataset",
]


@dataclass
class PatientRecording:
    """Container holding one patient's time series and optional metadata."""

    timeseries: np.ndarray
    parameters: Mapping[str, float]
    channel_metadata: Optional[pd.DataFrame] = None


def _extract_timeseries(mat: Mapping[str, object], patient: str, phase: str) -> np.ndarray:
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
    mat = load_mat_pat_data(patient, phase, root_path)
    return _extract_timeseries(mat, patient, phase)


def _load_parameters(mat: Mapping[str, object]) -> Mapping[str, float]:
    parameters: Dict[str, float] = {}
    params = mat.get("Parameters")
    if params is None:
        return parameters

    names = getattr(getattr(params, "dtype", None), "names", None) or ()
    for name in names:
        if name not in PARAMETER_KEYS:
            continue
        try:
            parameters[name] = float(params[name][0][0][0][0])
        except Exception:  # noqa: BLE001
            logging.debug("Could not parse parameter %s", name, exc_info=True)
    return parameters


def load_patient_metadata(patient: str, root_path: Path) -> Optional[pd.DataFrame]:
    patient_path = root_path / patient
    try:
        patnum = int(patient.split("_")[-1])
    except ValueError:
        patnum = 0

    implant_csv = patient_path / f"Implant_pat_{patnum:02d}.csv"
    channel_labels_csv = patient_path / "channel_labels.csv"

    if not implant_csv.exists() or not channel_labels_csv.exists():
        return None

    implant = pd.read_csv(implant_csv)
    labels = pd.read_csv(channel_labels_csv)
    metadata = labels.merge(implant, on="label", how="left")

    for column in ("x", "y", "z"):
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
    dataset: Dict[str, PatientRecording] = {}
    metadata = load_patient_metadata(patient, root_path)

    for phase in phases:
        mat = load_mat_pat_data(patient, phase, root_path)
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
    return {patient: load_patient_dataset(patient, root_path, phases) for patient in patients}
