"""Robust patient data loading with automatic format detection.

This module provides enhanced data loading that handles variations in .mat file
structure, data variable names, and metadata formats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py

from ...config.const import PHASE_LABELS

__all__ = ["PatientRecording", "load_timeseries_robust", "load_patient_dataset_robust"]

logger = logging.getLogger(__name__)


@dataclass
class PatientRecording:
    """Container for a patient's recording data and metadata."""

    timeseries: np.ndarray
    parameters: Dict[str, object]
    channel_metadata: Optional[pd.DataFrame]


def _try_load_mat_scipy(mat_path: Path) -> Optional[Mapping]:
    """Try loading with scipy.io.loadmat."""
    try:
        return loadmat(mat_path)
    except Exception as e:
        logger.debug(f"scipy.io.loadmat failed for {mat_path}: {e}")
        return None


def _try_load_mat_h5py(mat_path: Path) -> Optional[Mapping]:
    """Try loading with h5py (returns file handle - caller must close)."""
    try:
        f = h5py.File(mat_path, 'r')
        return f
    except Exception as e:
        logger.debug(f"h5py failed for {mat_path}: {e}")
        return None


def _extract_data_scipy(mat: Mapping, patient: str, phase: str) -> Optional[np.ndarray]:
    """Extract timeseries data from scipy-loaded .mat file."""
    # Try common data variable names
    data_keys = ['Data', 'data', 'EEG', 'timeseries', 'signal', 'eeg_data']

    for key in data_keys:
        if key in mat:
            data = np.asarray(mat[key])
            if data.ndim == 2:
                logger.info(f"{patient}/{phase}: Found data in '{key}', shape {data.shape}")
                return data
            else:
                logger.warning(f"{patient}/{phase}: '{key}' has unexpected dimensions: {data.shape}")

    # If none found, list available keys
    available = [k for k in mat.keys() if not k.startswith('__')]
    logger.error(f"{patient}/{phase}: No data variable found. Available keys: {available}")
    return None


def _extract_data_h5py(mat: h5py.File, patient: str, phase: str) -> Optional[np.ndarray]:
    """Extract timeseries data from h5py-loaded .mat file."""
    data_keys = ['Data', 'data', 'EEG', 'timeseries', 'signal', 'eeg_data']

    for key in data_keys:
        if key in mat:
            data = mat[key][()]
            if isinstance(data, np.ndarray) and data.ndim == 2:
                logger.info(f"{patient}/{phase}: Found data in '{key}' (h5py), shape {data.shape}")
                return data

    available = list(mat.keys())
    logger.error(f"{patient}/{phase}: No data variable found (h5py). Available keys: {available}")
    return None


def _extract_parameters_scipy(mat: Mapping) -> Dict[str, object]:
    """Extract parameters from scipy-loaded .mat file."""
    params = {}

    # Try Parameters struct
    if 'Parameters' in mat:
        param_struct = mat['Parameters']
        if isinstance(param_struct, np.ndarray) and param_struct.dtype.names:
            for name in param_struct.dtype.names:
                try:
                    value = param_struct[name][0, 0]
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            value = value.item()
                        elif value.size < 100:  # Don't store large arrays
                            value = value.tolist()
                        else:
                            continue  # Skip large arrays
                    params[name] = value
                except Exception as e:
                    logger.debug(f"Could not extract parameter '{name}': {e}")

    # Look for standalone common parameters
    standalone_params = {
        'fs': ['fs', 'Fs', 'sampling_rate', 'SamplingRate', 'srate'],
        'fcutHigh': ['fcutHigh', 'highcut', 'high_cutoff'],
        'fcutLow': ['fcutLow', 'lowcut', 'low_cutoff'],
        'filter_order': ['filter_order', 'filterOrder', 'FilterOrder'],
        'NotchFilter': ['NotchFilter', 'notch_filter', 'notch'],
    }

    for param_name, possible_keys in standalone_params.items():
        if param_name not in params:  # Only if not already found
            for key in possible_keys:
                if key in mat:
                    try:
                        value = mat[key]
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        params[param_name] = value
                        break
                    except:
                        pass

    return params


def _extract_parameters_h5py(mat: h5py.File) -> Dict[str, object]:
    """Extract parameters from h5py-loaded .mat file."""
    params = {}

    # Try Parameters group
    if 'Parameters' in mat:
        try:
            param_group = mat['Parameters']
            for key in param_group.keys():
                try:
                    value = param_group[key][()]
                    if isinstance(value, np.ndarray) and value.size == 1:
                        value = value.item()
                    params[key] = value
                except:
                    pass
        except:
            pass

    # Look for standalone parameters
    standalone_params = {
        'fs': ['fs', 'Fs', 'sampling_rate', 'SamplingRate'],
        'fcutHigh': ['fcutHigh', 'highcut'],
        'fcutLow': ['fcutLow', 'lowcut'],
        'filter_order': ['filter_order', 'filterOrder'],
        'NotchFilter': ['NotchFilter', 'notch_filter'],
    }

    for param_name, possible_keys in standalone_params.items():
        if param_name not in params:
            for key in possible_keys:
                if key in mat:
                    try:
                        value = mat[key][()]
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        params[param_name] = value
                        break
                    except:
                        pass

    return params


def load_timeseries_robust(
    patient: str,
    phase: str,
    root_path: Path = Path("data/stereoeeg_patients"),
    transpose_if_needed: bool = True,
) -> np.ndarray:
    """Robustly load timeseries data with automatic format detection.

    This function tries multiple loading strategies:
    1. scipy.io.loadmat (MATLAB v5-v7)
    2. h5py (MATLAB v7.3)

    And searches for data in common variable names:
    - Data, data, EEG, timeseries, signal, eeg_data

    Parameters
    ----------
    patient : str
        Patient identifier (e.g., "Pat_02")
    phase : str
        Recording phase (e.g., "rsPre")
    root_path : Path, optional
        Root directory containing patient subdirectories
    transpose_if_needed : bool, optional
        If True, transpose data if shape[0] > shape[1] to ensure
        (n_channels, n_samples) format (default: True)

    Returns
    -------
    np.ndarray
        Timeseries data with shape (n_channels, n_samples)

    Raises
    ------
    FileNotFoundError
        If the .mat file doesn't exist
    ValueError
        If data cannot be extracted from the file
    """
    mat_path = root_path / patient / f"{phase}.mat"

    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path}")

    # Try scipy first
    mat = _try_load_mat_scipy(mat_path)
    if mat is not None:
        data = _extract_data_scipy(mat, patient, phase)
        if data is not None:
            # Ensure (n_channels, n_samples) format
            if transpose_if_needed and data.shape[0] > data.shape[1]:
                logger.warning(f"{patient}/{phase}: Transposing data from {data.shape} to {data.T.shape}")
                data = data.T
            return data

    # Try h5py
    mat_h5 = _try_load_mat_h5py(mat_path)
    if mat_h5 is not None:
        try:
            data = _extract_data_h5py(mat_h5, patient, phase)
            if data is not None:
                # Ensure (n_channels, n_samples) format
                if transpose_if_needed and data.shape[0] > data.shape[1]:
                    logger.warning(f"{patient}/{phase}: Transposing data from {data.shape} to {data.T.shape}")
                    data = data.T
                return data
        finally:
            mat_h5.close()

    # If we get here, nothing worked
    raise ValueError(
        f"Could not extract timeseries data from {mat_path}. "
        f"File may be corrupted or have an unexpected format."
    )


def load_patient_metadata_robust(
    patient: str,
    root_path: Path = Path("data/stereoeeg_patients"),
) -> Optional[pd.DataFrame]:
    """Load patient metadata (channel labels, coordinates, etc.).

    Tries multiple file formats:
    - channel_labels.csv
    - channel_labels.txt
    - ChannelNames.mat
    - Implant_pat_XX.csv

    Parameters
    ----------
    patient : str
        Patient identifier
    root_path : Path, optional
        Root directory containing patient subdirectories

    Returns
    -------
    pd.DataFrame or None
        Metadata dataframe with columns like 'label', 'x', 'y', 'z', etc.
        Returns None if no metadata files found.
    """
    patient_dir = root_path / patient

    # Try channel_labels.csv
    csv_path = patient_dir / "channel_labels.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"{patient}: Loaded channel labels from CSV ({len(df)} channels)")
            return df
        except Exception as e:
            logger.warning(f"{patient}: Failed to load {csv_path}: {e}")

    # Try Implant CSV
    implant_csv = patient_dir / f"Implant_{patient.lower()}.csv"
    if implant_csv.exists():
        try:
            df = pd.read_csv(implant_csv)
            logger.info(f"{patient}: Loaded metadata from Implant CSV ({len(df)} channels)")
            return df
        except Exception as e:
            logger.warning(f"{patient}: Failed to load {implant_csv}: {e}")

    # Try ChannelNames.mat
    channel_mat = patient_dir / "ChannelNames.mat"
    if channel_mat.exists():
        try:
            mat = loadmat(channel_mat)
            if 'ChannelNames' in mat:
                names = mat['ChannelNames'][0]
                labels = [str(n[0]) if isinstance(n, np.ndarray) else str(n) for n in names]
                df = pd.DataFrame({'label': labels})
                logger.info(f"{patient}: Loaded channel names from MAT ({len(df)} channels)")
                return df
        except Exception as e:
            logger.warning(f"{patient}: Failed to load {channel_mat}: {e}")

    logger.warning(f"{patient}: No metadata files found")
    return None


def load_patient_dataset_robust(
    patient: str,
    root_path: Path = Path("data/stereoeeg_patients"),
    phases: Iterable[str] = PHASE_LABELS,
) -> Dict[str, PatientRecording]:
    """Load all phases for a patient with robust error handling.

    Parameters
    ----------
    patient : str
        Patient identifier
    root_path : Path, optional
        Root directory containing patient subdirectories
    phases : Iterable[str], optional
        Phases to load (default: all PHASE_LABELS)

    Returns
    -------
    Dict[str, PatientRecording]
        Dictionary mapping phase names to PatientRecording objects.
        Phases that fail to load are omitted (with warning logged).
    """
    metadata = load_patient_metadata_robust(patient, root_path)
    dataset = {}

    for phase in phases:
        try:
            # Load timeseries
            mat_path = root_path / patient / f"{phase}.mat"
            if not mat_path.exists():
                logger.warning(f"{patient}/{phase}: File not found, skipping")
                continue

            timeseries = load_timeseries_robust(patient, phase, root_path)

            # Load parameters
            mat = _try_load_mat_scipy(mat_path)
            if mat is not None:
                parameters = _extract_parameters_scipy(mat)
            else:
                mat_h5 = _try_load_mat_h5py(mat_path)
                if mat_h5 is not None:
                    try:
                        parameters = _extract_parameters_h5py(mat_h5)
                    finally:
                        mat_h5.close()
                else:
                    parameters = {}

            dataset[phase] = PatientRecording(
                timeseries=timeseries,
                parameters=parameters,
                channel_metadata=metadata,
            )

            logger.info(f"{patient}/{phase}: Successfully loaded ({timeseries.shape})")

        except Exception as e:
            logger.error(f"{patient}/{phase}: Failed to load - {e}")
            continue

    return dataset
