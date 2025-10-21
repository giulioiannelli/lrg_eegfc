import logging
import h5py

import numpy as np
import pandas as pd

from pathlib import Path

from typing import List, Tuple, Dict, Any
from scipy.io import loadmat

from lrg_eegfc.config.const import PARAMETER_KEYS, PHASE_LABELS,\
    sEEG_DATAPATH, PATIENTS_LIST

__all__ = [
    "load_mat_pat_data",
    "load_data_dict"
]

def load_mat_pat_data(
    patient: str,
    phase: str,
    mat_path: Path,
) -> Dict[str, Any]:
    """Load a patient's phase `.mat` file with a robust fallback.

    This function first tries `scipy.io.loadmat`. If that fails (e.g., due to
    v7.3 HDF5-based files), it falls back to reading via `h5py` and returns a
    plain dictionary of NumPy arrays.

    Parameters
    ----------
    patient : str
        Patient identifier, matching a subdirectory under `mat_path`.
    phase : str
        Phase name (e.g., "rsPre", "taskTest"). The file is `{phase}.mat`.
    mat_path : Path
        Base directory containing per-patient subdirectories with `.mat` files.

    Returns
    -------
    Dict[str, Any]
        Dictionary-like contents of the `.mat` file. Keys depend on the file
        schema (e.g., may include "Data" and "Parameters").
    """

    path = mat_path / patient / f"{phase}.mat"
    try:
        return loadmat(str(path))
    except Exception as exc:  # noqa: BLE001 - broad fallback is intentional
        logging.warning(
            "Failed to load %s via scipy (%s). Using h5py fallback.",
            path,
            type(exc).__name__,
        )
        # Ensure the file is closed and eagerly load datasets into memory.
        with h5py.File(path, "r") as f:
            return {k: np.array(v) for k, v in f.items()}

def load_data_dict(
    mat_path: Path = sEEG_DATAPATH,
    pat_list: List[str] = PATIENTS_LIST,
    phase_labels: List[str] = PHASE_LABELS,
    param_keys_list: List[str] = PARAMETER_KEYS,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """Load `.mat` data and channel metadata for multiple patients/phases.

    For each patient in `pat_list` and each phase in `phase_labels`, this
    function loads `{phase}.mat` using `load_mat_pat_data`, extracts the data
    matrix (transpose if needed so samples are rows), and pulls specified
    parameters. It also builds a label-to-position map by merging channel label
    and implant coordinate CSVs, coercing `x`, `y`, `z` to floats.

    Parameters
    ----------
    mat_path : Path, optional
        Root directory containing per-patient subdirectories with data.
    pat_list : List[str], optional
        Patient identifiers to process.
    phase_labels : List[str], optional
        Phase names to load for each patient.
    param_keys_list : List[str], optional
        Parameter keys to extract from the `Parameters` struct, if present.

    Returns
    -------
    Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]
        A nested dictionary `data_dict[patient][phase]` with `data` and any
        extracted parameters, and a dict mapping each patient to a DataFrame of
        channel labels and coordinates.
    """

    data_dict: Dict[str, Dict[str, Any]] = {}
    int_label_pos_map: Dict[str, pd.DataFrame] = {}

    for pat in pat_list:
        data_dict[pat] = {}
        patnum = int(pat.split('_')[-1])
        patpath = mat_path / pat

        # Merge channel labels with implant coordinates.
        ch_dat = pd.read_csv(patpath / f"Implant_pat_{patnum}.csv")
        ch_names = pd.read_csv(patpath / "channel_labels.csv")
        merged = ch_names.merge(ch_dat, on="label", how="left")

        # Coerce coordinate columns to float (handling commas as decimals).
        coord_cols = [c for c in ["x", "y", "z"] if c in merged.columns]
        if coord_cols:
            merged.loc[:, coord_cols] = (
                merged[coord_cols]
                .apply(lambda s: s.astype(str).str.replace(",", ".", 
                                                           regex=False))
                .apply(pd.to_numeric, errors="coerce")
            )

        int_label_pos_map[pat] = merged

        for phase in phase_labels:
            data_dict[pat][phase] = {}
            tmp_mat = load_mat_pat_data(pat, phase, mat_path)

            # Extract data matrix and ensure samples are rows.
            arr = tmp_mat["Data"]
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
            data_dict[pat][phase]["data"] = arr

            # Extract selected parameters when available.
            params = tmp_mat.get("Parameters")
            names = getattr(getattr(params, "dtype", None), "names", None)
            if params is not None and names:
                for name in names:
                    if name not in param_keys_list:
                        continue
                    try:
                        value = np.squeeze(params[name]).item()
                    except Exception:  # noqa: BLE001
                        try:
                            value = params[name][0][0][0][0]
                        except Exception:  # noqa: BLE001
                            logging.debug(
                                "Could not parse parameter %s for %s %s",
                                name,
                                pat,
                                phase,
                            )
                            continue
                    data_dict[pat][phase][name] = value
            else:
                logging.info("No parameters found for %s %s", pat, phase)

    return data_dict, int_label_pos_map
