"""Correlation matrix cleaning workflow with Marchenko-Pastur + percolation thresholding.

This module provides a workflow for cleaning correlation matrices using:
1. Marchenko-Pastur spectral filtering to remove noise eigenvalues
2. Percolation thresholding to remove weak connections just before network fragmentation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np
from lrgsglib.utils.basic.signals import bandpass_sos

from .config.const import BRAIN_BANDS, PHASE_LABELS
from .utils.corrmat.base import clean_correlation_matrix
from .utils.corrmat.thresholds import find_exact_detachment_threshold, find_threshold_jumps
from .utils.datamanag.patient_robust import load_patient_dataset_robust

__all__ = [
    "CleanedCorrResult",
    "clean_correlation_matrix_full",
    "load_cleaned_corr_matrix",
    "get_cleaned_corr_cache_path",
    "compute_cleaned_corr_for_patient",
]


@dataclass
class CleanedCorrResult:
    """Result from correlation cleaning workflow.

    Attributes
    ----------
    cleaned_matrix : np.ndarray
        Cleaned and thresholded correlation matrix (N x N)
    original_matrix : np.ndarray
        Original correlation matrix before cleaning (N x N)
    threshold : float
        Percolation threshold applied (just before first detachment)
    eigenvalues : np.ndarray
        Eigenvalues of original correlation matrix
    lambda_min : float
        Marchenko-Pastur lower bound
    lambda_max : float
        Marchenko-Pastur upper bound
    signal_eigenvalues : np.ndarray
        Eigenvalues above lambda_max (signal components)
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    n_signal_components : int
        Number of signal eigenvalues retained
    n_channels : int
        Number of channels
    detachment_info : Dict
        Information about percolation threshold selection
    """

    cleaned_matrix: np.ndarray
    original_matrix: np.ndarray
    threshold: float
    eigenvalues: np.ndarray
    lambda_min: float
    lambda_max: float
    signal_eigenvalues: np.ndarray
    patient: str
    phase: str
    band: str
    n_signal_components: int
    n_channels: int
    detachment_info: Dict


def get_cleaned_corr_cache_path(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/corr_cache"),
) -> tuple[Path, Path]:
    """Get cache file paths for cleaned correlation matrix and metadata.

    Parameters
    ----------
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    cache_root : Path, optional
        Root directory for cache files

    Returns
    -------
    tuple[Path, Path]
        (matrix_path, metadata_path) - Paths to cleaned matrix and metadata files
    """
    cache_dir = cache_root / patient
    cache_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = cache_dir / f"{band}_{phase}_corr_cleaned.npy"
    metadata_path = cache_dir / f"{band}_{phase}_corr_cleaned_meta.npz"

    return matrix_path, metadata_path


def load_cleaned_corr_matrix(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/corr_cache"),
    load_metadata: bool = False,
) -> Optional[np.ndarray] | tuple[Optional[np.ndarray], Optional[Dict]]:
    """Load cached cleaned correlation matrix if it exists.

    Parameters
    ----------
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    cache_root : Path, optional
        Root directory for cache files
    load_metadata : bool, optional
        If True, also load and return metadata

    Returns
    -------
    np.ndarray or None
        Cached cleaned correlation matrix, or None if not cached
    dict or None
        If load_metadata=True, returns (matrix, metadata) tuple
    """
    matrix_path, metadata_path = get_cleaned_corr_cache_path(
        patient, phase, band, cache_root
    )

    if not matrix_path.exists():
        return (None, None) if load_metadata else None

    matrix = np.load(matrix_path)

    if not load_metadata:
        return matrix

    # Load metadata if requested
    if metadata_path.exists():
        meta = np.load(metadata_path, allow_pickle=True)
        metadata = {
            "threshold": float(meta["threshold"]),
            "eigenvalues": meta["eigenvalues"],
            "lambda_min": float(meta["lambda_min"]),
            "lambda_max": float(meta["lambda_max"]),
            "signal_eigenvalues": meta["signal_eigenvalues"],
            "n_signal_components": int(meta["n_signal_components"]),
            "detachment_info": meta["detachment_info"].item(),
        }
        return matrix, metadata
    else:
        return matrix, None


def clean_correlation_matrix_full(
    patient: str,
    phase: str,
    band: str,
    dataset_root: Path = Path("data/stereoeeg_patients"),
    cache_root: Path = Path("data/corr_cache"),
    *,
    filter_order: int = 4,
    sample_rate: float = 2048.0,
    use_cache: bool = True,
    overwrite_cache: bool = False,
    verbose: bool = False,
) -> CleanedCorrResult:
    """Complete cleaning workflow: bandpass → correlate → MP clean → threshold.

    Steps:
    1. Load timeseries and bandpass filter
    2. Compute correlation matrix
    3. Apply Marchenko-Pastur spectral cleaning
    4. Find percolation threshold (just before first detachment)
    5. Apply threshold to cleaned matrix
    6. Cache results

    Parameters
    ----------
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    dataset_root : Path, optional
        Root directory for patient data
    cache_root : Path, optional
        Root directory for cache files
    filter_order : int, optional
        Butterworth filter order
    sample_rate : float, optional
        Sampling rate in Hz (will be auto-detected if available)
    use_cache : bool, optional
        Whether to load from cache if available
    overwrite_cache : bool, optional
        Whether to recompute even if cached
    verbose : bool, optional
        Print progress messages

    Returns
    -------
    CleanedCorrResult
        Result containing cleaned matrix and metadata

    Raises
    ------
    ValueError
        If data cannot be loaded or processed
    """
    # Check cache
    if use_cache and not overwrite_cache:
        matrix, metadata = load_cleaned_corr_matrix(
            patient, phase, band, cache_root, load_metadata=True
        )
        if matrix is not None and metadata is not None:
            if verbose:
                print(f"  Loaded from cache: {patient} {phase} {band}")
            # Reconstruct result from cached data
            result = CleanedCorrResult(
                cleaned_matrix=matrix,
                original_matrix=np.zeros_like(matrix),  # Not cached to save space
                threshold=metadata["threshold"],
                eigenvalues=metadata["eigenvalues"],
                lambda_min=metadata["lambda_min"],
                lambda_max=metadata["lambda_max"],
                signal_eigenvalues=metadata["signal_eigenvalues"],
                patient=patient,
                phase=phase,
                band=band,
                n_signal_components=metadata["n_signal_components"],
                n_channels=matrix.shape[0],
                detachment_info=metadata["detachment_info"],
            )
            return result

    if verbose:
        print(f"  Computing cleaned correlation: {patient} {phase} {band}")

    # Step 1: Load data and auto-detect sampling rate
    dataset = load_patient_dataset_robust(patient, dataset_root, phases=[phase])

    if phase not in dataset:
        raise ValueError(f"No data for {patient} {phase}")

    recording = dataset[phase]
    data = recording.timeseries

    # Extract sampling rate from parameters
    if "fs" in recording.parameters and recording.parameters["fs"] is not None:
        actual_fs = float(recording.parameters["fs"])
        sample_rate = actual_fs
        if verbose:
            print(f"    Using detected sampling rate: {sample_rate} Hz")
    elif verbose:
        print(f"    Using provided sampling rate: {sample_rate} Hz")

    # Get band limits
    if band not in BRAIN_BANDS:
        raise ValueError(f"Unknown band: {band}. Must be one of {list(BRAIN_BANDS.keys())}")

    f_low, f_high = BRAIN_BANDS[band]

    # Step 2: Bandpass filter
    filtered_data = bandpass_sos(
        data, f_low, f_high, fs=sample_rate, order=filter_order
    )

    # Step 3: Compute correlation matrix
    C_orig = np.corrcoef(filtered_data)

    # Handle non-finite values
    if not np.all(np.isfinite(C_orig)):
        C_orig = np.nan_to_num(C_orig, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 4: Marchenko-Pastur spectral cleaning
    C_clean, eigvals, eigvecs, lambda_min, lambda_max, signal_mask = (
        clean_correlation_matrix(filtered_data, rowvar=True)
    )

    signal_eigvals = eigvals[signal_mask]
    n_signal = len(signal_eigvals)

    if verbose:
        print(
            f"    MP cleaning: {n_signal}/{len(eigvals)} signal components "
            f"(λ_max={lambda_max:.3f})"
        )

    # Step 5: Find percolation threshold (just before first detachment)
    # We'll use absolute value for thresholding
    C_abs = np.abs(C_clean)

    # Create weighted graph for percolation analysis
    np.fill_diagonal(C_abs, 0)
    G = nx.from_numpy_array(C_abs)

    # Find thresholds where nodes detach
    try:
        Th, jumps, Einf, Pinf = find_threshold_jumps(G, return_stats=True)

        if len(jumps) > 0:
            # Use threshold just before first detachment
            # jumps[0] is the index where first detachment occurs
            # We want the threshold just before that
            detachment_idx = jumps[0]
            if detachment_idx > 0:
                threshold = Th[detachment_idx - 1]
            else:
                threshold = Th[detachment_idx]

            detachment_info = {
                "n_jumps": len(jumps),
                "first_detachment_idx": int(detachment_idx),
                "Pinf_at_threshold": float(Pinf[detachment_idx]),
                "Einf_at_threshold": float(Einf[detachment_idx]),
                "all_thresholds": Th.tolist(),
                "all_jumps": jumps.tolist(),
            }

            if verbose:
                print(
                    f"    Percolation threshold: {threshold:.4f} "
                    f"(P_inf={Pinf[detachment_idx]:.3f}, "
                    f"detachment at idx={detachment_idx}/{len(Th)})"
                )
        else:
            # No detachments found - use a conservative threshold
            threshold = find_exact_detachment_threshold(C_abs)
            detachment_info = {
                "n_jumps": 0,
                "method": "binary_search",
                "threshold": float(threshold),
            }
            if verbose:
                print(f"    No percolation jumps detected, using binary search: {threshold:.4f}")

    except Exception as e:
        if verbose:
            print(f"    Warning: Percolation analysis failed ({e}), using median threshold")
        # Fallback: use median of upper triangle
        triu_indices = np.triu_indices_from(C_abs, k=1)
        threshold = float(np.median(C_abs[triu_indices]))
        detachment_info = {
            "method": "median_fallback",
            "threshold": float(threshold),
            "error": str(e),
        }

    # Step 6: Apply threshold to cleaned matrix
    C_thresholded = C_clean.copy()
    C_thresholded[np.abs(C_thresholded) < threshold] = 0
    np.fill_diagonal(C_thresholded, 0)

    # Create result
    result = CleanedCorrResult(
        cleaned_matrix=C_thresholded,
        original_matrix=C_orig,
        threshold=float(threshold),
        eigenvalues=eigvals,
        lambda_min=float(lambda_min),
        lambda_max=float(lambda_max),
        signal_eigenvalues=signal_eigvals,
        patient=patient,
        phase=phase,
        band=band,
        n_signal_components=n_signal,
        n_channels=C_orig.shape[0],
        detachment_info=detachment_info,
    )

    # Step 7: Cache results
    matrix_path, metadata_path = get_cleaned_corr_cache_path(
        patient, phase, band, cache_root
    )

    np.save(matrix_path, C_thresholded)

    # Save metadata (don't save original matrix to save space)
    np.savez(
        metadata_path,
        threshold=threshold,
        eigenvalues=eigvals,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        signal_eigenvalues=signal_eigvals,
        n_signal_components=n_signal,
        detachment_info=detachment_info,
    )

    if verbose:
        print(f"    Cached to: {matrix_path.name}")

    return result


def compute_cleaned_corr_for_patient(
    patient: str,
    dataset_root: Path = Path("data/stereoeeg_patients"),
    cache_root: Path = Path("data/corr_cache"),
    *,
    bands: Optional[list[str]] = None,
    phases: Optional[list[str]] = None,
    filter_order: int = 4,
    sample_rate: float = 2048.0,
    use_cache: bool = True,
    overwrite_cache: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, Optional[CleanedCorrResult]]]:
    """Compute cleaned correlation matrices for all bands and phases of a patient.

    Parameters
    ----------
    patient : str
        Patient identifier
    dataset_root : Path, optional
        Root directory for patient data
    cache_root : Path, optional
        Root directory for cache files
    bands : list[str], optional
        Frequency bands to process (default: all BRAIN_BANDS)
    phases : list[str], optional
        Recording phases to process (default: all PHASE_LABELS)
    filter_order : int, optional
        Butterworth filter order
    sample_rate : float, optional
        Sampling rate in Hz (will be auto-detected if available)
    use_cache : bool, optional
        Whether to load from cache if available
    overwrite_cache : bool, optional
        Whether to recompute even if cached
    verbose : bool, optional
        Print progress messages

    Returns
    -------
    dict
        Nested dict: {band: {phase: CleanedCorrResult or None}}
    """
    if bands is None:
        bands = list(BRAIN_BANDS.keys())
    if phases is None:
        phases = list(PHASE_LABELS)

    results = {}

    for band in bands:
        results[band] = {}
        for phase in phases:
            try:
                result = clean_correlation_matrix_full(
                    patient,
                    phase,
                    band,
                    dataset_root=dataset_root,
                    cache_root=cache_root,
                    filter_order=filter_order,
                    sample_rate=sample_rate,
                    use_cache=use_cache,
                    overwrite_cache=overwrite_cache,
                    verbose=verbose,
                )
                results[band][phase] = result

            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed {patient} {phase} {band}: {e}")
                results[band][phase] = None

    return results
