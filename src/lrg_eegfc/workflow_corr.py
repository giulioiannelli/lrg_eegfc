"""Correlation-based functional connectivity workflow with caching.

This module provides a simple workflow for computing and caching correlation-based
functional connectivity matrices, matching the MSC workflow interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np
from lrgsglib.utils.basic.signals import bandpass_sos

from .config.const import BRAIN_BANDS
from .utils.corrmat import build_corr_network
from .utils.datamanag.patient import load_timeseries
from .utils.datamanag.patient_robust import load_patient_dataset_robust

__all__ = ["CorrResult", "compute_corr_matrix", "load_corr_matrix", "get_corr_cache_path"]


@dataclass
class CorrResult:
    """Result from correlation computation.

    Attributes
    ----------
    adjacency_matrix : np.ndarray
        Correlation adjacency matrix (N x N)
    graph : nx.Graph
        NetworkX graph from adjacency matrix
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    n_channels : int
        Number of channels
    mean_corr : float
        Mean correlation value (excluding diagonal)
    filter_type : str
        Filter type used ("abs", "pos", "neg", or "none")
    zero_diagonal : bool
        Whether diagonal was zeroed
    filter_order : int
        Bandpass filter order used
    """

    adjacency_matrix: np.ndarray
    graph: nx.Graph
    patient: str
    phase: str
    band: str
    n_channels: int
    mean_corr: float
    filter_type: str
    zero_diagonal: bool
    filter_order: int


def get_corr_cache_path(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/corr_cache"),
    filter_type: str = "abs",
    zero_diagonal: bool = True,
) -> Path:
    """Get cache file path for correlation matrix.

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
    filter_type : str, optional
        Filter type ("abs", "pos", "neg", "none")
    zero_diagonal : bool, optional
        Whether diagonal is zeroed

    Returns
    -------
    Path
        Path to cache file
    """
    cache_dir = cache_root / patient
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Include filter_type and zero_diagonal in filename for different cache keys
    suffix = f"ftype-{filter_type}_zdiag-{zero_diagonal}"
    return cache_dir / f"{band}_{phase}_corr_{suffix}.npy"


def load_corr_matrix(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/corr_cache"),
    filter_type: str = "abs",
    zero_diagonal: bool = True,
) -> Optional[np.ndarray]:
    """Load cached correlation matrix if it exists.

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
    filter_type : str, optional
        Filter type ("abs", "pos", "neg", "none")
    zero_diagonal : bool, optional
        Whether diagonal is zeroed

    Returns
    -------
    np.ndarray or None
        Cached correlation matrix, or None if not cached
    """
    cache_path = get_corr_cache_path(patient, phase, band, cache_root, filter_type, zero_diagonal)

    if cache_path.exists():
        return np.load(cache_path)
    return None


def compute_corr_matrix(
    patient: str,
    phase: str,
    band: str,
    dataset_root: Path = Path("data/stereoeeg_patients"),
    cache_root: Path = Path("data/corr_cache"),
    *,
    use_cache: bool = True,
    overwrite_cache: bool = False,
    sample_rate: float = 2048.0,
    filter_order: int = 4,
    filter_type: str = "abs",
    zero_diagonal: bool = True,
    filter_time: Optional[int] = None,
    verbose: bool = False,
) -> CorrResult:
    """Compute correlation-based functional connectivity matrix.

    This function computes Pearson correlation for a given patient, phase,
    and frequency band. Results are automatically cached for faster
    subsequent access.

    Parameters
    ----------
    patient : str
        Patient identifier (e.g., "Pat_02")
    phase : str
        Recording phase (e.g., "rsPre", "taskLearn")
    band : str
        Frequency band (must exist in BRAIN_BANDS)
    dataset_root : Path, optional
        Root directory containing patient data
    cache_root : Path, optional
        Root directory for cache files
    use_cache : bool, optional
        If True, load from cache if available (default: True)
    overwrite_cache : bool, optional
        If True, recompute even if cached (default: False)
    sample_rate : float, optional
        Sampling rate in Hz (default: 2048.0)
    filter_order : int, optional
        Bandpass filter order (default: 4)
    filter_type : str, optional
        Filter type: "abs" (absolute value), "pos" (positive only),
        "neg" (negative only), or "none" (raw correlation) (default: "abs")
    zero_diagonal : bool, optional
        Set diagonal to zero (no self-loops) (default: True)
    filter_time : int, optional
        Limit to first N samples (for testing)
    verbose : bool, optional
        Print progress information (default: False)

    Returns
    -------
    CorrResult
        Result object containing adjacency matrix and metadata

    Raises
    ------
    KeyError
        If band not defined in BRAIN_BANDS
    FileNotFoundError
        If patient/phase data doesn't exist

    Examples
    --------
    >>> # Compute correlation for Pat_02, resting pre-task, beta band
    >>> result = compute_corr_matrix("Pat_02", "rsPre", "beta")
    >>> print(result.adjacency_matrix.shape)
    (117, 117)
    >>> print(f"Mean correlation: {result.mean_corr:.4f}")
    Mean correlation: 0.1234

    >>> # Force recomputation
    >>> result = compute_corr_matrix("Pat_02", "rsPre", "beta", overwrite_cache=True)
    """

    # Validate band
    if band not in BRAIN_BANDS:
        available = ", ".join(sorted(BRAIN_BANDS))
        raise KeyError(f"Band '{band}' not defined. Available: {available}")

    # Check cache
    cache_path = get_corr_cache_path(patient, phase, band, cache_root, filter_type, zero_diagonal)

    if use_cache and not overwrite_cache and cache_path.exists():
        if verbose:
            print(f"  loading cached corr: {cache_path.name}")
        adj_matrix = np.load(cache_path)

        # Reconstruct result
        graph = nx.from_numpy_array(adj_matrix)
        n_channels = adj_matrix.shape[0]

        # Compute mean correlation (upper triangle, excluding diagonal)
        triu_indices = np.triu_indices_from(adj_matrix, k=1)
        mean_corr = float(np.mean(adj_matrix[triu_indices]))

        return CorrResult(
            adjacency_matrix=adj_matrix,
            graph=graph,
            patient=patient,
            phase=phase,
            band=band,
            n_channels=n_channels,
            mean_corr=mean_corr,
            filter_type=filter_type,
            zero_diagonal=zero_diagonal,
            filter_order=filter_order,
        )

    # Compute correlation from timeseries
    if verbose:
        print(f"Computing correlation for {patient} {phase} {band}...")
        print(f"  Loading timeseries and extracting sampling rate...")

    # Load data with robust loader to extract sampling rate
    dataset = load_patient_dataset_robust(patient, dataset_root, phases=[phase])

    if phase not in dataset:
        raise FileNotFoundError(f"Phase '{phase}' not found for patient '{patient}'")

    recording = dataset[phase]
    data = recording.timeseries

    # Extract sampling rate from parameters, or use provided default
    if 'fs' in recording.parameters and recording.parameters['fs'] is not None:
        actual_fs = float(recording.parameters['fs'])
        if verbose:
            print(f"  Using sampling rate from data: {actual_fs} Hz")
        # Override sample_rate with actual value from data
        sample_rate = actual_fs
    else:
        if verbose:
            print(f"  WARNING: No sampling rate found in data, using provided value: {sample_rate} Hz")

    if filter_time is not None and filter_time > 0:
        data = data[:, :filter_time]

    if verbose:
        print(f"  Data shape: {data.shape}")
        print(f"  Applying bandpass filter...")

    # Bandpass filter
    low, high = BRAIN_BANDS[band]
    filtered = bandpass_sos(data, low, high, sample_rate, filter_order)

    if verbose:
        print(f"  Computing correlation (filter_type={filter_type}, zero_diagonal={zero_diagonal})...")

    # Compute correlation with specified parameters
    adj_matrix = build_corr_network(filtered, filter_type=filter_type, zero_diagonal=zero_diagonal)

    # Cache result
    if verbose:
        print(f"  Saving to cache: {cache_path}")
    np.save(cache_path, adj_matrix)

    # Build graph
    graph = nx.from_numpy_array(adj_matrix)
    n_channels = adj_matrix.shape[0]

    # Compute statistics
    triu_indices = np.triu_indices_from(adj_matrix, k=1)
    mean_corr = float(np.mean(adj_matrix[triu_indices]))

    if verbose:
        print(f"  Done! Mean correlation: {mean_corr:.4f}")

    return CorrResult(
        adjacency_matrix=adj_matrix,
        graph=graph,
        patient=patient,
        phase=phase,
        band=band,
        n_channels=n_channels,
        mean_corr=mean_corr,
        filter_type=filter_type,
        zero_diagonal=zero_diagonal,
        filter_order=filter_order,
    )


def compute_corr_for_patient(
    patient: str,
    bands: Optional[list] = None,
    phases: Optional[list] = None,
    **kwargs
) -> Dict[str, Dict[str, CorrResult]]:
    """Compute correlation matrices for all band/phase combinations for a patient.

    Parameters
    ----------
    patient : str
        Patient identifier
    bands : list, optional
        List of band names (default: all BRAIN_BANDS)
    phases : list, optional
        List of phase names (default: all phases)
    **kwargs
        Additional arguments passed to compute_corr_matrix()

    Returns
    -------
    Dict[str, Dict[str, CorrResult]]
        Nested dict indexed by [band][phase]

    Examples
    --------
    >>> # Compute all correlation matrices for Pat_02
    >>> results = compute_corr_for_patient("Pat_02")
    >>>
    >>> # Access specific result
    >>> beta_rsPre = results["beta"]["rsPre"]
    >>> print(beta_rsPre.mean_corr)
    """
    from .config.const import PHASE_LABELS

    if bands is None:
        bands = list(BRAIN_BANDS.keys())
    if phases is None:
        phases = list(PHASE_LABELS)

    results = {band: {} for band in bands}

    for band in bands:
        for phase in phases:
            try:
                result = compute_corr_matrix(patient, phase, band, **kwargs)
                results[band][phase] = result
            except Exception as e:
                print(f"WARNING: Failed to compute {patient} {phase} {band}: {e}")
                results[band][phase] = None

    return results
