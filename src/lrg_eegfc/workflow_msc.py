"""MSC-based functional connectivity workflow.

This module provides a simple workflow for computing and caching MSC-based
functional connectivity matrices. It mirrors the correlation workflow but
uses magnitude-squared coherence instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np

from .config.const import BRAIN_BANDS
from .utils.coherence import coherence_fc_pipeline
from .utils.datamanag.patient import load_timeseries
from .utils.datamanag.patient_robust import load_patient_dataset_robust

__all__ = ["MSCResult", "compute_msc_matrix", "load_msc_matrix", "get_msc_cache_path"]


@dataclass
class MSCResult:
    """Result from MSC computation.

    Attributes
    ----------
    adjacency_matrix : np.ndarray
        MSC adjacency matrix (N x N)
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
    mean_msc : float
        Mean MSC value (excluding diagonal)
    sparsify : str
        Sparsification method ("none" or "soft")
    n_surrogates : int
        Number of surrogates used (0 if sparsify="none")
    nperseg : int
        Window length for Welch's method
    """

    adjacency_matrix: np.ndarray
    graph: nx.Graph
    patient: str
    phase: str
    band: str
    n_channels: int
    mean_msc: float
    sparsify: str
    n_surrogates: int
    nperseg: int


def get_msc_cache_path(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/msc_cache"),
    sparsify: str = "none",
    n_surrogates: int = 0,
    nperseg: int = 1024,
) -> Path:
    """Get cache file path for MSC matrix.

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
    sparsify : str, optional
        Sparsification method ("none" or "soft")
    n_surrogates : int, optional
        Number of surrogates (0 if sparsify="none")
    nperseg : int, optional
        Window length for Welch's method

    Returns
    -------
    Path
        Path to cache file
    """
    cache_dir = cache_root / patient
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Include sparsification parameters in filename
    if sparsify == "none":
        suffix = f"sparsify-none_nperseg-{nperseg}"
    else:
        suffix = f"sparsify-{sparsify}_nsurr-{n_surrogates}_nperseg-{nperseg}"

    return cache_dir / f"{band}_{phase}_msc_{suffix}.npy"


def load_msc_matrix(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/msc_cache"),
    sparsify: str = "none",
    n_surrogates: int = 0,
    nperseg: int = 1024,
) -> Optional[np.ndarray]:
    """Load cached MSC matrix if it exists.

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
    sparsify : str, optional
        Sparsification method ("none" or "soft")
    n_surrogates : int, optional
        Number of surrogates
    nperseg : int, optional
        Window length for Welch's method

    Returns
    -------
    np.ndarray or None
        Cached MSC matrix, or None if not cached
    """
    cache_path = get_msc_cache_path(patient, phase, band, cache_root, sparsify, n_surrogates, nperseg)

    if cache_path.exists():
        return np.load(cache_path)
    return None


def compute_msc_matrix(
    patient: str,
    phase: str,
    band: str,
    dataset_root: Path = Path("data/stereoeeg_patients"),
    cache_root: Path = Path("data/msc_cache"),
    *,
    use_cache: bool = True,
    overwrite_cache: bool = False,
    sample_rate: float = 2048.0,
    sparsify: str = "none",
    n_surrogates: int = 0,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    batch_size: int = 64,
    filter_time: Optional[int] = None,
    verbose: bool = False,
) -> MSCResult:
    """Compute MSC-based functional connectivity matrix.

    This function computes magnitude-squared coherence (MSC) for a given
    patient, phase, and frequency band. Results are automatically cached
    for faster subsequent access.

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
    sparsify : str, optional
        Sparsification method: "none" (dense MSC) or "soft" (surrogate-based
        soft sparsification) (default: "none")
    n_surrogates : int, optional
        Number of circular shift surrogates for validation. Only used if
        sparsify="soft". Set to 0 for no validation (default: 0)
    nperseg : int, optional
        Window length for Welch's method (default: 1024)
    noverlap : int, optional
        Overlap between windows (default: nperseg // 2)
    batch_size : int, optional
        Number of Welch segments to process per batch (reduces Python/FFT overhead;
        default: 64)
    filter_time : int, optional
        Limit to first N samples (for testing)
    verbose : bool, optional
        Print progress information (default: False)

    Returns
    -------
    MSCResult
        Result object containing adjacency matrix and metadata

    Raises
    ------
    KeyError
        If band not defined in BRAIN_BANDS
    FileNotFoundError
        If patient/phase data doesn't exist

    Examples
    --------
    >>> # Dense MSC (no validation)
    >>> result = compute_msc_matrix("Pat_02", "rsPre", "beta")
    >>> print(result.adjacency_matrix.shape)
    (117, 117)
    >>> print(f"Mean MSC: {result.mean_msc:.4f}")
    Mean MSC: 0.0523
    >>> print(f"Sparsify: {result.sparsify}, Surrogates: {result.n_surrogates}")
    Sparsify: none, Surrogates: 0

    >>> # Validated MSC with surrogates
    >>> result = compute_msc_matrix("Pat_02", "rsPre", "beta",
    ...                             sparsify="soft", n_surrogates=200)
    >>> print(f"Sparsify: {result.sparsify}, Surrogates: {result.n_surrogates}")
    Sparsify: soft, Surrogates: 200
    """

    # Validate band
    if band not in BRAIN_BANDS:
        available = ", ".join(sorted(BRAIN_BANDS))
        raise KeyError(f"Band '{band}' not defined. Available: {available}")

    # Check cache
    cache_path = get_msc_cache_path(patient, phase, band, cache_root, sparsify, n_surrogates, nperseg)

    if use_cache and not overwrite_cache and cache_path.exists():
        if verbose:
            print(f"  loading cached msc: {cache_path.name}")
        adj_matrix = np.load(cache_path)

        # Reconstruct result
        graph = nx.from_numpy_array(adj_matrix)
        n_channels = adj_matrix.shape[0]

        # Compute mean MSC (upper triangle, excluding diagonal)
        triu_indices = np.triu_indices_from(adj_matrix, k=1)
        mean_msc = float(np.mean(adj_matrix[triu_indices]))

        return MSCResult(
            adjacency_matrix=adj_matrix,
            graph=graph,
            patient=patient,
            phase=phase,
            band=band,
            n_channels=n_channels,
            mean_msc=mean_msc,
            sparsify=sparsify,
            n_surrogates=n_surrogates,
            nperseg=nperseg,
        )

    # Compute MSC from timeseries
    if verbose:
        print(f"Computing MSC for {patient} {phase} {band}...")
        print(f"  Sparsify: {sparsify}, n_surrogates: {n_surrogates}")
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
        print(f"  Computing MSC with Welch's method...")

    # Compute MSC for single band
    bands_dict = {band: BRAIN_BANDS[band]}

    adj_matrices = coherence_fc_pipeline(
        data,
        fs=sample_rate,
        bands=bands_dict,
        sparsify=sparsify,
        n_surrogates=n_surrogates if sparsify == "soft" else 0,
        nperseg=nperseg,
        noverlap=noverlap,
        batch_size=batch_size,
        zero_diagonal=True,
        verbose=verbose,
    )

    adj_matrix = adj_matrices[band]

    # Cache result
    if verbose:
        print(f"  Saving to cache: {cache_path}")
    np.save(cache_path, adj_matrix)

    # Build graph
    graph = nx.from_numpy_array(adj_matrix)
    n_channels = adj_matrix.shape[0]

    # Compute statistics
    triu_indices = np.triu_indices_from(adj_matrix, k=1)
    mean_msc = float(np.mean(adj_matrix[triu_indices]))

    if verbose:
        print(f"  Done! Mean MSC: {mean_msc:.4f}")

    return MSCResult(
        adjacency_matrix=adj_matrix,
        graph=graph,
        patient=patient,
        phase=phase,
        band=band,
        n_channels=n_channels,
        mean_msc=mean_msc,
        sparsify=sparsify,
        n_surrogates=n_surrogates if sparsify == "soft" else 0,
        nperseg=nperseg,
    )


def compute_msc_for_patient(
    patient: str,
    bands: Optional[list] = None,
    phases: Optional[list] = None,
    **kwargs
) -> Dict[str, Dict[str, MSCResult]]:
    """Compute MSC matrices for all band/phase combinations for a patient.

    Parameters
    ----------
    patient : str
        Patient identifier
    bands : list, optional
        List of band names (default: all BRAIN_BANDS)
    phases : list, optional
        List of phase names (default: all phases)
    **kwargs
        Additional arguments passed to compute_msc_matrix()

    Returns
    -------
    Dict[str, Dict[str, MSCResult]]
        Nested dict indexed by [band][phase]

    Examples
    --------
    >>> # Compute all MSC matrices for Pat_02
    >>> results = compute_msc_for_patient("Pat_02")
    >>>
    >>> # Access specific result
    >>> beta_rsPre = results["beta"]["rsPre"]
    >>> print(beta_rsPre.mean_msc)
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
                result = compute_msc_matrix(patient, phase, band, **kwargs)
                results[band][phase] = result
            except Exception as e:
                print(f"WARNING: Failed to compute {patient} {phase} {band}: {e}")
                results[band][phase] = None

    return results
