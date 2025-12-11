"""LRG (Laplacian Renormalization Group) analysis workflow.

This module provides workflows for computing LRG-based ultrametric distances
from functional connectivity matrices (MSC or correlation-based).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np
from scipy.spatial.distance import squareform

from lrgsglib.core import (
    compute_laplacian_properties,
    compute_normalized_linkage,
    compute_optimal_threshold,
    entropy,
    extract_ultrametric_matrix,
    get_giant_component,
)

__all__ = ["LRGResult", "compute_lrg_analysis", "load_lrg_result", "get_lrg_cache_path"]


@dataclass
class LRGResult:
    """Result from LRG analysis.

    Attributes
    ----------
    ultrametric_matrix : np.ndarray
        Ultrametric distance matrix (condensed form, 1D vector)
    linkage_matrix : np.ndarray
        Hierarchical clustering linkage matrix
    entropy_tau : np.ndarray
        Tau values for entropy curve
    entropy_1_minus_S : np.ndarray
        1-S (normalized entropy) values
    entropy_C : np.ndarray
        C (spectral complexity) values
    optimal_threshold : float
        Optimal dendrogram cutting threshold
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    fc_method : str
        Functional connectivity method ("msc" or "corr")
    n_nodes : int
        Number of nodes in giant component
    """

    ultrametric_matrix: np.ndarray
    linkage_matrix: np.ndarray
    entropy_tau: np.ndarray
    entropy_1_minus_S: np.ndarray
    entropy_C: np.ndarray
    optimal_threshold: float
    patient: str
    phase: str
    band: str
    fc_method: str
    n_nodes: int


def get_lrg_cache_path(
    patient: str,
    phase: str,
    band: str,
    fc_method: str,
    cache_root: Path = Path("data/lrg_cache"),
) -> Path:
    """Get cache file path for LRG analysis results.

    Parameters
    ----------
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    fc_method : str
        FC method ("msc" or "corr")
    cache_root : Path, optional
        Root directory for cache files

    Returns
    -------
    Path
        Path to cache file
    """
    cache_dir = cache_root / patient
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"{band}_{phase}_lrg_{fc_method}.npz"


def load_lrg_result(
    patient: str,
    phase: str,
    band: str,
    fc_method: str,
    cache_root: Path = Path("data/lrg_cache"),
) -> Optional[LRGResult]:
    """Load cached LRG analysis result if it exists.

    Parameters
    ----------
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    fc_method : str
        FC method ("msc" or "corr")
    cache_root : Path, optional
        Root directory for cache files

    Returns
    -------
    LRGResult or None
        Cached LRG result, or None if not cached
    """
    cache_path = get_lrg_cache_path(patient, phase, band, fc_method, cache_root)

    if cache_path.exists():
        data = np.load(cache_path)
        return LRGResult(
            ultrametric_matrix=data["ultrametric_matrix"],
            linkage_matrix=data["linkage_matrix"],
            entropy_tau=data["entropy_tau"],
            entropy_1_minus_S=data["entropy_1_minus_S"],
            entropy_C=data["entropy_C"],
            optimal_threshold=float(data["optimal_threshold"]),
            patient=str(data["patient"]),
            phase=str(data["phase"]),
            band=str(data["band"]),
            fc_method=str(data["fc_method"]),
            n_nodes=int(data["n_nodes"]),
        )
    return None


def compute_lrg_analysis(
    adjacency_matrix: np.ndarray,
    patient: str,
    phase: str,
    band: str,
    fc_method: str,
    cache_root: Path = Path("data/lrg_cache"),
    *,
    use_cache: bool = True,
    overwrite_cache: bool = False,
    entropy_steps: int = 400,
    entropy_t1: float = -3.0,
    entropy_t2: float = 5.0,
    verbose: bool = False,
) -> LRGResult:
    """Compute LRG analysis (ultrametric distances, entropy) from FC matrix.

    This function performs Laplacian Renormalization Group analysis on a
    functional connectivity adjacency matrix. It computes:
    - Ultrametric distance matrix via hierarchical clustering
    - Entropy and spectral complexity curves
    - Optimal dendrogram cutting threshold

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix (N x N)
    patient : str
        Patient identifier (for caching)
    phase : str
        Recording phase (for caching)
    band : str
        Frequency band (for caching)
    fc_method : str
        FC method: "msc" or "corr"
    cache_root : Path, optional
        Root directory for cache files
    use_cache : bool, optional
        If True, load from cache if available (default: True)
    overwrite_cache : bool, optional
        If True, recompute even if cached (default: False)
    entropy_steps : int, optional
        Number of steps for entropy computation (default: 400)
    entropy_t1 : float, optional
        Start of tau range (log scale) for entropy (default: -3.0)
    entropy_t2 : float, optional
        End of tau range (log scale) for entropy (default: 5.0)
    verbose : bool, optional
        Print progress information (default: False)

    Returns
    -------
    LRGResult
        Result object containing ultrametric distances and entropy data

    Raises
    ------
    ValueError
        If fc_method not in ["msc", "corr"]
        If adjacency matrix is invalid

    Examples
    --------
    >>> from lrg_eegfc import compute_corr_matrix, compute_lrg_analysis
    >>> # Get correlation matrix
    >>> corr_result = compute_corr_matrix("Pat_02", "rsPre", "beta")
    >>> # Compute LRG analysis
    >>> lrg_result = compute_lrg_analysis(
    ...     corr_result.adjacency_matrix,
    ...     "Pat_02", "rsPre", "beta", "corr"
    ... )
    >>> print(lrg_result.ultrametric_matrix.shape)
    (6786,)  # Condensed form: 117 * 116 / 2 = 6786 unique distances
    >>> print(f"Optimal threshold: {lrg_result.optimal_threshold:.4f}")
    Optimal threshold: 0.2345
    """
    # Validate fc_method
    if fc_method not in ["msc", "corr"]:
        raise ValueError(f"fc_method must be 'msc' or 'corr', got '{fc_method}'")

    # Validate adjacency matrix
    if not isinstance(adjacency_matrix, np.ndarray):
        raise ValueError("adjacency_matrix must be a numpy array")
    if adjacency_matrix.ndim != 2:
        raise ValueError(f"adjacency_matrix must be 2D, got shape {adjacency_matrix.shape}")
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError(f"adjacency_matrix must be square, got shape {adjacency_matrix.shape}")

    # Check cache
    cache_path = get_lrg_cache_path(patient, phase, band, fc_method, cache_root)

    if use_cache and not overwrite_cache and cache_path.exists():
        if verbose:
            print(f"Loading cached LRG analysis: {cache_path}")
        return load_lrg_result(patient, phase, band, fc_method, cache_root)

    # Compute LRG analysis
    if verbose:
        print(f"Computing LRG analysis for {patient} {phase} {band} ({fc_method})...")

    # Create graph and extract giant component
    graph = nx.from_numpy_array(adjacency_matrix)
    giant = get_giant_component(graph)
    n_nodes = giant.number_of_nodes()

    if verbose:
        print(f"  Giant component: {n_nodes}/{graph.number_of_nodes()} nodes")

    # Compute entropy
    if verbose:
        print(f"  Computing entropy (steps={entropy_steps})...")
    sm1, spec, *_, tau = entropy(giant, steps=entropy_steps, t1=entropy_t1, t2=entropy_t2)

    # Compute Laplacian properties and ultrametric distances
    if verbose:
        print(f"  Computing Laplacian properties and ultrametric distances...")
    _, _, _, Trho, _ = compute_laplacian_properties(giant)

    # Convert to distance matrix (condensed form)
    dists = squareform(Trho)

    # Compute hierarchical clustering linkage
    linkage_matrix, labels, _ = compute_normalized_linkage(dists, giant)

    # Compute optimal threshold for dendrogram cutting
    threshold, *_ = compute_optimal_threshold(linkage_matrix)

    # Extract ultrametric matrix from linkage (returns square form)
    ultrametric_square = extract_ultrametric_matrix(linkage_matrix, n_nodes)

    # Convert to condensed form for storage (saves space and matches visualization expectations)
    ultrametric_matrix = squareform(ultrametric_square)

    if verbose:
        print(f"  Optimal threshold: {threshold:.4f}")
        print(f"  Ultrametric matrix shape: {ultrametric_matrix.shape} (condensed)")

    # Create result
    result = LRGResult(
        ultrametric_matrix=ultrametric_matrix,
        linkage_matrix=linkage_matrix,
        entropy_tau=tau,
        entropy_1_minus_S=sm1,
        entropy_C=spec,
        optimal_threshold=float(threshold),
        patient=patient,
        phase=phase,
        band=band,
        fc_method=fc_method,
        n_nodes=n_nodes,
    )

    # Cache result
    if verbose:
        print(f"  Saving to cache: {cache_path}")

    np.savez_compressed(
        cache_path,
        ultrametric_matrix=result.ultrametric_matrix,
        linkage_matrix=result.linkage_matrix,
        entropy_tau=result.entropy_tau,
        entropy_1_minus_S=result.entropy_1_minus_S,
        entropy_C=result.entropy_C,
        optimal_threshold=result.optimal_threshold,
        patient=result.patient,
        phase=result.phase,
        band=result.band,
        fc_method=result.fc_method,
        n_nodes=result.n_nodes,
    )

    return result


def compute_lrg_for_patient(
    patient: str,
    fc_method: str,
    bands: Optional[list] = None,
    phases: Optional[list] = None,
    **kwargs
) -> Dict[str, Dict[str, LRGResult]]:
    """Compute LRG analysis for all band/phase combinations for a patient.

    Parameters
    ----------
    patient : str
        Patient identifier
    fc_method : str
        FC method: "msc" or "corr"
    bands : list, optional
        List of band names (default: all BRAIN_BANDS)
    phases : list, optional
        List of phase names (default: all phases)
    **kwargs
        Additional arguments passed to compute_lrg_analysis()

    Returns
    -------
    Dict[str, Dict[str, LRGResult]]
        Nested dict indexed by [band][phase]

    Examples
    --------
    >>> # Compute LRG analysis for all correlation matrices
    >>> from lrg_eegfc import compute_corr_for_patient
    >>> corr_results = compute_corr_for_patient("Pat_02")
    >>> lrg_results = compute_lrg_for_patient("Pat_02", "corr")
    >>>
    >>> # Access specific result
    >>> beta_rsPre_lrg = lrg_results["beta"]["rsPre"]
    >>> print(beta_rsPre_lrg.optimal_threshold)
    """
    from .config.const import BRAIN_BANDS, PHASE_LABELS

    if bands is None:
        bands = list(BRAIN_BANDS.keys())
    if phases is None:
        phases = list(PHASE_LABELS)

    # Import the appropriate workflow
    if fc_method == "msc":
        from .workflow_msc import load_msc_matrix
        load_fc_matrix = load_msc_matrix
    elif fc_method == "corr":
        from .workflow_corr import load_corr_matrix
        load_fc_matrix = load_corr_matrix
    else:
        raise ValueError(f"fc_method must be 'msc' or 'corr', got '{fc_method}'")

    results = {band: {} for band in bands}

    for band in bands:
        for phase in phases:
            try:
                # Load FC matrix
                fc_matrix = load_fc_matrix(patient, phase, band)
                if fc_matrix is None:
                    print(f"WARNING: No cached FC matrix for {patient} {phase} {band} ({fc_method})")
                    results[band][phase] = None
                    continue

                # Compute LRG analysis
                result = compute_lrg_analysis(
                    fc_matrix, patient, phase, band, fc_method, **kwargs
                )
                results[band][phase] = result

            except Exception as e:
                print(f"WARNING: Failed to compute LRG for {patient} {phase} {band} ({fc_method}): {e}")
                results[band][phase] = None

    return results
