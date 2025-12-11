"""Comparison metrics for functional connectivity networks.

This module provides functions to compare FC networks using ultrametric distances,
correlation matrices, and other network properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lrgsglib.core import (
    ultrametric_matrix_distance,
    ultrametric_multiscale_distance,
    ultrametric_quantile_rmse,
    ultrametric_rank_correlation,
    ultrametric_scale_profile_distance,
    ultrametric_scaled_distance,
    compare_ultrametric_trees,
)

__all__ = [
    "UltrametricComparison",
    "compare_ultrametric_matrices",
    "compare_fc_methods",
    "compare_phases",
    "aggregate_comparisons",
]


@dataclass
class UltrametricComparison:
    """Result from comparing two ultrametric distance matrices.

    Attributes
    ----------
    matrix_distance : float
        Element-wise Frobenius distance between matrices
    multiscale_distance : float
        Multiscale comparison across hierarchical levels
    quantile_rmse : float
        RMSE of quantile distributions
    rank_correlation : float
        Spearman rank correlation of distances
    scale_profile_distance : float
        Distance between scale profiles
    scaled_distance : float
        Normalized distance metric
    tree_similarity : float
        Tree topology similarity (0=different, 1=identical)
    label_1 : str
        Label for first matrix
    label_2 : str
        Label for second matrix
    """

    matrix_distance: float
    multiscale_distance: float
    quantile_rmse: float
    rank_correlation: float
    scale_profile_distance: float
    scaled_distance: float
    tree_similarity: float
    label_1: str
    label_2: str


def compare_ultrametric_matrices(
    ultrametric_1: np.ndarray,
    ultrametric_2: np.ndarray,
    linkage_1: Optional[np.ndarray] = None,
    linkage_2: Optional[np.ndarray] = None,
    label_1: str = "Matrix 1",
    label_2: str = "Matrix 2",
) -> UltrametricComparison:
    """Compare two ultrametric distance matrices using multiple metrics.

    Parameters
    ----------
    ultrametric_1 : np.ndarray
        First ultrametric distance matrix (condensed 1D or square 2D form)
    ultrametric_2 : np.ndarray
        Second ultrametric distance matrix (condensed 1D or square 2D form)
    linkage_1 : np.ndarray, optional
        Linkage matrix for first tree (for tree comparison)
    linkage_2 : np.ndarray, optional
        Linkage matrix for second tree (for tree comparison)
    label_1 : str, optional
        Label for first matrix (default: "Matrix 1")
    label_2 : str, optional
        Label for second matrix (default: "Matrix 2")

    Returns
    -------
    UltrametricComparison
        Comparison result with multiple distance metrics

    Examples
    --------
    >>> from lrg_eegfc import compute_lrg_analysis, compute_corr_matrix, compute_msc_matrix
    >>> # Get FC matrices
    >>> corr = compute_corr_matrix("Pat_02", "rsPre", "beta")
    >>> msc = compute_msc_matrix("Pat_02", "rsPre", "beta")
    >>> # Compute LRG
    >>> lrg_corr = compute_lrg_analysis(corr.adjacency_matrix, "Pat_02", "rsPre", "beta", "corr")
    >>> lrg_msc = compute_lrg_analysis(msc.adjacency_matrix, "Pat_02", "rsPre", "beta", "msc")
    >>> # Compare
    >>> comparison = compare_ultrametric_matrices(
    ...     lrg_corr.ultrametric_matrix,
    ...     lrg_msc.ultrametric_matrix,
    ...     lrg_corr.linkage_matrix,
    ...     lrg_msc.linkage_matrix,
    ...     label_1="Correlation",
    ...     label_2="MSC"
    ... )
    >>> print(f"Matrix distance: {comparison.matrix_distance:.4f}")
    >>> print(f"Tree similarity: {comparison.tree_similarity:.4f}")
    """
    from scipy.spatial.distance import squareform

    # Convert condensed to square form if needed
    if ultrametric_1.ndim == 1:
        ultrametric_1 = squareform(ultrametric_1)
    if ultrametric_2.ndim == 1:
        ultrametric_2 = squareform(ultrametric_2)

    # Validate inputs
    if ultrametric_1.shape != ultrametric_2.shape:
        raise ValueError(
            f"Ultrametric matrices must have same shape: "
            f"{ultrametric_1.shape} vs {ultrametric_2.shape}"
        )

    # Compute various distance metrics
    matrix_dist = ultrametric_matrix_distance(ultrametric_1, ultrametric_2)

    # Some metrics may fail with certain matrix configurations - handle gracefully
    try:
        multiscale_dist = ultrametric_multiscale_distance(ultrametric_1, ultrametric_2)
    except Exception:
        multiscale_dist = np.nan

    quantile_rmse = ultrametric_quantile_rmse(ultrametric_1, ultrametric_2)
    rank_corr = ultrametric_rank_correlation(ultrametric_1, ultrametric_2)

    try:
        scale_profile_dist = ultrametric_scale_profile_distance(ultrametric_1, ultrametric_2)
    except Exception:
        scale_profile_dist = np.nan

    scaled_dist = ultrametric_scaled_distance(ultrametric_1, ultrametric_2)

    # Tree similarity (if linkage matrices provided)
    if linkage_1 is not None and linkage_2 is not None:
        try:
            tree_sim = compare_ultrametric_trees(linkage_1, linkage_2)
        except Exception:
            tree_sim = np.nan
    else:
        tree_sim = np.nan

    return UltrametricComparison(
        matrix_distance=float(matrix_dist),
        multiscale_distance=float(multiscale_dist),
        quantile_rmse=float(quantile_rmse),
        rank_correlation=float(rank_corr),
        scale_profile_distance=float(scale_profile_dist),
        scaled_distance=float(scaled_dist),
        tree_similarity=float(tree_sim),
        label_1=label_1,
        label_2=label_2,
    )


def compare_fc_methods(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/lrg_cache"),
) -> Optional[UltrametricComparison]:
    """Compare MSC vs correlation FC for a specific patient/phase/band.

    Parameters
    ----------
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    cache_root : Path, optional
        Root directory for LRG cache files

    Returns
    -------
    UltrametricComparison or None
        Comparison result, or None if either method missing

    Examples
    --------
    >>> comparison = compare_fc_methods("Pat_02", "rsPre", "beta")
    >>> if comparison:
    ...     print(f"MSC vs Correlation distance: {comparison.matrix_distance:.4f}")
    """
    from .workflow_lrg import load_lrg_result

    # Load LRG results for both methods
    lrg_corr = load_lrg_result(patient, phase, band, "corr", cache_root)
    lrg_msc = load_lrg_result(patient, phase, band, "msc", cache_root)

    if lrg_corr is None or lrg_msc is None:
        return None

    return compare_ultrametric_matrices(
        lrg_corr.ultrametric_matrix,
        lrg_msc.ultrametric_matrix,
        lrg_corr.linkage_matrix,
        lrg_msc.linkage_matrix,
        label_1=f"{patient}_{phase}_{band}_corr",
        label_2=f"{patient}_{phase}_{band}_msc",
    )


def compare_phases(
    patient: str,
    phase_1: str,
    phase_2: str,
    band: str,
    fc_method: str,
    cache_root: Path = Path("data/lrg_cache"),
) -> Optional[UltrametricComparison]:
    """Compare two phases for same patient/band/method.

    Useful for analyzing memory effects (e.g., rsPre vs rsPost).

    Parameters
    ----------
    patient : str
        Patient identifier
    phase_1 : str
        First recording phase (e.g., "rsPre")
    phase_2 : str
        Second recording phase (e.g., "rsPost")
    band : str
        Frequency band
    fc_method : str
        FC method: "msc" or "corr"
    cache_root : Path, optional
        Root directory for LRG cache files

    Returns
    -------
    UltrametricComparison or None
        Comparison result, or None if either phase missing

    Examples
    --------
    >>> # Compare pre vs post resting state
    >>> comparison = compare_phases("Pat_02", "rsPre", "rsPost", "beta", "corr")
    >>> if comparison:
    ...     print(f"Pre-Post distance: {comparison.matrix_distance:.4f}")
    """
    from .workflow_lrg import load_lrg_result

    # Load LRG results for both phases
    lrg_1 = load_lrg_result(patient, phase_1, band, fc_method, cache_root)
    lrg_2 = load_lrg_result(patient, phase_2, band, fc_method, cache_root)

    if lrg_1 is None or lrg_2 is None:
        return None

    return compare_ultrametric_matrices(
        lrg_1.ultrametric_matrix,
        lrg_2.ultrametric_matrix,
        lrg_1.linkage_matrix,
        lrg_2.linkage_matrix,
        label_1=f"{patient}_{phase_1}_{band}_{fc_method}",
        label_2=f"{patient}_{phase_2}_{band}_{fc_method}",
    )


def aggregate_comparisons(
    comparisons: List[UltrametricComparison],
) -> pd.DataFrame:
    """Aggregate multiple comparisons into a DataFrame for analysis.

    Parameters
    ----------
    comparisons : List[UltrametricComparison]
        List of comparison results

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per comparison, columns for each metric

    Examples
    --------
    >>> from lrg_eegfc.config.const import BRAIN_BANDS
    >>> # Compare MSC vs corr for all bands
    >>> comparisons = []
    >>> for band in BRAIN_BANDS:
    ...     comp = compare_fc_methods("Pat_02", "rsPre", band)
    ...     if comp:
    ...         comparisons.append(comp)
    >>> df = aggregate_comparisons(comparisons)
    >>> print(df[["label_1", "label_2", "matrix_distance", "rank_correlation"]])
    """
    data = []
    for comp in comparisons:
        data.append({
            "label_1": comp.label_1,
            "label_2": comp.label_2,
            "matrix_distance": comp.matrix_distance,
            "multiscale_distance": comp.multiscale_distance,
            "quantile_rmse": comp.quantile_rmse,
            "rank_correlation": comp.rank_correlation,
            "scale_profile_distance": comp.scale_profile_distance,
            "scaled_distance": comp.scaled_distance,
            "tree_similarity": comp.tree_similarity,
        })

    return pd.DataFrame(data)


def batch_compare_fc_methods(
    patients: List[str],
    phases: List[str],
    bands: List[str],
    cache_root: Path = Path("data/lrg_cache"),
    verbose: bool = False,
) -> pd.DataFrame:
    """Batch compare MSC vs correlation across patients/phases/bands.

    Parameters
    ----------
    patients : List[str]
        List of patient IDs
    phases : List[str]
        List of phases
    bands : List[str]
        List of frequency bands
    cache_root : Path, optional
        Root directory for LRG cache files
    verbose : bool, optional
        Print progress (default: False)

    Returns
    -------
    pd.DataFrame
        Comparison results with additional columns for patient/phase/band

    Examples
    --------
    >>> from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS
    >>> df = batch_compare_fc_methods(
    ...     ["Pat_02", "Pat_03"],
    ...     PHASE_LABELS,
    ...     list(BRAIN_BANDS.keys())
    ... )
    >>> # Analyze mean distance per band
    >>> print(df.groupby("band")["matrix_distance"].mean())
    """
    comparisons = []

    for patient in patients:
        for phase in phases:
            for band in bands:
                if verbose:
                    print(f"Comparing {patient} {phase} {band}...")

                comp = compare_fc_methods(patient, phase, band, cache_root)
                if comp is not None:
                    comparisons.append((patient, phase, band, comp))

    # Create DataFrame
    data = []
    for patient, phase, band, comp in comparisons:
        data.append({
            "patient": patient,
            "phase": phase,
            "band": band,
            "matrix_distance": comp.matrix_distance,
            "multiscale_distance": comp.multiscale_distance,
            "quantile_rmse": comp.quantile_rmse,
            "rank_correlation": comp.rank_correlation,
            "scale_profile_distance": comp.scale_profile_distance,
            "scaled_distance": comp.scaled_distance,
            "tree_similarity": comp.tree_similarity,
        })

    return pd.DataFrame(data)


def batch_compare_phases(
    patients: List[str],
    phase_pairs: List[Tuple[str, str]],
    bands: List[str],
    fc_method: str,
    cache_root: Path = Path("data/lrg_cache"),
    verbose: bool = False,
) -> pd.DataFrame:
    """Batch compare phase pairs (e.g., pre vs post) for memory effects.

    Parameters
    ----------
    patients : List[str]
        List of patient IDs
    phase_pairs : List[Tuple[str, str]]
        List of (phase_1, phase_2) pairs to compare
    bands : List[str]
        List of frequency bands
    fc_method : str
        FC method: "msc" or "corr"
    cache_root : Path, optional
        Root directory for LRG cache files
    verbose : bool, optional
        Print progress (default: False)

    Returns
    -------
    pd.DataFrame
        Comparison results

    Examples
    --------
    >>> # Compare pre vs post for memory effects
    >>> phase_pairs = [("rsPre", "rsPost"), ("taskLearn", "taskTest")]
    >>> df = batch_compare_phases(
    ...     ["Pat_02", "Pat_03"],
    ...     phase_pairs,
    ...     ["beta", "alpha"],
    ...     "corr"
    ... )
    >>> # Analyze memory effects
    >>> print(df.groupby("phase_pair")["matrix_distance"].mean())
    """
    comparisons = []

    for patient in patients:
        for phase_1, phase_2 in phase_pairs:
            for band in bands:
                if verbose:
                    print(f"Comparing {patient} {phase_1} vs {phase_2} {band}...")

                comp = compare_phases(patient, phase_1, phase_2, band, fc_method, cache_root)
                if comp is not None:
                    comparisons.append((patient, phase_1, phase_2, band, comp))

    # Create DataFrame
    data = []
    for patient, phase_1, phase_2, band, comp in comparisons:
        data.append({
            "patient": patient,
            "phase_1": phase_1,
            "phase_2": phase_2,
            "phase_pair": f"{phase_1}_vs_{phase_2}",
            "band": band,
            "fc_method": fc_method,
            "matrix_distance": comp.matrix_distance,
            "multiscale_distance": comp.multiscale_distance,
            "quantile_rmse": comp.quantile_rmse,
            "rank_correlation": comp.rank_correlation,
            "scale_profile_distance": comp.scale_profile_distance,
            "scaled_distance": comp.scaled_distance,
            "tree_similarity": comp.tree_similarity,
        })

    return pd.DataFrame(data)
