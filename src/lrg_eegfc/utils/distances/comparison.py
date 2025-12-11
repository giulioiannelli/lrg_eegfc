"""Distance measure comparison and cross-patient consistency analysis.

This module provides utilities for comparing different distance measures across
patients and experimental phases.

Based on code from DSTCMP_all_distance_measures.ipynb.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_phase_distance_matrix(
    phase_labels: List[str],
    compute_fn: Callable[[str, str], float],
    diag_value: float = 0.0,
    patient: Optional[str] = None,
    band: Optional[str] = None,
) -> np.ndarray:
    """Compute pairwise distance matrix across experimental phases.

    This function creates an N×N matrix where N is the number of phases,
    and entry (i,j) contains the distance between phase i and phase j
    according to the provided distance function.

    Parameters
    ----------
    phase_labels : list of str
        List of phase names (e.g., ['rsPre', 'taskLearn', 'taskTest', 'rsPost'])
    compute_fn : callable
        Function that takes two phase names (str, str) and returns a distance (float)
        Example: lambda pi, pj: ultrametric_matrix_distance(U[pi], U[pj])
    diag_value : float, optional
        Value to use on the diagonal (self-distance). Default 0.0
    patient : str, optional
        Patient ID for logging/debugging. Default None
    band : str, optional
        Frequency band for logging/debugging. Default None

    Returns
    -------
    distance_matrix : np.ndarray
        (N, N) symmetric matrix of phase-phase distances

    Examples
    --------
    >>> from lrgsglib.utils.basic.linalg import ultrametric_matrix_distance
    >>> # Assume U is dict mapping phase -> ultrametric matrix
    >>> phases = ['rsPre', 'taskLearn', 'taskTest', 'rsPost']
    >>> def dist_fn(pi, pj):
    ...     return ultrametric_matrix_distance(U[pi], U[pj], metric='euclidean')
    >>> dm = compute_phase_distance_matrix(phases, dist_fn)
    >>> print(f"Distance rsPre->taskLearn: {dm[0, 1]:.3f}")

    Notes
    -----
    The function handles exceptions gracefully - if compute_fn raises an exception
    or returns None/NaN, the corresponding matrix entry is set to NaN.
    """
    n_phases = len(phase_labels)
    dm = np.full((n_phases, n_phases), np.nan, dtype=float)

    for i, pi in enumerate(phase_labels):
        for j, pj in enumerate(phase_labels):
            if i == j:
                dm[i, j] = diag_value
                continue

            try:
                value = compute_fn(pi, pj)
            except Exception as exc:
                # Log warning but continue
                context = f"{patient} {band}" if patient and band else ""
                print(f"[WARN] {context} {pi}->{pj}: {exc}")
                value = np.nan

            if value is None:
                dm[i, j] = np.nan
            else:
                val = float(value)
                dm[i, j] = val if np.isfinite(val) else np.nan

    return dm


def compute_cross_patient_consistency(
    M1: np.ndarray,
    M2: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute consistency metrics between two patient distance matrices.

    This function quantifies how similar the distance patterns are between
    two patients, using correlation and normalized difference metrics.

    Parameters
    ----------
    M1 : np.ndarray
        Distance matrix for patient 1 (N, N)
    M2 : np.ndarray
        Distance matrix for patient 2 (N, N)

    Returns
    -------
    corr_pearson : float
        Pearson correlation of upper triangular elements
    corr_spearman : float
        Spearman rank correlation of upper triangular elements
    norm_diff : float
        Normalized Frobenius distance: ||M1-M2|| / mean(||M1||, ||M2||)

    Examples
    --------
    >>> # Compare alpha band distance matrices for two patients
    >>> corr_p, corr_s, norm_diff = compute_cross_patient_consistency(
    ...     M_pat02_alpha, M_pat03_alpha
    ... )
    >>> print(f"Pearson correlation: {corr_p:.3f}")
    >>> print(f"Spearman correlation: {corr_s:.3f}")
    >>> print(f"Normalized difference: {norm_diff:.3f}")

    Notes
    -----
    Higher correlations (closer to 1.0) indicate more similar patterns.
    Lower normalized difference (closer to 0.0) indicates more comparable magnitudes.
    Only finite values are included in the analysis - NaN pairs are excluded.
    """
    # Get upper triangular elements (excluding diagonal)
    mask = np.triu(np.ones_like(M1, dtype=bool), k=1)
    v1 = M1[mask]
    v2 = M2[mask]

    # Remove NaN pairs
    valid = np.isfinite(v1) & np.isfinite(v2)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan

    v1_clean = v1[valid]
    v2_clean = v2[valid]

    # Pearson correlation
    try:
        corr_p, _ = pearsonr(v1_clean, v2_clean)
    except Exception:
        corr_p = np.nan

    # Spearman correlation
    try:
        corr_s, _ = spearmanr(v1_clean, v2_clean)
    except Exception:
        corr_s = np.nan

    # Normalized Frobenius difference
    diff = np.linalg.norm(v1_clean - v2_clean)
    avg_norm = (np.linalg.norm(v1_clean) + np.linalg.norm(v2_clean)) / 2
    norm_diff = diff / avg_norm if avg_norm > 0 else np.nan

    return corr_p, corr_s, norm_diff


def rank_distance_measures(
    measure_results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    patients: List[str],
    bands: List[str],
    consistency_fn: Optional[Callable] = None,
) -> Dict[str, Dict]:
    """Rank distance measures by cross-patient consistency.

    This function evaluates multiple distance measures and ranks them based on
    how consistently they perform across different patients.

    Parameters
    ----------
    measure_results : dict
        Nested dict: {measure_key: {band: {patient: distance_matrix}}}
    patients : list of str
        List of patient IDs (must have at least 2)
    bands : list of str
        List of frequency bands
    consistency_fn : callable, optional
        Function to compute consistency between two matrices.
        Default: compute_cross_patient_consistency

    Returns
    -------
    rankings : dict
        Dictionary with:
        - 'scores': dict mapping measure_key -> combined consistency score
        - 'details': dict mapping measure_key -> detailed metrics per band
        - 'best_measure': str, key of the best-ranked measure

    Examples
    --------
    >>> # Assume measure_results contains multiple distance measures
    >>> rankings = rank_distance_measures(
    ...     measure_results,
    ...     patients=['Pat_02', 'Pat_03'],
    ...     bands=['alpha', 'beta', 'gamma']
    ... )
    >>> print(f"Best measure: {rankings['best_measure']}")
    >>> for measure, score in sorted(rankings['scores'].items(),
    ...                              key=lambda x: x[1], reverse=True):
    ...     print(f"  {measure}: {score:.4f}")

    Notes
    -----
    Combined score = (avg_pearson + avg_spearman) / 2 - 0.5 * avg_normdiff
    Higher scores indicate better cross-patient consistency.
    """
    if consistency_fn is None:
        consistency_fn = compute_cross_patient_consistency

    if len(patients) < 2:
        raise ValueError("Need at least 2 patients for cross-patient comparison")

    measure_scores = {}
    measure_details = {}

    for measure_key, band_dict in measure_results.items():
        pearson_vals = []
        spearman_vals = []
        normdiff_vals = []

        details_per_band = {}

        for band in bands:
            # Get matrices for first two patients
            M1 = band_dict[band].get(patients[0])
            M2 = band_dict[band].get(patients[1])

            if M1 is None or M2 is None:
                continue

            corr_p, corr_s, norm_d = consistency_fn(M1, M2)

            if np.isfinite(corr_p):
                pearson_vals.append(corr_p)
            if np.isfinite(corr_s):
                spearman_vals.append(corr_s)
            if np.isfinite(norm_d):
                normdiff_vals.append(norm_d)

            details_per_band[band] = {
                "pearson": corr_p,
                "spearman": corr_s,
                "norm_diff": norm_d,
            }

        # Compute averages
        avg_pearson = np.mean(pearson_vals) if pearson_vals else np.nan
        avg_spearman = np.mean(spearman_vals) if spearman_vals else np.nan
        avg_normdiff = np.mean(normdiff_vals) if normdiff_vals else np.nan

        # Combined score: higher correlation + lower difference = better
        if np.isfinite([avg_pearson, avg_spearman, avg_normdiff]).all():
            combined_score = (avg_pearson + avg_spearman) / 2 - 0.5 * avg_normdiff
        else:
            combined_score = np.nan

        measure_scores[measure_key] = combined_score
        measure_details[measure_key] = {
            "avg_pearson": avg_pearson,
            "avg_spearman": avg_spearman,
            "avg_normdiff": avg_normdiff,
            "combined_score": combined_score,
            "bands": details_per_band,
        }

    # Find best measure
    valid_scores = {k: v for k, v in measure_scores.items() if np.isfinite(v)}
    best_measure = max(valid_scores, key=valid_scores.get) if valid_scores else None

    return {
        "scores": measure_scores,
        "details": measure_details,
        "best_measure": best_measure,
    }


def compute_measure_summary_stats(
    measure_results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    measure_key: str,
) -> Dict[str, Dict]:
    """Compute summary statistics for a distance measure across all patients/bands.

    Parameters
    ----------
    measure_results : dict
        Nested dict: {measure_key: {band: {patient: distance_matrix}}}
    measure_key : str
        Key of the measure to summarize

    Returns
    -------
    stats : dict
        Dictionary with summary statistics per band:
        - mean: Mean distance across all phase pairs
        - std: Standard deviation
        - min: Minimum distance
        - max: Maximum distance
        - sparsity: Fraction of NaN values

    Examples
    --------
    >>> stats = compute_measure_summary_stats(
    ...     measure_results, 'ultrametric_matrix_distance'
    ... )
    >>> for band, band_stats in stats.items():
    ...     print(f"{band}: mean={band_stats['mean']:.3f}")
    """
    if measure_key not in measure_results:
        raise KeyError(f"Measure '{measure_key}' not found in results")

    band_dict = measure_results[measure_key]
    stats = {}

    for band, patient_dict in band_dict.items():
        all_values = []

        for patient, matrix in patient_dict.items():
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
            values = matrix[mask]
            all_values.extend(values)

        all_values = np.array(all_values)
        finite_values = all_values[np.isfinite(all_values)]

        stats[band] = {
            "mean": np.mean(finite_values) if len(finite_values) > 0 else np.nan,
            "std": np.std(finite_values) if len(finite_values) > 0 else np.nan,
            "min": np.min(finite_values) if len(finite_values) > 0 else np.nan,
            "max": np.max(finite_values) if len(finite_values) > 0 else np.nan,
            "sparsity": np.mean(~np.isfinite(all_values)),
            "n_values": len(all_values),
            "n_finite": len(finite_values),
        }

    return stats
