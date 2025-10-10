"""High level helpers for building and cleaning correlation networks."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, MutableMapping, Optional, Tuple
import warnings

import networkx as nx
import numpy as np

from lrgsglib.utils import bandpass_sos, compute_threshold_stats_fast

from .constants import BRAIN_BANDS

__all__ = [
    "apply_threshold_filter",
    "build_correlation_network",
    "build_corr_network",
    "find_exact_detachment_threshold",
    "find_threshold_jumps",
    "build_band_correlation_matrices",
    "clean_correlation_matrix",
]


def apply_threshold_filter(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Return a copy of ``matrix`` where values below ``threshold`` are zeroed."""

    filtered = matrix.copy()
    filtered[filtered < threshold] = 0
    return filtered


def clean_correlation_matrix(X: np.ndarray, rowvar: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Denoise a correlation matrix using the Marchenko–Pastur spectrum."""

    if rowvar:
        n_observations, n_variables = X.shape[1], X.shape[0]
    else:
        n_observations, n_variables = X.shape
    C = np.corrcoef(X, rowvar=rowvar)
    eigvals, eigvecs = np.linalg.eigh(C)

    Q = n_observations / n_variables
    lambda_min = (1 - np.sqrt(1 / Q)) ** 2
    lambda_max = (1 + np.sqrt(1 / Q)) ** 2

    signal_mask = eigvals > lambda_max
    V = eigvecs[:, signal_mask]
    L = np.diag(eigvals[signal_mask])
    C_clean = V @ L @ V.T
    np.fill_diagonal(C_clean, 1.0)
    return C_clean, eigvals, eigvecs, lambda_min, lambda_max, signal_mask


def build_correlation_network(
    timeseries: np.ndarray,
    *,
    filter_type: Optional[str] = None,
    threshold: Optional[float] = None,
    zero_diagonal: bool = True,
    spectral_cleaning: bool = False,
) -> np.ndarray:
    """Compute a processed correlation matrix from ``timeseries``."""

    C = np.corrcoef(timeseries)
    if filter_type == "abs":
        C = np.abs(C)
    if threshold is not None:
        C = apply_threshold_filter(C, threshold)
    if spectral_cleaning:
        C, *_ = clean_correlation_matrix(C.T, rowvar=False)
    if zero_diagonal:
        np.fill_diagonal(C, 0)
    return C


# Backwards compatible alias used by legacy notebooks
build_corr_network = build_correlation_network


def find_threshold_jumps(
    graph: nx.Graph,
    *,
    return_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute percolation jumps for ``graph`` using ``compute_threshold_stats_fast``."""

    Th, Einf, Pinf = compute_threshold_stats_fast(graph)
    Pinf_diff = np.diff(Pinf)
    jumps = np.where(Pinf_diff != 0)[0]

    if return_stats:
        return Th, jumps, Einf, Pinf
    return Th, jumps, None, None


def find_exact_detachment_threshold(corr_mat: np.ndarray) -> float:
    """Return the smallest threshold where the network splits into multiple components."""

    n = corr_mat.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    sorted_weights = np.sort(np.abs(corr_mat[triu_indices]))

    if sorted_weights.size == 0:
        return 0.0

    left, right = 0, sorted_weights.size - 1
    while left < right:
        mid = (left + right) // 2
        threshold = sorted_weights[mid]

        adjacency = np.abs(corr_mat) >= threshold
        np.fill_diagonal(adjacency, False)

        graph = nx.from_numpy_array(adjacency.astype(float))
        components = nx.number_connected_components(graph)

        if components == 1:
            left = mid + 1
        else:
            right = mid
    return float(sorted_weights[right])


def _select_threshold_from_jumps(
    corr_mat: np.ndarray,
    *,
    jump_index: int,
    band_name: str,
) -> Dict[str, object]:
    graph = nx.from_numpy_array(corr_mat)
    Th, jumps, *_ = find_threshold_jumps(graph)

    if jumps.size == 0:
        warnings.warn(
            f"Band '{band_name}': no percolation jumps were detected. Using minimum threshold.",
            UserWarning,
        )
        chosen_jump = 0
        effective_index = 0
        threshold = Th[0] if Th.size else 0.0
    elif jump_index >= jumps.size:
        warnings.warn(
            f"Band '{band_name}': jump_index={jump_index} exceeds {jumps.size}. Using last available jump.",
            UserWarning,
        )
        chosen_jump = int(jumps[-1])
        effective_index = jumps.size - 1
        threshold = Th[chosen_jump]
    else:
        chosen_jump = int(jumps[jump_index])
        effective_index = jump_index
        threshold = Th[chosen_jump]

    return {
        "jumps": jumps,
        "chosen_jump": chosen_jump,
        "chosen_threshold": float(threshold),
        "jump_index": effective_index,
        "expected_components": effective_index + 1,
    }


def build_band_correlation_matrices(
    data_ts: np.ndarray,
    fs: float,
    *,
    bandpass_func: Callable[[np.ndarray, float, float, float, int], np.ndarray] = bandpass_sos,
    brain_bands: Mapping[str, Tuple[float, float]] = BRAIN_BANDS,
    return_jump_info: bool = False,
    apply_threshold_filtering: bool = True,
    corr_network_params: Optional[MutableMapping[str, object]] = None,
    jump_index: int = 0,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Dict[str, object]]]]:
    """Compute correlation matrices for each frequency band in ``brain_bands``."""

    params = dict(corr_network_params or {})
    corr_mats: Dict[str, np.ndarray] = {}
    jump_results: Dict[str, Dict[str, object]] = {}

    for band_name, (low, high) in brain_bands.items():
        filtered = bandpass_func(data_ts, low, high, fs, 1)
        corr_initial = build_correlation_network(filtered, **params)

        if apply_threshold_filtering:
            threshold_info = _select_threshold_from_jumps(
                corr_initial,
                jump_index=jump_index,
                band_name=band_name,
            )
            jump_results[band_name] = threshold_info.copy()

            final_params = dict(params)
            final_params["threshold"] = threshold_info["chosen_threshold"]
            corr = build_correlation_network(filtered, **final_params)

            graph = nx.from_numpy_array(corr)
            graph.remove_nodes_from(list(nx.isolates(graph)))
            if graph.number_of_nodes() > 0:
                n_components = nx.number_connected_components(graph)
                expected = int(threshold_info["expected_components"])
                jump_results[band_name]["actual_components"] = n_components
                jump_results[band_name]["validation_passed"] = n_components == expected
                if n_components != expected:
                    warnings.warn(
                        f"Band '{band_name}': expected {expected} components but found {n_components}.",
                        UserWarning,
                    )
            else:
                warnings.warn(
                    f"Band '{band_name}': threshold {threshold_info['chosen_threshold']:.4f} removes all edges.",
                    UserWarning,
                )
                jump_results[band_name]["actual_components"] = 0
                jump_results[band_name]["validation_passed"] = False
        else:
            corr = corr_initial
            if return_jump_info:
                jump_results[band_name] = {
                    "jumps": None,
                    "chosen_jump": None,
                    "chosen_threshold": params.get("threshold", 0.0),
                    "jump_index": None,
                    "expected_components": None,
                    "actual_components": None,
                    "validation_passed": None,
                }

        corr_mats[band_name] = corr

    return corr_mats, (jump_results if return_jump_info else None)
