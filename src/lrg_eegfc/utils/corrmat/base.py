"""Core utilities for correlation-matrix construction and cleaning."""

from __future__ import annotations

from typing import Optional

import warnings

import numpy as np
from numpy.typing import NDArray


def apply_threshold_filter(matrix: NDArray, threshold: float) -> NDArray:
    """Return a copy of ``matrix`` with entries below ``threshold`` zeroed."""

    filtered_matrix = matrix.copy()
    filtered_matrix[filtered_matrix < threshold] = 0
    return filtered_matrix


def build_corr_network(
    timeseries: NDArray,
    filter_type: Optional[str] = None,
    threshold: Optional[float] = None,
    zero_diagonal: bool = True,
    spectral_cleaning: bool = False,
) -> NDArray:
    """Compute a processed correlation network from multivariate time series."""

    C = np.corrcoef(timeseries)

    if not np.all(np.isfinite(C)):
        warnings.warn(
            "Correlation matrix contains non-finite values. Replacing with zeros.",
            RuntimeWarning,
        )
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    if spectral_cleaning:
        C_clean, *_ = clean_correlation_matrix(C.T, rowvar=False)
        C = C_clean

    if filter_type == "abs":
        C = np.abs(C)

    if threshold is not None:
        C = apply_threshold_filter(C, threshold)

    if zero_diagonal:
        np.fill_diagonal(C, 0)

    return C


def clean_correlation_matrix(X: NDArray, rowvar: bool = True):
    """Denoise a correlation matrix via Marchenko–Pastur spectral filtering."""

    N, T = X.shape
    C = np.corrcoef(X, rowvar=rowvar)
    eigvals, eigvecs = np.linalg.eigh(C)

    Q = T / N
    lambda_min = (1 - np.sqrt(1 / Q)) ** 2
    lambda_max = (1 + np.sqrt(1 / Q)) ** 2

    signal_mask = eigvals > lambda_max
    V = eigvecs[:, signal_mask]
    L = np.diag(eigvals[signal_mask])

    C_clean = V @ L @ V.T
    np.fill_diagonal(C_clean, 1.0)

    return C_clean, eigvals, eigvecs, lambda_min, lambda_max, signal_mask
