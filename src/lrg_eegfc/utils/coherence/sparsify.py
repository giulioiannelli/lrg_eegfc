"""Soft sparsification of coherence matrices using surrogate-based null models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


__all__ = [
    'soft_sparsify_surrogate',
]


def soft_sparsify_surrogate(
    W: NDArray,
    W_null: NDArray,
) -> NDArray:
    """
    Apply soft sparsification to a coherence matrix using surrogate null distribution.

    This function computes empirical p-values for each edge based on a surrogate null
    distribution and uses them to downweight non-significant edges while preserving
    the weight geometry.

    Parameters
    ----------
    W : NDArray
        Observed band-averaged MSC matrix of shape (N, N)
    W_null : NDArray
        Null distribution from surrogates of shape (n_surrogates, N, N)

    Returns
    -------
    A : NDArray
        Soft-sparsified adjacency matrix of shape (N, N)

    Notes
    -----
    The sparsification is computed as:

    1. For each edge (i, j), compute empirical p-value:
       p_ij = (#surrogates r with W_null[r, i, j] >= W[i, j]) / n_surrogates

    2. Compute soft weight:
       g_ij = 1 - p_ij

    3. Apply soft sparsification:
       A_ij = g_ij * W_ij

    This approach preserves the full weighted matrix but downweights non-significant
    edges without binarization or strict backbone extraction.
    """
    n_surrogates = W_null.shape[0]
    N = W.shape[0]

    # Initialize p-values and adjacency
    p_values = np.zeros_like(W)
    A = np.zeros_like(W)

    # Compute empirical p-values for each edge
    for i in range(N):
        for j in range(N):
            # Count surrogates with equal or greater coherence
            n_exceeding = np.sum(W_null[:, i, j] >= W[i, j])

            # Empirical p-value
            p_values[i, j] = n_exceeding / n_surrogates

    # Soft sparsification: g_ij = 1 - p_ij
    g = 1.0 - p_values

    # Apply soft weights
    A = g * W

    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(A, 0.0)

    return A
