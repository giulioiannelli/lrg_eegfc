"""Threshold selection utilities for correlation networks."""

from __future__ import annotations

from typing import Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from lrgsglib.utils import compute_threshold_stats_fast


def find_exact_detachment_threshold(corr_mat: NDArray) -> float:
    """Binary-search the correlation level where the first node detaches."""

    n = corr_mat.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    edge_weights = np.abs(corr_mat[triu_indices])
    sorted_weights = np.sort(edge_weights)

    left, right = 0, len(sorted_weights) - 1
    while left < right:
        mid = (left + right) // 2
        threshold = sorted_weights[mid]

        adj_matrix = np.abs(corr_mat) >= threshold
        np.fill_diagonal(adj_matrix, False)

        G = nx.from_numpy_array(adj_matrix.astype(int))
        if nx.number_connected_components(G) > 1:
            right = mid
        else:
            left = mid + 1

    return sorted_weights[left] if left < len(sorted_weights) else sorted_weights[-1]


def find_threshold_jumps(
    G_filt: nx.Graph,
    return_stats: bool = False,
) -> Union[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray, NDArray, NDArray]]:
    """Compute percolation jumps for a weighted graph."""

    Th, Einf, Pinf = compute_threshold_stats_fast(G_filt)
    jumps = np.where(np.diff(Pinf) != 0)[0]

    if return_stats:
        return Th, jumps, Einf, Pinf
    return Th, jumps
