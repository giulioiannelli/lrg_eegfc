"""Network-level processing for correlation matrices."""

from __future__ import annotations

from typing import Optional, Tuple

import networkx as nx
import numpy as np
import warnings
from numpy.typing import NDArray
from scipy.spatial.distance import squareform

from lrgsglib.utils import get_giant_component_leftoff
from lrgsglib.utils.lrg import (
    compute_laplacian_properties,
    compute_normalized_linkage,
    compute_optimal_threshold,
)

from ...config.const import BRAIN_BANDS
from .bands import build_corrmat_single_band


def process_network_for_phase(
    data_pat_phase_ts: NDArray,
    fs: float,
    band_name: str,
    correlation_protocol: dict,
    all_labels,
    jump_index_to_use: int = 0,
    scaling_factor: float = 0.98,
    linkage_method: str = "ward",
    filter_order: int = 4,
) -> Tuple[
    Optional[nx.Graph],
    Optional[dict],
    Optional[NDArray],
    Optional[float],
    Optional[NDArray],
    Optional[NDArray],
]:
    """Run the full correlation-network pipeline for a single phase/band."""

    corr_mat = build_corrmat_single_band(
        data_pat_phase_ts,
        fs,
        BRAIN_BANDS[band_name],
        corr_network_params=correlation_protocol,
        jump_index=jump_index_to_use,
        filter_order=filter_order,
        band_name=band_name,
    )

    if isinstance(corr_mat, tuple):
        corr_mat, _ = corr_mat

    if corr_mat is None:
        return None, None, None, None, None, None

    G = nx.from_numpy_array(corr_mat)
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G.nodes()) == 0:
        return None, None, None, None, None, None

    G_giant, _ = get_giant_component_leftoff(G)
    if len(G_giant.nodes()) == 0:
        return None, None, None, None, None, None

    labeldict = {k: v for k, v in all_labels.to_dict().items() if k in G_giant.nodes()}

    spect, L, rho, Trho, tau = compute_laplacian_properties(G_giant, tau=None)

    if not np.all(np.isfinite(Trho)):
        warnings.warn(
            f"Band '{band_name}': Non-finite values detected in resistance distance matrix.",
            RuntimeWarning,
        )
        return None, None, None, None, None, None

    dists = squareform(Trho)
    if not np.all(np.isfinite(dists)):
        warnings.warn(
            f"Band '{band_name}': Non-finite values detected in condensed distance matrix.",
            RuntimeWarning,
        )
        return None, None, None, None, None, None

    lnkgM, _, _ = compute_normalized_linkage(dists, G_giant, method=linkage_method)
    clTh, *_ = compute_optimal_threshold(lnkgM, scaling_factor=scaling_factor)

    return G_giant, labeldict, lnkgM, clTh, corr_mat, dists
