"""High level workflows that combine the IO and correlation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np

from lrgsglib.utils.basic.signals import bandpass_sos

from .constants import BRAIN_BANDS
from .utils.corrmat import apply_threshold_filter, build_corr_network, find_threshold_jumps
from .utils.datamanag.patient import load_timeseries

__all__ = ["BandComputationResult", "compute_band_connectivity"]


@dataclass
class BandComputationResult:
    """Return value for :func:`compute_band_connectivity`."""

    correlation_matrix: np.ndarray
    graph: nx.Graph
    threshold: float
    jump_statistics: Dict[str, object]


def compute_band_connectivity(
    patient: str,
    phase: str,
    band: str,
    dataset_root: Path,
    *,
    filter_time: Optional[int] = None,
    sample_rate: float = 2048.0,
    filter_order: int = 4,
    jump_index: int = 0,
) -> BandComputationResult:
    """Compute a thresholded correlation network for a specific band."""

    if band not in BRAIN_BANDS:
        available = ", ".join(sorted(BRAIN_BANDS))
        raise KeyError(f"Band '{band}' is not defined. Available bands: {available}.")

    data = load_timeseries(patient, phase, dataset_root)
    if filter_time is not None and filter_time > 0:
        data = data[:, :filter_time]

    low, high = BRAIN_BANDS[band]
    filtered = bandpass_sos(data, low, high, sample_rate, filter_order)

    corr = build_corr_network(filtered, filter_type="abs", zero_diagonal=True)
    graph = nx.from_numpy_array(corr)

    Th, jumps = find_threshold_jumps(graph)
    if jumps.size == 0:
        threshold = 0.0
        jump_stats = {
            "jumps": jumps,
            "chosen_jump": None,
            "jump_index": None,
            "chosen_threshold": threshold,
            "expected_components": None,
        }
    else:
        effective_index = min(max(jump_index, 0), jumps.size - 1)
        chosen_jump = int(jumps[effective_index])
        threshold = float(Th[chosen_jump])
        jump_stats = {
            "jumps": jumps,
            "chosen_jump": chosen_jump,
            "jump_index": effective_index,
            "chosen_threshold": threshold,
            "expected_components": effective_index + 1,
        }

    corr_thresholded = apply_threshold_filter(corr, threshold)
    graph_thresholded = nx.from_numpy_array(corr_thresholded)

    return BandComputationResult(
        correlation_matrix=corr_thresholded,
        graph=graph_thresholded,
        threshold=threshold,
        jump_statistics=jump_stats,
    )
