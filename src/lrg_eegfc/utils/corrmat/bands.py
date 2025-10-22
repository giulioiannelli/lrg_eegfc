"""Band-specific correlation-matrix builders."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, MutableMapping, Optional, Tuple, Union

import networkx as nx
import numpy as np
import warnings
from numpy.typing import NDArray

from lrgsglib.utils import bandpass_sos

from ...config.const import BRAIN_BANDS
from .base import build_corr_network
from .thresholds import find_threshold_jumps


def _select_threshold_from_jumps(
    corr_mat_initial: NDArray,
    jump_index: int = 0,
    band_name: str = "",
) -> Dict[str, Union[int, float, NDArray]]:
    """Choose a threshold based on percolation jumps."""

    Th, jumps = find_threshold_jumps(nx.from_numpy_array(corr_mat_initial))

    if len(jumps) == 0:
        warnings.warn(
            f"Band '{band_name}': No jumps found in percolation analysis. Using minimum threshold.",
            UserWarning,
        )
        chosen_idx = 0
    elif jump_index >= len(jumps):
        warnings.warn(
            f"Band '{band_name}': jump_index={jump_index} exceeds available jumps ({len(jumps)})."
            f" Using last available jump.",
            UserWarning,
        )
        chosen_idx = jumps[-1]
        jump_index = len(jumps) - 1
    else:
        chosen_idx = jumps[jump_index]

    chosen_threshold = Th[chosen_idx] if len(Th) > 0 else 0.0
    return {
        "jumps": jumps,
        "chosen_jump": chosen_idx,
        "chosen_threshold": chosen_threshold,
        "jump_index": jump_index,
        "expected_components": jump_index + 1,
    }


def build_corrmat_perband(
    data_ts: NDArray,
    fs: float,
    bandpass_func: Callable = bandpass_sos,
    brain_bands: Mapping[str, Tuple[float, float]] = BRAIN_BANDS,
    return_jump_info: bool = False,
    apply_threshold_filtering: bool = True,
    corr_network_params: Optional[MutableMapping[str, Union[int, float, str]]] = None,
    jump_index: int = 0,
    filter_order: int = 4,
) -> Union[Dict[str, NDArray], Tuple[Dict[str, NDArray], Dict[str, Dict]]]:
    """Build correlation matrices for every band defined in ``brain_bands``."""

    corr_network_params = corr_network_params or {"threshold": 0}
    corr_mat_band: Dict[str, NDArray] = {}
    band_jump_info: Dict[str, Dict] = {}

    for band_name, (low, high) in brain_bands.items():
        nyquist = fs / 2.0
        bandwidth_ratio = (high - low) / nyquist
        effective_order = 1 if bandwidth_ratio < 0.01 and filter_order > 1 else filter_order

        if effective_order != filter_order:
            warnings.warn(
                f"Band '{band_name}': Reducing filter order from {filter_order} to {effective_order}"
                f" for narrow band ({low}-{high} Hz).",
                RuntimeWarning,
            )

        filtered = bandpass_func(data_ts, low, high, fs, effective_order)
        if not np.all(np.isfinite(filtered)):
            warnings.warn(
                f"Band '{band_name}': Bandpass filtering produced non-finite values. Skipping.",
                RuntimeWarning,
            )
            continue

        if apply_threshold_filtering:
            corr_initial = build_corr_network(filtered, **corr_network_params)
            info = _select_threshold_from_jumps(corr_initial, jump_index, band_name)
            band_jump_info[band_name] = info.copy()

            final_params = corr_network_params.copy()
            final_params["threshold"] = info["chosen_threshold"]
            corr_mat = build_corr_network(filtered, **final_params)

            G_final = nx.from_numpy_array(corr_mat)
            G_final.remove_nodes_from(list(nx.isolates(G_final)))
            if G_final.number_of_nodes() > 0:
                n_components = nx.number_connected_components(G_final)
                expected = info["expected_components"]
                if n_components != expected:
                    warnings.warn(
                        f"Band '{band_name}': Expected {expected} components but found {n_components}.",
                        UserWarning,
                    )
                band_jump_info[band_name]["actual_components"] = n_components
                band_jump_info[band_name]["validation_passed"] = n_components == expected
            else:
                warnings.warn(
                    f"Band '{band_name}': Threshold {info['chosen_threshold']:.6f} removed all edges.",
                    UserWarning,
                )
                band_jump_info[band_name]["actual_components"] = 0
                band_jump_info[band_name]["validation_passed"] = False
        else:
            corr_mat = build_corr_network(filtered, **corr_network_params)
            if return_jump_info:
                band_jump_info[band_name] = {
                    "jumps": None,
                    "chosen_jump": None,
                    "chosen_threshold": corr_network_params.get("threshold", 0),
                    "jump_index": None,
                    "expected_components": None,
                    "actual_components": None,
                    "validation_passed": None,
                }

        corr_mat_band[band_name] = corr_mat

    if return_jump_info:
        return corr_mat_band, band_jump_info
    return corr_mat_band


def build_band_correlation_matrices(
    data_ts: NDArray,
    fs: float,
    *,
    bandpass_func: Callable = bandpass_sos,
    brain_bands: Mapping[str, Tuple[float, float]] = BRAIN_BANDS,
    return_jump_info: bool = False,
    apply_threshold_filtering: bool = True,
    corr_network_params: Optional[MutableMapping[str, Union[int, float, str]]] = None,
    jump_index: int = 0,
) -> Tuple[Dict[str, NDArray], Optional[Dict[str, Dict]]]:
    """Backward-compatible wrapper returning both matrices and optional jump info."""

    result = build_corrmat_perband(
        data_ts,
        fs,
        bandpass_func=bandpass_func,
        brain_bands=brain_bands,
        return_jump_info=return_jump_info,
        apply_threshold_filtering=apply_threshold_filtering,
        corr_network_params=corr_network_params,
        jump_index=jump_index,
    )

    if return_jump_info:
        corr_mats, jump_info = result  # type: ignore[misc]
        return corr_mats, jump_info

    return result, None


def build_corrmat_single_band(
    data_ts: NDArray,
    fs: float,
    band: Tuple[float, float],
    bandpass_func: Callable = bandpass_sos,
    return_jump_info: bool = False,
    apply_threshold_filtering: bool = True,
    corr_network_params: Optional[Dict[str, Union[int, float, str]]] = None,
    jump_index: int = 0,
    band_name: str = "",
    filter_order: int = 4,
) -> Union[NDArray, Tuple[NDArray, Dict]]:
    """Build a single-band correlation matrix."""

    corr_network_params = corr_network_params or {"threshold": 0}
    low, high = band
    nyquist = fs / 2.0
    bandwidth_ratio = (high - low) / nyquist
    effective_order = 1 if bandwidth_ratio < 0.01 and filter_order > 1 else filter_order

    if effective_order != filter_order:
        warnings.warn(
            f"Band '{band_name}' ({low}-{high} Hz): reducing filter order from {filter_order} to {effective_order}.",
            RuntimeWarning,
        )

    filtered = bandpass_func(data_ts, low, high, fs, effective_order)
    if not np.all(np.isfinite(filtered)):
        warnings.warn(
            f"Band '{band_name}' ({low}-{high} Hz): Bandpass filtering produced non-finite values.",
            RuntimeWarning,
        )
        return (None, None) if return_jump_info else None

    if apply_threshold_filtering:
        corr_initial = build_corr_network(filtered, **corr_network_params)
        info = _select_threshold_from_jumps(corr_initial, jump_index, band_name)
        final_params = corr_network_params.copy()
        final_params["threshold"] = info["chosen_threshold"]
        corr_mat = build_corr_network(filtered, **final_params)
        return (corr_mat, info) if return_jump_info else corr_mat

    corr_mat = build_corr_network(filtered, **corr_network_params)
    if return_jump_info:
        info = {
            "jumps": None,
            "chosen_jump": None,
            "chosen_threshold": corr_network_params.get("threshold", 0),
            "jump_index": None,
            "expected_components": None,
            "actual_components": None,
            "validation_passed": None,
        }
        return corr_mat, info
    return corr_mat
