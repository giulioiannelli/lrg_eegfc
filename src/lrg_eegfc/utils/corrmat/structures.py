"""Helpers to assemble ultrametric/linkage structures per patient."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from lrgsglib.utils.lrg.infocomm import extract_ultrametric_matrix

from ...config.const import BRAIN_BANDS, PHASE_LABELS
from .network import process_network_for_phase


def compute_structures_for_single(
    patient: str,
    phase: str,
    band: str,
    *,
    data_dict: dict,
    int_label_map: dict,
    correlation_protocol: Optional[dict] = None,
    filter_order: int = 1,
    linkage_method: str = "average",
):
    """Compute ultrametric, linkage and condensed distances for one trio."""

    if correlation_protocol is None:
        correlation_protocol = dict(filter_type="abs", spectral_cleaning=False)

    pin_labels = int_label_map[patient]["label"]
    entry = data_dict[patient][phase]
    data_ts = entry["data"]
    fs_raw = entry.get("fs", None)
    fs = float(np.asarray(fs_raw).flat[0]) if fs_raw is not None else 1000.0

    G, _, lnkgM, _, _, dists = process_network_for_phase(
        data_ts,
        fs,
        band,
        correlation_protocol,
        pin_labels,
        filter_order=filter_order,
        linkage_method=linkage_method,
    )

    if lnkgM is None or G is None:
        return None, None, None

    try:
        U = extract_ultrametric_matrix(lnkgM, G.number_of_nodes())
    except Exception:
        U = None

    if isinstance(dists, dict):
        condensed = dists.get("condensed")
    else:
        condensed = dists

    return U, lnkgM, condensed


def compute_structures_for_patient_phase(
    patient: str,
    phase: str,
    *,
    data_dict: dict,
    int_label_map: dict,
    bands: Optional[Iterable[str]] = None,
    correlation_protocol: Optional[dict] = None,
    filter_order: int = 1,
    linkage_method: str = "average",
):
    """Compute structures for all bands of a single patient/phase."""

    if bands is None:
        bands = BRAIN_BANDS.keys()

    result = {}
    for band in bands:
        result[band] = compute_structures_for_single(
            patient,
            phase,
            band,
            data_dict=data_dict,
            int_label_map=int_label_map,
            correlation_protocol=correlation_protocol,
            filter_order=filter_order,
            linkage_method=linkage_method,
        )
    return result


def compute_structures_for_patient_band(
    patient: str,
    band: str,
    *,
    data_dict: dict,
    int_label_map: dict,
    phases: Optional[Iterable[str]] = None,
    correlation_protocol: Optional[dict] = None,
    filter_order: int = 1,
    linkage_method: str = "average",
):
    """Compute structures for all phases of a single patient/band."""

    if phases is None:
        phases = PHASE_LABELS

    result = {}
    for phase in phases:
        result[phase] = compute_structures_for_single(
            patient,
            phase,
            band,
            data_dict=data_dict,
            int_label_map=int_label_map,
            correlation_protocol=correlation_protocol,
            filter_order=filter_order,
            linkage_method=linkage_method,
        )
    return result


def compute_structures_for_patient(
    patient: str,
    *,
    data_dict: dict,
    int_label_map: dict,
    bands: Optional[Iterable[str]] = None,
    phases: Optional[Iterable[str]] = None,
    correlation_protocol: Optional[dict] = None,
    filter_order: int = 1,
    linkage_method: str = "average",
):
    """Compute structures across every band/phase combination."""

    if bands is None:
        bands = list(BRAIN_BANDS.keys())
    if phases is None:
        phases = list(PHASE_LABELS)
    if correlation_protocol is None:
        correlation_protocol = dict(filter_type="abs", spectral_cleaning=False)

    U = {b: {} for b in bands}
    Z = {b: {} for b in bands}
    D = {b: {} for b in bands}

    for phase in phases:
        for band in bands:
            u, z, d = compute_structures_for_single(
                patient,
                phase,
                band,
                data_dict=data_dict,
                int_label_map=int_label_map,
                correlation_protocol=correlation_protocol,
                filter_order=filter_order,
                linkage_method=linkage_method,
            )
            U[band][phase] = u
            Z[band][phase] = z
            D[band][phase] = d

    return U, Z, D
