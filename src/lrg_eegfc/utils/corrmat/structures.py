"""Helpers to assemble ultrametric/linkage structures per patient."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from lrgsglib.utils.lrg.infocomm import extract_ultrametric_matrix

from ...config.const import BRAIN_BANDS, PHASE_LABELS
from .network import process_network_for_phase


def compute_structures_for_single(
    data_dict: dict,
    patient: str,
    phase: str,
    band: str,
    int_label_map: Optional[dict] = None,
    *,
    correlation_protocol: Optional[dict] = None,
    filter_order: int = 1,
    linkage_method: str = "average",
):
    """Compute structures for a single (patient, phase, band) combination.

    Builds the correlation network for the requested band/phase, extracts the
    giant component, computes the linkage (hierarchical clustering) and returns
    the associated ultrametric (cophenetic) matrix together with the linkage
    matrix and the condensed pairwise distance vector.

    Parameters
    ----------
    data_dict : dict
        Nested dictionary as returned by the dataset loaders where
        ``data_dict[patient][phase]`` contains the time series and  ``'fs'`` 
        (sampling frequency).
    patient : str
        Patient identifier present in ``data_dict`` and (optionally)
        ``int_label_map``.
    phase : str
        Phase label to process (e.g., ``"rsPre"``, ``"taskLearn"``).
    band : str
        Frequency band key (must exist in ``BRAIN_BANDS``).
    int_label_map : dict, optional
        Mapping ``patient -> DataFrame`` with a ``'label'`` column used as
        canonical node labels. If omitted, numeric labels are used.
    correlation_protocol : dict, optional
        Parameters forwarded to the correlation/network construction, e.g.
        ``{"filter_type": "abs", "spectral_cleaning": False}``.
    filter_order : int, optional
        Band-pass filter order (default: 1).
    linkage_method : str, optional
        Linkage method for hierarchical clustering (default: ``"average"``).

    Returns
    -------
    tuple[ndarray|None, ndarray|None, ndarray|None]
        ``(U, Z, D)`` where ``U`` is the ultrametric matrix (square), ``Z`` is
        the SciPy linkage matrix, and ``D`` is the condensed pairwise distance
        vector. Any of them may be ``None`` when the network is empty or
        numerically invalid.
    """

    if correlation_protocol is None:
        correlation_protocol = dict(filter_type='abs', spectral_cleaning=False)
    #
    entry = data_dict[patient][phase]
    data_ts = entry['data']
    #
    if int_label_map is None:
        pin_labels = {i: str(i) for i in range(int(data_ts.shape[0]))}
    else:
        pin_labels = int_label_map[patient]["label"]
    fs_raw = entry.get("fs", None)
    fs = float(np.asarray(fs_raw).flat[0]) if fs_raw is not None else 1024

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

    condensed = dists

    return U, lnkgM, condensed


def compute_structures_for_patient_phase(
    data_dict: dict,
    patient: str,
    phase: str,
    int_label_map: Optional[dict] = None,
    *,
    bands: Optional[Iterable[str]] = None,
    correlation_protocol: Optional[dict] = None,
    filter_order: int = 1,
    linkage_method: str = "average",
):
    """Compute structures across all bands for a patient/phase.

    Parameters
    ----------
    patient, phase : str
        Identify the data entry in ``data_dict`` to process.
    data_dict : dict
        Nested dataset dictionary (see ``compute_structures_for_single``).
    int_label_map : dict, optional
        Map of canonical node labels per patient (with a ``'label'`` column).
    bands : iterable of str, optional
        Band names to process (defaults to all ``BRAIN_BANDS``).
    correlation_protocol : dict, optional
        Passed to network construction.
    filter_order : int, optional
        Filter order used in band-pass filter.
    linkage_method : str, optional
        Linkage method for hierarchical clustering.

    Returns
    -------
    dict[str, tuple]
        Mapping ``band -> (U, Z, D)`` as defined in
        ``compute_structures_for_single``.
    """

    if bands is None:
        bands = BRAIN_BANDS.keys()

    result = {}
    for band in bands:
        result[band] = compute_structures_for_single(
            data_dict,
            patient,
            phase,
            band,
            int_label_map,
            correlation_protocol=correlation_protocol,
            filter_order=filter_order,
            linkage_method=linkage_method,
        )
    return result


def compute_structures_for_patient_band(
    data_dict: dict,
    patient: str,
    band: str,
    int_label_map: Optional[dict] = None,
    *,
    phases: Optional[Iterable[str]] = None,
    correlation_protocol: Optional[dict] = None,
    filter_order: int = 1,
    linkage_method: str = "average",
):
    """Compute structures across all phases for a single patient and band.

    Parameters
    ----------
    patient : str
        Patient identifier.
    band : str
        Frequency band to process.
    data_dict, int_label_map, phases, correlation_protocol, filter_order, linkage_method
        See ``compute_structures_for_patient_phase``.

    Returns
    -------
    dict[str, tuple]
        Mapping ``phase -> (U, Z, D)``.
    """

    if phases is None:
        phases = PHASE_LABELS

    result = {}
    for phase in phases:
        result[phase] = compute_structures_for_single(
            data_dict,
            patient,
            phase,
            band,
            int_label_map,
            correlation_protocol=correlation_protocol,
            filter_order=filter_order,
            linkage_method=linkage_method,
        )
    return result


def compute_structures_for_patient(
    data_dict: dict,
    patient: str,
    int_label_map: Optional[dict] = None,
    *,
    bands: Optional[Iterable[str]] = None,
    phases: Optional[Iterable[str]] = None,
    correlation_protocol: Optional[dict] = None,
    filter_order: int = 1,
    linkage_method: str = "average",
):
    """Compute structures across every band/phase combination for a patient.

    Parameters
    ----------
    patient : str
        Patient identifier to evaluate.
    data_dict, int_label_map, bands, phases, correlation_protocol, filter_order, linkage_method
        As described in ``compute_structures_for_single`` and
        ``compute_structures_for_patient_phase``.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(U, Z, D)`` three nested dictionaries indexed by ``[band][phase]``.
    """

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
                data_dict,
                patient,
                phase,
                band,
                int_label_map,
                correlation_protocol=correlation_protocol,
                filter_order=filter_order,
                linkage_method=linkage_method,
            )
            U[band][phase] = u
            Z[band][phase] = z
            D[band][phase] = d

    return U, Z, D
