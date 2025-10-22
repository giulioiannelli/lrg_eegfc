"""Core constants used across :mod:`lrg_eegfc`.

The values defined in this module are intentionally lightweight and import
without touching the filesystem.  They are primarily used by the correlation
and data-loading utilities so keeping them together avoids circular
dependencies.
"""
#
from __future__ import annotations
#
from pathlib import Path
from typing import Dict, Iterable, Tuple, List
#
__all__ = [
    "BRAIN_BANDS",
    "BRAIN_BAND_TEX_DICT",
    "sEEG_DATAPATH",
    "phase_labels",
    "patients_list",
    "param_keys_list",
    "PATIENTS_LIST",
    "PHASE_LABELS",
    "PARAMETER_KEYS"
]
#
sEEG_DATAPATH = Path('data') / 'stereoeeg_patients'
patients_list = [p.name for p in Path(sEEG_DATAPATH).iterdir() 
                 if p.is_dir() and p.name.startswith('Pat_')]
#
phase_labels = ['rsPre', 'taskLearn', 'taskTest', 'rsPost']
param_keys_list = [
    'fs', 'fcutHigh', 'fcutLow', 'filter_order', 
    'NotchFilter', 'DataDimensions'
]
#: Patients available in the LRG EEG-FC dataset.
PATIENTS_LIST: List[str] = [p.name for p in Path(sEEG_DATAPATH).iterdir() 
                 if p.is_dir() and p.name.startswith('Pat_')]

#: Recording phases expected in the publicly shared SEEG datasets.
PHASE_LABELS: Tuple[str, ...] = ("rsPre", "taskLearn", "taskTest", "rsPost")

#: Keys typically embedded inside the ``Parameters`` struct of the ``.mat``
#: files distributed with the LRG datasets.
PARAMETER_KEYS: Tuple[str, ...] = (
    "fs",
    "fcutHigh",
    "fcutLow",
    "filter_order",
    "NotchFilter",
    "DataDimensions",
)
#
#: Brain bands keys
BRAIN_BANDS_NAMES: List[str] = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "low_gamma",
    "high_gamma"
]
#: Brain bands frequency tuples
BRAIN_BANDS_FREQ: List[Tuple[float, float]] = [
    (0.53, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 80.0),
    (80.0, 300.0),
]
#: Canonical EEG frequency bands expressed as ``(low, high)`` Hz pairs.
BRAIN_BANDS: Dict[str, Tuple[float, float]] = {
    band: freq for band, freq in zip(BRAIN_BANDS_NAMES, BRAIN_BANDS_FREQ)
}
#: LaTeX-friendly labels for each EEG band. Useful when generating plots.
BRAIN_BANDS_TEX_NAMES: List[str] = [
    r"$\delta$",
    r"$\theta$",
    r"$\alpha$",
    r"$\beta$",
    r"$\gamma_{\mathrm{l}}$",
    r"$\gamma_{\mathrm{h}}$",
]
BRAIN_BAND_TEX_DICT: Dict[str, str] = {
    band: tex_label
    for band, tex_label in zip(BRAIN_BANDS_NAMES, BRAIN_BANDS_TEX_NAMES)
}