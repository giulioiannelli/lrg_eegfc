"""Correlation-matrix utilities (modularised package)."""

from .base import apply_threshold_filter, build_corr_network, clean_correlation_matrix
from .bands import build_band_correlation_matrices, build_corrmat_perband, build_corrmat_single_band
from .network import process_network_for_phase
from .structures import (
    compute_structures_for_patient,
    compute_structures_for_patient_band,
    compute_structures_for_patient_phase,
    compute_structures_for_single,
)
from .thresholds import find_exact_detachment_threshold, find_threshold_jumps

__all__ = [
    "apply_threshold_filter",
    "build_corr_network",
    "clean_correlation_matrix",
    "build_corrmat_perband",
    "build_corrmat_single_band",
    "build_band_correlation_matrices",
    "process_network_for_phase",
    "compute_structures_for_single",
    "compute_structures_for_patient_phase",
    "compute_structures_for_patient_band",
    "compute_structures_for_patient",
    "find_exact_detachment_threshold",
    "find_threshold_jumps",
]
