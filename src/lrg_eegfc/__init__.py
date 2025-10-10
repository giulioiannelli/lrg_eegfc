"""LRG EEG Functional Connectivity toolkit."""

from .constants import BRAIN_BANDS, BRAIN_BAND_LABELS, PARAMETER_KEYS, PHASE_LABELS
from .correlation import (
    apply_threshold_filter,
    build_correlation_network,
    build_corr_network,
    build_band_correlation_matrices,
    clean_correlation_matrix,
    find_exact_detachment_threshold,
    find_threshold_jumps,
)
from .io import (
    PatientRecording,
    load_dataset,
    load_mat_file,
    load_patient_dataset,
    load_patient_metadata,
    load_timeseries,
)
from .workflow import BandComputationResult, compute_band_connectivity

__all__ = [
    "BRAIN_BANDS",
    "BRAIN_BAND_LABELS",
    "PARAMETER_KEYS",
    "PHASE_LABELS",
    "apply_threshold_filter",
    "build_correlation_network",
    "build_corr_network",
    "build_band_correlation_matrices",
    "clean_correlation_matrix",
    "find_exact_detachment_threshold",
    "find_threshold_jumps",
    "PatientRecording",
    "load_dataset",
    "load_mat_file",
    "load_patient_dataset",
    "load_patient_metadata",
    "load_timeseries",
    "BandComputationResult",
    "compute_band_connectivity",
]
