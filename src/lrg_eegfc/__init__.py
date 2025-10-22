"""LRG EEG Functional Connectivity toolkit."""

from .constants import BRAIN_BANDS, BRAIN_BAND_LABELS, PARAMETER_KEYS, PHASE_LABELS
from .utils.datamanag.loaders import load_data_dict, load_mat_pat_data as load_mat_file
from .utils.datamanag.patient import (
    PatientRecording,
    load_dataset,
    load_patient_dataset,
    load_patient_metadata,
    load_timeseries,
)
from .utils.corrmat import (
    apply_threshold_filter,
    build_band_correlation_matrices,
    build_corr_network,
    build_corrmat_perband,
    build_corrmat_single_band,
    clean_correlation_matrix,
    compute_structures_for_patient,
    compute_structures_for_patient_band,
    compute_structures_for_patient_phase,
    compute_structures_for_single,
    find_exact_detachment_threshold,
    find_threshold_jumps,
    process_network_for_phase,
)
from .workflow import BandComputationResult, compute_band_connectivity
from .utils import *  # noqa: F401,F403
from .config.const import *  # noqa: F401,F403