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
from .workflow_msc import MSCResult, compute_msc_matrix, load_msc_matrix, get_msc_cache_path, compute_msc_for_patient
from .workflow_corr import CorrResult, compute_corr_matrix, load_corr_matrix, get_corr_cache_path, compute_corr_for_patient
from .workflow_lrg import LRGResult, compute_lrg_analysis, load_lrg_result, get_lrg_cache_path, compute_lrg_for_patient
from .workflow_cleaning import (
    CleanedCorrResult,
    clean_correlation_matrix_full,
    load_cleaned_corr_matrix,
    get_cleaned_corr_cache_path,
    compute_cleaned_corr_for_patient,
)
from .compare import (
    UltrametricComparison,
    compare_ultrametric_matrices,
    compare_fc_methods,
    compare_phases,
    aggregate_comparisons,
    batch_compare_fc_methods,
    batch_compare_phases,
)
from .visuals.compare import plot_fc_comparison
from .visuals.reorganization import (
    compute_cluster_membership_correlation,
    plot_phase_reorganization,
    plot_reorganization_distance_matrix,
)
from .batch_compute import compute_all_matrices
from .utils import *  # noqa: F401,F403
from .config.const import *  # noqa: F401,F403
