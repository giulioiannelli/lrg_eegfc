"""Convenience imports and utilities for Jupyter notebooks.

This module provides a curated set of imports and helper functions for
working with the lrg_eegfc package in Jupyter notebooks.

Usage
-----
In a notebook, start with:

    from lrgsglib import move_to_rootf
    move_to_rootf(pathname='lrg_eegfc')

    from lrg_eegfc.notebook import *
    path_figs = setup_notebook('my_analysis')

This imports all commonly-used functions and sets up the output directory.
"""

from pathlib import Path

import numpy as np

# Core configuration
from .config import *
from .config.const import BRAIN_BANDS, BRAIN_BANDS_NAMES, PHASE_LABELS, PATIENTS_LIST

# Data loading
from .utils.datamanag.loaders import load_data_dict
from .utils.datamanag.patient import PatientRecording, load_timeseries

# Correlation FC
from .utils.corrmat import build_corr_network, find_threshold_jumps
from .utils.corrmat.bands import process_band_correlation
from .utils.corrmat.structures import compute_structures_for_patient

# Coherence FC
from .utils.coherence import coherence_fc_pipeline

# Clustering utilities
from .utils.clustering import (
    compute_optimal_clusters_auto,
    fcluster_with_outliers,
    get_dendrogram_consistent_clusters,
)

# Distance utilities
from .utils.distances import (
    compute_cross_patient_consistency,
    compute_phase_distance_matrix,
    rank_distance_measures,
)

# Visualization - Correlation
from .visuals import (
    plot_correlation_and_network,
    plot_correlation_heatmap,
    plot_marchenko_pastur_comparison,
    plot_percolation_curves,
)

# Visualization - MSC/Coherence
from .visuals import (
    plot_msc_and_network,
    plot_msc_comparison_dense_vs_validated,
    plot_msc_heatmap,
)

# Visualization - LRG
from .visuals import (
    plot_lrg_dendrogram,
    plot_lrg_entropy_curves,
    plot_lrg_full_panel,
    plot_ultrametric_heatmap,
)

# Visualization - Metastable/Sankey
from .visuals.metastable import (
    analyze_node_trajectories,
    compute_clustering_across_tau,
    create_sankey_diagram,
    track_cluster_changes,
)

# Visualization - Reorganization
from .visuals.reorganization import (
    compute_cluster_membership_correlation,
    plot_phase_reorganization,
    plot_reorganization_distance_matrix,
)

# Visualization - Comparison
from .visuals.compare import plot_fc_comparison

# Workflows
from .workflow_corr import compute_band_connectivity
from .compare import compare_fc_methods

# Core plotting utilities
from .plotting import (
    plot_correlation_matrix,
    plot_dendrogram,
    plot_entropy,
    plot_graph,
    prepare_dendrogram,
)

# Re-export commonly used items from utils
from .utils import *

# __all__ = [
#     # Setup
#     "setup_notebook",
#     # Constants
#     "BRAIN_BANDS",
#     "BRAIN_BANDS_NAMES",
#     "PHASE_LABELS",
#     "PATIENTS_LIST",
#     # Data loading
#     "load_data_dict",
#     "PatientRecording",
#     "load_timeseries",
#     # Correlation FC
#     "build_corr_network",
#     "find_threshold_jumps",
#     "process_band_correlation",
#     "compute_structures_for_patient",
#     # Coherence FC
#     "coherence_fc_pipeline",
#     # Clustering
#     "fcluster_with_outliers",
#     "get_dendrogram_consistent_clusters",
#     "compute_optimal_clusters_auto",
#     # Distance comparison
#     "compute_phase_distance_matrix",
#     "compute_cross_patient_consistency",
#     "rank_distance_measures",
#     # Visualization - Correlation
#     "plot_correlation_heatmap",
#     "plot_correlation_and_network",
#     "plot_marchenko_pastur_comparison",
#     "plot_percolation_curves",
#     # Visualization - MSC
#     "plot_msc_heatmap",
#     "plot_msc_and_network",
#     "plot_msc_comparison_dense_vs_validated",
#     # Visualization - LRG
#     "plot_lrg_entropy_curves",
#     "plot_lrg_dendrogram",
#     "plot_ultrametric_heatmap",
#     "plot_lrg_full_panel",
#     # Visualization - Metastable
#     "compute_clustering_across_tau",
#     "analyze_node_trajectories",
#     "track_cluster_changes",
#     "create_sankey_diagram",
#     # Visualization - Reorganization
#     "compute_cluster_membership_correlation",
#     "plot_phase_reorganization",
#     "plot_reorganization_distance_matrix",
#     # Visualization - Comparison
#     "plot_fc_comparison",
#     # Workflows
#     "compute_band_connectivity",
#     "compare_fc_methods",
#     # Core plotting
#     "plot_correlation_matrix",
#     "plot_entropy",
#     "prepare_dendrogram",
#     "plot_dendrogram",
#     "plot_graph",
# ]


def setup_notebook(
    figure_dir: str = "figures",
    *,
    dpi: int = 150,
    figsize: tuple = (10, 6),
    inline: bool = True,
) -> Path:
    """Set up notebook environment with standard configuration.

    This function:
    1. Creates the output directory for figures
    2. Configures matplotlib settings
    3. Returns the path for saving figures

    Parameters
    ----------
    figure_dir : str, optional
        Name of figure subdirectory under data/. Default 'figures'
    dpi : int, optional
        Figure DPI for saving. Default 150
    figsize : tuple, optional
        Default figure size (width, height) in inches. Default (10, 6)
    inline : bool, optional
        Whether to configure for inline plotting. Default True

    Returns
    -------
    path_figs : Path
        Path object pointing to data/{figure_dir}/

    Examples
    --------
    Basic usage:

        >>> from lrgsglib import move_to_rootf
        >>> move_to_rootf(pathname='lrg_eegfc')
        >>> from lrg_eegfc.notebook import *
        >>> path_figs = setup_notebook('my_analysis')
        ✓ Notebook setup complete
          Figure output: data/my_analysis

    Custom configuration:

        >>> path_figs = setup_notebook(
        ...     'paper_figures',
        ...     dpi=300,
        ...     figsize=(12, 8)
        ... )

    Notes
    -----
    This function should be called near the start of every notebook after
    importing lrg_eegfc.notebook. It ensures consistent figure output
    locations and matplotlib settings.
    """
    import matplotlib.pyplot as plt

    # Create figure directory
    path_figs = Path("data") / figure_dir
    path_figs.mkdir(parents=True, exist_ok=True)

    # Configure matplotlib
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["savefig.bbox"] = "tight"

    # Configure for inline plotting if requested
    if inline:
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            if ipython is not None:
                ipython.run_line_magic("matplotlib", "inline")
        except (ImportError, AttributeError):
            pass  # Not in IPython/Jupyter

    print("✓ Notebook setup complete")
    print(f"  Figure output: {path_figs}")

    return path_figs
