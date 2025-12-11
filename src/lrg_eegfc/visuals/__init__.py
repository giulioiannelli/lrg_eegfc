"""Visualization functions for LRG EEG FC analysis."""

from .correlation import (
    plot_correlation_heatmap,
    plot_correlation_and_network,
    plot_marchenko_pastur_comparison,
    plot_percolation_curves,
)
from .msc import (
    plot_msc_heatmap,
    plot_msc_and_network,
    plot_msc_comparison_dense_vs_validated,
)
from .lrg import (
    plot_lrg_entropy_curves,
    plot_lrg_dendrogram,
    plot_ultrametric_heatmap,
    plot_lrg_full_panel,
)

__all__ = [
    # Correlation visualizations
    "plot_correlation_heatmap",
    "plot_correlation_and_network",
    "plot_marchenko_pastur_comparison",
    "plot_percolation_curves",
    # MSC visualizations
    "plot_msc_heatmap",
    "plot_msc_and_network",
    "plot_msc_comparison_dense_vs_validated",
    # LRG visualizations
    "plot_lrg_entropy_curves",
    "plot_lrg_dendrogram",
    "plot_ultrametric_heatmap",
    "plot_lrg_full_panel",
]
