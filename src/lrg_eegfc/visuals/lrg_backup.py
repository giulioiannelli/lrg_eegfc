"""LRG (Laplacian Renormalization Group) visualization functions.

This module provides visualization functions for LRG analysis results including
entropy curves, dendrograms, ultrametric distances, and comprehensive panels.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from ..workflow_lrg import load_lrg_result
from ..workflow_corr import load_corr_matrix
from ..workflow_msc import load_msc_matrix

__all__ = [
    "plot_lrg_entropy_curves",
    "plot_lrg_dendrogram",
    "plot_ultrametric_heatmap",
    "plot_lrg_full_panel",
]


def _load_channel_labels(patient: str, dataset_root: Path) -> List[str]:
    """Helper function to load channel labels."""
    try:
        label_file = dataset_root / patient / "channel_labels.mat"
        if label_file.exists():
            label_data = scipy.io.loadmat(label_file)
            # Try different possible field names
            if "channel_labels" in label_data:
                labels_raw = label_data["channel_labels"].flatten()
            elif "ChannelNames" in label_data:
                labels_raw = label_data["ChannelNames"].flatten()
            else:
                # Use first non-metadata field
                for key in label_data:
                    if not key.startswith("__"):
                        labels_raw = label_data[key].flatten()
                        break
            return [str(label[0]) if hasattr(label, '__getitem__') else str(label) for label in labels_raw]
    except Exception:
        pass
    return []


def plot_lrg_entropy_curves(
    patient: str,
    phase: str,
    band: str,
    fc_method: str,
    cache_root: Path = Path("data/lrg_cache"),
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> Path:
    """Plot LRG entropy curves (1-S and C) vs tau.

    This visualization shows how entropy metrics evolve across scales,
    revealing hierarchical transitions in the network.

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    fc_method : str
        FC method ("corr" or "msc")
    cache_root : Path, optional
        Root directory for LRG cache
    output_path : Optional[Path], optional
        Path to save figure (auto-generated if None)
    figsize : tuple, optional
        Figure size

    Returns
    -------
    Path
        Path to the saved figure
    """
    # Load LRG result
    result = load_lrg_result(patient, phase, band, fc_method, cache_root)

    if result is None:
        from ..workflow_lrg import get_lrg_cache_path
        cache_path = get_lrg_cache_path(patient, phase, band, fc_method, cache_root)
        raise FileNotFoundError(f"LRG result not found: {cache_path}")

    # Extract entropy data
    tau = result.entropy_tau
    S_norm = result.entropy_1_minus_S  # 1-S (normalized entropy)
    C = result.entropy_C  # Spectral complexity

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")

    # Plot both curves
    ax.plot(tau, S_norm, label=r"$1-S$ (Normalized Entropy)", color="blue", lw=2)
    ax.plot(tau[:-1], C, label=r"$C$ (Spectral Complexity)", color="red", lw=2)

    # Formatting
    ax.set_xlabel(r"$\tau$ (Scale Parameter)", fontsize=12)
    ax.set_ylabel("Entropy Metrics", fontsize=12)
    ax.set_title(f"LRG Entropy - {patient} {phase} {band} ({fc_method})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)  # Both metrics normalized to [0, 1]

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/lrg") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_lrg_{fc_method}_entropy.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_lrg_dendrogram(
    patient: str,
    phase: str,
    band: str,
    fc_method: str,
    cache_root: Path = Path("data/lrg_cache"),
    output_path: Optional[Path] = None,
    orientation: str = "top",
    figsize: tuple = (12, 8),
    dataset_root: Path = Path("data/stereoeeg_patients"),
    optimal_leaf_order: bool = True,
    show_labels: bool = True,
    max_labels: int = 50,
) -> Path:
    """Plot hierarchical dendrogram with optimal threshold line.

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    fc_method : str
        FC method ("corr" or "msc")
    cache_root : Path, optional
        Root directory for LRG cache
    output_path : Optional[Path], optional
        Path to save figure (auto-generated if None)
    orientation : str, optional
        Dendrogram orientation ("top", "right", "bottom", "left")
    figsize : tuple, optional
        Figure size
    dataset_root : Path, optional
        Root directory for patient data
    optimal_leaf_order : bool, optional
        Use optimal leaf ordering for better visualization
    show_labels : bool, optional
        Show channel labels
    max_labels : int, optional
        Maximum number of labels to show

    Returns
    -------
    Path
        Path to the saved figure
    """
    # Load LRG result
    result = load_lrg_result(patient, phase, band, fc_method, cache_root)

    if result is None:
        from ..workflow_lrg import get_lrg_cache_path
        cache_path = get_lrg_cache_path(patient, phase, band, fc_method, cache_root)
        raise FileNotFoundError(f"LRG result not found: {cache_path}")

    linkage = result.linkage_matrix
    ultrametric_condensed = result.ultrametric_matrix
    optimal_th = result.optimal_threshold

    # Optional: Optimal leaf ordering for better visualization
    if optimal_leaf_order:
        try:
            linkage = optimal_leaf_ordering(linkage, ultrametric_condensed)
        except Exception:
            pass  # Fall back to default ordering if optimization fails

    # Load channel labels
    labels = _load_channel_labels(patient, dataset_root)
    if not labels:
        labels = [str(i) for i in range(result.n_nodes)]

    # Create dendrogram
    fig, ax = plt.subplots(figsize=figsize)

    dendro = dendrogram(
        linkage,
        ax=ax,
        orientation=orientation,
        labels=labels if (show_labels and len(labels) <= max_labels) else None,
        no_labels=(not show_labels or len(labels) > max_labels),
        color_threshold=optimal_th,
        above_threshold_color="black",  # Clusters above threshold are black
        leaf_font_size=8 if len(labels) <= max_labels else 5,
    )

    # Add optimal threshold line
    if orientation == "top":
        ax.axhline(
            optimal_th,
            color="blue",
            linestyle="--",
            lw=2,
            label=f"Optimal Threshold = {optimal_th:.3f}",
        )
        ax.set_ylabel("Ultrametric Distance", fontsize=12)
        ax.set_xlabel("Channel Index", fontsize=12)
        ax.set_yscale("log")
    elif orientation == "right":
        ax.axvline(
            optimal_th,
            color="blue",
            linestyle="--",
            lw=2,
            label=f"Optimal Threshold = {optimal_th:.3f}",
        )
        ax.set_xlabel("Ultrametric Distance", fontsize=12)
        ax.set_ylabel("Channel Index", fontsize=12)
        ax.set_xscale("log")
    elif orientation == "bottom":
        ax.axhline(
            optimal_th,
            color="blue",
            linestyle="--",
            lw=2,
            label=f"Optimal Threshold = {optimal_th:.3f}",
        )
        ax.set_ylabel("Ultrametric Distance", fontsize=12)
        ax.set_xlabel("Channel Index", fontsize=12)
        ax.set_yscale("log")
    elif orientation == "left":
        ax.axvline(
            optimal_th,
            color="blue",
            linestyle="--",
            lw=2,
            label=f"Optimal Threshold = {optimal_th:.3f}",
        )
        ax.set_xlabel("Ultrametric Distance", fontsize=12)
        ax.set_ylabel("Channel Index", fontsize=12)
        ax.set_xscale("log")

    ax.set_title(
        f"LRG Dendrogram - {patient} {phase} {band} ({fc_method})", fontsize=14
    )
    ax.legend(fontsize=10)

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/lrg") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_lrg_{fc_method}_dendrogram.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_ultrametric_heatmap(
    patient: str,
    phase: str,
    band: str,
    fc_method: str,
    cache_root: Path = Path("data/lrg_cache"),
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 10),
    dataset_root: Path = Path("data/stereoeeg_patients"),
    cmap: str = "viridis",
) -> Path:
    """Plot ultrametric distance matrix as heatmap.

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    fc_method : str
        FC method ("corr" or "msc")
    cache_root : Path, optional
        Root directory for LRG cache
    output_path : Optional[Path], optional
        Path to save figure (auto-generated if None)
    figsize : tuple, optional
        Figure size
    dataset_root : Path, optional
        Root directory for patient data
    cmap : str, optional
        Colormap name

    Returns
    -------
    Path
        Path to the saved figure
    """
    # Load LRG result
    result = load_lrg_result(patient, phase, band, fc_method, cache_root)

    if result is None:
        from ..workflow_lrg import get_lrg_cache_path
        cache_path = get_lrg_cache_path(patient, phase, band, fc_method, cache_root)
        raise FileNotFoundError(f"LRG result not found: {cache_path}")

    ultrametric_condensed = result.ultrametric_matrix

    # Convert condensed to square form
    ultrametric_square = squareform(ultrametric_condensed)

    # Load channel labels
    labels = _load_channel_labels(patient, dataset_root)
    if not labels:
        labels = [str(i) for i in range(ultrametric_square.shape[0])]

    # Plot symmetric heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        ultrametric_square, cmap=cmap, interpolation="none", aspect="auto"
    )

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Ultrametric Distance", fontsize=12)

    # Add labels if not too many
    if len(labels) <= 50:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(
        f"Ultrametric Distance - {patient} {phase} {band} ({fc_method})",
        fontsize=14,
    )

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/lrg") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_lrg_{fc_method}_ultrametric.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_lrg_full_panel(
    patient: str,
    phase: str,
    band: str,
    fc_method: str,
    cache_root: Path = Path("data/lrg_cache"),
    output_path: Optional[Path] = None,
    figsize: tuple = (20, 14),
    dataset_root: Path = Path("data/stereoeeg_patients"),
) -> Path:
    """Create 4-panel comprehensive LRG visualization.

    Layout:
    ┌─────────────┬─────────────┐
    │  Entropy    │  Dendrogram │
    │  (1-S, C)   │  (log-scale)│
    ├─────────────┼─────────────┤
    │  Ultrametric│  Network    │
    │  Heatmap    │  (colored)  │
    └─────────────┴─────────────┘

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    fc_method : str
        FC method ("corr" or "msc")
    cache_root : Path, optional
        Root directory for LRG cache
    output_path : Optional[Path], optional
        Path to save figure (auto-generated if None)
    figsize : tuple, optional
        Figure size
    dataset_root : Path, optional
        Root directory for patient data

    Returns
    -------
    Path
        Path to the saved figure
    """
    # Load LRG result
    result = load_lrg_result(patient, phase, band, fc_method, cache_root)

    if result is None:
        from ..workflow_lrg import get_lrg_cache_path
        cache_path = get_lrg_cache_path(patient, phase, band, fc_method, cache_root)
        raise FileNotFoundError(f"LRG result not found: {cache_path}")

    # Load FC matrix
    if fc_method == "corr":
        fc_matrix = load_corr_matrix(patient, phase, band)
    elif fc_method == "msc":
        fc_matrix = load_msc_matrix(patient, phase, band)
    else:
        raise ValueError(f"Unknown fc_method: {fc_method}")

    if fc_matrix is None:
        raise FileNotFoundError(f"FC matrix not found for {patient} {phase} {band} ({fc_method})")

    # Load channel labels
    labels = _load_channel_labels(patient, dataset_root)
    if not labels:
        labels = [str(i) for i in range(result.n_nodes)]

    # Create 2x2 subplot layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]

    # Panel [0, 0]: Entropy curves
    tau = result.entropy_tau
    S_norm = result.entropy_1_minus_S
    C = result.entropy_C

    ax[0][0].set_xscale("log")
    ax[0][0].plot(tau, S_norm, label=r"$1-S$", color="blue", lw=2)
    ax[0][0].plot(tau[:-1], C, label=r"$C$", color="red", lw=2)
    ax[0][0].set_xlabel(r"$\tau$", fontsize=11)
    ax[0][0].set_ylabel("Entropy", fontsize=11)
    ax[0][0].set_title("Entropy Curves", fontsize=12)
    ax[0][0].legend(fontsize=9)
    ax[0][0].grid(True, alpha=0.3)
    ax[0][0].set_ylim(0, 1.05)

    # Panel [0, 1]: Dendrogram
    linkage = result.linkage_matrix
    optimal_th = result.optimal_threshold

    dendro = dendrogram(
        linkage,
        ax=ax[0][1],
        orientation="top",
        labels=labels if len(labels) <= 30 else None,
        no_labels=(len(labels) > 30),
        color_threshold=optimal_th,
        above_threshold_color="black",
        leaf_font_size=7,
    )

    ax[0][1].axhline(optimal_th, color="blue", linestyle="--", lw=2)
    ax[0][1].set_yscale("log")
    ax[0][1].set_ylabel("Distance", fontsize=11)
    ax[0][1].set_title("Hierarchical Dendrogram", fontsize=12)

    # Panel [1, 0]: Ultrametric heatmap
    ultrametric_square = squareform(result.ultrametric_matrix)

    im = ax[1][0].imshow(ultrametric_square, cmap="viridis", interpolation="none")
    divider = make_axes_locatable(ax[1][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1][0].set_title("Ultrametric Distances", fontsize=12)
    ax[1][0].axis("off")

    # Panel [1, 1]: Network colored by dendrogram clusters
    G = nx.from_numpy_array(fc_matrix)

    # Get dendrogram leaf colors (compute on temporary hidden axis for color assignment)
    temp_fig, temp_ax = plt.subplots()
    temp_dendro = dendrogram(
        linkage,
        ax=temp_ax,
        no_plot=True,
        color_threshold=optimal_th,
    )
    plt.close(temp_fig)

    # Map leaves to colors
    leaf_to_color = {}
    if "leaves_color_list" in temp_dendro:
        for leaf_idx, color in zip(temp_dendro["leaves"], temp_dendro["leaves_color_list"]):
            leaf_to_color[leaf_idx] = color
    else:
        # Fallback if color info not available
        leaf_to_color = {i: "lightblue" for i in range(len(labels))}

    # Color nodes by cluster
    node_colors = [leaf_to_color.get(i, "gray") for i in G.nodes()]

    # Compute layout
    try:
        pos = nx.spring_layout(G, seed=42, k=1.0, iterations=50)
    except Exception:
        pos = nx.circular_layout(G)

    # Extract edge widths
    edges = list(G.edges())
    if len(edges) > 0:
        widths = [G[u][v]["weight"] for u, v in edges]
        wmin, wmax = min(widths), max(widths)
        if wmax > wmin:
            widths_scaled = [0.1 + 4.9 * (w - wmin) / (wmax - wmin) for w in widths]
        else:
            widths_scaled = [2.5] * len(widths)
    else:
        widths_scaled = []

    # Draw network
    nx.draw(
        G,
        pos=pos,
        ax=ax[1][1],
        node_color=node_colors,
        width=widths_scaled,
        node_size=100,
        edge_color="gray",
        with_labels=False,
    )

    ax[1][1].set_title("Network (Cluster-Colored)", fontsize=12)

    # Overall title
    fig.suptitle(
        f"LRG Analysis - {patient} {phase} {band} ({fc_method})",
        fontsize=16,
        y=0.98,
    )

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/lrg") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_lrg_{fc_method}_full.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save high-resolution figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path
