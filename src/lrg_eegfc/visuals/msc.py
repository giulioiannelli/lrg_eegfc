"""MSC-based functional connectivity visualization functions.

This module provides visualization functions for MSC (Magnitude-Squared Coherence)
matrices and networks. Unlike correlation matrices, MSC does not require spectral
cleaning as it is already computed in the frequency domain.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..workflow_msc import load_msc_matrix, get_msc_cache_path

__all__ = [
    "plot_msc_heatmap",
    "plot_msc_and_network",
    "plot_msc_comparison_dense_vs_validated",
    "plot_msc_summary",
]


def plot_msc_heatmap(
    msc_matrix: np.ndarray,
    output_path: Path,
    title: str = "MSC Matrix",
    vmin: float = 0.0,
    vmax: float = 1.0,
    channel_labels: Optional[List[str]] = None,
    figsize: tuple = (8, 8),
    cmap: str = "viridis",
) -> Path:
    """Plot MSC matrix heatmap.

    MSC values range from 0 to 1, unlike correlation which ranges from -1 to 1.

    Parameters
    ----------
    msc_matrix : np.ndarray
        MSC matrix (N x N)
    output_path : Path
        Path to save the figure
    title : str, optional
        Plot title
    vmin : float, optional
        Minimum value for colormap (default: 0.0)
    vmax : float, optional
        Maximum value for colormap (default: 1.0)
    channel_labels : List[str], optional
        Channel labels for axes
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap name (default: 'viridis')

    Returns
    -------
    Path
        Path to the saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(
        msc_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="auto",
    )

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # Set labels if provided and not too many channels
    if channel_labels is not None and len(channel_labels) <= 50:
        ax.set_xticks(range(len(channel_labels)))
        ax.set_yticks(range(len(channel_labels)))
        ax.set_xticklabels(channel_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(channel_labels, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title, fontsize=14)

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_msc_and_network(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/msc_cache"),
    output_path: Optional[Path] = None,
    sparsify: str = "none",
    n_surrogates: int = 0,
    nperseg: int = 1024,
    figsize: tuple = (16, 8),
    dataset_root: Path = Path("data/stereoeeg_patients"),
) -> Path:
    """Plot MSC matrix and network graph side-by-side.

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    cache_root : Path, optional
        Root directory for MSC cache
    output_path : Optional[Path], optional
        Path to save figure (auto-generated if None)
    sparsify : str, optional
        Sparsification method: "none" (dense) or "soft" (validated)
    n_surrogates : int, optional
        Number of surrogates for validation (if sparsify="soft")
    nperseg : int, optional
        Window length for Welch's method
    figsize : tuple, optional
        Figure size
    dataset_root : Path, optional
        Root directory for patient data

    Returns
    -------
    Path
        Path to the saved figure
    """
    # Load MSC matrix
    msc_matrix = load_msc_matrix(
        patient, phase, band, cache_root, sparsify, n_surrogates, nperseg
    )

    if msc_matrix is None:
        cache_path = get_msc_cache_path(
            patient, phase, band, cache_root, sparsify, n_surrogates, nperseg
        )
        raise FileNotFoundError(f"MSC matrix not found: {cache_path}")

    # Load channel labels
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
            channel_labels = [str(label[0]) if hasattr(label, '__getitem__') else str(label) for label in labels_raw]
        else:
            channel_labels = [str(i) for i in range(msc_matrix.shape[0])]
    except Exception:
        channel_labels = [str(i) for i in range(msc_matrix.shape[0])]

    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 0.1, 1.5]})

    # Left: MSC heatmap (force square)
    im = ax[0].imshow(msc_matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax[0].set_box_aspect(1)  # Force square axes
    ax[0].set_title(f"MSC Matrix - {band} {phase}", fontsize=12)
    ax[0].axis("off")
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    # Middle: hide spacer
    ax[1].axis("off")

    # Right: Network graph
    G = nx.from_numpy_array(msc_matrix)

    # Use edge weights directly (already in [0, 1] range)
    widths = [G[u][v]["weight"] for u, v in G.edges()]

    # Compute layout (k=0.1 for fully connected weighted networks shows structure better)
    try:
        pos = nx.spring_layout(G, seed=42, k=0.1, iterations=50)
    except Exception:
        # Fallback to circular layout
        pos = nx.circular_layout(G)

    # Create label dictionary
    label_dict = {i: channel_labels[i] for i in range(len(channel_labels))}

    # Draw network
    nx.draw(
        G,
        pos=pos,
        ax=ax[2],
        width=widths,
        node_size=100,
        node_color="lightblue",
        edge_color="gray",
        with_labels=(len(channel_labels) <= 30),
        labels=label_dict if len(channel_labels) <= 30 else {},
        font_size=8,
    )

    ax[2].set_title(f"MSC Network - {band} {phase}", fontsize=12)

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/msc") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "dense" if sparsify == "none" else f"validated_nsurr{n_surrogates}"
        output_path = output_dir / f"{band}_{phase}_msc_{suffix}_network.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_msc_comparison_dense_vs_validated(
    patient: str,
    phase: str,
    band: str,
    n_surrogates: int = 200,
    cache_root: Path = Path("data/msc_cache"),
    output_path: Optional[Path] = None,
    nperseg: int = 1024,
    figsize: tuple = (18, 6),
) -> Path:
    """Compare dense vs validated MSC matrices side-by-side.

    This creates a 3-panel visualization showing:
    - Panel 0: Dense MSC matrix (no validation)
    - Panel 1: Validated MSC matrix (surrogate-based sparsification)
    - Panel 2: Difference matrix (what was removed)

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    n_surrogates : int, optional
        Number of surrogates used for validation
    cache_root : Path, optional
        Root directory for MSC cache
    output_path : Optional[Path], optional
        Path to save figure (auto-generated if None)
    nperseg : int, optional
        Window length for Welch's method
    figsize : tuple, optional
        Figure size

    Returns
    -------
    Path
        Path to the saved figure
    """
    # Load both versions
    dense_matrix = load_msc_matrix(patient, phase, band, cache_root, "none", 0, nperseg)
    validated_matrix = load_msc_matrix(
        patient, phase, band, cache_root, "soft", n_surrogates, nperseg
    )

    if dense_matrix is None:
        cache_path = get_msc_cache_path(patient, phase, band, cache_root, "none", 0, nperseg)
        raise FileNotFoundError(f"Dense MSC matrix not found: {cache_path}")

    if validated_matrix is None:
        cache_path = get_msc_cache_path(
            patient, phase, band, cache_root, "soft", n_surrogates, nperseg
        )
        raise FileNotFoundError(f"Validated MSC matrix not found: {cache_path}")

    # Compute difference
    difference = dense_matrix - validated_matrix

    # Compute statistics
    mean_dense = np.mean(dense_matrix[np.triu_indices_from(dense_matrix, k=1)])
    mean_validated = np.mean(validated_matrix[np.triu_indices_from(validated_matrix, k=1)])
    n_edges_dense = np.sum(dense_matrix > 0) - dense_matrix.shape[0]  # Exclude diagonal
    n_edges_validated = np.sum(validated_matrix > 0) - validated_matrix.shape[0]
    pct_retained = 100 * n_edges_validated / n_edges_dense if n_edges_dense > 0 else 0

    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # Panel 0: Dense matrix
    im0 = ax[0].imshow(dense_matrix, cmap="viridis", vmin=0, vmax=1, interpolation="none")
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax0)
    ax[0].set_title(f"Dense MSC\n(mean={mean_dense:.3f})", fontsize=12)
    ax[0].axis("off")

    # Panel 1: Validated matrix
    im1 = ax[1].imshow(
        validated_matrix, cmap="viridis", vmin=0, vmax=1, interpolation="none"
    )
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    ax[1].set_title(
        f"Validated MSC\n(mean={mean_validated:.3f}, {pct_retained:.1f}% edges)",
        fontsize=12,
    )
    ax[1].axis("off")

    # Panel 2: Difference
    im2 = ax[2].imshow(difference, cmap="Reds", vmin=0, vmax=difference.max(), interpolation="none")
    divider2 = make_axes_locatable(ax[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    ax[2].set_title(f"Removed by Validation", fontsize=12)
    ax[2].axis("off")

    # Overall title
    fig.suptitle(
        f"{patient} - {band} {phase} - MSC Comparison (n_surr={n_surrogates})",
        fontsize=14,
        y=0.98,
    )

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/msc") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_msc_comparison_nsurr{n_surrogates}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_msc_summary(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/msc_cache"),
    output_path: Optional[Path] = None,
    nperseg: int = 1024,
    figsize: tuple = (18, 6),
    dataset_root: Path = Path("data/stereoeeg_patients"),
) -> Path:
    """Create 3-panel MSC summary visualization.

    Layout:
    ┌─────────────┬─────────────┬─────────────┐
    │ MSC Matrix  │ Full Network│ Percolation │
    │ (0-1 range) │ (unthresh.) │ Curves (info)│
    └─────────────┴─────────────┴─────────────┘

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    cache_root : Path, optional
        Root directory for MSC cache
    output_path : Optional[Path], optional
        Path to save figure (auto-generated if None)
    nperseg : int, optional
        Window length for Welch's method
    figsize : tuple, optional
        Figure size
    dataset_root : Path, optional
        Root directory for patient data

    Returns
    -------
    Path
        Path to the saved figure
    """
    # Load MSC matrix
    msc_matrix = load_msc_matrix(
        patient, phase, band, cache_root, sparsify="none", nperseg=nperseg
    )

    if msc_matrix is None:
        from ..workflow_msc import get_msc_cache_path
        cache_path = get_msc_cache_path(
            patient, phase, band, cache_root, sparsify="none", nperseg=nperseg
        )
        raise FileNotFoundError(f"MSC matrix not found: {cache_path}")

    # Load channel labels
    try:
        label_file = dataset_root / patient / "channel_labels.mat"
        if label_file.exists():
            label_data = scipy.io.loadmat(label_file)
            if "channel_labels" in label_data:
                labels_raw = label_data["channel_labels"].flatten()
            elif "ChannelNames" in label_data:
                labels_raw = label_data["ChannelNames"].flatten()
            else:
                for key in label_data:
                    if not key.startswith("__"):
                        labels_raw = label_data[key].flatten()
                        break
            channel_labels = [
                str(label[0]) if hasattr(label, "__getitem__") else str(label)
                for label in labels_raw
            ]
        else:
            channel_labels = [str(i) for i in range(msc_matrix.shape[0])]
    except Exception:
        channel_labels = [str(i) for i in range(msc_matrix.shape[0])]

    # Create figure with 3 panels
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # =========================================================================
    # Panel 0: MSC Matrix Heatmap (1:1 aspect ratio)
    # =========================================================================
    im0 = ax[0].imshow(
        msc_matrix, cmap="viridis", vmin=0, vmax=1, interpolation="none", aspect="equal"
    )
    ax[0].set_box_aspect(1)
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax0)
    ax[0].set_title(f"MSC Matrix\n{band} {phase}", fontsize=12, fontweight="bold")
    ax[0].axis("off")

    # =========================================================================
    # Panel 1: Network Graph (Full, Unthresholded)
    # =========================================================================
    # Build graph from full MSC matrix
    G = nx.from_numpy_array(msc_matrix)

    # Use edge weights directly (already in [0, 1] range)
    widths = [G[u][v]["weight"] for u, v in G.edges()]

    # Compute layout with increased k to spread nodes and reveal edge width structure
    try:
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    except Exception:
        pos = nx.circular_layout(G)

    # Create label dictionary
    label_dict = {i: channel_labels[i] for i in range(len(channel_labels))}

    # Draw network
    nx.draw(
        G,
        pos=pos,
        ax=ax[1],
        width=widths,
        node_size=100,
        node_color="lightblue",
        edge_color="gray",
        with_labels=(len(channel_labels) <= 30),
        labels=label_dict if len(channel_labels) <= 30 else {},
        font_size=8,
        alpha=0.7,
    )

    ax[1].set_title(
        f"MSC Network\n(Full, {G.number_of_edges()} edges)",
        fontsize=12,
        fontweight="bold",
    )

    # =========================================================================
    # Panel 2: Percolation Curves (Informational Only)
    # =========================================================================
    # Compute percolation statistics using lrgsglib
    try:
        from lrgsglib.nx_patches.funcs.thresholding import compute_threshold_stats

        # Compute percolation curves
        Th, Einf, Pinf = compute_threshold_stats(G)

        # Find jumps (where components change)
        jumps = np.where(np.diff(Pinf) != 0)[0]

        # Plot P_inf (giant component fraction)
        ax[2].plot(
            Th, Pinf, "kH-", markersize=4, linewidth=1.5, label=r"$P_\infty$ (Giant)"
        )

        # Plot E_inf (edge fraction in giant)
        ax[2].plot(Th, Einf, "b-", linewidth=1.5, label=r"$E_\infty$ (Edges)")

        # Mark jumps with vertical lines
        for jump_idx in jumps[:5]:  # Show first 5 jumps
            ax[2].axvline(
                Th[jump_idx], color="red", linestyle="--", alpha=0.5, linewidth=0.8
            )

        ax[2].set_xlabel("Threshold (MSC value)", fontsize=10)
        ax[2].set_ylabel("Fraction", fontsize=10)
        ax[2].set_title(
            "Percolation Curves\n(Informational)", fontsize=12, fontweight="bold"
        )
        ax[2].legend(fontsize=9, loc="upper right")
        ax[2].grid(True, alpha=0.3)
        ax[2].set_xlim(0, 1)
        ax[2].set_ylim(-0.05, 1.05)

        # Add text note
        ax[2].text(
            0.5,
            0.02,
            "Note: Threshold NOT applied to network",
            transform=ax[2].transAxes,
            fontsize=8,
            ha="center",
            style="italic",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    except Exception as e:
        # Fallback if lrgsglib not available or error occurs
        ax[2].text(
            0.5,
            0.5,
            f"Percolation analysis\nunavailable\n\n({str(e)[:50]}...)",
            transform=ax[2].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax[2].set_title(
            "Percolation Curves\n(Unavailable)", fontsize=12, fontweight="bold"
        )
        ax[2].axis("off")

    # =========================================================================
    # Overall Title and Layout
    # =========================================================================
    fig.suptitle(
        f"{patient} - MSC Summary - {band.upper()} {phase}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/msc") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_msc_summary.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path
