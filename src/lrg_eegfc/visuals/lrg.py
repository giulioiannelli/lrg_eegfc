"""LRG (Laplacian Renormalization Group) visualization functions.

This module provides visualization functions for LRG analysis results including
entropy curves, dendrograms, ultrametric distances, and comprehensive panels.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, fcluster, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from ..workflow_lrg import load_lrg_result
from ..workflow_corr import load_corr_matrix
from ..workflow_msc import load_msc_matrix

__all__ = [
    "compute_partition_stability_index",
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


def compute_partition_stability_index(linkage_matrix: np.ndarray):
    """Compute Partition Stability Index from hierarchical clustering.

    Implements Eq. (2) from the multiscale community-detection paper:
        Ψ(n; τ) = N_norm * (log10 Δ_n - log10 Δ_{n+1})
    where Δ_n are dendrogram branch thresholds ordered from the root (single
    cluster) downward and
        N_norm = [log10 Δ_1 - log10 Δ_{n_max}]^{-1}
    normalizes by the total dendrogram length in log-space.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering (N-1, 4).

    Returns
    -------
    psi_values : np.ndarray
        Partition Stability Index values using the paper's normalization.
    n_communities : np.ndarray
        Number of communities corresponding to each PSI value.
    """
    # Merge heights are non-decreasing; reverse so Δ_n starts at the root (largest gap)
    deltas = linkage_matrix[:, 2][::-1]

    # Guard against zeros/negatives before taking logs
    eps = 1e-12
    deltas = np.clip(deltas, eps, None)
    log_deltas = np.log10(deltas)

    if len(log_deltas) < 2:
        return np.array([]), np.array([])

    # Normalization constant over total dendrogram length (Eq. 2)
    denom = log_deltas[0] - log_deltas[-1]
    norm = 1.0 / denom if denom > 0 else 0.0

    # Ψ for consecutive gaps; n_communities starts at 2 (first split) up to N-1
    psi_values = norm * (log_deltas[:-1] - log_deltas[1:])
    n_communities = np.arange(2, 2 + len(psi_values))

    return psi_values, n_communities


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
    dataset_root: Path = Path("data/stereoeeg_patients"),
    output_path: Optional[Path] = None,
    figsize: tuple = (20, 12),
    verbose: bool = False,
) -> Path:
    """Create comprehensive LRG analysis visualization - CORRECTED VERSION.

    Generates a 5-panel layout following FIGMNTGN notebooks exactly:
    - Panel (a): FC Matrix Heatmap
    - Panel (b): Entropy and Specific Heat (NO N multiplication)
    - Panel (c): Dendrogram (correct tmin/tmax/log scale from FIGMNTGN03)
    - Panel (d): Network with partition colors
    - Panel (e): PSI Plot (Partition Stability Index)

    Parameters
    ----------
    patient : str
        Patient identifier
    phase : str
        Recording phase
    band : str
        Frequency band
    fc_method : str
        FC method: "corr" or "msc"
    cache_root : Path
        Root directory for LRG cache
    dataset_root : Path
        Root directory for patient data
    output_path : Path, optional
        Output file path
    figsize : tuple
        Figure size
    verbose : bool
        Print progress

    Returns
    -------
    Path
        Path to saved figure
    """
    # Load LRG result
    lrg_result = load_lrg_result(patient, phase, band, fc_method, cache_root)

    if lrg_result is None:
        raise FileNotFoundError(
            f"LRG result not found: {patient} {phase} {band} ({fc_method})\n"
            f"Run: python src/compute_lrg_analysis.py --patient {patient} --fc-method {fc_method}"
        )

    if verbose:
        print(f"Loaded LRG result: {lrg_result.n_nodes} nodes")

    # Load FC matrix for visualization
    percolation_threshold = None  # Will be set for correlation method
    if fc_method == "corr":
        fc_matrix = load_corr_matrix(
            patient, phase, band, cache_root=Path("data/corr_cache"), filter_type="abs", zero_diagonal=True
        )
        method_label = "Correlation"

        # Load percolation threshold from correlation metadata
        # This is the threshold where first node detaches from giant component
        meta_path = Path("data/corr_cache") / patient / f"{band}_{phase}_corr_cleaned_meta.npz"
        if meta_path.exists():
            meta_data = np.load(meta_path, allow_pickle=True)
            percolation_threshold = float(meta_data["threshold"])
            if verbose:
                print(f"Loaded percolation threshold: {percolation_threshold:.6f}")
        else:
            if verbose:
                print("Warning: No percolation metadata found, will use all edges for layout")
    elif fc_method == "msc":
        fc_matrix = load_msc_matrix(
            patient, phase, band, cache_root=Path("data/msc_cache"), sparsify="none", n_surrogates=0
        )
        method_label = "MSC"
    else:
        raise ValueError(f"Unknown fc_method: {fc_method}")

    if fc_matrix is None:
        raise FileNotFoundError(f"FC matrix not found for {patient} {phase} {band} ({fc_method})")

    # Extract data from LRG result
    linkage_matrix = lrg_result.linkage_matrix
    ultrametric_matrix = lrg_result.ultrametric_matrix
    optimal_threshold = lrg_result.optimal_threshold
    entropy_tau = lrg_result.entropy_tau
    entropy_1_minus_S = lrg_result.entropy_1_minus_S
    entropy_C = lrg_result.entropy_C
    n_nodes = lrg_result.n_nodes

    # Compute PSI
    psi_values, n_communities = compute_partition_stability_index(linkage_matrix)

    # Get optimal partition
    optimal_clusters = fcluster(linkage_matrix, t=optimal_threshold, criterion="distance")
    n_clusters = len(np.unique(optimal_clusters))

    if verbose:
        print(f"Optimal threshold: {optimal_threshold:.6f}")
        print(f"Number of clusters: {n_clusters}")

    # Load channel labels
    try:
        from ..utils.datamanag.patient_robust import load_patient_dataset_robust

        dataset = load_patient_dataset_robust(patient, dataset_root, phases=[phase])
        recording = dataset[phase]

        if hasattr(recording, "channel_labels") and recording.channel_labels is not None:
            channel_labels = {i: label for i, label in enumerate(recording.channel_labels)}
        else:
            channel_labels = {i: f"Ch{i}" for i in range(n_nodes)}
    except Exception as e:
        if verbose:
            print(f"Could not load channel labels: {e}")
        channel_labels = {i: f"Ch{i}" for i in range(n_nodes)}

    # Create figure with custom grid layout (matching FIGMNTGN01)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.4)

    # Define subplot areas
    ax_matrix = fig.add_subplot(gs[0:2, 0:2])  # Top-left: Matrix
    ax_entropy = fig.add_subplot(gs[0:2, 2:4])  # Top-mid: Entropy/Heat
    ax_dendro = fig.add_subplot(gs[0:4, 4:6])  # Right: Dendrogram (full height)
    ax_network = fig.add_subplot(gs[2:4, 0:3])  # Bottom-left: Network
    ax_psi = fig.add_subplot(gs[2:4, 3:4])  # Bottom-mid-right: PSI

    # Overall title
    fig.suptitle(
        f"{patient} {phase} {band.upper()} - LRG Analysis ({method_label})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # -------------------------------------------------------------------------
    # Panel (a): FC Matrix Heatmap
    # -------------------------------------------------------------------------
    im_matrix = ax_matrix.imshow(
        fc_matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        origin="upper",
        aspect="equal",
    )

    ax_matrix.set_title(f"(a) {method_label} Matrix", fontsize=12, fontweight="bold")
    ax_matrix.set_xlabel("Channel", fontsize=10)
    ax_matrix.set_ylabel("Channel", fontsize=10)

    # Colorbar
    cbar_matrix = plt.colorbar(im_matrix, ax=ax_matrix, fraction=0.046, pad=0.04)
    cbar_matrix.set_label(method_label, fontsize=10)

    # -------------------------------------------------------------------------
    # Panel (b): Entropy and Specific Heat (CORRECTED - NO N multiplication)
    # -------------------------------------------------------------------------
    ax_entropy_twin = ax_entropy.twinx()

    # Plot specific heat (CORRECTED: use entropy_C directly, NO multiplication)
    line_heat = ax_entropy.plot(
        entropy_tau[1:], entropy_C, "-", label="C (Specific Heat)", color="blue", linewidth=2
    )
    ax_entropy.set_ylabel("C (Specific Heat)", color="blue", fontsize=11)
    ax_entropy.tick_params(axis="y", labelcolor="blue")

    # Plot entropy (1 - S normalized)
    line_entropy = ax_entropy_twin.plot(
        entropy_tau, entropy_1_minus_S, "-", label="1-S (Entropy)", color="red", linewidth=2
    )
    ax_entropy_twin.set_ylabel("1-S (Normalized Entropy)", color="red", fontsize=11)
    ax_entropy_twin.tick_params(axis="y", labelcolor="red")

    # Set x-axis (log scale for tau)
    ax_entropy.set_xscale("log")
    ax_entropy.set_xlabel(r"$\tau$ (Diffusion Time)", fontsize=11)
    ax_entropy.set_title("(b) Thermodynamic Observables", fontsize=12, fontweight="bold")

    # Combined legend
    lines_heat, labels_heat = ax_entropy.get_legend_handles_labels()
    lines_entropy, labels_entropy = ax_entropy_twin.get_legend_handles_labels()
    ax_entropy_twin.legend(lines_heat + lines_entropy, labels_heat + labels_entropy, loc="best", fontsize=9).set_zorder(
        200
    )

    ax_entropy.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel (c): Dendrogram (CORRECTED settings from FIGMNTGN03)
    # -------------------------------------------------------------------------
    # Generate color palette
    n_colors = min(n_clusters + 3, 20)
    palette = plt.cm.tab20(np.linspace(0, 1, n_colors))
    palette = [plt.matplotlib.colors.to_hex(c) for c in palette]

    from scipy.cluster import hierarchy

    hierarchy.set_link_color_palette(palette)

    # Create dendrogram
    node_list = list(range(n_nodes))
    labels_for_dendro = [channel_labels.get(n, f"Ch{n}") for n in node_list]

    dendro = dendrogram(
        linkage_matrix,
        ax=ax_dendro,
        color_threshold=optimal_threshold,
        labels=labels_for_dendro,
        above_threshold_color="k",
        leaf_font_size=5,
        orientation="right",
    )

    # CORRECTED: Set log scale and limits exactly as in FIGMNTGN03
    tmin = linkage_matrix[:, 2][0] * 0.8
    tmax = linkage_matrix[:, 2][-1] * 1.01
    ax_dendro.set_xscale("log")
    ax_dendro.axvline(
        optimal_threshold, color="b", linestyle="--", linewidth=2, label=r"$\mathcal{D}_{\rm th}$"
    )
    ax_dendro.set_xlim(tmin, tmax)
    ax_dendro.set_xlabel(r"$\mathcal{D}/\mathcal{D}_{\max}$", fontsize=11)
    ax_dendro.set_title("(c) Hierarchical Tree", fontsize=12, fontweight="bold")
    ax_dendro.legend(fontsize=9)

    # -------------------------------------------------------------------------
    # Panel (d): Network with Partition Colors (k=0.1 for weighted networks)
    # -------------------------------------------------------------------------
    # Extract node colors from dendrogram
    leaf_label_colors = {lbl: col for lbl, col in zip(dendro["ivl"], dendro["leaves_color_list"])}

    # Map colors to nodes
    node_colors = [leaf_label_colors.get(channel_labels.get(n, f"Ch{n}"), "gray") for n in node_list]

    # Build full graph from FC matrix
    G = nx.from_numpy_array(fc_matrix)

    # Create layout: use percolation threshold for sparse backbone if available
    if percolation_threshold is not None:
        # Create sparse backbone using percolation threshold for layout
        # Keep only edges above threshold (first detachment from giant component)
        # This reveals community structure by removing weak edges
        G_backbone = nx.Graph()
        G_backbone.add_nodes_from(G.nodes())
        for u, v in G.edges():
            if G[u][v]["weight"] >= percolation_threshold:
                G_backbone.add_edge(u, v, weight=G[u][v]["weight"])

        # Compute layout on sparse backbone (reveals hierarchical structure)
        pos = nx.spring_layout(G_backbone, seed=43, scale=1, k=0.1, iterations=50)
    else:
        # No percolation threshold available (e.g., MSC method)
        # Use full graph for layout
        pos = nx.spring_layout(G, seed=43, scale=1, k=0.1, iterations=50)

    # Draw FULL network (all edges) using backbone-based positions
    # This shows all connectivity with hierarchically-organized layout
    widths = [G[u][v]["weight"] for u, v in G.edges()]

    # Draw network
    nx.draw(
        G,  # Full graph, not MST!
        pos=pos,
        ax=ax_network,
        width=widths,
        node_color=node_colors,
        node_size=80,
        with_labels=False,
        font_size=6,
        font_color="k",
    )

    ax_network.set_title(
        f"(d) Network (Optimal Partition: {n_clusters} communities)", fontsize=12, fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Panel (e): PSI Plot
    # -------------------------------------------------------------------------
    ax_psi.plot(n_communities, psi_values, "-o", color="green", linewidth=2, markersize=4)

    # Find most stable partition from PSI
    if len(psi_values) > 0:
        max_psi_idx = np.argmax(psi_values)
        optimal_n_communities = n_communities[max_psi_idx]

        # Mark optimal partition from PSI
        ax_psi.axvline(
            optimal_n_communities,
            ls="--",
            c="red",
            linewidth=2,
            label=f"PSI optimal: n={optimal_n_communities}",
        )

        # Mark partition we used
        ax_psi.axvline(
            n_clusters, ls=":", c="blue", linewidth=2, label=f"Used: n={n_clusters}"
        )

    ax_psi.set_xlabel(r"$n$ (Number of Communities)", fontsize=11)
    ax_psi.set_ylabel(r"$\Psi(n, \tau)$ (PSI)", fontsize=11)
    ax_psi.set_title("(e) Partition Stability", fontsize=12, fontweight="bold")
    ax_psi.legend(fontsize=9)
    ax_psi.grid(alpha=0.3)
    if len(n_communities) > 0:
        ax_psi.set_xlim(1, min(20, max(n_communities)))

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    if output_path is None:
        output_dir = Path("data/figures/lrg") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_lrg_{fc_method}_full.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"Saved LRG analysis: {output_path}")

    return output_path
