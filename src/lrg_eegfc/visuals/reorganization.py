"""Phase reorganization visualizations for brain network analysis.

This module provides visualizations for analyzing how brain network structure
reorganizes across experimental phases (rsPre → taskLearn → taskTest → rsPost).
This is used to assess cognitive effects of learning tasks on multiscale
hierarchical network organization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import to_hex
from scipy.cluster.hierarchy import dendrogram, fcluster, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from lrgsglib.utils.basic.linalg import (
    ultrametric_matrix_distance,
    ultrametric_scaled_distance,
    ultrametric_rank_correlation,
    ultrametric_quantile_rmse,
    ultrametric_distance_permutation_robust,
    tree_robinson_foulds_distance,
    tree_cophenetic_correlation,
    tree_baker_gamma,
    tree_fowlkes_mallows_index,
)
from ..workflow_corr import load_corr_matrix
from ..workflow_msc import load_msc_matrix
from ..workflow_lrg import load_lrg_result
from ..utils.datamanag.patient_robust import load_patient_dataset_robust

__all__ = [
    "compute_cluster_membership_correlation",
    "plot_phase_reorganization",
    "plot_reorganization_distance_matrix",
]


def _cluster_palette(labels: np.ndarray) -> Dict[int, str]:
    unique = sorted(np.unique(labels))
    cmap = cm.get_cmap("tab20")
    return {lab: to_hex(cmap((i % 20) / 20.0)) for i, lab in enumerate(unique)}


def _link_color_func(linkage: np.ndarray, leaf_labels: np.ndarray, palette: Dict[int, str]):
    """Return a color function for dendrogram so branches match cluster colors."""
    n_leaves = len(leaf_labels)
    node_labels = [leaf_labels[i] for i in range(n_leaves)] + [None] * linkage.shape[0]

    for idx, (c1, c2, _, _) in enumerate(linkage.astype(int)):
        l1, l2 = node_labels[c1], node_labels[c2]
        node_labels[n_leaves + idx] = l1 if l1 == l2 else None

    def color_func(k: int) -> str:
        lbl = node_labels[k]
        return palette.get(lbl, "#000000") if lbl is not None else "#000000"

    return color_func


def compute_cluster_membership_correlation(
    linkage_1: np.ndarray,
    linkage_2: np.ndarray,
    threshold_1: Optional[float] = None,
    threshold_2: Optional[float] = None,
    method: str = "spearman",
) -> float:
    """Compute correlation of cluster memberships between two hierarchies.

    This measures how similarly two hierarchical clusterings partition
    the nodes into clusters.

    Parameters
    ----------
    linkage_1 : np.ndarray
        First linkage matrix
    linkage_2 : np.ndarray
        Second linkage matrix
    threshold_1 : float, optional
        Threshold for cutting first dendrogram (default: 0.5 * max_distance)
    threshold_2 : float, optional
        Threshold for cutting second dendrogram (default: 0.5 * max_distance)
    method : str, optional
        Correlation method: "spearman", "pearson", or "kendall"

    Returns
    -------
    float
        Correlation coefficient [-1, 1]
        1.0: Identical clustering (perfect agreement)
        0.0: No relationship
        -1.0: Inverse clustering
    """
    # Determine thresholds if not provided
    if threshold_1 is None:
        max_dist_1 = linkage_1[:, 2].max()
        threshold_1 = 0.5 * max_dist_1

    if threshold_2 is None:
        max_dist_2 = linkage_2[:, 2].max()
        threshold_2 = 0.5 * max_dist_2

    # Cut dendrograms to get cluster labels
    labels_1 = fcluster(linkage_1, threshold_1, criterion="distance")
    labels_2 = fcluster(linkage_2, threshold_2, criterion="distance")

    # Compute correlation or ARI
    if method == "spearman":
        corr, _ = spearmanr(labels_1, labels_2)
        return float(corr)
    elif method == "pearson":
        from scipy.stats import pearsonr

        corr, _ = pearsonr(labels_1, labels_2)
        return float(corr)
    elif method == "kendall":
        from scipy.stats import kendalltau

        corr, _ = kendalltau(labels_1, labels_2)
        return float(corr)
    elif method == "ari":
        return float(adjusted_rand_score(labels_1, labels_2))
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_phase_reorganization(
    patient: str,
    band: str,
    fc_method: str,
    phases: List[str] = None,
    cache_root: Path = Path("data/lrg_cache"),
    output_path: Optional[Path] = None,
    figsize: tuple = (34, 20),
    verbose: bool = False,
) -> Path:
    """Visualize network reorganization across experimental phases.

    Creates a compact 3x4 + 1 layout showing, for each phase:
      (row 0) Network view with cluster colors (cut at optimal threshold)
      (row 1) Ultrametric distance matrix heatmap
      (row 2) Dendrogram with optimal cut line (styled like LRG visuals)
    The extra column aggregates phase-to-phase cluster agreement.

    Parameters
    ----------
    patient : str
        Patient identifier
    band : str
        Frequency band
    fc_method : str
        FC method: "corr" or "msc"
    phases : List[str], optional
        List of phases (default: ["rsPre", "taskLearn", "taskTest", "rsPost"])
    cache_root : Path, optional
        Root directory for LRG cache
    output_path : Path, optional
        Output file path (auto-generated if None)
    figsize : tuple, optional
        Figure size
    verbose : bool, optional
        Print progress

    Returns
    -------
    Path
        Path to saved figure

    Examples
    --------
    >>> # Analyze phase reorganization for correlation-based networks
    >>> output = plot_phase_reorganization("Pat_02", "beta", "corr")
    >>> # Check cluster correlation matrix to see stability across phases
    """
    if phases is None:
        from ..config.const import PHASE_LABELS

        phases = list(PHASE_LABELS)

    if verbose:
        print(f"Loading LRG results for {patient} {band} ({fc_method})...")

    # Load all LRG results
    results = {}
    for phase in phases:
        result = load_lrg_result(patient, phase, band, fc_method, cache_root)
        if result is None:
            raise FileNotFoundError(
                f"LRG result not found for {patient} {phase} {band} ({fc_method}). "
                f"Run: python src/compute_lrg_analysis.py --patient {patient} --fc-method {fc_method}"
            )
        results[phase] = result

    if verbose:
        print("Creating visualization...")

    # Load FC matrices and channel labels for network panels
    fc_matrices: Dict[str, np.ndarray] = {}
    channel_labels: Dict[int, str] = {}
    try:
        dataset = load_patient_dataset_robust(patient, Path("data/stereoeeg_patients"), phases=phases)
        recording = next(iter(dataset.values()))
        if hasattr(recording, "channel_labels") and recording.channel_labels:
            channel_labels = {i: lbl for i, lbl in enumerate(recording.channel_labels)}
    except Exception:
        channel_labels = {}

    for phase in phases:
        if fc_method == "corr":
            fc_matrix = load_corr_matrix(
                patient, phase, band, cache_root=Path("data/corr_cache"), filter_type="abs", zero_diagonal=True
            )
        else:
            fc_matrix = load_msc_matrix(
                patient, phase, band, cache_root=Path("data/msc_cache"), sparsify="none", n_surrogates=0
            )
        if fc_matrix is None:
            raise FileNotFoundError(f"FC matrix not found for {patient} {phase} {band} ({fc_method})")
        np.fill_diagonal(fc_matrix, 0)
        fc_matrices[phase] = fc_matrix

    n_phases = len(phases)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        3,
        n_phases + 1,
        width_ratios=[1] * n_phases + [4],
        height_ratios=[1.1, 1.1, 1.3],
        hspace=0.25,
        wspace=0.25,
    )

    axes_net = [fig.add_subplot(gs[0, i]) for i in range(n_phases)]
    axes_ultra = [fig.add_subplot(gs[1, i]) for i in range(n_phases)]
    axes_dendro = [fig.add_subplot(gs[2, i]) for i in range(n_phases)]
    ax_summary = fig.add_subplot(gs[:, -1])
    ax_summary.set_position(ax_summary.get_position())  # force occupy all three rows

    # Global scales
    all_ultras = [squareform(results[p].ultrametric_matrix) for p in phases]
    ultra_vmin = min(u.min() for u in all_ultras)
    ultra_vmax = max(u.max() for u in all_ultras)
    max_height = max(results[p].linkage_matrix[:, 2].max() for p in phases)

    # Colormap for clusters
    cluster_cmap = cm.get_cmap("tab20")

    def _cluster_labels(linkage_mat: np.ndarray, threshold: float) -> np.ndarray:
        return fcluster(linkage_mat, threshold, criterion="distance")

    def _draw_network(ax, fc_matrix: np.ndarray, labels: np.ndarray, palette: Dict[int, str], title: str) -> None:
        G = nx.from_numpy_array(fc_matrix)
        weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
        if len(weights) == 0:
            ax.text(0.5, 0.5, "No edges", ha="center", va="center")
            ax.axis("off")
            return
        cutoff = np.quantile(weights, 0.9) if len(weights) > 8 else weights.min()
        backbone = nx.Graph()
        backbone.add_nodes_from(G.nodes())
        backbone.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= cutoff)
        # Spring layout on backbone, but ensure all nodes get coordinates
        pos = nx.spring_layout(backbone, seed=42, k=0.2, iterations=20)
        node_colors = [palette.get(lab, "#999999") for lab in labels]
        widths = [max(d["weight"], 1e-3) * 2 for _, _, d in G.edges(data=True)]
        nx.draw(
            G,
            pos=pos,
            ax=ax,
            node_color=node_colors,
            node_size=60,
            width=widths,
            edge_color="lightgray",
            with_labels=False,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Fill per-phase panels
    for idx, phase in enumerate(phases):
        result = results[phase]
        labels = _cluster_labels(result.linkage_matrix, result.optimal_threshold)
        palette = _cluster_palette(labels)

        # Network
        _draw_network(
            axes_net[idx],
            fc_matrices[phase],
            labels,
            palette,
            f"{phase} — Network",
        )

        # Ultrametric heatmap
        ultra_sq = all_ultras[idx]
        im_ultra = axes_ultra[idx].imshow(
            ultra_sq,
            cmap="viridis",
            vmin=ultra_vmin,
            vmax=ultra_vmax,
            interpolation="none",
            aspect="equal",
        )
        axes_ultra[idx].set_title(f"{phase} — Ultrametric", fontsize=12)
        axes_ultra[idx].set_xticks([])
        axes_ultra[idx].set_yticks([])

        # Dendrogram (styled like LRG)
        linkage = result.linkage_matrix
        try:
            linkage = optimal_leaf_ordering(linkage, result.ultrametric_matrix)
        except Exception:
            pass

        link_color = _link_color_func(linkage, labels, palette)
        dendro_obj = dendrogram(
            linkage,
            ax=axes_dendro[idx],
            no_labels=True,
            color_threshold=0,  # use custom coloring
            above_threshold_color="black",
            link_color_func=link_color,
        )
        axes_dendro[idx].axhline(
            result.optimal_threshold,
            color="blue",
            linestyle="--",
            lw=2,
            alpha=0.7,
        )
        # Match LRG dendrogram scaling (log, tight bounds)
        heights = linkage[:, 2]
        tmin = max(heights[0] * 0.8, 1e-9)
        tmax = heights[-1] * 1.01
        axes_dendro[idx].set_ylim(tmin, tmax)
        axes_dendro[idx].set_yscale("log")
        axes_dendro[idx].set_title(f"{phase} — Hierarchy", fontsize=12)
        axes_dendro[idx].set_ylabel("Ultrametric Distance", fontsize=10)
        axes_dendro[idx].grid(alpha=0.2)

    # Shared colorbar for ultrametric heatmaps
    fig.colorbar(
        im_ultra,
        ax=axes_ultra,
        orientation="horizontal",
        fraction=0.05,
        pad=0.1,
        label="Ultrametric Distance",
    )

    # Summary: cluster membership correlation
    n = len(phases)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr = compute_cluster_membership_correlation(
                results[phases[i]].linkage_matrix,
                results[phases[j]].linkage_matrix,
                results[phases[i]].optimal_threshold,
                results[phases[j]].optimal_threshold,
                method="ari",
            )
            corr_matrix[i, j] = corr_matrix[j, i] = corr

    im_corr = ax_summary.imshow(
        corr_matrix,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="none",
        aspect="equal",
    )
    ax_summary.set_xticks(range(n))
    ax_summary.set_yticks(range(n))
    ax_summary.set_xticklabels(phases, fontsize=11, rotation=45, ha="right")
    ax_summary.set_yticklabels(phases, fontsize=11)
    ax_summary.set_title("Cluster Agreement", fontsize=13, fontweight="bold")
    # Mask lower triangle for clarity
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
    corr_masked = np.ma.masked_where(mask, corr_matrix)
    cmap_corr = plt.get_cmap("RdBu_r").copy()
    cmap_corr.set_bad(color="none")
    im_corr.set_data(corr_masked)
    im_corr.set_cmap(cmap_corr)
    ax_summary.set_box_aspect(1)

    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax_summary.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    divider = make_axes_locatable(ax_summary)
    cax = divider.append_axes("bottom", size="6%", pad=0.4)
    plt.colorbar(im_corr, cax=cax, orientation="horizontal", label="Correlation")

    # Overall title
    fig.suptitle(
        f"Phase Reorganization Analysis - {patient} {band} ({fc_method})",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/reorganization") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir / f"{band}_{fc_method}_phase_reorganization.png"
        )
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    # Save figure
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"✓ Saved: {output_path}")

    return output_path


def plot_reorganization_distance_matrix(
    patient: str,
    band: str,
    fc_method: str,
    phases: List[str] = None,
    cache_root: Path = Path("data/lrg_cache"),
    output_path: Optional[Path] = None,
    figsize: tuple = (22, 22),
    verbose: bool = False,
) -> Path:
    """Plot pairwise distance matrix between all phase combinations.

    Creates a comprehensive matrix showing all distance metrics between
    every pair of experimental phases.

    Parameters
    ----------
    patient : str
        Patient identifier
    band : str
        Frequency band
    fc_method : str
        FC method: "corr" or "msc"
    phases : List[str], optional
        List of phases
    cache_root : Path, optional
        Root directory for LRG cache
    output_path : Path, optional
        Output file path
    figsize : tuple, optional
        Figure size
    verbose : bool, optional
        Print progress

    Returns
    -------
    Path
        Path to saved figure
    """
    if phases is None:
        from ..config.const import PHASE_LABELS

        phases = list(PHASE_LABELS)

    if verbose:
        print(f"Loading LRG results for {patient} {band} ({fc_method})...")

    # Load all LRG results
    results = {}
    for phase in phases:
        result = load_lrg_result(patient, phase, band, fc_method, cache_root)
        if result is None:
            raise FileNotFoundError(
                f"LRG result not found for {patient} {phase} {band} ({fc_method})"
            )
        results[phase] = result

    if verbose:
        print("Computing pairwise distances...")

    # Compute all pairwise distances once with full metric suite
    n = len(phases)
    metric_keys = [
        ("ultrametric_matrix_distance", "Ultrametric Matrix Distance", "YlOrRd", 0.0, None, False),
        ("ultrametric_scaled_distance_log", "Ultrametric Scaled Distance (log)", "YlOrRd", 0.0, None, False),
        ("ultrametric_rank_correlation_spearman", "Ultrametric Rank Distance (1 - Spearman)", "YlOrRd", 0.0, None, False),
        ("ultrametric_quantile_rmse_log", "Ultrametric Quantile RMSE (log)", "YlOrRd", 0.0, None, False),
        ("ultrametric_distance_permutation_robust", "Permutation-Robust Ultrametric Distance", "YlOrRd", 0.0, None, False),
        ("tree_robinson_foulds", "Robinson–Foulds Distance", "YlOrRd", 0.0, None, False),
        ("tree_cophenetic_distance", "Cophenetic Distance (1 - corr)", "YlOrRd", 0.0, None, False),
        ("tree_baker_gamma_distance", "Baker's Gamma Distance (1 - gamma)", "YlOrRd", 0.0, None, False),
        ("tree_fowlkes_mallows_distance", "Fowlkes–Mallows Distance (1 - fm)", "YlOrRd", 0.0, None, False),
    ]

    metric_mats: Dict[str, np.ndarray] = {}
    for key, _, _, _, _, _ in metric_keys:
        mat = np.full((n, n), np.nan, dtype=float)
        np.fill_diagonal(mat, 0.0)
        metric_mats[key] = mat

    def compute_pair(phase1: str, phase2: str) -> Dict[str, float]:
        r1, r2 = results[phase1], results[phase2]
        U1 = squareform(r1.ultrametric_matrix) if r1.ultrametric_matrix.ndim == 1 else r1.ultrametric_matrix
        U2 = squareform(r2.ultrametric_matrix) if r2.ultrametric_matrix.ndim == 1 else r2.ultrametric_matrix
        Z1, Z2 = r1.linkage_matrix, r2.linkage_matrix

        vals = {}
        try:
            vals["ultrametric_matrix_distance"] = ultrametric_matrix_distance(U1, U2, metric="euclidean")
        except Exception:
            vals["ultrametric_matrix_distance"] = np.nan
        try:
            vals["ultrametric_scaled_distance_log"] = ultrametric_scaled_distance(U1, U2, metric="euclidean", scale="log", normalize=True)
        except Exception:
            vals["ultrametric_scaled_distance_log"] = np.nan
        try:
            rc = ultrametric_rank_correlation(U1, U2, method="spearman")
            vals["ultrametric_rank_correlation_spearman"] = 1.0 - rc if np.isfinite(rc) else np.nan
        except Exception:
            vals["ultrametric_rank_correlation_spearman"] = np.nan
        try:
            vals["ultrametric_quantile_rmse_log"] = ultrametric_quantile_rmse(U1, U2, scale="log")
        except Exception:
            vals["ultrametric_quantile_rmse_log"] = np.nan
        try:
            vals["ultrametric_distance_permutation_robust"] = ultrametric_distance_permutation_robust(
                Z1, Z2, None, None, None, metric="euclidean"
            )
        except Exception:
            vals["ultrametric_distance_permutation_robust"] = np.nan
        try:
            vals["tree_robinson_foulds"] = tree_robinson_foulds_distance(Z1, Z2, normalized=True)
        except Exception:
            vals["tree_robinson_foulds"] = np.nan
        try:
            cc = tree_cophenetic_correlation(Z1, Z2)
            vals["tree_cophenetic_distance"] = 1.0 - cc if np.isfinite(cc) else np.nan
        except Exception:
            vals["tree_cophenetic_distance"] = np.nan
        try:
            bg = tree_baker_gamma(Z1, Z2)
            vals["tree_baker_gamma_distance"] = 1.0 - bg if np.isfinite(bg) else np.nan
        except Exception:
            vals["tree_baker_gamma_distance"] = np.nan
        try:
            fm = tree_fowlkes_mallows_index(Z1, Z2)
            vals["tree_fowlkes_mallows_distance"] = 1.0 - fm if np.isfinite(fm) else np.nan
        except Exception:
            vals["tree_fowlkes_mallows_distance"] = np.nan
        return vals

    for i, phase1 in enumerate(phases):
        for j in range(i + 1, n):
            vals = compute_pair(phase1, phases[j])
            for key in metric_mats:
                val = vals.get(key, np.nan)
                if not np.isfinite(val):
                    continue
                metric_mats[key][i, j] = metric_mats[key][j, i] = val

    if verbose:
        print("Creating visualization...")

    n_metrics = len(metric_keys)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax in axes[n_metrics:]:
        ax.axis("off")

    for idx, (key, label, cmap, vmin_default, vmax_default, diverging) in enumerate(metric_keys):
        mat = metric_mats[key]
        finite_vals = mat[np.isfinite(mat) & ~np.eye(n, dtype=bool)]
        if finite_vals.size:
            data_min, data_max = float(finite_vals.min()), float(finite_vals.max())
        else:
            data_min = vmin_default if vmin_default is not None else 0.0
            data_max = vmax_default if vmax_default is not None else 1.0

        if diverging:
            max_abs = max(abs(data_min), abs(data_max), 1e-9)
            vmin, vmax = -max_abs, max_abs
        else:
            vmin = data_min if vmin_default is None else vmin_default
            vmax = data_max if vmax_default is None else vmax_default

        im = axes[idx].imshow(
            np.ma.masked_where(~np.triu(np.ones_like(mat, dtype=bool), k=0), mat),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
            aspect="equal",
        )
        axes[idx].set_title(label, fontsize=12, fontweight="bold")
        axes[idx].set_xticks(range(n))
        axes[idx].set_yticks(range(n))
        axes[idx].set_xticklabels(phases, fontsize=10, rotation=45, ha="right")
        axes[idx].set_yticklabels(phases, fontsize=10)

        mask_upper = np.triu(np.ones_like(mat, dtype=bool), k=0)
        for i in range(n):
            for j in range(n):
                if not mask_upper[i, j]:
                    continue
                if not np.isfinite(mat[i, j]):
                    continue
                color = "white" if (diverging and abs(mat[i, j]) > 0.5) or (not diverging and mat[i, j] > (vmax if vmax is not None else 0) * 0.6) else "black"
                axes[idx].text(
                    j,
                    i,
                    f"{mat[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Phase-to-Phase Distance Matrix - {patient} {band} ({fc_method})",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/reorganization") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{fc_method}_distance_matrix.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"✓ Saved: {output_path}")

    return output_path
