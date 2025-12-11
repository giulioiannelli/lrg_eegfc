"""Correlation-based functional connectivity visualization functions.

This module provides visualization functions for correlation matrices and networks,
following patterns from ipynb/pat02_corrnet_bands.ipynb and ipynb/poster01.ipynb.
"""

from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from lrgsglib.utils.basic.probability import marchenko_pastur
from lrgsglib.nx_patches.funcs.thresholding import compute_threshold_stats

from lrg_eegfc import load_corr_matrix, get_corr_cache_path
from lrg_eegfc.workflow_cleaning import load_cleaned_corr_matrix
from lrg_eegfc.workflow_corr import compute_corr_matrix
from lrg_eegfc.utils.datamanag.patient import load_patient_metadata
from lrg_eegfc.utils.corrmat.thresholds import find_exact_detachment_threshold, find_threshold_jumps


def imshow_colorbar_caxdivider(im, ax, position='right', orientation='vertical', size='5%', pad=0.05):
    """Add colorbar to imshow plot using axes divider.

    Pattern from: lrgsglib plotting utilities
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    clb = plt.colorbar(im, cax=cax, orientation=orientation)
    return divider, cax, clb


def _load_or_compute_corr(
    patient: str,
    phase: str,
    band: str,
    *,
    cache_root: Path,
    dataset_root: Path,
    filter_type: str,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """Load cached correlation matrix or compute and cache it on demand."""
    matrix = load_corr_matrix(
        patient,
        phase,
        band,
        cache_root=cache_root,
        filter_type=filter_type,
        zero_diagonal=zero_diagonal,
    )

    if matrix is not None:
        return matrix

    result = compute_corr_matrix(
        patient,
        phase,
        band,
        dataset_root=dataset_root,
        cache_root=cache_root,
        use_cache=True,
        overwrite_cache=False,
        filter_type=filter_type,
        zero_diagonal=zero_diagonal,
    )
    return result.adjacency_matrix


def _get_channel_labels(patient: str, dataset_root: Path, n_channels: int) -> dict:
    """Return channel labels if available, otherwise integer labels."""
    try:
        patient_meta = load_patient_metadata(patient, dataset_root)
        labels = patient_meta.get("channel_labels", {})
        # Ensure integer keys for indexing
        return {int(k): v for k, v in labels.items()}
    except Exception:
        return {i: str(i) for i in range(n_channels)}


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    output_path: Path,
    title: str = "Correlation Matrix",
    vmin: float = -1.0,
    vmax: float = 1.0,
    channel_labels: Optional[List[str]] = None,
    figsize: tuple = (8, 8),
) -> Path:
    """Plot correlation matrix heatmap with colorbar.

    Pattern from: ipynb/pat02_corrnet_bands.ipynb
    - Uses viridis colormap
    - Adds colorbar with divider
    - No axis labels if >50 channels

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix to visualize
    output_path : Path
        Path to save the figure
    title : str, optional
        Title for the plot
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
    channel_labels : Optional[List[str]], optional
        Channel labels for axes
    figsize : tuple, optional
        Figure size

    Returns
    -------
    Path
        Path to the saved figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    kwargs_imshow = dict(cmap='viridis', interpolation='none', vmin=vmin, vmax=vmax)
    im = ax.imshow(corr_matrix, **kwargs_imshow)

    # Add colorbar
    divider, cax, clb = imshow_colorbar_caxdivider(im, ax)

    # Only show channel labels if not too many channels
    n_channels = corr_matrix.shape[0]
    if channel_labels is not None and n_channels <= 50:
        ax.set_xticks(range(n_channels))
        ax.set_yticks(range(n_channels))
        ax.set_xticklabels(channel_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(channel_labels, fontsize=8)
    else:
        ax.axis('off')

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_correlation_summary(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/corr_cache"),
    output_path: Optional[Path] = None,
    dataset_root: Path = Path("data/stereoeeg_patients"),
    figsize: tuple = (14, 10),
) -> Path:
    """Build a single figure with raw, |corr|, percolation/threshold, and network.

    Panels:
        A: Raw signed correlation matrix (diverging colormap, centered at 0)
        B: Absolute-value correlation matrix
        C: Thresholded matrix (mask via percolation threshold) with inset percolation curves
        D: Network graph built from the thresholded matrix (edges scaled by |weight|)
    """
    # Resolve output path
    patient_dir = Path("data/figures/correlation") / patient
    patient_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = patient_dir / f"{band}_{phase}_summary.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    raw_matrix = _load_or_compute_corr(
        patient,
        phase,
        band,
        cache_root=cache_root,
        dataset_root=dataset_root,
        filter_type="none",
        zero_diagonal=True,
    )
    abs_matrix = _load_or_compute_corr(
        patient,
        phase,
        band,
        cache_root=cache_root,
        dataset_root=dataset_root,
        filter_type="abs",
        zero_diagonal=True,
    )

    # Ensure diagonals are zeroed for visualization and network construction
    np.fill_diagonal(raw_matrix, 0)
    np.fill_diagonal(abs_matrix, 0)

    # Compute percolation stats on the cached abs-filtered matrix
    G_perc = nx.from_numpy_array(abs_matrix)

    threshold = None
    detachment_idx = None
    Pinf = None
    Einf = None
    Th = None

    try:
        Th, jumps, Einf, Pinf = find_threshold_jumps(G_perc, return_stats=True)
        if len(jumps) > 0:
            detachment_idx = int(jumps[0])
            threshold = float(Th[detachment_idx - 1] if detachment_idx > 0 else Th[detachment_idx])
        else:
            # No jump detected: use robust fallback
            triu = np.triu_indices_from(abs_matrix, k=1)
            threshold = float(np.median(abs_matrix[triu]))
    except Exception:
        # Fallback if percolation computation fails
        triu = np.triu_indices_from(abs_matrix, k=1)
        threshold = float(np.median(abs_matrix[triu]))

    # Apply threshold to the abs-filtered matrix
    thresholded_matrix = abs_matrix.copy()
    thresholded_matrix[thresholded_matrix < threshold] = 0
    np.fill_diagonal(thresholded_matrix, 0)

    # Channel labels
    n_channels = raw_matrix.shape[0]
    channel_labels = _get_channel_labels(patient, dataset_root, n_channels)
    small_graph = n_channels <= 50

    # Color limits
    vmax_raw = float(np.max(np.abs(raw_matrix))) if raw_matrix.size else 1.0
    vmax_raw = vmax_raw if vmax_raw > 0 else 1.0
    vmin_raw = -vmax_raw

    vmax_abs = float(np.max(abs_matrix)) if abs_matrix.size else 1.0
    vmax_abs = vmax_abs if vmax_abs > 0 else 1.0

    # Figure layout
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel A: Raw correlation
    im0 = axes[0, 0].imshow(
        raw_matrix,
        cmap="coolwarm",
        interpolation="none",
        vmin=vmin_raw,
        vmax=vmax_raw,
    )
    imshow_colorbar_caxdivider(im0, axes[0, 0])
    if small_graph:
        axes[0, 0].set_xticks(range(n_channels))
        axes[0, 0].set_yticks(range(n_channels))
        labels = [channel_labels.get(i, str(i)) for i in range(n_channels)]
        axes[0, 0].set_xticklabels(labels, rotation=90, fontsize=7)
        axes[0, 0].set_yticklabels(labels, fontsize=7)
    else:
        axes[0, 0].axis('off')
    axes[0, 0].set_title(f"{patient} {phase} {band} — raw corr")

    # Panel B: Absolute value correlation
    im1 = axes[0, 1].imshow(
        abs_matrix,
        cmap="magma",
        interpolation="none",
        vmin=0.0,
        vmax=vmax_abs,
    )
    imshow_colorbar_caxdivider(im1, axes[0, 1])
    if small_graph:
        axes[0, 1].set_xticks(range(n_channels))
        axes[0, 1].set_yticks(range(n_channels))
        labels = [channel_labels.get(i, str(i)) for i in range(n_channels)]
        axes[0, 1].set_xticklabels(labels, rotation=90, fontsize=7)
        axes[0, 1].set_yticklabels(labels, fontsize=7)
    else:
        axes[0, 1].axis('off')
    axes[0, 1].set_title("|corr| (abs)")

    # Panel C: Thresholded matrix (abs) with percolation inset
    im2 = axes[1, 0].imshow(
        thresholded_matrix,
        cmap="magma",
        interpolation="none",
        vmin=0.0,
        vmax=vmax_abs,
    )
    imshow_colorbar_caxdivider(im2, axes[1, 0])
    axes[1, 0].set_title(f"Thresholded @ θ≈{threshold:.3f}")
    axes[1, 0].axis('off')

    # Percolation inset
    inset_width = "48%"
    inset_height = "48%"
    ax_inset = inset_axes(axes[1, 0], width=inset_width, height=inset_height, loc="upper right")
    if Th is not None and Pinf is not None and Einf is not None:
        ax_inset.plot(Th, Pinf, 'k-', lw=1.5, label='P_inf')
        ax_inset.plot(Th, Einf, color='C0', lw=1.3, label='E_inf')
        ax_inset.axvline(threshold, color='r', linestyle='--', lw=1)
        if detachment_idx is not None:
            ax_inset.plot(
                Th[detachment_idx],
                Pinf[detachment_idx],
                marker='o',
                color='r',
                markersize=4,
                label='first jump',
            )
        ax_inset.set_xlabel("θ", fontsize=8)
        ax_inset.set_ylabel("Fraction", fontsize=8)
        ax_inset.tick_params(axis='both', labelsize=7)
        ax_inset.grid(alpha=0.3)
    else:
        ax_inset.text(
            0.5,
            0.5,
            "Percolation\nunavailable",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax_inset.transAxes,
        )
        ax_inset.axis('off')

    # Panel D: Network from thresholded matrix
    G_net = nx.from_numpy_array(thresholded_matrix)
    axes[1, 1].set_title("Network (thresholded)")
    if G_net.number_of_edges() == 0:
        axes[1, 1].text(
            0.5,
            0.5,
            "No edges above threshold",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].axis('off')
    else:
        # Use edge weights directly (already in [0, 1] range)
        widths = [G_net[u][v]['weight'] for u, v in G_net.edges()]
        edge_colors = ['gray' for _ in G_net.edges()]

        pos = nx.spring_layout(G_net, seed=42)
        node_labels = {i: channel_labels.get(i, str(i)) for i in G_net.nodes()}
        nx.draw(
            G_net,
            pos=pos,
            ax=axes[1, 1],
            node_size=100,
            width=widths,
            with_labels=True if n_channels <= 30 else False,
            labels=node_labels,
            font_size=8,
            node_color='lightblue',
            edge_color=edge_colors,
        )
        axes[1, 1].axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_correlation_and_network(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/corr_cache"),
    output_path: Optional[Path] = None,
    cleaned: bool = False,
    figsize: tuple = (16, 8),
    dataset_root: Path = Path("data/stereoeeg_patients"),
) -> Path:
    """Side-by-side: correlation matrix + network graph with edge weights.

    Pattern from: ipynb/pat02_corrnet_bands.ipynb (cell f9bd5d3c)

    Left panel: Heatmap of correlation matrix
    Right panel: Network graph with:
        - Edge widths proportional to correlation strength
        - widths = [G[u][v]['weight'] for u, v in G.edges()]
        - Scaled to visual range [0.05, 0.35]
        - Node labels from channel mapping

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    cache_root : Path, optional
        Root directory for correlation cache
    output_path : Optional[Path], optional
        Path to save figure (auto-generated if None)
    cleaned : bool, optional
        Whether to use cleaned correlation matrix
    figsize : tuple, optional
        Figure size
    dataset_root : Path, optional
        Root directory for patient data

    Returns
    -------
    Path
        Path to the saved figure
    """
    # Load correlation matrix
    if cleaned:
        corr_matrix, metadata = load_cleaned_corr_matrix(
            patient, phase, band, cache_root, load_metadata=True
        )
    else:
        corr_matrix = load_corr_matrix(patient, phase, band, cache_root)

    # Load channel labels
    try:
        patient_meta = load_patient_metadata(patient, dataset_root)
        channel_labels = patient_meta['channel_labels']
    except:
        # Fallback to integer labels
        n_channels = corr_matrix.shape[0]
        channel_labels = {i: str(i) for i in range(n_channels)}

    # Generate output path if not provided
    if output_path is None:
        suffix = "_cleaned" if cleaned else ""
        output_dir = Path("data/figures/correlation") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_network{suffix}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Left panel: Correlation matrix heatmap
    vmin, vmax = corr_matrix.min(), corr_matrix.max()
    kwargs_imshow = dict(cmap='viridis', interpolation='none', vmin=vmin, vmax=vmax)
    im = ax[0].imshow(corr_matrix, **kwargs_imshow)
    divider, cax, clb = imshow_colorbar_caxdivider(im, ax[0])

    n_channels = corr_matrix.shape[0]
    if n_channels <= 50:
        ax[0].set_xticks(range(n_channels))
        ax[0].set_yticks(range(n_channels))
        labels = [channel_labels.get(i, str(i)) for i in range(n_channels)]
        ax[0].set_xticklabels(labels, rotation=90, fontsize=8)
        ax[0].set_yticklabels(labels, fontsize=8)
    else:
        ax[0].axis('off')

    title_suffix = " (Cleaned)" if cleaned else ""
    ax[0].set_title(f"{patient} {phase} {band}{title_suffix}")

    # Right panel: Network graph
    G = nx.from_numpy_array(corr_matrix)

    # Use edge weights directly (already in [0, 1] range)
    widths = [G[u][v]['weight'] for u, v in G.edges()]

    # Create node labels dict
    node_labels = {i: channel_labels.get(i, str(i)) for i in G.nodes()}

    # Draw network
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos=pos,
        ax=ax[1],
        node_size=100,
        width=widths,
        with_labels=True if n_channels <= 30 else False,
        labels=node_labels,
        font_size=8,
        node_color='lightblue',
        edge_color='gray',
    )
    ax[1].set_title("Network Graph")
    ax[1].axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_marchenko_pastur_comparison(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/corr_cache"),
    output_path: Optional[Path] = None,
    figsize: tuple = (18, 6),
    dataset_root: Path = Path("data/stereoeeg_patients"),
) -> Path:
    """Plot eigenvalue histogram vs Marchenko-Pastur distribution.

    Pattern from: ipynb/pat02_corrnet_bands.ipynb (cell 49e51f2b)

    3-panel layout:
    Panel 0: Original correlation matrix heatmap
    Panel 1: Cleaned correlation matrix heatmap
    Panel 2: Eigenvalue comparison:
        - Empirical eigenvalue histogram (original)
        - Theoretical MP curve: marchenko_pastur(bins, gamma)
        - Cleaned eigenvalue histogram
        - Vertical lines at lambda_min, lambda_max (red/blue dashed)
        - Shaded gray region between lambda_min and lambda_max
        - Log scale on y-axis, symlog on x-axis

    Key calculations:
    - gamma = time_steps / n_channels
    - lambda_min = (1 - sqrt(1/gamma))^2
    - lambda_max = (1 + sqrt(1/gamma))^2
    - MP_dist = marchenko_pastur(bins, gamma)

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    cache_root : Path, optional
        Root directory for correlation cache
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
    # Load original and cleaned matrices
    corr_matrix = load_corr_matrix(patient, phase, band, cache_root)

    try:
        cleaned_matrix, metadata = load_cleaned_corr_matrix(
            patient, phase, band, cache_root, load_metadata=True
        )
        has_cleaned = True
    except (FileNotFoundError, KeyError):
        cleaned_matrix = None
        metadata = None
        has_cleaned = False

    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("data/figures/correlation") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_mp_comparison.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute eigenvalues
    eigvals_original = np.linalg.eigvalsh(corr_matrix)

    # Compute Marchenko-Pastur parameters
    n_channels = corr_matrix.shape[0]

    # Try to get time_steps from metadata, otherwise estimate
    if has_cleaned and metadata is not None and 'n_channels' in metadata:
        # Metadata should have aspect ratio info
        gamma = metadata.get('gamma', 1.0)
        lambda_min = metadata.get('lambda_min', 0.0)
        lambda_max = metadata.get('lambda_max', 2.0)
    else:
        # Estimate gamma from eigenvalue spread (rough approximation)
        # For typical EEG: time_steps ~ 1000-10000, n_channels ~ 100
        gamma = 100.0  # Conservative estimate
        lambda_min = (1 - np.sqrt(1/gamma))**2
        lambda_max = (1 + np.sqrt(1/gamma))**2

    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # Panel 0: Original matrix
    if has_cleaned and cleaned_matrix is not None:
        vmin = min(corr_matrix.min(), cleaned_matrix.min())
        vmax = max(corr_matrix.max(), cleaned_matrix.max())
    else:
        vmin = corr_matrix.min()
        vmax = corr_matrix.max()
    kwargs_imshow = dict(cmap='viridis', interpolation='none', vmin=vmin, vmax=vmax)

    im = ax[0].imshow(corr_matrix, **kwargs_imshow)
    divider, cax, clb = imshow_colorbar_caxdivider(im, ax[0])
    ax[0].axis('off')
    ax[0].set_title("Original")

    # Panel 1: Cleaned matrix
    if has_cleaned and cleaned_matrix is not None:
        im = ax[1].imshow(cleaned_matrix, **kwargs_imshow)
        divider, cax, clb = imshow_colorbar_caxdivider(im, ax[1])
        ax[1].axis('off')
        ax[1].set_title("Cleaned")

        eigvals_cleaned = np.linalg.eigvalsh(cleaned_matrix)
    else:
        ax[1].text(0.5, 0.5, "No cleaned\nmatrix available",
                   ha='center', va='center', transform=ax[1].transAxes,
                   fontsize=12)
        ax[1].axis('off')
        eigvals_cleaned = None

    # Panel 2: Eigenvalue comparison
    counts, bins = np.histogram(eigvals_original, density=True, bins=500)

    # Compute Marchenko-Pastur distribution
    MP_dist = marchenko_pastur(bins, gamma)

    # Plot
    ax[2].plot(bins, MP_dist, label='Marchenko-Pastur', lw=2, color='orange')
    ax[2].plot(bins[1:], counts, label='Empirical', lw=2, color='blue', alpha=0.7)

    if has_cleaned and eigvals_cleaned is not None:
        counts2, bins2 = np.histogram(eigvals_cleaned, density=True, bins=1000)
        ax[2].plot(bins2[1:], counts2, label='Cleaned', lw=2, color='green', alpha=0.7)

    # Add lambda_min and lambda_max lines
    ax[2].axvline(x=lambda_min, color='red', linestyle='--', label='lambda_min', lw=1.5)
    ax[2].axvline(x=lambda_max, color='blue', linestyle='--', label='lambda_max', lw=1.5)

    # Shade noise region
    ax[2].fill_between(
        bins, 0, 1,
        where=(bins > lambda_min) & (bins < lambda_max),
        color='gray', alpha=0.3,
        transform=ax[2].get_xaxis_transform()
    )

    ax[2].set_yscale('log')
    ax[2].set_xscale('symlog')
    ax[2].set_xlim(left=0)
    ax[2].set_xlabel('Eigenvalue')
    ax[2].set_ylabel('Density')
    ax[2].set_title(f"{patient} {phase} {band} - Eigenvalue Distribution")
    ax[2].legend(fontsize=8)
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_percolation_curves(
    patient: str,
    phase: str,
    band: str,
    cache_root: Path = Path("data/corr_cache"),
    output_path: Optional[Path] = None,
    cleaned: bool = False,
    figsize: tuple = (10, 7),
    dataset_root: Path = Path("data/stereoeeg_patients"),
) -> Path:
    """Plot P_inf and E_inf vs threshold θ.

    Pattern from: ipynb/poster01.ipynb (cell ad38bec7)

    Shows:
    - P_inf (fraction of nodes in giant component) vs θ
        - Plotted with 'kH' markers (black hexagons), ms=15, hollow
    - E_inf (fraction of edges in giant component) vs θ
        - Plotted with blue line, lw=3
    - Vertical red dashed lines at percolation jumps (node detachments)
    - Annotations showing which nodes detached at each jump
        - Rotated 90°, bbox with rounded corners
        - Arrow pointing to jump location

    Algorithm:
    1. Load correlation matrix (cleaned or original)
    2. Build network G = nx.from_numpy_array(C)
    3. Compute Th, Einf, Pinf = compute_threshold_stats(G)
    4. Find jumps: np.where(np.diff(Pinf) != 0)[0]
    5. For each jump, identify detached nodes:
        - Filter graph at Th[jump]
        - Find giant component
        - Nodes not in giant = detached nodes
    6. Annotate with channel labels

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    cache_root : Path, optional
        Root directory for correlation cache
    output_path : Optional[Path], optional
        Path to save figure (auto-generated if None)
    cleaned : bool, optional
        Whether to use cleaned correlation matrix
    figsize : tuple, optional
        Figure size
    dataset_root : Path, optional
        Root directory for patient data

    Returns
    -------
    Path
        Path to the saved figure
    """
    # Load correlation matrix
    if cleaned:
        corr_matrix, metadata = load_cleaned_corr_matrix(
            patient, phase, band, cache_root, load_metadata=True
        )
    else:
        corr_matrix = load_corr_matrix(patient, phase, band, cache_root)

    # Load channel labels
    try:
        patient_meta = load_patient_metadata(patient, dataset_root)
        channel_labels = patient_meta['channel_labels']
    except:
        # Fallback to integer labels
        n_channels = corr_matrix.shape[0]
        channel_labels = {i: str(i) for i in range(n_channels)}

    # Generate output path if not provided
    if output_path is None:
        suffix = "_cleaned" if cleaned else ""
        output_dir = Path("data/figures/correlation") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_percolation{suffix}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build network
    G = nx.from_numpy_array(corr_matrix)

    # Compute percolation statistics
    Th, Einf, Pinf = compute_threshold_stats(G)

    # Find jumps in Pinf (node detachments)
    Pinf_diff = np.diff(Pinf)
    jumps = np.where(Pinf_diff != 0)[0]

    # Identify nodes that detach at each jump
    not_in_giant = {}
    for i, jump in enumerate(jumps):
        C_tmp = corr_matrix.copy()
        C_tmp[C_tmp < Th[jump] * 1.1] = 0
        G_tmp = nx.from_numpy_array(C_tmp)

        # Find giant component
        if G_tmp.number_of_nodes() > 0:
            try:
                Gcc = max(nx.connected_components(G_tmp), key=len)
                # Nodes not in giant component
                not_in_giant[i] = set(G_tmp.nodes()) - Gcc

                # Remove nodes that detached in previous jumps
                if i > 0:
                    not_in_giant[i] = not_in_giant[i] - not_in_giant[i-1]
            except:
                not_in_giant[i] = set()
        else:
            not_in_giant[i] = set()

    # Create figure
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    # Plot P_inf and E_inf
    maxlabels = 5
    ax.plot(Th, Pinf, 'kH', ms=15, mec='k', mfc='none', label='P_inf')
    ax.plot(Th, Einf, lw=3, color='b', label='E_inf')

    # Add vertical lines and annotations at jumps
    for i, jump in enumerate(jumps):
        ax.axvline(Th[jump], color='r', linestyle='--', lw=1)

        # Create label string from detached nodes
        lablist = [str(channel_labels.get(n, str(n))) for n in not_in_giant[i]]
        if len(lablist) < maxlabels:
            not_in_giant_str = ', '.join(lablist)
        else:
            not_in_giant_str = ', '.join(lablist[:maxlabels]) + '...'

        # Annotate
        ax.annotate(
            not_in_giant_str,
            xy=(Th[jump], Pinf[jump]),
            xytext=(Th[jump], Pinf[jump] + 0.1 if Pinf[jump] < 0.5 else Pinf[jump] - 0.03),
            rotation=90,
            fontsize=10,
            ha='center',
            va='bottom' if Pinf[jump] < 0.5 else 'top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'),
            arrowprops=dict(arrowstyle="->", lw=1.5, color='black')
        )

    ax.set_xlabel('Threshold θ', fontsize=14)
    ax.set_ylabel('Fraction', fontsize=14)
    ax.set_title(f"{patient} {phase} {band} - Percolation Analysis", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_path
