"""Comparison visualizations for functional connectivity methods."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lrg_eegfc.constants import BRAIN_BANDS
from lrg_eegfc.workflow_corr import load_corr_matrix
from lrg_eegfc.workflow_msc import load_msc_matrix

__all__ = ["plot_fc_comparison"]


def _validate_matrices(corr_matrix: np.ndarray, msc_matrix: np.ndarray, band: str) -> None:
    """Validate shapes, symmetry, and band availability."""
    if band not in BRAIN_BANDS:
        raise ValueError(f"Unknown band '{band}'. Available bands: {list(BRAIN_BANDS.keys())}")

    if corr_matrix.shape != msc_matrix.shape:
        raise ValueError(
            f"Shape mismatch between correlation {corr_matrix.shape} and MSC {msc_matrix.shape}"
        )

    if not (np.allclose(corr_matrix, corr_matrix.T) and np.allclose(msc_matrix, msc_matrix.T)):
        raise ValueError("Both FC matrices must be symmetric.")


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle values excluding the diagonal."""
    n = matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    return matrix[triu_idx]


def plot_fc_comparison(
    patient: str,
    phase: str,
    band: str,
    cache_root_corr: Path = Path("data/corr_cache"),
    cache_root_msc: Path = Path("data/msc_cache"),
    output_path: Optional[Path] = None,
    nperseg: int = 1024,
    figsize: tuple = (18, 6),
) -> Path:
    """Create comparison visualization between Correlation and MSC FC methods.

    Generates a 3-panel horizontal layout comparing two FC methods:
    - Panel 0: Correlation FC matrix (absolute value)
    - Panel 1: MSC FC matrix
    - Panel 2: Method agreement analysis (scatter + distributions)

    Parameters
    ----------
    patient : str
        Patient identifier (e.g., "Pat_02")
    phase : str
        Recording phase (e.g., "rsPre", "taskLearn")
    band : str
        Frequency band (e.g., "beta", "alpha")
    cache_root_corr : Path
        Root directory for correlation cache
    cache_root_msc : Path
        Root directory for MSC cache
    output_path : Path, optional
        Output file path. If None, uses default location
    nperseg : int
        Window length for MSC (must match cached files)
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    Path
        Path to saved figure

    Raises
    ------
    FileNotFoundError
        If cache files not found for either method
    ValueError
        If matrices are incompatible
    """
    # Load correlation matrix (abs will be taken later)
    corr_matrix = load_corr_matrix(
        patient,
        phase,
        band,
        cache_root=cache_root_corr,
        filter_type="abs",
        zero_diagonal=True,
    )
    if corr_matrix is None:
        raise FileNotFoundError(
            f"Correlation matrix not found for {patient} {phase} {band}. "
            f"Run: python src/compute_corr_matrices.py --patient {patient}"
        )

    # Load MSC matrix (dense)
    msc_matrix = load_msc_matrix(
        patient,
        phase,
        band,
        cache_root=cache_root_msc,
        sparsify="none",
        n_surrogates=0,
        nperseg=nperseg,
    )
    if msc_matrix is None:
        raise FileNotFoundError(
            f"MSC matrix not found for {patient} {phase} {band}. "
            f"Run: python src/compute_msc_matrices.py --patient {patient}"
        )

    corr_matrix_abs = corr_matrix.copy()

    # Validate non-negativity; warn and fix if cache content is not absolute
    tol = 1e-9
    if np.any(corr_matrix_abs < -tol):
        import warnings
        warnings.warn(
            f"Correlation cache for {patient} {phase} {band} contains negative entries; "
            "taking absolute value for fair comparison."
        )
        corr_matrix_abs = np.abs(corr_matrix_abs)
    elif np.any(corr_matrix_abs < 0):
        import warnings
        warnings.warn(
            f"Correlation cache for {patient} {phase} {band} contains small negative entries; "
            "clipping to zero for fair comparison."
        )
        corr_matrix_abs = np.clip(corr_matrix_abs, 0, None)

    if np.any(msc_matrix < -tol):
        import warnings
        warnings.warn(
            f"MSC cache for {patient} {phase} {band} contains negative entries; clipping to zero."
        )
        msc_matrix = np.clip(msc_matrix, 0, None)

    np.fill_diagonal(corr_matrix_abs, 0)
    np.fill_diagonal(msc_matrix, 0)
    _validate_matrices(corr_matrix_abs, msc_matrix, band)

    # Extract edge values (upper triangle)
    corr_values = _upper_triangle_values(corr_matrix_abs)
    msc_values = _upper_triangle_values(msc_matrix)

    # Agreement metric
    if corr_values.std() == 0 or msc_values.std() == 0:
        agreement = float("nan")
    else:
        agreement = float(np.corrcoef(corr_values, msc_values)[0, 1])

    corr_stats = {
        "mean": float(corr_values.mean()),
        "std": float(corr_values.std()),
        "median": float(np.median(corr_values)),
    }
    msc_stats = {
        "mean": float(msc_values.mean()),
        "std": float(msc_values.std()),
        "median": float(np.median(msc_values)),
    }

    # Prepare output path
    if output_path is None:
        output_dir = Path("data/figures/comparison") / patient
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{band}_{phase}_fc_comparison.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"{patient} {phase} {band.upper()} - FC Method Comparison", fontsize=14, fontweight='bold')

    # Panel 0: Correlation (abs)
    ax0 = axes[0]
    im0 = ax0.imshow(corr_matrix_abs, cmap='viridis', vmin=0, vmax=1, origin='upper', aspect='auto')
    ax0.set_title('Correlation FC\n(Absolute)', fontsize=12, fontweight='bold')
    ax0.set_xlabel('Channel', fontsize=10)
    ax0.set_ylabel('Channel', fontsize=10)
    cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label('|Correlation|', fontsize=10)
    stats_text0 = f"Mean: {corr_stats['mean']:.3f}\nStd: {corr_stats['std']:.3f}"
    ax0.text(
        0.02, 0.98, stats_text0,
        transform=ax0.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    # Panel 1: MSC
    ax1 = axes[1]
    im1 = ax1.imshow(msc_matrix, cmap='viridis', vmin=0, vmax=1, origin='upper', aspect='auto')
    ax1.set_title('MSC FC\n(Magnitude-Squared Coherence)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Channel', fontsize=10)
    ax1.set_ylabel('Channel', fontsize=10)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('MSC', fontsize=10)
    stats_text1 = f"Mean: {msc_stats['mean']:.3f}\nStd: {msc_stats['std']:.3f}"
    ax1.text(
        0.02, 0.98, stats_text1,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    # Panel 2: Agreement (distributions + scatter)
    ax2 = axes[2]
    divider = make_axes_locatable(ax2)
    ax2_top = divider.append_axes("top", size="30%", pad=0.1, sharex=ax2)

    # Distributions (top)
    ax2_top.hist(
        corr_values, bins=40, alpha=0.6, label='Correlation',
        color='steelblue', density=True, edgecolor='none',
    )
    ax2_top.hist(
        msc_values, bins=40, alpha=0.6, label='MSC',
        color='coral', density=True, edgecolor='none',
    )
    ax2_top.set_ylabel('Density', fontsize=9)
    ax2_top.legend(fontsize=9, loc='upper right')
    ax2_top.grid(alpha=0.3)
    ax2_top.tick_params(labelbottom=False)

    # Scatter (main)
    n_points = len(corr_values)
    if n_points > 5000:
        sample_idx = np.random.choice(n_points, 5000, replace=False)
        corr_plot = corr_values[sample_idx]
        msc_plot = msc_values[sample_idx]
    else:
        corr_plot = corr_values
        msc_plot = msc_values

    ax2.scatter(corr_plot, msc_plot, alpha=0.3, s=2, color='steelblue')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=1.5, alpha=0.7, label='Identity')
    ax2.set_xlabel('Correlation FC', fontsize=10)
    ax2.set_ylabel('MSC FC', fontsize=10)
    ax2.set_title('Method Agreement', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    agreement_text = f'Pearson r = {agreement:.3f}\nN edges = {n_points}'
    ax2.text(
        0.05, 0.95, agreement_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path
