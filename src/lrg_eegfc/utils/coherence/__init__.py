"""Coherence-based functional connectivity network construction.

This module provides a standardized pipeline for constructing adjacency matrices
from multivariate SEEG time series using magnitude-squared coherence (MSC) with
optional soft sparsification via surrogate-based null models.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from ...config.const import (
    BRAIN_BANDS,
    DEFAULT_N_SURROGATES,
)
from .bands import validate_bands
from .msc import compute_msc_welch, band_average_msc
from .sparsify import soft_sparsify_surrogate
from .surrogates import surrogate_msc_null


__all__ = [
    'coherence_fc_pipeline',
    'compute_msc_welch',
    'band_average_msc',
    'soft_sparsify_surrogate',
    'surrogate_msc_null',
]


def coherence_fc_pipeline(
    X: NDArray,
    fs: float,
    bands: Dict[str, Tuple[float, float]] | None = None,
    n_surrogates: int | None = None,
    sparsify: str = "soft",
    nperseg: int = 256,
    noverlap: int | None = None,
    zero_diagonal: bool = True,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> Dict[str, NDArray]:
    """
    Coherence-based functional connectivity pipeline.

    This is the main entry point for computing coherence-based FC networks. It matches
    the interface of the correlation-based FC pipeline but uses magnitude-squared
    coherence (MSC) instead of Pearson correlation.

    Parameters
    ----------
    X : NDArray
        Input time series of shape (N, L) where N is the number of channels and L is
        the number of samples
    fs : float
        Sampling frequency in Hz
    bands : Dict[str, Tuple[float, float]], optional
        Dictionary mapping band names to (fmin, fmax) tuples in Hz. If None, uses
        BRAIN_BANDS from config.const (default: None)
    n_surrogates : int, optional
        Number of circular shift surrogates for null model estimation. If None, uses
        DEFAULT_N_SURROGATES from config.const. Set to 0 to skip surrogate generation
        (default: None)
    sparsify : str, optional
        Sparsification method: "soft" for surrogate-based soft sparsification,
        "none" for no sparsification (returns dense MSC matrices) (default: "soft")
    nperseg : int, optional
        Length of each segment (window) for spectral estimation in samples
        (default: 256)
    noverlap : int, optional
        Number of overlapping samples between segments. If None, uses nperseg // 2
        (default: None)
    zero_diagonal : bool, optional
        If True, set diagonal elements to zero (no self-loops) (default: True)
    rng : np.random.Generator, optional
        Random number generator for reproducibility in surrogate generation
    verbose : bool, optional
        If True, print timing information for each step of the pipeline (default: False)

    Returns
    -------
    adjacency_matrices : Dict[str, NDArray]
        Dictionary mapping band names to adjacency matrices of shape (N, N). If
        sparsify="none", returns dense MSC matrices. If sparsify="soft", returns
        soft-sparsified matrices.

    Notes
    -----
    The pipeline consists of the following steps:

    1. Compute magnitude-squared coherence (MSC) using Welch's method
    2. Average MSC over each frequency band
    3. (Optional) Soft sparsification using surrogate null models

    Uses Welch's method (scipy.signal.coherence) for fast and robust spectral
    estimation. This is significantly faster than multitaper methods (10-50x)
    while still providing good spectral estimates for most applications.

    The output format matches the correlation-based FC pipeline, so downstream LRG
    analysis code remains unchanged.

    Examples
    --------
    >>> # Basic usage with defaults
    >>> adj_matrices = coherence_fc_pipeline(X, fs=512.0)

    >>> # Without sparsification
    >>> msc_matrices = coherence_fc_pipeline(X, fs=512.0, sparsify="none")

    >>> # Custom bands and surrogates
    >>> custom_bands = {"slow": (0.5, 4.0), "fast": (30.0, 100.0)}
    >>> adj_matrices = coherence_fc_pipeline(
    ...     X, fs=512.0, bands=custom_bands, n_surrogates=100
    ... )
    """
    # Start timing
    t_start = time.time()
    if verbose:
        print(f"[coherence_fc_pipeline] Starting pipeline for {X.shape[0]} channels, {X.shape[1]} samples")

    # Set defaults
    if bands is None:
        bands = BRAIN_BANDS

    if n_surrogates is None:
        n_surrogates = DEFAULT_N_SURROGATES

    # Validate bands
    t0 = time.time()
    validate_bands(bands, fs)
    if verbose:
        print(f"  [1/5] Band validation: {time.time() - t0:.3f}s")

    # Step 1: Compute magnitude-squared coherence using Welch's method
    t0 = time.time()
    freqs, Coh = compute_msc_welch(
        X,
        fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    if verbose:
        print(f"  [2/5] MSC computation (Welch): {time.time() - t0:.3f}s")

    # Step 2: Band average
    t0 = time.time()
    W_bands = band_average_msc(Coh, freqs, bands)
    if verbose:
        print(f"  [3/5] Band averaging: {time.time() - t0:.3f}s")

    # Step 4: Sparsification (optional)
    t0 = time.time()
    if sparsify == "none":
        # Return dense MSC matrices
        adjacency_matrices = W_bands
        if verbose:
            print(f"  [4/5] Sparsification (none): {time.time() - t0:.3f}s")
    elif sparsify == "soft":
        # Compute surrogate null distribution
        if n_surrogates > 0:
            t_surr = time.time()
            W_null = surrogate_msc_null(
                X,
                fs,
                bands,
                n_surrogates,
                nperseg=nperseg,
                noverlap=noverlap,
                rng=rng,
            )
            if verbose:
                print(f"  [4/5] Surrogate generation ({n_surrogates} surrogates): {time.time() - t_surr:.3f}s")

            # Apply soft sparsification to each band
            t_spars = time.time()
            adjacency_matrices = {}
            for band_name in bands:
                A = soft_sparsify_surrogate(W_bands[band_name], W_null[band_name])
                adjacency_matrices[band_name] = A
            if verbose:
                print(f"        Soft sparsification: {time.time() - t_spars:.3f}s")
                print(f"        Total sparsification step: {time.time() - t0:.3f}s")
        else:
            # No surrogates requested, return dense MSC
            adjacency_matrices = W_bands
            if verbose:
                print(f"  [4/5] Sparsification (skipped, n_surrogates=0): {time.time() - t0:.3f}s")
    else:
        raise ValueError(f"Unknown sparsify method: {sparsify}. Use 'soft' or 'none'.")

    # Zero diagonal if requested
    t0 = time.time()
    if zero_diagonal:
        for band_name in adjacency_matrices:
            np.fill_diagonal(adjacency_matrices[band_name], 0.0)
    if verbose:
        print(f"  [5/5] Zero diagonal: {time.time() - t0:.3f}s")
        print(f"[coherence_fc_pipeline] Total time: {time.time() - t_start:.3f}s")

    return adjacency_matrices
