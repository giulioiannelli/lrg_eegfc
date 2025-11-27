"""Utilities for handling frequency bands in coherence analysis."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


__all__ = [
    'validate_bands',
    'get_band_indices',
]


def validate_bands(
    bands: Dict[str, Tuple[float, float]],
    fs: float,
) -> None:
    """
    Validate that frequency bands are well-formed and compatible with sampling rate.

    Parameters
    ----------
    bands : Dict[str, Tuple[float, float]]
        Dictionary mapping band names to (fmin, fmax) tuples in Hz
    fs : float
        Sampling frequency in Hz

    Raises
    ------
    ValueError
        If any band is invalid or incompatible with the sampling rate
    """
    nyquist = fs / 2.0

    for band_name, (fmin, fmax) in bands.items():
        # Check that fmin < fmax
        if fmin >= fmax:
            raise ValueError(
                f"Band '{band_name}': fmin ({fmin}) must be less than fmax ({fmax})"
            )

        # Check that frequencies are positive
        if fmin < 0 or fmax < 0:
            raise ValueError(
                f"Band '{band_name}': frequencies must be non-negative (got {fmin}, {fmax})"
            )

        # Check that fmax doesn't exceed Nyquist frequency
        if fmax > nyquist:
            raise ValueError(
                f"Band '{band_name}': fmax ({fmax} Hz) exceeds Nyquist frequency "
                f"({nyquist} Hz) for sampling rate {fs} Hz"
            )


def get_band_indices(
    freqs: NDArray,
    fmin: float,
    fmax: float,
) -> NDArray:
    """
    Get indices of frequencies within a specified band.

    Parameters
    ----------
    freqs : NDArray
        Array of frequencies
    fmin : float
        Minimum frequency of band (inclusive)
    fmax : float
        Maximum frequency of band (inclusive)

    Returns
    -------
    indices : NDArray
        Boolean array indicating which frequencies fall within the band
    """
    return (freqs >= fmin) & (freqs <= fmax)
