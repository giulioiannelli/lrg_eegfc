#!/usr/bin/env python3
"""Clean correlation matrices using Marchenko-Pastur + percolation thresholding.

This script computes cleaned correlation matrices for all patients, phases, and bands.
The cleaning pipeline applies:
1. Marchenko-Pastur spectral filtering to remove noise eigenvalues
2. Percolation thresholding to preserve network connectivity

Usage:
    # Clean all patients
    python src/clean_correlation_matrices.py --verbose

    # Specific patients
    python src/clean_correlation_matrices.py --patients Pat_02 Pat_03 --verbose

    # Force recompute
    python src/clean_correlation_matrices.py --overwrite --verbose
"""

import argparse
from pathlib import Path
from lrg_eegfc import compute_cleaned_corr_for_patient
from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS


def main():
    parser = argparse.ArgumentParser(
        description="Clean correlation matrices using Marchenko-Pastur + percolation"
    )

    # Patient selection
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["Pat_02", "Pat_03", "Pat_05", "Pat_07", "Pat_08"],
        help="Patient IDs to process (default: Pat_02 Pat_03 Pat_05 Pat_07 Pat_08)",
    )

    # Filtering parameters
    parser.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Butterworth filter order (default: 4)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2048.0,
        help="Sampling rate in Hz (default: 2048.0, auto-detected if available)",
    )

    # Cache control
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/corr_cache"),
        help="Cache directory (default: data/corr_cache)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if cached",
    )

    # Output control
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 70)
    print("Correlation Matrix Cleaning (Marchenko-Pastur + Percolation)")
    print("=" * 70)
    print(f"Patients: {', '.join(args.patients)}")
    print(f"Bands: {', '.join(BRAIN_BANDS.keys())}")
    print(f"Phases: {', '.join(PHASE_LABELS)}")
    print(f"Filter order: {args.filter_order}")
    print(f"Default sample rate: {args.sample_rate} Hz (auto-detected if available)")
    print(f"Cache root: {args.cache_root}")
    expected = len(args.patients) * len(BRAIN_BANDS) * len(PHASE_LABELS)
    print(f"Expected cleaned matrices: {expected}")
    print("=" * 70)

    total_computed = 0
    total_failed = 0

    show_totals = len(args.patients) > 1

    for patient in args.patients:
        print(f"\nProcessing {patient}...")
        print("-" * 70)

        results = compute_cleaned_corr_for_patient(
            patient,
            filter_order=args.filter_order,
            sample_rate=args.sample_rate,
            cache_root=args.cache_root,
            overwrite_cache=args.overwrite,
            verbose=args.verbose,
        )

        # Count successes
        n_computed = sum(
            1
            for band in results
            for phase in results[band]
            if results[band][phase] is not None
        )
        n_failed = len(BRAIN_BANDS) * len(PHASE_LABELS) - n_computed

        total_computed += n_computed
        total_failed += n_failed

        print(f"  ✓ Computed: {n_computed}/{len(BRAIN_BANDS) * len(PHASE_LABELS)}")
        if n_failed > 0:
            print(f"  ✗ Failed: {n_failed}")

        # Print summary statistics for this patient
        if n_computed > 0:
            print(f"\n  Summary for {patient}:")
            print(f"  {'Band':<15} {'Phase':<15} {'N_signal':<10} {'Threshold':<12}")
            print("  " + "-" * 55)

            for band in results:
                for phase in results[band]:
                    result = results[band][phase]
                    if result is not None:
                        print(
                            f"  {band:<15} {phase:<15} "
                            f"{result.n_signal_components:<10} "
                            f"{result.threshold:<12.4f}"
                        )

    print("\n" + "=" * 70)
    print("✓ Correlation cleaning complete!")
    print("=" * 70)
    if show_totals:
        print(f"Total computed: {total_computed}/{expected}")
        if total_failed > 0:
            print(f"Total failed: {total_failed}")
    elif total_failed > 0:
        print(f"Failed: {total_failed}")

    # Print cache information
    print(f"\nCache location: {args.cache_root}/{{patient}}/")
    print("\nFilename format:")
    print("  Matrix: {band}_{phase}_corr_cleaned.npy")
    print("  Metadata: {band}_{phase}_corr_cleaned_meta.npz")

    print("\nMetadata contains:")
    print("  - threshold: Percolation threshold applied")
    print("  - eigenvalues: All eigenvalues of original correlation matrix")
    print("  - lambda_min, lambda_max: Marchenko-Pastur bounds")
    print("  - signal_eigenvalues: Eigenvalues above lambda_max")
    print("  - n_signal_components: Number of signal components retained")
    print("  - detachment_info: Percolation analysis details")


if __name__ == "__main__":
    main()
