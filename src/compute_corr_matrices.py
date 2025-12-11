#!/usr/bin/env python3
"""Compute correlation-based functional connectivity matrices.

This script computes correlation matrices for all patients, phases, and bands.

Usage:
    # Standard (absolute correlation, zero diagonal)
    python src/compute_corr_matrices.py

    # Raw correlation (keep negative values)
    python src/compute_corr_matrices.py --filter-type none

    # Specific patients
    python src/compute_corr_matrices.py --patients Pat_02 Pat_03

    # Custom filter order
    python src/compute_corr_matrices.py --filter-order 6 --verbose
"""

import argparse
from pathlib import Path
from lrg_eegfc import compute_corr_for_patient
from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS


def main():
    parser = argparse.ArgumentParser(
        description="Compute correlation-based functional connectivity matrices"
    )

    # Patient selection
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["Pat_02", "Pat_03", "Pat_05", "Pat_07", "Pat_08"],
        help="Patient IDs to process (default: Pat_02 Pat_03 Pat_05 Pat_07 Pat_08)"
    )

    # Correlation parameters
    parser.add_argument(
        "--filter-type",
        choices=["abs", "pos", "neg", "none"],
        default="abs",
        help="Filter type: abs (absolute), pos (positive), neg (negative), none (raw) (default: abs)"
    )
    parser.add_argument(
        "--zero-diagonal",
        action="store_true",
        default=True,
        help="Set diagonal to zero (default: True)"
    )
    parser.add_argument(
        "--keep-diagonal",
        dest="zero_diagonal",
        action="store_false",
        help="Keep diagonal values"
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Bandpass filter order (default: 4)"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2048.0,
        help="Sampling rate in Hz (default: 2048.0)"
    )

    # Cache control
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/corr_cache"),
        help="Cache directory (default: data/corr_cache)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if cached"
    )

    # Output control
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 70)
    print("Correlation Functional Connectivity Matrix Computation")
    print("=" * 70)
    print(f"Patients: {', '.join(args.patients)}")
    print(f"Bands: {', '.join(BRAIN_BANDS.keys())}")
    print(f"Phases: {', '.join(PHASE_LABELS)}")
    print(f"Filter type: {args.filter_type}")
    print(f"Zero diagonal: {args.zero_diagonal}")
    print(f"Filter order: {args.filter_order}")
    print(f"Cache root: {args.cache_root}")
    expected = len(args.patients) * len(BRAIN_BANDS) * len(PHASE_LABELS)
    print(f"Expected matrices: {expected}")
    print("=" * 70)

    total_computed = 0
    total_failed = 0

    show_totals = len(args.patients) > 1

    for patient in args.patients:
        patient_cache = args.cache_root / patient
        print(f"\nProcessing {patient}... ({patient_cache}/)")

        results = compute_corr_for_patient(
            patient,
            verbose=args.verbose,
            filter_type=args.filter_type,
            zero_diagonal=args.zero_diagonal,
            filter_order=args.filter_order,
            sample_rate=args.sample_rate,
            cache_root=args.cache_root,
            overwrite_cache=args.overwrite,
        )

        # Count successes
        n_computed = sum(
            1 for band in results
            for phase in results[band]
            if results[band][phase] is not None
        )
        n_failed = len(BRAIN_BANDS) * len(PHASE_LABELS) - n_computed

        total_computed += n_computed
        total_failed += n_failed

        print(f"  ✓ Computed: {n_computed}/{len(BRAIN_BANDS) * len(PHASE_LABELS)}")
        if n_failed > 0:
            print(f"  ✗ Failed: {n_failed}")

    print("\n" + "=" * 70)
    print("✓ Correlation matrix computation complete!")
    print("=" * 70)
    if show_totals:
        print(f"Total computed: {total_computed}/{expected}")
        if total_failed > 0:
            print(f"Total failed: {total_failed}")
    elif total_failed > 0:
        print(f"Failed: {total_failed}")

    # Print cache information
    print("\nCache")
    print(f"  root:        {args.cache_root}/")
    print(f"  per-patient: {args.cache_root}/{{patient}}/")
    print("  filename:    "
          f"{{band}}_{{phase}}_corr_ftype-{args.filter_type}_zdiag-{args.zero_diagonal}.npy")


if __name__ == "__main__":
    main()
