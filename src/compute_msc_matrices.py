#!/usr/bin/env python3
"""Compute MSC-based functional connectivity matrices.

This script computes MSC matrices for all patients, phases, and bands.
Supports both dense MSC (no validation) and validated MSC (with surrogates).

Usage:
    # Dense MSC (no validation)
    python src/compute_msc_matrices.py

    # Validated MSC with surrogates
    python src/compute_msc_matrices.py --sparsify soft --n-surrogates 200

    # Specific patients
    python src/compute_msc_matrices.py --patients Pat_02 Pat_03

    # Custom parameters
    python src/compute_msc_matrices.py --nperseg 512 --verbose
"""

import argparse
from pathlib import Path
from lrg_eegfc import compute_msc_for_patient
from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS


def main():
    parser = argparse.ArgumentParser(
        description="Compute MSC-based functional connectivity matrices"
    )

    # Patient selection
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["Pat_02", "Pat_03", "Pat_05", "Pat_07", "Pat_08"],
        help="Patient IDs to process (default: Pat_02 Pat_03 Pat_05 Pat_07 Pat_08)"
    )

    # MSC parameters
    parser.add_argument(
        "--sparsify",
        choices=["none", "soft"],
        default="none",
        help="Sparsification method (default: none for dense MSC)"
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=0,
        help="Number of surrogates for validation (only if sparsify=soft, default: 0)"
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=1024,
        help="Window length for Welch's method (default: 1024)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of Welch segments to process per batch (default: 64)"
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
        default=Path("data/msc_cache"),
        help="Cache directory (default: data/msc_cache)"
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

    # Validate arguments
    if args.sparsify == "soft" and args.n_surrogates == 0:
        print("WARNING: sparsify='soft' but n_surrogates=0. Setting n_surrogates=200.")
        args.n_surrogates = 200

    # Print configuration
    print("=" * 70)
    print("MSC Functional Connectivity Matrix Computation")
    print("=" * 70)
    print(f"Patients: {', '.join(args.patients)}")
    print(f"Bands: {', '.join(BRAIN_BANDS.keys())}")
    print(f"Phases: {', '.join(PHASE_LABELS)}")
    print(f"Sparsify: {args.sparsify}")
    if args.sparsify == "soft":
        print(f"N surrogates: {args.n_surrogates}")
    print(f"nperseg: {args.nperseg}")
    print(f"batch_size: {args.batch_size}")
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

        results = compute_msc_for_patient(
            patient,
            verbose=args.verbose,
            sparsify=args.sparsify,
            n_surrogates=args.n_surrogates,
            nperseg=args.nperseg,
            batch_size=args.batch_size,
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
    print("✓ MSC matrix computation complete!")
    print("=" * 70)
    if show_totals:
        print(f"Total computed: {total_computed}/{expected}")
        if total_failed > 0:
            print(f"Total failed: {total_failed}")
    elif total_failed > 0:
        print(f"Failed: {total_failed}")

    # Print cache information
    if args.sparsify == "none":
        filename_pattern = f"{{band}}_{{phase}}_msc_sparsify-none_nperseg-{args.nperseg}.npy"
    else:
        filename_pattern = (
            f"{{band}}_{{phase}}_msc_sparsify-{args.sparsify}_"
            f"nsurr-{args.n_surrogates}_nperseg-{args.nperseg}.npy"
        )

    print("\nCache")
    print(f"  root:        {args.cache_root}/")
    print(f"  per-patient: {args.cache_root}/{{patient}}/")
    print(f"  filename:    {filename_pattern}")


if __name__ == "__main__":
    main()
