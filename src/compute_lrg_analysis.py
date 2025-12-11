#!/usr/bin/env python3
"""Compute LRG analysis (ultrametric distances) from FC matrices.

This script computes LRG (Laplacian Renormalization Group) analysis for
functional connectivity matrices. It can process both MSC and correlation-based
FC matrices.

Usage:
    # Compute LRG for correlation matrices
    python src/compute_lrg_analysis.py --fc-method corr

    # Compute LRG for MSC matrices
    python src/compute_lrg_analysis.py --fc-method msc

    # Specific patients
    python src/compute_lrg_analysis.py --fc-method corr --patients Pat_02 Pat_03

    # Custom entropy parameters
    python src/compute_lrg_analysis.py --fc-method corr --entropy-steps 600 --verbose
"""

import argparse
from pathlib import Path
from lrg_eegfc.workflow_lrg import compute_lrg_for_patient
from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS


def main():
    parser = argparse.ArgumentParser(
        description="Compute LRG analysis (ultrametric distances) from FC matrices"
    )

    # Patient selection
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["Pat_02", "Pat_03", "Pat_05", "Pat_07", "Pat_08"],
        help="Patient IDs to process (default: Pat_02 Pat_03 Pat_05 Pat_07 Pat_08)"
    )

    # FC method selection
    parser.add_argument(
        "--fc-method",
        choices=["msc", "corr"],
        required=True,
        help="Functional connectivity method: msc or corr"
    )

    # LRG parameters
    parser.add_argument(
        "--entropy-steps",
        type=int,
        default=400,
        help="Number of steps for entropy computation (default: 400)"
    )
    parser.add_argument(
        "--entropy-t1",
        type=float,
        default=-3.0,
        help="Start of tau range (log scale) for entropy (default: -3.0)"
    )
    parser.add_argument(
        "--entropy-t2",
        type=float,
        default=5.0,
        help="End of tau range (log scale) for entropy (default: 5.0)"
    )

    # Cache control
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/lrg_cache"),
        help="Cache directory (default: data/lrg_cache)"
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
    print("LRG Analysis Computation")
    print("=" * 70)
    print(f"Patients: {', '.join(args.patients)}")
    print(f"Bands: {', '.join(BRAIN_BANDS.keys())}")
    print(f"Phases: {', '.join(PHASE_LABELS)}")
    print(f"FC method: {args.fc_method}")
    print(f"Entropy steps: {args.entropy_steps}")
    print(f"Entropy tau range: [{args.entropy_t1}, {args.entropy_t2}]")
    print(f"Cache root: {args.cache_root}")
    expected = len(args.patients) * len(BRAIN_BANDS) * len(PHASE_LABELS)
    print(f"Expected LRG analyses: {expected}")
    print("=" * 70)

    total_computed = 0
    total_failed = 0

    show_totals = len(args.patients) > 1

    for patient in args.patients:
        print(f"\nProcessing {patient}...")
        print("-" * 70)

        results = compute_lrg_for_patient(
            patient,
            fc_method=args.fc_method,
            verbose=args.verbose,
            entropy_steps=args.entropy_steps,
            entropy_t1=args.entropy_t1,
            entropy_t2=args.entropy_t2,
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
    print("✓ LRG analysis computation complete!")
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
    print(f"  {{band}}_{{phase}}_lrg_{args.fc_method}.npz")

    print("\nOutputs in each .npz file:")
    print("  - ultrametric_matrix: Condensed distance matrix")
    print("  - linkage_matrix: Hierarchical clustering linkage")
    print("  - entropy_tau: Tau values for entropy curve")
    print("  - entropy_1_minus_S: 1-S (normalized entropy)")
    print("  - entropy_C: C (spectral complexity)")
    print("  - optimal_threshold: Dendrogram cutting threshold")


if __name__ == "__main__":
    main()
