#!/usr/bin/env python3
"""Compare MSC vs correlation functional connectivity networks.

This script compares MSC and correlation-based FC networks using ultrametric
distance metrics computed via LRG analysis.

Usage:
    # Compare all patients/phases/bands
    python src/compare_fc_methods.py

    # Specific patients
    python src/compare_fc_methods.py --patients Pat_02 Pat_03

    # Save to CSV
    python src/compare_fc_methods.py --output results/fc_comparison.csv

    # Memory effects analysis (pre vs post)
    python src/compare_fc_methods.py --mode memory --fc-method corr
"""

import argparse
from pathlib import Path
import pandas as pd
from lrg_eegfc.compare import batch_compare_fc_methods, batch_compare_phases
from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS


def main():
    parser = argparse.ArgumentParser(
        description="Compare MSC vs correlation FC networks"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["fc-methods", "memory"],
        default="fc-methods",
        help="Comparison mode: fc-methods (MSC vs corr) or memory (pre vs post)"
    )

    # Patient selection
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["Pat_02", "Pat_03", "Pat_05", "Pat_07", "Pat_08"],
        help="Patient IDs to process (default: Pat_02 Pat_03 Pat_05 Pat_07 Pat_08)"
    )

    # Phase/band selection
    parser.add_argument(
        "--phases",
        nargs="+",
        default=None,
        help="Phases to compare (default: all PHASE_LABELS)"
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="Bands to compare (default: all BRAIN_BANDS)"
    )

    # Memory mode specific
    parser.add_argument(
        "--fc-method",
        choices=["msc", "corr"],
        default="corr",
        help="FC method for memory mode (default: corr)"
    )

    # Cache control
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/lrg_cache"),
        help="LRG cache directory (default: data/lrg_cache)"
    )

    # Output control
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file (default: print to stdout)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Set defaults
    phases = args.phases if args.phases else list(PHASE_LABELS)
    bands = args.bands if args.bands else list(BRAIN_BANDS.keys())

    # Print configuration
    print("=" * 70)
    print("FC Network Comparison")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Patients: {', '.join(args.patients)}")
    print(f"Phases: {', '.join(phases)}")
    print(f"Bands: {', '.join(bands)}")
    if args.mode == "memory":
        print(f"FC method: {args.fc_method}")
    print(f"LRG cache: {args.cache_root}")
    print("=" * 70)

    # Run comparison
    if args.mode == "fc-methods":
        print("\nComparing MSC vs Correlation...")
        df = batch_compare_fc_methods(
            args.patients,
            phases,
            bands,
            cache_root=args.cache_root,
            verbose=args.verbose,
        )
    else:
        # Memory effects: compare pre vs post
        phase_pairs = [("rsPre", "rsPost"), ("taskLearn", "taskTest")]
        print(f"\nAnalyzing memory effects (pre vs post) using {args.fc_method}...")
        df = batch_compare_phases(
            args.patients,
            phase_pairs,
            bands,
            fc_method=args.fc_method,
            cache_root=args.cache_root,
            verbose=args.verbose,
        )

    # Print summary
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    print(f"Total comparisons: {len(df)}")
    print()

    if len(df) > 0:
        # Summary statistics
        print("Summary Statistics:")
        print("-" * 70)
        metrics = [
            "matrix_distance",
            "multiscale_distance",
            "quantile_rmse",
            "rank_correlation",
            "tree_similarity",
        ]

        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                print(f"  {metric:30s} mean={mean_val:8.4f}  std={std_val:8.4f}")

        # Per-band analysis
        if "band" in df.columns:
            print("\nMean Matrix Distance by Band:")
            print("-" * 70)
            band_means = df.groupby("band")["matrix_distance"].mean().sort_values()
            for band, val in band_means.items():
                print(f"  {band:15s} {val:.4f}")

        # Per-patient analysis
        if "patient" in df.columns:
            print("\nMean Matrix Distance by Patient:")
            print("-" * 70)
            patient_means = df.groupby("patient")["matrix_distance"].mean().sort_values()
            for patient, val in patient_means.items():
                print(f"  {patient:15s} {val:.4f}")

        # Memory effects analysis
        if "phase_pair" in df.columns:
            print("\nMean Matrix Distance by Phase Pair:")
            print("-" * 70)
            pair_means = df.groupby("phase_pair")["matrix_distance"].mean().sort_values()
            for pair, val in pair_means.items():
                print(f"  {pair:25s} {val:.4f}")

        # Output to file
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")
        else:
            print("\n(Use --output <file.csv> to save the full table.)")

    else:
        print("⚠ No comparisons computed. Check that LRG analyses are cached.")
        print("\nMake sure you've run:")
        if args.mode == "fc-methods":
            print("  python src/compute_lrg_analysis.py --fc-method corr")
            print("  python src/compute_lrg_analysis.py --fc-method msc")
        else:
            print(f"  python src/compute_lrg_analysis.py --fc-method {args.fc_method}")


if __name__ == "__main__":
    main()
