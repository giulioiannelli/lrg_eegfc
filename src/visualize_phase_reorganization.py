#!/usr/bin/env python3
"""Visualize phase reorganization of brain network hierarchical structure.

This script creates visualizations analyzing how brain network organization
changes across experimental phases (rsPre → taskLearn → taskTest → rsPost).
This is used to assess cognitive effects of learning tasks on multiscale
hierarchical network structure.

Usage:
    # Single patient, single band, correlation-based networks
    python src/visualize_phase_reorganization.py --patient Pat_02 --band beta --fc-method corr

    # MSC-based networks
    python src/visualize_phase_reorganization.py --patient Pat_02 --band beta --fc-method msc

    # Both main visualization and distance matrix
    python src/visualize_phase_reorganization.py --patient Pat_02 --band beta --fc-method corr --plot-type all

    # Batch mode: all bands for a patient
    python src/visualize_phase_reorganization.py --patient Pat_02 --fc-method corr --batch

Examples:
    # Analyze reorganization for beta band correlations
    python src/visualize_phase_reorganization.py --patient Pat_02 --band beta --fc-method corr --verbose

    # Compare all bands
    python src/visualize_phase_reorganization.py --patient Pat_02 --fc-method corr --batch --verbose
"""

import argparse
from pathlib import Path

from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS
from lrg_eegfc.visuals.reorganization import (
    plot_phase_reorganization,
    plot_reorganization_distance_matrix,
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize phase reorganization of brain network structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--patient",
        required=True,
        help="Patient identifier (e.g., Pat_02)",
    )
    parser.add_argument(
        "--fc-method",
        required=True,
        choices=["corr", "msc"],
        help="FC method: corr (correlation) or msc (magnitude-squared coherence)",
    )

    # Optional arguments
    parser.add_argument(
        "--band",
        help="Frequency band (required unless --batch)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        help="List of phases to analyze (default: all phases)",
    )

    # Plot type
    parser.add_argument(
        "--plot-type",
        choices=["reorganization", "distance-matrix", "all"],
        default="reorganization",
        help="Type of visualization to generate",
    )

    # Paths
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/lrg_cache"),
        help="Root directory for LRG cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/figures/reorganization"),
        help="Output directory for figures",
    )

    # Batch mode
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all frequency bands",
    )

    # Options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()

    # Validation
    if not args.batch and not args.band:
        parser.error("--band required unless using --batch")

    # Determine bands to process
    if args.batch:
        bands = list(BRAIN_BANDS.keys())
    else:
        bands = [args.band]

    # Determine phases
    if args.phases:
        phases = args.phases
    else:
        phases = list(PHASE_LABELS)

    if args.verbose:
        print(f"\n=== Phase reorganization for {args.patient} ({args.fc_method}) ===")
        print(f"  plot: {args.plot_type}")
        print(f"  bands: {', '.join(bands)}")
        print(f"  phases: {', '.join(phases)}")
        print(f"  cache: {args.cache_root}/{args.patient}/")
        print(f"  output: {args.output_dir}/{args.patient}/")

    # Process each band
    success_count = 0
    fail_count = 0

    for band in bands:
        try:
            # Generate reorganization visualization
            if args.plot_type in ["reorganization", "all"]:
                reorg_path = args.output_dir / args.patient / f"{band}_{args.fc_method}_phase_reorganization.png"
                if reorg_path.exists():
                    if args.verbose:
                        print(f"• Skipping (exists): {reorg_path.name}")
                else:
                    output_path = plot_phase_reorganization(
                        patient=args.patient,
                        band=band,
                        fc_method=args.fc_method,
                        phases=phases,
                        cache_root=args.cache_root,
                        verbose=args.verbose,
                        output_path=reorg_path,
                    )

                    if args.verbose:
                        print(f"✓ Reorganization: {band} -> {output_path.name}")

            # Generate distance matrix visualization
            if args.plot_type in ["distance-matrix", "all"]:
                dist_path = args.output_dir / args.patient / f"{band}_{args.fc_method}_distance_matrix.png"
                if dist_path.exists():
                    if args.verbose:
                        print(f"• Skipping (exists): {dist_path.name}")
                else:
                    output_path = plot_reorganization_distance_matrix(
                        patient=args.patient,
                        band=band,
                        fc_method=args.fc_method,
                        phases=phases,
                        cache_root=args.cache_root,
                        verbose=args.verbose,
                        output_path=dist_path,
                    )

                    if args.verbose:
                        print(f"✓ Distance matrix: {band} -> {output_path.name}")

            success_count += 1

        except FileNotFoundError as e:
            print(f"  ✗ ERROR: {e}")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ ERROR: Unexpected error for {band}: {e}")
            fail_count += 1

        if args.verbose:
            print()

    # Summary
    if args.verbose:
        print(f"\n✓ Completed: {success_count}/{len(bands)}")
        if fail_count > 0:
            print(f"✗ Failed: {fail_count}/{len(bands)}")


if __name__ == "__main__":
    main()
