#!/usr/bin/env python3
"""CLI script for MSC-based functional connectivity visualizations.

This script provides command-line access to MSC visualization functions,
allowing users to generate plots from cached MSC matrices without
recomputing the analysis.

Examples
--------
# Generate network visualization for Pat_02 rsPre beta (dense MSC)
python src/visualize_msc.py --patient Pat_02 --phase rsPre --band beta --plot-type network --verbose

# Generate summary visualization (3-panel: Matrix | Network | Percolation)
python src/visualize_msc.py --patient Pat_02 --phase rsPre --band beta --plot-type summary --verbose

# Generate all plots for Pat_02 rsPre beta (validated MSC with 200 surrogates)
python src/visualize_msc.py --patient Pat_02 --phase rsPre --band beta --plot-type all --sparsify soft --n-surrogates 200 --verbose

# Compare dense vs validated MSC
python src/visualize_msc.py --patient Pat_02 --phase rsPre --band beta --plot-type comparison --n-surrogates 200 --verbose

# Batch mode: generate network plots for all bands in a phase
python src/visualize_msc.py --patient Pat_02 --phase rsPre --batch --verbose

# Batch mode with validation
python src/visualize_msc.py --patient Pat_02 --phase rsPre --batch --sparsify soft --n-surrogates 200 --verbose
"""

import argparse
import sys
from pathlib import Path

from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS
from lrg_eegfc.visuals.msc import (
    plot_msc_and_network,
    plot_msc_comparison_dense_vs_validated,
    plot_msc_heatmap,
    plot_msc_summary,
)
from lrg_eegfc.workflow_msc import load_msc_matrix


def visualize_single(
    patient: str,
    phase: str,
    band: str,
    plot_type: str,
    sparsify: str,
    n_surrogates: int,
    nperseg: int,
    cache_root: Path,
    output_dir: Path,
    dataset_root: Path,
    verbose: bool,
) -> None:
    """Generate visualizations for a single patient/phase/band combination."""
    output_subdir = output_dir / patient
    output_subdir.mkdir(parents=True, exist_ok=True)

    try:
        # Matrix plot
        if plot_type in ["matrix", "all"]:
            suffix = "dense" if sparsify == "none" else f"validated_nsurr{n_surrogates}"
            matrix_path = output_subdir / f"{band}_{phase}_msc_{suffix}_matrix.png"
            if matrix_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {matrix_path.name}")
            else:
                msc_matrix = load_msc_matrix(
                    patient, phase, band, cache_root, sparsify, n_surrogates, nperseg
                )
                if msc_matrix is None:
                    print(f"  ERROR: MSC matrix not found in cache for {band} {phase}")
                    return

                plot_msc_heatmap(
                    msc_matrix,
                    matrix_path,
                    title=f"MSC Matrix - {band} {phase} ({suffix})",
                )
                if verbose:
                    print(f"  ✓ Saved matrix plot: {matrix_path.name}")

        # Network plot
        if plot_type in ["network", "all"]:
            suffix = "dense" if sparsify == "none" else f"validated_nsurr{n_surrogates}"
            network_path = output_subdir / f"{band}_{phase}_msc_{suffix}_network.png"
            if network_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {network_path.name}")
            else:
                output_path = plot_msc_and_network(
                    patient,
                    phase,
                    band,
                    cache_root=cache_root,
                    sparsify=sparsify,
                    n_surrogates=n_surrogates,
                    nperseg=nperseg,
                    dataset_root=dataset_root,
                    output_path=network_path,
                )
                if verbose:
                    print(f"  ✓ Saved network plot: {output_path.name}")

        # Summary plot (3-panel: Matrix | Network | Percolation)
        if plot_type in ["summary", "all"]:
            summary_path = output_subdir / f"{band}_{phase}_msc_summary.png"
            if summary_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {summary_path.name}")
            else:
                output_path = plot_msc_summary(
                    patient,
                    phase,
                    band,
                    cache_root=cache_root,
                    nperseg=nperseg,
                    dataset_root=dataset_root,
                    output_path=summary_path,
                )
                if verbose:
                    print(f"  ✓ Saved summary plot: {output_path.name}")

        # Comparison plot
        if plot_type == "comparison":
            comparison_path = output_subdir / f"{band}_{phase}_msc_comparison_nsurr{n_surrogates}.png"
            if comparison_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {comparison_path.name}")
            else:
                output_path = plot_msc_comparison_dense_vs_validated(
                    patient,
                    phase,
                    band,
                    n_surrogates=n_surrogates,
                    cache_root=cache_root,
                    nperseg=nperseg,
                    output_path=comparison_path,
                )
                if verbose:
                    print(f"  ✓ Saved comparison plot: {output_path.name}")

    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
    except Exception as e:
        print(f"  ERROR: Failed to generate {plot_type} plot for {patient} {phase} {band}: {e}")


def main() -> int:
    """Main entry point for MSC visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize MSC functional connectivity networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument("--patient", required=True, help="Patient ID (e.g., Pat_02)")

    # Optional positional arguments
    parser.add_argument("--phase", help="Recording phase (e.g., rsPre, taskLearn)")
    parser.add_argument("--band", help="Frequency band (e.g., beta, alpha)")

    # Plot type selection
    parser.add_argument(
        "--plot-type",
        choices=["matrix", "network", "summary", "comparison", "all"],
        default="network",
        help="Type of plot to generate (default: network)",
    )

    # MSC version selection
    parser.add_argument(
        "--sparsify",
        choices=["none", "soft"],
        default="none",
        help="MSC version: 'none' (dense) or 'soft' (surrogate-validated). Default: none",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=200,
        help="Number of surrogates for validation (if sparsify='soft'). Default: 200",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=1024,
        help="Window length for Welch's method. Default: 1024",
    )

    # Paths
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/msc_cache"),
        help="MSC cache directory. Default: data/msc_cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/figures/msc"),
        help="Output directory for figures. Default: data/figures/msc",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/stereoeeg_patients"),
        help="Root directory for patient data. Default: data/stereoeeg_patients",
    )

    # Batch mode
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate plots for all bands (requires --phase)",
    )

    # Verbosity
    parser.add_argument("--verbose", action="store_true", help="Print progress information")

    args = parser.parse_args()

    # Validation
    if not args.batch:
        if not args.phase or not args.band:
            parser.error("--phase and --band required unless using --batch")

    # Validate band and phase
    if args.band and args.band not in BRAIN_BANDS:
        print(f"ERROR: Unknown band '{args.band}'. Valid bands: {list(BRAIN_BANDS.keys())}")
        return 1

    if args.phase and args.phase not in PHASE_LABELS:
        print(f"ERROR: Unknown phase '{args.phase}'. Valid phases: {PHASE_LABELS}")
        return 1

    # Determine combinations to process
    if args.batch:
        bands = list(BRAIN_BANDS.keys())
        phases = list(PHASE_LABELS) if not args.phase else [args.phase]
    else:
        bands = [args.band]
        phases = [args.phase]

    if args.verbose:
        desc = "dense" if args.sparsify == "none" else f"validated (n_surr={args.n_surrogates})"
        print(f"\n=== MSC visualization for {args.patient} ===")
        print(f"  plot: {args.plot_type} ({desc})")
        print(f"  bands: {', '.join(bands)}")
        print(f"  phases: {', '.join(phases)}")
        print(f"  cache: {args.cache_root}/{args.patient}/")
        print(f"  output: {args.output_dir}/{args.patient}/")

    # Process each combination
    for band in bands:
        for phase in phases:
            visualize_single(
                patient=args.patient,
                phase=phase,
                band=band,
                plot_type=args.plot_type,
                sparsify=args.sparsify,
                n_surrogates=args.n_surrogates,
                nperseg=args.nperseg,
                cache_root=args.cache_root,
                output_dir=args.output_dir,
                dataset_root=args.dataset_root,
                verbose=args.verbose,
            )

    if args.verbose:
        print("\n✓ Visualization complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
