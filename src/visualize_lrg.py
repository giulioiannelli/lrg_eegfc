#!/usr/bin/env python3
"""CLI script for LRG (Laplacian Renormalization Group) visualizations.

This script provides command-line access to LRG visualization functions,
allowing users to generate plots from cached LRG analysis results without
recomputing the analysis.

Examples
--------
# Generate entropy curves for Pat_02 rsPre beta (correlation-based)
python src/visualize_lrg.py --patient Pat_02 --phase rsPre --band beta --fc-method corr --plot-type entropy --verbose

# Generate full 4-panel visualization
python src/visualize_lrg.py --patient Pat_02 --phase rsPre --band beta --fc-method corr --plot-type full --verbose

# Generate all plot types
python src/visualize_lrg.py --patient Pat_02 --phase rsPre --band beta --fc-method corr --plot-type all --verbose

# Batch mode: generate visualizations for all bands in a phase
python src/visualize_lrg.py --patient Pat_02 --phase rsPre --fc-method corr --batch --verbose

# Using MSC-based FC
python src/visualize_lrg.py --patient Pat_02 --phase rsPre --band beta --fc-method msc --plot-type full --verbose
"""

import argparse
import sys
from pathlib import Path

from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS
from lrg_eegfc.visuals.lrg import (
    plot_lrg_dendrogram,
    plot_lrg_entropy_curves,
    plot_lrg_full_panel,
    plot_ultrametric_heatmap,
)


def visualize_single(
    patient: str,
    phase: str,
    band: str,
    fc_method: str,
    plot_type: str,
    orientation: str,
    cache_root: Path,
    output_dir: Path,
    dataset_root: Path,
    verbose: bool,
) -> None:
    """Generate visualizations for a single patient/phase/band combination."""
    output_subdir = output_dir / patient
    output_subdir.mkdir(parents=True, exist_ok=True)

    try:
        # Entropy curves
        if plot_type in ["entropy", "all"]:
            entropy_path = output_subdir / f"{band}_{phase}_lrg_{fc_method}_entropy.png"
            if entropy_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {entropy_path.name}")
            else:
                output_path = plot_lrg_entropy_curves(
                    patient,
                    phase,
                    band,
                    fc_method,
                    cache_root=cache_root,
                    dataset_root=dataset_root,
                    output_path=entropy_path,
                )
                if verbose:
                    print(f"  ✓ Saved entropy curves: {output_path.name}")

        # Dendrogram
        if plot_type in ["dendrogram", "all"]:
            dendro_path = output_subdir / f"{band}_{phase}_lrg_{fc_method}_dendrogram.png"
            if dendro_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {dendro_path.name}")
            else:
                output_path = plot_lrg_dendrogram(
                    patient,
                    phase,
                    band,
                    fc_method,
                    cache_root=cache_root,
                    orientation=orientation,
                    dataset_root=dataset_root,
                    output_path=dendro_path,
                )
                if verbose:
                    print(f"  ✓ Saved dendrogram: {output_path.name}")

        # Ultrametric heatmap
        if plot_type in ["ultrametric", "all"]:
            ultra_path = output_subdir / f"{band}_{phase}_lrg_{fc_method}_ultrametric.png"
            if ultra_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {ultra_path.name}")
            else:
                output_path = plot_ultrametric_heatmap(
                    patient,
                    phase,
                    band,
                    fc_method,
                    cache_root=cache_root,
                    dataset_root=dataset_root,
                    output_path=ultra_path,
                )
                if verbose:
                    print(f"  ✓ Saved ultrametric heatmap: {output_path.name}")

        # Full 4-panel visualization
        if plot_type in ["full", "all"]:
            full_path = output_subdir / f"{band}_{phase}_lrg_{fc_method}_full.png"
            if full_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {full_path.name}")
            else:
                output_path = plot_lrg_full_panel(
                    patient,
                    phase,
                    band,
                    fc_method,
                    cache_root=cache_root,
                    dataset_root=dataset_root,
                    output_path=full_path,
                )
                if verbose:
                    print(f"  ✓ Saved full panel: {output_path.name}")

    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
    except Exception as e:
        print(f"  ERROR: Failed to generate {plot_type} plot for {patient} {phase} {band}: {e}")


def main() -> int:
    """Main entry point for LRG visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize LRG hierarchical network analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument("--patient", required=True, help="Patient ID (e.g., Pat_02)")
    parser.add_argument("--fc-method", required=True, choices=["corr", "msc"],
                       help="Functional connectivity method")

    # Optional positional arguments
    parser.add_argument("--phase", help="Recording phase (e.g., rsPre, taskLearn)")
    parser.add_argument("--band", help="Frequency band (e.g., beta, alpha)")

    # Plot type selection
    parser.add_argument(
        "--plot-type",
        choices=["entropy", "dendrogram", "ultrametric", "full", "all"],
        default="full",
        help="Type of plot to generate (default: full)",
    )

    # Dendrogram options
    parser.add_argument(
        "--orientation",
        choices=["top", "right", "bottom", "left"],
        default="top",
        help="Dendrogram orientation (default: top)",
    )

    # Paths
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/lrg_cache"),
        help="LRG cache directory. Default: data/lrg_cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/figures/lrg"),
        help="Output directory for figures. Default: data/figures/lrg",
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
        print(f"\n=== LRG visualization for {args.patient} ({args.fc_method}) ===")
        print(f"  plot: {args.plot_type}")
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
                fc_method=args.fc_method,
                plot_type=args.plot_type,
                orientation=args.orientation,
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
