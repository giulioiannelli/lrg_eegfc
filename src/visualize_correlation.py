#!/usr/bin/env python3
"""CLI script for correlation-based functional connectivity visualizations.

This script provides command-line access to correlation visualization functions,
allowing users to generate plots from cached correlation matrices without
recomputing the analysis.

Examples
--------
# Generate combined summary for Pat_02 rsPre beta
python src/visualize_correlation.py --patient Pat_02 --phase rsPre --band beta --plot-type summary --verbose

# Generate network visualization only (uncleaned)
python src/visualize_correlation.py --patient Pat_02 --phase rsPre --band beta --plot-type network --verbose

# Batch mode: generate all plots for all bands/phases of a patient
python src/visualize_correlation.py --patient Pat_02 --batch --verbose

# Generate Marchenko-Pastur comparison
python src/visualize_correlation.py --patient Pat_02 --phase rsPre --band beta --plot-type mp --verbose
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from lrg_eegfc.constants import BRAIN_BANDS
from lrg_eegfc.visuals.correlation import (
    plot_correlation_heatmap,
    plot_correlation_and_network,
    plot_marchenko_pastur_comparison,
    plot_percolation_curves,
    plot_correlation_summary,
)
from lrg_eegfc import load_corr_matrix


PHASE_LABELS = ['rsPre', 'taskLearn', 'taskTest', 'rsPost']


def _indent_warnings() -> None:
    """Indent warnings to align with verbose output."""

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        formatted = warnings.formatwarning(message, category, filename, lineno, line).rstrip("\n")
        formatted = "\n".join(f"  {row}" for row in formatted.splitlines())
        (file or sys.stderr).write(formatted + "\n")

    warnings.showwarning = _showwarning


def visualize_single(
    patient: str,
    phase: str,
    band: str,
    plot_type: str,
    cleaned: bool,
    cache_root: Path,
    output_dir: Path,
    dataset_root: Path,
    verbose: bool,
) -> None:
    """Generate visualization for a single patient/phase/band combination.

    Parameters
    ----------
    patient : str
        Patient ID
    phase : str
        Experimental phase
    band : str
        Frequency band
    plot_type : str
        Type of plot: 'summary', 'matrix', 'network', 'mp', 'percolation', 'all'
    cleaned : bool
        Whether to use cleaned matrices
    cache_root : Path
        Root directory for correlation cache
    output_dir : Path
        Output directory for figures
    dataset_root : Path
        Root directory for patient data
    verbose : bool
        Whether to print progress messages
    """
    # Check if cache file exists
    try:
        if cleaned:
            from lrg_eegfc.workflow_cleaning import load_cleaned_corr_matrix
            _ = load_cleaned_corr_matrix(patient, phase, band, cache_root)
        else:
            _ = load_corr_matrix(patient, phase, band, cache_root)
    except FileNotFoundError as e:
        print(f"  WARNING: Cache file not found for {patient} {phase} {band}: {e}")
        return
    except Exception as e:
        print(f"  ERROR: Failed to load cache for {patient} {phase} {band}: {e}")
        return

    # Create output directory
    patient_output_dir = output_dir / patient
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots based on plot_type
    try:
        if plot_type in ['summary', 'all']:
            summary_path = patient_output_dir / f"{band}_{phase}_summary.png"
            if summary_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {summary_path.name}")
                return
            output_path = plot_correlation_summary(
                patient,
                phase,
                band,
                cache_root=cache_root,
                dataset_root=dataset_root,
                output_path=summary_path,
            )
            if verbose:
                print(f"  ✓ Saved summary plot: {output_path.name}")
            return

        if plot_type == 'matrix':
            suffix = "_cleaned" if cleaned else ""
            target_path = patient_output_dir / f"{band}_{phase}_matrix{suffix}.png"
            if target_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {target_path.name}")
                return

            if cleaned:
                from lrg_eegfc.workflow_cleaning import load_cleaned_corr_matrix
                corr_matrix, _ = load_cleaned_corr_matrix(patient, phase, band, cache_root)
            else:
                corr_matrix = load_corr_matrix(patient, phase, band, cache_root)

            plot_correlation_heatmap(
                corr_matrix,
                target_path,
                title=f"{patient} {phase} {band}{' (Cleaned)' if cleaned else ''}",
            )
            if verbose:
                print(f"  ✓ Saved matrix plot: {target_path.name}")

        if plot_type == 'network':
            suffix = "_cleaned" if cleaned else ""
            target_path = patient_output_dir / f"{band}_{phase}_network{suffix}.png"
            if target_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {target_path.name}")
                return
            output_path = plot_correlation_and_network(
                patient, phase, band,
                cache_root=cache_root,
                cleaned=cleaned,
                dataset_root=dataset_root,
                output_path=target_path,
            )
            if verbose:
                print(f"  ✓ Saved network plot: {output_path.name}")

        if plot_type == 'mp':
            target_path = patient_output_dir / f"{band}_{phase}_mp_comparison.png"
            if target_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {target_path.name}")
                return
            output_path = plot_marchenko_pastur_comparison(
                patient, phase, band,
                cache_root=cache_root,
                dataset_root=dataset_root,
                output_path=target_path,
            )
            if verbose:
                print(f"  ✓ Saved Marchenko-Pastur comparison: {output_path.name}")

        if plot_type == 'percolation':
            suffix = "_cleaned" if cleaned else ""
            target_path = patient_output_dir / f"{band}_{phase}_percolation{suffix}.png"
            if target_path.exists():
                if verbose:
                    print(f"  • Skipping (exists): {target_path.name}")
                return
            output_path = plot_percolation_curves(
                patient, phase, band,
                cache_root=cache_root,
                cleaned=cleaned,
                dataset_root=dataset_root,
                output_path=target_path,
            )
            if verbose:
                print(f"  ✓ Saved percolation curves: {output_path.name}")

    except Exception as e:
        print(f"  ERROR: Failed to generate {plot_type} plot for {patient} {phase} {band}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def visualize_batch(
    patient: str,
    cache_root: Path,
    output_dir: Path,
    dataset_root: Path,
    cleaned: bool,
    verbose: bool,
) -> None:
    """Generate all visualizations for a patient (all bands and phases).

    Parameters
    ----------
    patient : str
        Patient ID
    cache_root : Path
        Root directory for correlation cache
    output_dir : Path
        Output directory for figures
    dataset_root : Path
        Root directory for patient data
    cleaned : bool
        Whether to use cleaned matrices
    verbose : bool
        Whether to print progress messages
    """
    if verbose:
        print(f"\n=== Batch visualization for {patient} ===")
        print(f"  plot: summary (cleaned={cleaned})")
        print(f"  bands: {', '.join(BRAIN_BANDS.keys())}")
        print(f"  phases: {', '.join(PHASE_LABELS)}")
        print(f"  cache: {cache_root}/{patient}/")
        print(f"  output: {output_dir}/{patient}/")

    # Iterate over all bands and phases
    for band in BRAIN_BANDS.keys():
        for phase in PHASE_LABELS:
            visualize_single(
                patient, phase, band,
                plot_type='summary',
                cleaned=cleaned,
                cache_root=cache_root,
                output_dir=output_dir,
                dataset_root=dataset_root,
                verbose=verbose,
            )

    if verbose:
        print(f"\n=== Completed batch visualization for {patient} ===")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize correlation-based functional connectivity matrices and networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--patient',
        type=str,
        required=True,
        help='Patient ID (e.g., Pat_02)',
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=PHASE_LABELS,
        help='Experimental phase (required unless --batch)',
    )
    parser.add_argument(
        '--band',
        type=str,
        choices=list(BRAIN_BANDS.keys()),
        help='Frequency band (required unless --batch)',
    )
    parser.add_argument(
        '--plot-type',
        type=str,
        choices=['summary', 'matrix', 'network', 'mp', 'percolation', 'all'],
        default='summary',
        help="Plot to generate. Use 'summary' (or 'all') for the combined figure (default: summary)",
    )
    parser.add_argument(
        '--cleaned',
        action='store_true',
        help='Use cleaned correlation matrices (Marchenko-Pastur + percolation)',
    )
    parser.add_argument(
        '--cache-root',
        type=Path,
        default=Path('data/corr_cache'),
        help='Root directory for correlation cache (default: data/corr_cache)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/figures/correlation'),
        help='Output directory for figures (default: data/figures/correlation)',
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('data/stereoeeg_patients'),
        help='Root directory for patient data (default: data/stereoeeg_patients)',
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Generate all plots for all bands/phases of the patient',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages',
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.batch:
        if args.phase is None or args.band is None:
            parser.error("--phase and --band are required unless --batch is specified")

    _indent_warnings()

    # Run visualization
    try:
        if args.batch:
            visualize_batch(
                args.patient,
                args.cache_root,
                args.output_dir,
                args.dataset_root,
                args.cleaned,
                args.verbose,
            )
        else:
            if args.verbose:
                print(f"\n=== Visualization for {args.patient} {args.phase} {args.band} ===")
                print(f"  plot: {args.plot_type} (cleaned={args.cleaned})")
                print(f"  cache: {args.cache_root}/{args.patient}/")
                print(f"  output: {args.output_dir}/{args.patient}/")

            visualize_single(
                args.patient,
                args.phase,
                args.band,
                args.plot_type,
                args.cleaned,
                args.cache_root,
                args.output_dir,
                args.dataset_root,
                args.verbose,
            )

        if args.verbose:
            print("\n✓ Visualization complete!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
