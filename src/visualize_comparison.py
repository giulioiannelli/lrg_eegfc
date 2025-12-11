#!/usr/bin/env python3
"""CLI script for FC method comparison visualizations.

Examples
--------
# Single comparison
python src/visualize_comparison.py --patient Pat_02 --phase rsPre --band beta --verbose

# Batch mode: all bands for a phase
python src/visualize_comparison.py --patient Pat_02 --phase rsPre --batch --verbose
"""

import argparse
import sys
from pathlib import Path

from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS
from lrg_eegfc.visuals.compare import plot_fc_comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare correlation vs MSC functional connectivity methods"
    )

    parser.add_argument("--patient", required=True, help="Patient ID")
    parser.add_argument("--phase", choices=PHASE_LABELS, help="Recording phase")
    parser.add_argument("--band", choices=list(BRAIN_BANDS.keys()), help="Frequency band")

    parser.add_argument(
        "--nperseg",
        type=int,
        default=1024,
        help="MSC window length (must match cache). Default: 1024",
    )

    parser.add_argument(
        "--cache-root-corr",
        type=Path,
        default=Path("data/corr_cache"),
        help="Correlation cache directory",
    )

    parser.add_argument(
        "--cache-root-msc",
        type=Path,
        default=Path("data/msc_cache"),
        help="MSC cache directory",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate for all bands (requires --phase)",
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Validation
    if not args.batch and (not args.phase or not args.band):
        parser.error("--phase and --band required unless using --batch")

    # Determine combinations
    bands = list(BRAIN_BANDS.keys()) if args.batch else [args.band]
    phases = list(PHASE_LABELS) if (args.batch and not args.phase) else ([args.phase] if args.phase else [])

    if args.verbose:
        output_dir = Path("data/figures/comparison") / args.patient
        print(f"\n=== FC comparison for {args.patient} ===")
        print(f"  bands: {', '.join(bands)}")
        print(f"  phases: {', '.join(phases)}")
        print(f"  corr cache: {args.cache_root_corr}/{args.patient}/")
        print(f"  msc cache:  {args.cache_root_msc}/{args.patient}/")
        print(f"  output:     {output_dir}/")

    # Process
    for band in bands:
        for phase in phases:
            expected_path = Path("data/figures/comparison") / args.patient / f"{band}_{phase}_fc_comparison.png"
            if expected_path.exists():
                if args.verbose:
                    print(f"• Skipping (exists): {expected_path.name}")
                continue
            try:
                output_path = plot_fc_comparison(
                    patient=args.patient,
                    phase=phase,
                    band=band,
                    cache_root_corr=args.cache_root_corr,
                    cache_root_msc=args.cache_root_msc,
                    nperseg=args.nperseg,
                    output_path=expected_path,
                )

                if args.verbose:
                    print(f"✓ FC comparison: {args.patient} {phase} {band} -> {output_path.name}")

            except FileNotFoundError as e:
                print(f"  ERROR: {e}")
            except Exception as e:
                print(f"  ERROR: Failed for {args.patient} {phase} {band}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
