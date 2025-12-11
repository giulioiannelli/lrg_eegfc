#!/usr/bin/env python3
"""Inspect patient data and generate comprehensive report.

This script inspects all patient .mat files and metadata, generating a detailed
report about data completeness, sampling rates, file formats, and issues.

Usage:
    python src/inspect_patient_data.py
    python src/inspect_patient_data.py --output data_report.txt
    python src/inspect_patient_data.py --patients Pat_02 Pat_03
"""

import argparse
from pathlib import Path
from lrg_eegfc.utils.datamanag.inspect import inspect_all_patients, generate_report, save_report


def main():
    parser = argparse.ArgumentParser(
        description="Inspect patient data and generate report"
    )

    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("data/stereoeeg_patients"),
        help="Root directory containing patient data (default: data/stereoeeg_patients)"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Specific patients to inspect (default: auto-detect all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PATIENT_DATA_REPORT.txt"),
        help="Output report file (default: PATIENT_DATA_REPORT.txt)"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print report to console only (don't save file)"
    )

    args = parser.parse_args()

    print("Inspecting patient data...")
    print(f"Root path: {args.root_path}")

    # Run inspection
    results = inspect_all_patients(args.root_path, args.patients)

    # Generate report
    report = generate_report(results)

    # Output
    print(report)

    if not args.print_only:
        save_report(results, args.output)
        print(f"\n✓ Report saved to: {args.output}")
        print(f"\nSummary:")
        print(f"  Total patients inspected: {results['summary']['total_patients']}")
        print(f"  Patients with issues: {results['summary']['patients_with_issues']}")
        print(f"  Total issues found: {results['summary']['total_issues']}")


if __name__ == "__main__":
    main()
