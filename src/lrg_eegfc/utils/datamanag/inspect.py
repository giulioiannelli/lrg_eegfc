"""Data inspection utilities for patient datasets.

This module provides tools to inspect and validate patient data, generating
comprehensive reports about data completeness and structure.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.io import loadmat
import h5py

from ...config.const import PHASE_LABELS


def inspect_mat_file(mat_path: Path) -> Dict:
    """Inspect a single .mat file and extract all metadata.

    Parameters
    ----------
    mat_path : Path
        Path to .mat file

    Returns
    -------
    Dict
        Dictionary with inspection results
    """
    result = {
        'exists': mat_path.exists(),
        'path': str(mat_path),
        'file_format': None,
        'keys': [],
        'data_shape': None,
        'data_variable': None,
        'fs': None,
        'filter_info': {},
        'parameters': {},
        'error': None,
    }

    if not mat_path.exists():
        result['error'] = 'File does not exist'
        return result

    # Try scipy first (standard MATLAB format)
    try:
        mat = loadmat(mat_path)
        result['file_format'] = 'scipy (MATLAB v5-v7)'
        result['keys'] = [k for k in mat.keys() if not k.startswith('__')]

        # Look for data variable
        data_keys = ['Data', 'data', 'EEG', 'timeseries', 'signal']
        for key in data_keys:
            if key in mat:
                result['data_variable'] = key
                data = np.asarray(mat[key])
                result['data_shape'] = data.shape
                break

        # Extract parameters
        if 'Parameters' in mat:
            params = mat['Parameters']
            if isinstance(params, np.ndarray) and params.dtype.names:
                for name in params.dtype.names:
                    try:
                        value = params[name][0, 0]
                        if isinstance(value, np.ndarray):
                            if value.size == 1:
                                value = value.item()
                            else:
                                value = value.tolist()
                        result['parameters'][name] = value

                        # Capture specific parameters
                        if name in ['fs', 'Fs', 'sampling_rate', 'SamplingRate']:
                            result['fs'] = float(value) if value is not None else None
                        elif 'filter' in name.lower():
                            result['filter_info'][name] = value
                    except:
                        pass

        # Look for standalone fs/Fs
        for fs_key in ['fs', 'Fs', 'sampling_rate', 'SamplingRate']:
            if fs_key in mat and result['fs'] is None:
                try:
                    result['fs'] = float(mat[fs_key])
                except:
                    pass

    except Exception as e_scipy:
        # Try h5py (MATLAB v7.3)
        try:
            with h5py.File(mat_path, 'r') as f:
                result['file_format'] = 'h5py (MATLAB v7.3)'
                result['keys'] = list(f.keys())

                # Look for data variable
                data_keys = ['Data', 'data', 'EEG', 'timeseries', 'signal']
                for key in data_keys:
                    if key in f:
                        result['data_variable'] = key
                        result['data_shape'] = f[key].shape
                        break

                # Extract parameters
                if 'Parameters' in f:
                    try:
                        params = f['Parameters']
                        for key in params.keys():
                            try:
                                value = params[key][()]
                                if isinstance(value, np.ndarray):
                                    if value.size == 1:
                                        value = value.item()
                                result['parameters'][key] = value

                                if key in ['fs', 'Fs', 'sampling_rate']:
                                    result['fs'] = float(value)
                                elif 'filter' in key.lower():
                                    result['filter_info'][key] = value
                            except:
                                pass
                    except:
                        pass

                # Look for standalone fs
                for fs_key in ['fs', 'Fs', 'sampling_rate']:
                    if fs_key in f and result['fs'] is None:
                        try:
                            result['fs'] = float(f[fs_key][()])
                        except:
                            pass

        except Exception as e_h5py:
            result['error'] = f'scipy: {str(e_scipy)[:50]}, h5py: {str(e_h5py)[:50]}'

    return result


def inspect_patient(patient: str, root_path: Path) -> Dict:
    """Inspect all data for a single patient.

    Parameters
    ----------
    patient : str
        Patient identifier
    root_path : Path
        Root directory containing patient subdirectories

    Returns
    -------
    Dict
        Comprehensive inspection results
    """
    patient_dir = root_path / patient
    result = {
        'patient': patient,
        'directory_exists': patient_dir.exists(),
        'phases': {},
        'metadata_files': {},
        'issues': [],
    }

    if not patient_dir.exists():
        result['issues'].append(f'Directory {patient_dir} does not exist')
        return result

    # Check phase files
    for phase in PHASE_LABELS:
        mat_file = patient_dir / f'{phase}.mat'
        phase_result = inspect_mat_file(mat_file)
        result['phases'][phase] = phase_result

        if not phase_result['exists']:
            result['issues'].append(f'Missing phase: {phase}')
        elif phase_result['error']:
            result['issues'].append(f'{phase}: {phase_result["error"]}')
        elif phase_result['data_variable'] is None:
            result['issues'].append(f'{phase}: No data variable found (keys: {phase_result["keys"]})')
        elif phase_result['fs'] is None:
            result['issues'].append(f'{phase}: No sampling rate (fs) found')

    # Check metadata files
    metadata_files = {
        'channel_labels_csv': patient_dir / 'channel_labels.csv',
        'channel_labels_txt': patient_dir / 'channel_labels.txt',
        'implant_csv': patient_dir / f'Implant_{patient.lower()}.csv',
        'implant_xlsx': patient_dir / f'Implant_{patient.lower()}.xlsx',
        'channel_names_mat': patient_dir / 'ChannelNames.mat',
    }

    for name, path in metadata_files.items():
        result['metadata_files'][name] = {
            'exists': path.exists(),
            'path': str(path) if path.exists() else None,
        }

    # Check if any channel labels exist
    has_channel_labels = any(
        result['metadata_files'][k]['exists']
        for k in ['channel_labels_csv', 'channel_labels_txt', 'channel_names_mat']
    )
    if not has_channel_labels:
        result['issues'].append('No channel label files found')

    return result


def inspect_all_patients(root_path: Path, patients: Optional[List[str]] = None) -> Dict:
    """Inspect all patients in the dataset.

    Parameters
    ----------
    root_path : Path
        Root directory containing patient subdirectories
    patients : List[str], optional
        List of patient IDs to inspect. If None, auto-detects from directory.

    Returns
    -------
    Dict
        Complete inspection results for all patients
    """
    if patients is None:
        # Auto-detect patients
        patients = sorted([
            p.name for p in root_path.iterdir()
            if p.is_dir() and p.name.startswith('Pat_')
        ])

    results = {
        'root_path': str(root_path),
        'patients': {},
        'summary': {
            'total_patients': len(patients),
            'patients_with_issues': 0,
            'total_issues': 0,
        }
    }

    for patient in patients:
        patient_result = inspect_patient(patient, root_path)
        results['patients'][patient] = patient_result

        if patient_result['issues']:
            results['summary']['patients_with_issues'] += 1
            results['summary']['total_issues'] += len(patient_result['issues'])

    return results


def generate_report(results: Dict) -> str:
    """Generate a human-readable text report from inspection results.

    Parameters
    ----------
    results : Dict
        Results from inspect_all_patients()

    Returns
    -------
    str
        Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STEREOEEG PATIENT DATA INSPECTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Root path: {results['root_path']}")
    lines.append(f"Total patients: {results['summary']['total_patients']}")
    lines.append(f"Patients with issues: {results['summary']['patients_with_issues']}")
    lines.append(f"Total issues: {results['summary']['total_issues']}")
    lines.append("=" * 80)

    for patient, patient_data in results['patients'].items():
        lines.append(f"\n{'=' * 80}")
        lines.append(f"PATIENT: {patient}")
        lines.append("=" * 80)

        if not patient_data['directory_exists']:
            lines.append("✗ Directory does not exist")
            continue

        # Phase data
        lines.append("\nPhase Data:")
        lines.append("-" * 80)
        for phase in PHASE_LABELS:
            phase_data = patient_data['phases'][phase]

            if not phase_data['exists']:
                lines.append(f"  {phase:12s} ✗ File missing")
                continue

            status = "✓" if phase_data['data_variable'] else "✗"
            lines.append(f"  {phase:12s} {status} File: {phase_data['file_format']}")

            if phase_data['data_variable']:
                lines.append(f"               Data variable: {phase_data['data_variable']}")
                lines.append(f"               Shape: {phase_data['data_shape']}")

                if phase_data['fs']:
                    lines.append(f"               Sampling rate: {phase_data['fs']} Hz")
                else:
                    lines.append(f"               Sampling rate: ⚠ NOT FOUND")

                if phase_data['filter_info']:
                    lines.append(f"               Filter info:")
                    for key, val in phase_data['filter_info'].items():
                        lines.append(f"                 - {key}: {val}")

                if phase_data['parameters']:
                    lines.append(f"               Other parameters: {', '.join(phase_data['parameters'].keys())}")

            if phase_data['error']:
                lines.append(f"               Error: {phase_data['error']}")

            if phase_data['keys']:
                lines.append(f"               Available keys: {', '.join(phase_data['keys'][:10])}")

        # Metadata files
        lines.append("\nMetadata Files:")
        lines.append("-" * 80)
        for name, info in patient_data['metadata_files'].items():
            status = "✓" if info['exists'] else "✗"
            display_name = name.replace('_', ' ').title()
            lines.append(f"  {display_name:25s} {status}")
            if info['exists']:
                lines.append(f"                            {info['path']}")

        # Issues summary
        if patient_data['issues']:
            lines.append("\nIssues Found:")
            lines.append("-" * 80)
            for issue in patient_data['issues']:
                lines.append(f"  ⚠ {issue}")
        else:
            lines.append("\n✓ No issues found for this patient")

    # Global summary
    lines.append(f"\n{'=' * 80}")
    lines.append("SUMMARY OF DIVERGENCES BETWEEN PATIENTS")
    lines.append("=" * 80)

    # Sampling rates
    lines.append("\nSampling Rates:")
    fs_by_patient = {}
    for patient, patient_data in results['patients'].items():
        if patient_data['directory_exists']:
            fs_values = set()
            for phase, phase_data in patient_data['phases'].items():
                if phase_data['exists'] and phase_data['fs']:
                    fs_values.add(phase_data['fs'])
            if fs_values:
                fs_by_patient[patient] = fs_values

    unique_fs = set()
    for fs_set in fs_by_patient.values():
        unique_fs.update(fs_set)

    for fs in sorted(unique_fs):
        patients_with_fs = [p for p, fs_set in fs_by_patient.items() if fs in fs_set]
        lines.append(f"  {fs} Hz: {', '.join(patients_with_fs)}")

    # Data variable names
    lines.append("\nData Variable Names Used:")
    var_names = {}
    for patient, patient_data in results['patients'].items():
        if patient_data['directory_exists']:
            for phase, phase_data in patient_data['phases'].items():
                if phase_data['exists'] and phase_data['data_variable']:
                    var_name = phase_data['data_variable']
                    if var_name not in var_names:
                        var_names[var_name] = []
                    var_names[var_name].append(f"{patient}/{phase}")

    for var_name, occurrences in var_names.items():
        lines.append(f"  '{var_name}': {len(occurrences)} occurrences")
        if len(occurrences) <= 5:
            lines.append(f"           {', '.join(occurrences)}")

    # File formats
    lines.append("\nFile Formats:")
    formats = {}
    for patient, patient_data in results['patients'].items():
        if patient_data['directory_exists']:
            for phase, phase_data in patient_data['phases'].items():
                if phase_data['exists'] and phase_data['file_format']:
                    fmt = phase_data['file_format']
                    if fmt not in formats:
                        formats[fmt] = []
                    formats[fmt].append(f"{patient}/{phase}")

    for fmt, occurrences in formats.items():
        lines.append(f"  {fmt}: {len(occurrences)} files")

    # Recommendations
    lines.append(f"\n{'=' * 80}")
    lines.append("RECOMMENDATIONS FOR COLLABORATORS")
    lines.append("=" * 80)

    missing_data = []
    missing_fs = []
    missing_labels = []

    for patient, patient_data in results['patients'].items():
        if not patient_data['directory_exists']:
            continue

        # Check missing phases
        for phase in PHASE_LABELS:
            phase_data = patient_data['phases'][phase]
            if not phase_data['exists'] or phase_data['data_variable'] is None:
                missing_data.append(f"{patient}/{phase}")
            elif phase_data['fs'] is None:
                missing_fs.append(f"{patient}/{phase}")

        # Check missing labels
        has_labels = any(
            patient_data['metadata_files'][k]['exists']
            for k in ['channel_labels_csv', 'channel_labels_txt', 'channel_names_mat']
        )
        if not has_labels:
            missing_labels.append(patient)

    if missing_data:
        lines.append("\n1. Missing or corrupted phase data:")
        for item in missing_data:
            lines.append(f"   - {item}")

    if missing_fs:
        lines.append("\n2. Missing sampling rate (fs) information:")
        for item in missing_fs:
            lines.append(f"   - {item}")
        lines.append("   Please add 'fs' field to Parameters struct or as standalone variable")

    if missing_labels:
        lines.append("\n3. Missing channel label files:")
        for patient in missing_labels:
            lines.append(f"   - {patient}: Please provide channel_labels.csv or ChannelNames.mat")

    if not (missing_data or missing_fs or missing_labels):
        lines.append("\n✓ All data appears complete!")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def save_report(results: Dict, output_path: Path):
    """Save inspection report to a text file.

    Parameters
    ----------
    results : Dict
        Results from inspect_all_patients()
    output_path : Path
        Path to output file
    """
    report = generate_report(results)
    output_path.write_text(report)
    print(f"Report saved to: {output_path}")
