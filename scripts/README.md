# Pipeline Scripts Guide

## Overview

This directory contains modular scripts for running the LRG EEG FC analysis pipeline. You can run the entire pipeline or execute individual steps for quick testing.

## Scripts

### `config.sh`
Configuration file with default parameters:
- **Test mode defaults**: `Pat_03`, `rsPre`, `beta`
- **Batch patients**: `Pat_02`, `Pat_03`
- **Frequency bands**: All 6 bands (delta through high_gamma)

Edit this file to change defaults.

### `run_step.sh`
**Modular step runner** - Run individual pipeline steps with test or batch mode.

### `run_full_analysis.sh`
**Complete pipeline** - Run all 12 steps for all patients (takes hours).

## Usage

### Quick Testing (Single Case)

Test a visualization step on one case (`Pat_03/rsPre/beta`):

```bash
# Test MSC visualization
bash scripts/run_step.sh --test --step 5b

# Test LRG visualization (correlation-based)
bash scripts/run_step.sh --test --step 9

# Test metastable nodes visualization
bash scripts/run_step.sh --test --step meta
```

### Custom Test Case

Override defaults with custom patient/phase/band:

```bash
# Test on Pat_02, taskLearn, alpha band
bash scripts/run_step.sh --test --step 9 --patient Pat_02 --phase taskLearn --band alpha

# Test MSC-based LRG
bash scripts/run_step.sh --test --step 10 --fc-method msc
```

### Batch Mode (All Bands/Phases for a Patient)

Run a step for all bands and phases of one patient:

```bash
# Compute all correlation matrices for Pat_03
bash scripts/run_step.sh --step 1 --patient Pat_03

# Generate all MSC visualizations for Pat_02
bash scripts/run_step.sh --step 5b --patient Pat_02

# All LRG visualizations (correlation) for Pat_03
bash scripts/run_step.sh --step 9 --patient Pat_03
```

### Full Pipeline (All Patients)

Run the complete 12-step pipeline:

```bash
bash scripts/run_full_analysis.sh
```

**Warning**: This processes both patients through all steps and takes several hours.

## Available Steps

List all available steps:

```bash
bash scripts/run_step.sh --list
```

**Pipeline Steps:**

| Step | Description | Fast/Slow |
|------|-------------|-----------|
| `1` | Compute correlation matrices | Fast (~1min) |
| `2` | Clean correlation matrices | Fast (~30s) |
| `3` | Visualize correlations | Fast (~1min) |
| `5` | Compute MSC matrices | Medium (~5min) |
| `5b` | Visualize MSC (summary) | Fast (~30s) |
| `5c` | Visualize FC comparison | Fast (~30s) |
| `6` | Compare FC methods (distances) | Medium (~2min) |
| `7` | Compute LRG (Correlation) | Slow (~10min) |
| `8` | Compute LRG (MSC) | Slow (~10min) |
| `9` | Visualize LRG (Correlation) | Fast (~30s) |
| `10` | Visualize LRG (MSC) | Fast (~30s) |
| `11` | Phase reorganization (Correlation) | Medium (~2min) |
| `12` | Phase reorganization (MSC) | Medium (~2min) |
| `meta` | Metastable nodes (Sankey) | Medium (~3min) |

## Common Workflows

### 1. Testing New Visualization

```bash
# Quick test on single case
bash scripts/run_step.sh --test --step 5b

# Check output
ls -lh data/figures/msc/Pat_03/beta_rsPre_msc_summary.png

# If good, run for all bands/phases
bash scripts/run_step.sh --step 5b --patient Pat_03
```

### 2. Debugging Computation Issues

```bash
# Test computation on single band
bash scripts/run_step.sh --test --step 7 --band beta

# Check cache
ls -lh data/lrg_cache/Pat_03/beta_rsPre_lrg_corr.npz

# If successful, run full computation
bash scripts/run_step.sh --step 7 --patient Pat_03
```

### 3. Comparing FC Methods

```bash
# Ensure both FC methods are computed (steps 1 and 5)
bash scripts/run_step.sh --step 1 --patient Pat_03
bash scripts/run_step.sh --step 5 --patient Pat_03

# Compute LRG for both methods (steps 7 and 8)
bash scripts/run_step.sh --step 7 --patient Pat_03
bash scripts/run_step.sh --step 8 --patient Pat_03

# Generate comparison visualizations
bash scripts/run_step.sh --step 5c --patient Pat_03
bash scripts/run_step.sh --step 6 --patient Pat_03
```

### 4. Complete Analysis for One Patient

Run all steps sequentially for a single patient:

```bash
for step in 1 2 3 5 5b 5c 6 7 8 9 10 11 12; do
  bash scripts/run_step.sh --step $step --patient Pat_03
done
```

## Output Locations

All outputs follow the pattern: `data/{type}/{patient}/`

- **Correlation cache**: `data/corr_cache/Pat_XX/`
- **Correlation figures**: `data/figures/correlation/Pat_XX/`
- **MSC cache**: `data/msc_cache/Pat_XX/`
- **MSC figures**: `data/figures/msc/Pat_XX/`
- **FC comparisons**: `data/figures/comparison/Pat_XX/`
- **LRG cache**: `data/lrg_cache/Pat_XX/`
- **LRG figures**: `data/figures/lrg/Pat_XX/`
- **Reorganization**: `data/figures/reorganization/Pat_XX/`
- **Metastable**: `data/figures/metastable/Pat_XX/`

## Tips

1. **Always test first**: Use `--test` mode before batch processing
2. **Check outputs**: Verify figures before running full pipeline
3. **Monitor time**: Fast steps (<1min) can run batch, slow steps test first
4. **Use caching**: Most scripts cache results, rerunning is fast
5. **Verbose mode**: Scripts run with `--verbose` for progress tracking

## Editing Defaults

Edit `scripts/config.sh` to change:
- Test patient/phase/band
- Batch patient list
- Default FC method
- Color scheme for output

## Examples

```bash
# Quick visualization check (30 seconds)
bash scripts/run_step.sh --test --step 5b

# Full MSC pipeline for one patient (15 minutes)
bash scripts/run_step.sh --step 5 --patient Pat_03
bash scripts/run_step.sh --step 5b --patient Pat_03

# Test LRG on different bands (3 minutes each)
bash scripts/run_step.sh --test --step 9 --band alpha
bash scripts/run_step.sh --test --step 9 --band beta
bash scripts/run_step.sh --test --step 9 --band gamma

# Complete pipeline (hours)
bash scripts/run_full_analysis.sh
```
