# TCP fMRI Preprocessing Pipeline

Rust fMRI preprocessing pipeline for TCP (Transdiagnostic Connectomes Project) dataset. 

## Pipeline Stages

Execution order (crates 00-09):
1. **00tcp_subject_selection**: Filter subjects
2. **01fmri_parcellation**: Parcellate regions
3. **02fmri_segment_trials**: Segment timeseries
4. **03cwt**: Wavelet transform
5. **04mvmd**: Multivariate Mode Decomposition 
6. **05hilbert**: Hilbert transform
7. **06fc**: Functional connectivity
8. **07feature_extraction**: Extract features (CNN)
9. **08data_splitting**: Train/test split
10. **09classification**: Model classification

**utils**: Shared helpers.
**cli**: Pipeline CLI.

## Prerequisites
Rust 1.82.0+. Git-annex for large data. HDF5 for timeseries.

## IDUN Cluster Setup

Remote IDUN needs environment prep before run. Source script to load modules, paths, build HDF5 if absent.

```bash
source setup_env.sh
```

**Cluster Paths logic:**
- Data & large files (`.nii`, `.pt`, `fmriprep_output_dir`): `/cluster/work/{username}/`
- Repos: `/cluster/home/{username}/`
- fMRIPrep files: `/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4`

## Building

```bash
cargo build --release
```

## Usage

```bash
cargo run --release -p cli -- --help
cargo run --release -p cli -- tcp-select-subjects
```

## Configuration

Copy example config. Set paths for platform.

```bash
cp config.toml.example config.toml
```

Use `tcp_repo_dir` = `/cluster/home/{username}/masters-thesis`
Use `fmriprep_output_dir` = `/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4`
Use output dirs in `/cluster/work/{username}/...`
