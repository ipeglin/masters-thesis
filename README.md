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
10. **08classification**: Model classification

**utils**: Shared helpers.
**cli**: Pipeline CLI.

## Prerequisites
Rust 1.82.0+. Git-annex for large data. HDF5 for timeseries. Libtorch (PyTorch C++ API) for deep learning features.

## Setup & Initialization

Run main initialization script to prepare paths, atlases, and system environment.

```bash
# Initialize project paths, download atlases, and build dependencies
bash ./scripts/init.sh
```

### IDUN Cluster Setup

IDUN needs specific modules and environment prep. The `init.sh` script handles this if it detects IDUN. Trigger cluster mode by passing `idun` argument OR creating `.sys-idun` file.

```bash
# Option 1: Pass idun argument
bash ./scripts/init.sh idun

# Option 2: Empty trigger file
touch .sys-idun
bash ./scripts/init.sh
```
This auto-loads IDUN specific config defaults and builds HDF5 (if missing). 

**CRITICAL (IDUN):** After running `init.sh`, you must source the environment script to load Rust and CUDA modules into your current shell session:
```bash
source ./scripts/sys-idun_env.sh
```

This script will automatically detect if `LIBTORCH` is exported in your environment (e.g. from your `~/.bashrc`). If missing, it will automatically download the correct PyTorch CUDA binaries (`libtorch`) into `$HOME/libtorch`, and configure `LD_LIBRARY_PATH` for your session.

### Local Machine Setup

**CRITICAL:** Run `bash scripts/init.sh` on local (Mac/Windows/Linux). It copies `config.toml` and fetches atlases. But **you MUST manually edit `config.toml`** and set your local directory paths (`tcp_repo_dir`, `fmriprep_output_dir`, etc.) after script finishes.

**Local Libtorch Setup:** You must manually download and configure Libtorch to compile the CNN models (Stage 07).
1. Download the appropriate [PyTorch C++ Library (LibTorch)](https://pytorch.org/) (CPU, MPS/Apple Silicon, or CUDA depending on your system).
2. Extract the archive (e.g., to `$HOME/libtorch`).
3. Set the `LIBTORCH` and relative Linker paths in your shell environment before running `cargo build` or the pipeline scripts:
   ```bash
   export LIBTORCH=$HOME/libtorch
   export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH # (macOS)
   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH   # (Linux)
   ```

## Building

```bash
cargo build --release
```

## Usage

### Running on accelerated environment on IDUN using Slurm (Recommended)
For accelerated and significantly improved processing runtime, you should try to always use Slurm on IDUN.
The current repo will during initialization — as described above — fetch preconfigured slurm schemas made by the author, if and only if the system is detected to be running on IDUN.
The schemas will be written to `./slurm` relative to the project root, and username injection will be run automatically to insert your NTNU username on IDUN into paths in the schema.

These schemas may be added to the IDUN resource queue using the `sbatch` command. For example
```bash
sbatch ./slurm/run_pipeline.slurm
```

For additional information on using Slurm on IDUN, see [these student-written articles](https://www.hpc.ntnu.no/idun/documentation/#:~:text=Articles) in the official NTNU [IDUN Documentation](https://www.hpc.ntnu.no/idun/documentation/).

__You are of course welcome to modify your slurm schemas after installation. However, beware that rerunning `scripts/init.sh` may again overwrite your own configuration. Therefore, we recommend you to either rename your self-configured `.slurm` files, or move them to a new directory.__

### Running the pipeline locally
```bash
bash scripts/run-pipeline.sh
```

### Single-crate execution
__NB: It is important to note that pipeline steps are highly dependent on previous crates. The user is responsible for running pipeline steps in the appropriate order when executing crates separately.__

```bash
cargo run -- select-subjects
cargo run -- parcellate-bold --force # forcefully recompute parcellating even if precomputed
```

Complete list of CLI commands per crate:
| Crate                 | CLI Command        |
|-----------------------|--------------------|
| 00subject_selection   | select-subjects    |
| 01fmri_parcellation   | parcellate-bold    |
| 02fmri_segment_trials | segment-trials     |
| 03cwt                 | cwt                |
| 04mvmd                | mvmd               |
| 05hilbert             | hht                |
| 06fc                  | fc                 |
| 07feature_extraction  | feature-extraction |
| 08classification      | classify           |

## Configuration

For local machines, copy example config if `init.sh` didn't. Set paths for platform manually.

```bash
cp config.toml.example config.toml
```

Example local paths:
- `tcp_repo_dir` = `/Users/ipeglin/data/ds005237`
- `fmriprep_output_dir` = `/Users/ipeglin/data/fmriprep`
