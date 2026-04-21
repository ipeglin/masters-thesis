#!/bin/bash
# sys-idun_env.sh - Setup environment for Rust project with HDF5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sys-logger.sh"

HDF5_INSTALL_DIR="$HOME/hdf5"

# Check if HDF5 is already installed
if [ ! -f "$HDF5_INSTALL_DIR/lib/libhdf5.so" ]; then
    log_warn "HDF5 not found at $HDF5_INSTALL_DIR"
    log_info "Run init.sh or sys-idun_build-hdf5.sh to build."
fi

# Load Rust module
log_info "Loading Rust module (1.91.1)"
module load Rust/1.91.1-GCCcore-14.2.0

# Load PyTorch module
log_info "Loading CUDA module (12.8.0)"
module load CUDA/12.8.0

# Respect user-defined LIBTORCH, default to $HOME/libtorch if not set
LIBTORCH_DIR="${LIBTORCH:-$HOME/libtorch}"

if [ ! -d "$LIBTORCH_DIR" ]; then
  log_info "Libtorch not found at $LIBTORCH_DIR. Downloading..."
  TARGET_PARENT="$(dirname "$LIBTORCH_DIR")"
  mkdir -p "$TARGET_PARENT"
  cd "$TARGET_PARENT"
  wget -q https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.11.0%2Bcu128.zip \
    || wget -q https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.11.0%2Bcu128.zip \
    || wget -q https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.11.0%2Bcpu.zip
  unzip -q libtorch-*.zip
  cd - >/dev/null
fi

log_info "Setting environment variables"
export LIBTORCH="$LIBTORCH_DIR"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
export HDF5_DIR="$HDF5_INSTALL_DIR"

log_success "Environment ready!"
log_info "HDF5_DIR=$HDF5_DIR"
log_info "Rust version: $(rustc --version)"