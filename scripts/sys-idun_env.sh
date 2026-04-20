#!/bin/bash
# sys-idun_env.sh - Setup environment for Rust project with HDF5

HDF5_INSTALL_DIR="$HOME/hdf5"

# Check if HDF5 is already installed
if [ ! -f "$HDF5_INSTALL_DIR/lib/libhdf5.so" ]; then
    echo "Warning: HDF5 not found at $HDF5_INSTALL_DIR"
    echo "Run init.sh or sys-idun_build-hdf5.sh to build."
fi

# Load Rust module
echo "Loading Rust module..."
module load Rust/1.91.1-GCCcore-14.2.0

# Load PyTorch modole
module load CUDA/12.8.0

LIBTORCH_DIR="$HOME/libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
  cd $HOME
  # CUDA 12.8 (cxx11-abi variant)
  wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.11.0%2Bcu128.zip \
    || wget https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.11.0%2Bcu128.zip \
    || wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.11.0%2Bcpu.zip
  unzip libtorch-*.zip
  cd -
fi
export LIBTORCH="$LIBTORCH_DIR"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"


# Set HDF5 paths
echo "Setting HDF5 environment variables..."
export HDF5_DIR="$HDF5_INSTALL_DIR"

echo ""
echo "Environment ready!"
echo "HDF5_DIR=$HDF5_DIR"
echo "Rust version: $(rustc --version)"