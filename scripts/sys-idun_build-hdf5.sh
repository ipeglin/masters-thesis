#!/bin/bash
# sys-idun_build-hdf5.sh - Build HDF5 1.12.3 without parallel support

set -e  # Exit on error

echo "=== Building HDF5 1.12.3 ==="

# Load required modules
echo "Loading CMake..."
module load CMake/3.31.3-GCCcore-14.2.0

# Set paths
HDF5_SRC_DIR="$HOME/Downloads/hdfsrc"
HDF5_INSTALL_DIR="$HOME/hdf5"

# Check if source exists
if [ ! -f "$HDF5_SRC_DIR/CMakeLists.txt" ]; then
    echo "Error: HDF5 source not found at $HDF5_SRC_DIR"
    echo "Please download and extract HDF5 to $HDF5_SRC_DIR/"
    exit 1
fi

# Create build directory
cd "$HDF5_SRC_DIR"
rm -rf build
mkdir build
cd build

echo "Configuring HDF5..."
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$HDF5_INSTALL_DIR" \
  -DBUILD_SHARED_LIBS=ON \
  -DHDF5_ENABLE_PARALLEL=OFF \
  -DHDF5_BUILD_TOOLS=OFF \
  -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_CPP_LIB=OFF \
  -DHDF5_BUILD_JAVA=OFF \
  -DHDF5_BUILD_DOC=OFF \
  -DHDF5_ENABLE_PLUGIN_SUPPORT=OFF \
  -DHDF5_ALLOW_EXTERNAL_SUPPORT=NONE \
  -DHDF5_ENABLE_ZLIB_SUPPORT=ON

echo "Building HDF5 (this may take a few minutes)..."
make -j8

echo "Installing HDF5 to $HDF5_INSTALL_DIR..."
make install

echo ""
echo "=== HDF5 build complete ==="
echo "Installation directory: $HDF5_INSTALL_DIR"