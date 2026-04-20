#!/bin/bash
# sys-local_install-deps.sh - Install HDF5 and other dependencies on local machines

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sys-logger.sh"

log_step "Checking local dependencies"

if command -v h5cc >/dev/null 2>&1 || command -v h5pcc >/dev/null 2>&1; then
    log_success "HDF5 already installed"
    exit 0
fi

log_info "HDF5 not found. Attempting to install"
OS="$(uname -s)"
case "$OS" in
    Linux*)
        if command -v apt-get >/dev/null 2>&1; then
            log_info "Detected Debian/Ubuntu. Installing libhdf5-dev"
            sudo apt-get update && sudo apt-get install -y libhdf5-dev
        elif command -v dnf >/dev/null 2>&1; then
            log_info "Detected Fedora/RHEL. Installing hdf5-devel"
            sudo dnf install -y hdf5-devel
        elif command -v yum >/dev/null 2>&1; then
            log_info "Detected legacy RHEL/CentOS. Installing hdf5-devel"
            sudo yum install -y hdf5-devel
        elif command -v pacman >/dev/null 2>&1; then
            log_info "Detected Arch Linux. Installing hdf5"
            sudo pacman -S --noconfirm hdf5
        else
            log_warn "Unsupported Linux package manager. Install HDF5 manually"
        fi
        ;;
    Darwin*)
        if command -v brew >/dev/null 2>&1; then
            log_info "Detected macOS. Installing hdf5@1.10 via brew"
            brew install hdf5@1.10
        else
            log_warn "Homebrew not found. Install Homebrew or install HDF5 manually"
        fi
        ;;
    CYGWIN*|MINGW*|MSYS*|MINGW32*|MINGW64*)
        log_warn "Detected Windows. Please install HDF5 manually via vcpkg or download binaries:"
        log_info "https://support.hdfgroup.org/HDF5/release/obtain5.html"
        log_info "Set HDF5_DIR environment variable after installation."
        ;;
    *)
        log_err "Unknown OS: $OS. Install HDF5 manually"
        ;;
esac
