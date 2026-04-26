#!/bin/bash
# sys-idun_fetch-slurmjobs.sh - Fetch slurm job schemas for running on IDUN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sys-logger.sh"

PROJECT_ROOT="$1"
SLURM_DEST_DIR="$2"

# --- JQ INSTALLATION ---
if ! command -v jq &> /dev/null; then
    log_info "jq not found. Attempting to install..."
    sudo yum install epel-release -y && sudo yum install jq -y
    if [ $? -ne 0 ]; then
        log_err "Failed to install jq. Please install it manually."
        exit 1
    fi
    log_success "jq installed successfully."
fi

# --- CONFIGURATION ---
GH_USER="ipeglin"
GH_REPO="masters-thesis-supplementary"
GH_REMOTE_DIR="slurm"

# Add "Key = Value" pairs to replace in the .slurm files
declare -A REPLACEMENTS
REPLACEMENTS["PROJECT_ROOT"]="$PROJECT_ROOT"

if [ -z "$PROJECT_ROOT" ] || [ -z "$SLURM_DEST_DIR" ]; then
    log_err "Usage: $0 <PROJECT_ROOT> <SLURM_DEST_DIR>"
    exit 1
fi

# 1. Get the list of files
API_URL="https://api.github.com/repos/$GH_USER/$GH_REPO/contents/$GH_REMOTE_DIR"
log_info "Fetching file list from $GH_REMOTE_DIR..."

FILES_JSON=$(curl -s -f "$API_URL")
if [ $? -ne 0 ]; then
    log_err "Failed to fetch directory listing from GitHub API."
    exit 1
fi

found_files=false

while read -r DOWNLOAD_URL; do
    if [ -n "$DOWNLOAD_URL" ] && [ "$DOWNLOAD_URL" != "null" ]; then
        found_files=true

        if [ ! -d "$SLURM_DEST_DIR" ]; then
            log_info "Creating destination directory: $SLURM_DEST_DIR"
            mkdir -p "$SLURM_DEST_DIR"
        fi

        FILENAME=$(basename "$DOWNLOAD_URL")
        TARGET_FILE="$SLURM_DEST_DIR/$FILENAME"

        log_info "Processing $FILENAME..."
        curl -L -s -f "$DOWNLOAD_URL" -o "$TARGET_FILE"

        if [ $? -eq 0 ]; then
            # --- VARIABLE INJECTION LOGIC ---
            # Loop through the REPLACEMENTS array and update the file
            for KEY in "${!REPLACEMENTS[@]}"; do
                VALUE="${REPLACEMENTS[$KEY]}"
                # This matches "KEY = " with or without spaces and replaces the whole line
                sed -i "s|^$KEY = .*|$KEY=\"$VALUE\"|g" "$TARGET_FILE"
            done

            log_success "Downloaded and configured $FILENAME"
            chmod +x "$TARGET_FILE"
        else
            log_err "Failed to download $FILENAME"
        fi
    fi
done < <(echo "$FILES_JSON" | jq -r '.[] | select(.type=="file") | .download_url')

if [ "$found_files" = false ]; then
    log_warn "No files found in $GH_REMOTE_DIR."
fi
