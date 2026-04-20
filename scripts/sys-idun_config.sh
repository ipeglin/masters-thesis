#!/bin/bash
# sys-idun_config.sh - Set IDUN default paths in config.toml

CONFIG_FILE="$1"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file $CONFIG_FILE not found."
    exit 1
fi

echo ">> Configuring IDUN specific paths in config.toml..."

# Use sed to replace empty paths with IDUN defaults
sed -i.bak \
    -e "s|^tcp_repo_dir = .*|tcp_repo_dir = \"/cluster/work/$USER/ds005237\"|" \
    -e "s|^fmriprep_output_dir = .*|fmriprep_output_dir = \"/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4\"|" \
    -e "s|^parcellated_ts_dir = .*|parcellated_ts_dir = \"/cluster/work/$USER/preprocessing/tcp/parcellations\"|" \
    -e "s|^subject_filter_dir = .*|subject_filter_dir = \"/cluster/work/$USER/preprocessing/tcp/subject_filters\"|" \
    -e "s|^task_regressors_output_dir = .*|task_regressors_output_dir = \"/cluster/work/$USER/preprocessing/tcp/task_regressors\"|" \
    -e "s|^training_subjects_path = .*|training_subjects_path = \"/cluster/work/$USER/processing/tcp/classifier_models/subjects_train.csv\"|" \
    -e "s|^test_subjects_path = .*|test_subjects_path = \"/cluster/work/$USER/processing/tcp/classifier_models/subjects_test.csv\"|" \
    -e "s|^validation_subjects_path = .*|validation_subjects_path = \"/cluster/work/$USER/processing/tcp/classifier_models/subjects_validation.csv\"|" \
    "$CONFIG_FILE"

rm -f "${CONFIG_FILE}.bak"
echo ">> IDUN config defaults applied."
