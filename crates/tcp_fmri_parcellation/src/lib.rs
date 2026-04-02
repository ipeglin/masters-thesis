use anyhow::Result;
use config::bids_filename::{BidsFilename, find_bids_files};
use config::bids_subject_id::BidsSubjectId;
use config::{TcpFmriParcellationConfig, polars_csv};
use ndarray::{Array2, Axis, concatenate, s};
use nifti_masker::{LabelsMasker, MaskerSignalConfig, Standardize, preprocess_signals};
use polars::prelude::*;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, error, info, info_span, warn};

pub fn run(cfg: &TcpFmriParcellationConfig) -> Result<()> {
    // Check that the fmri dir is even present.
    // If not, fail gracefully and inform the user that the
    // disk might not be connected, or the network disk is not opened.
    let fmri_dir = &cfg.fmri_dir;
    match fs::read_dir(fmri_dir) {
        Ok(_) => { /* Process entries */ }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            error!(
                fmri_dir = %fmri_dir.display(),
                "Directory not found: {}. Make sure to have the disk connected, or connecting to the network drive", fmri_dir.display()
            );
            return Ok(());
        }
        Err(e) => panic!("Failed to read directory: {}", e),
    }

    let run_start = Instant::now();

    info!(
        fmri_dir = %cfg.fmri_dir.display(),
        filter_dir = %cfg.filter_dir.display(),
        output_dir = %cfg.output_dir.display(),
        cortical_atlas = %cfg.cortical_atlas.display(),
        subcortical_atlas = %cfg.subcortical_atlas.display(),
        dry_run = cfg.dry_run,
        force = cfg.force,
        "starting fMRI preprocessing pipeline"
    );

    // Load subject filter files
    let filter_dir = &cfg.filter_dir;
    let filtered_subjects = [
        filter_dir.join("healthy_controls.csv"),
        filter_dir.join("anhedonic.csv"),
    ];

    let dataframes: Vec<LazyFrame> = filtered_subjects
        .iter()
        .filter_map(|file| {
            polars_csv::read_dataframe(file)
                .map_err(|e| {
                    warn!(
                        file = %file.display(),
                        error = %e,
                        "failed to read filter file"
                    )
                })
                .ok()
                .map(|df| df.lazy())
        })
        .collect();

    let subjects = concat(dataframes, UnionArgs::default())?
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any)
        .collect()?;
    let subject_keys = subjects.column("subjectkey")?.str()?;
    let total_subjects = subject_keys.len();

    info!(
        total_subjects = total_subjects,
        filter_files_loaded = filtered_subjects.len(),
        "loaded subject keys"
    );

    // Validate atlases
    if !cfg.cortical_atlas.exists() || !cfg.subcortical_atlas.exists() {
        panic!("failed to locate atlases");
    }

    std::fs::create_dir_all(&cfg.output_dir)?;

    // Processing state
    let mut processed_count = 0usize;
    let mut skipped_count = 0usize;
    let error_count = 0usize;

    for (i, subject_key) in subject_keys.into_iter().flatten().enumerate() {
        let subject_idx = i + 1;
        let dir_name = BidsSubjectId::parse(subject_key).to_dir_name();
        let subject_dir = fmri_dir.join(&dir_name);

        // Create a span for the entire subject processing - this is the "wide event"
        let _subject_span = info_span!(
            "process_subject",
            subject_key = subject_key,
            subject_idx = subject_idx,
            total_subjects = total_subjects,
            subject_dir = %subject_dir.display(),
        )
        .entered();

        if !subject_dir.is_dir() {
            skipped_count += 1;
            warn!(
                subject_key = subject_key,
                subject_idx = subject_idx,
                total_subjects = total_subjects,
                reason = "missing_fmri_data",
                subject_dir = %subject_dir.display(),
                "skipping subject"
            );
            continue;
        }

        let mni_results_dir = subject_dir.join("func");

        let hammer_scan_files = find_bids_files(
            &mni_results_dir,
            &[
                ("task", "hammerAP"),
                ("space", "MNI152NLin2009cAsym"),
                ("res", "2"),
                ("desc", "preproc"),
            ],
            Some("bold"),
            Some(".nii.gz"),
        );
        let resting_scan_files = find_bids_files(
            &mni_results_dir,
            &[
                ("task", "restAP"),
                ("space", "MNI152NLin2009cAsym"),
                ("res", "2"),
                ("desc", "preproc"),
            ],
            Some("bold"),
            Some(".nii.gz"),
        );
        let files_to_preprocess: Vec<PathBuf> = hammer_scan_files
            .into_iter()
            .chain(resting_scan_files.into_iter())
            .collect();

        for file_path in files_to_preprocess {
            if !file_path.exists() {
                skipped_count += 1;
                warn!(
                    subject_key = subject_key,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    reason = "missing_bold_file",
                    file_path = %file_path.display(),
                    "skipping subject file"
                );
                continue;
            }

            let bids_filename = BidsFilename::parse(
                file_path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
            );
            let task_name = bids_filename.get("task").unwrap_or("unknown");
            let output_stem = bids_filename.to_stem();

            // Check if output already exists (skip unless --force)
            let output_path = cfg
                .output_dir
                .join(BidsSubjectId::parse(subject_key).to_dir_name())
                .join(format!("{}.h5", output_stem));
            if output_path.exists() && !cfg.force {
                skipped_count += 1;
                info!(
                    subject_key = subject_key,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    task_name = task_name,
                    reason = "already_preprocessed",
                    output_file = %output_path.display(),
                    "skipping file (already preprocessed, use --force to reprocess)"
                );
                continue;
            }

            let file_start = Instant::now();

            debug!(
                subject_key = subject_key,
                task_name = task_name,
                file_path = %file_path.display(),
                "starting parcellation"
            );

            // Extract raw (no preprocessing) parcellated signals from each atlas.
            // All preprocessing variants are derived from these raw arrays so that
            // the NIfTI file is only read and resampled once per atlas.
            let raw_config = MaskerSignalConfig::default(); // detrend=false, standardize=None

            // Cortical parcellation (raw)
            let cortical_start = Instant::now();
            let cortical_masker =
                LabelsMasker::with_config(&cfg.cortical_atlas, raw_config.clone())?;
            let cortical_raw = cortical_masker.fit_transform(&file_path)?;
            let cortical_duration_ms = cortical_start.elapsed().as_millis();

            debug!(
                subject_key = subject_key,
                atlas_type = "cortical",
                n_rois = cortical_raw.shape()[0],
                n_timepoints = cortical_raw.shape()[1],
                duration_ms = cortical_duration_ms,
                atlas_path = %cfg.cortical_atlas.display(),
                "raw parcellation completed"
            );

            // Subcortical parcellation (raw)
            let subcortical_start = Instant::now();
            let subcortical_masker =
                LabelsMasker::with_config(&cfg.subcortical_atlas, raw_config.clone())?;
            let subcortical_raw = subcortical_masker.fit_transform(&file_path)?;
            let subcortical_duration_ms = subcortical_start.elapsed().as_millis();

            debug!(
                subject_key = subject_key,
                atlas_type = "subcortical",
                n_rois = subcortical_raw.shape()[0],
                n_timepoints = subcortical_raw.shape()[1],
                duration_ms = subcortical_duration_ms,
                atlas_path = %cfg.subcortical_atlas.display(),
                "raw parcellation completed"
            );

            // Derive all preprocessing permutations from the raw signal.
            //
            // Variant 1: detrend=false, standardize=None  (raw — already computed above)
            // Variant 2: detrend=true,  standardize=None
            // Variant 3: detrend=false, standardize=ZscoreSample
            // Variant 4: detrend=true,  standardize=ZscoreSample  (nilearn default)

            // let detrended_config = MaskerSignalConfig::default().detrend(true);
            // let standardized_config =
            //     MaskerSignalConfig::default().standardize(Standardize::ZscoreSample);
            // let detrended_standardized_config = MaskerSignalConfig::with_defaults();

            // let cortical_detrended = preprocess_signals(&cortical_raw, &detrended_config);
            // let cortical_standardized = preprocess_signals(&cortical_raw, &standardized_config);
            // let cortical_detrended_standardized =
            //     preprocess_signals(&cortical_raw, &detrended_standardized_config);

            // let subcortical_detrended = preprocess_signals(&subcortical_raw, &detrended_config);
            // let subcortical_standardized =
            //     preprocess_signals(&subcortical_raw, &standardized_config);
            // let subcortical_detrended_standardized =
            //     preprocess_signals(&subcortical_raw, &detrended_standardized_config);

            // Debug: Print first few raw values
            debug!(
                subject_key = subject_key,
                cortical_raw_first_roi_first_5_timepoints = ?cortical_raw.slice(s![0, ..5]),
                subcortical_raw_first_roi_first_5_timepoints = ?subcortical_raw.slice(s![0, ..5]),
                "raw timeseries sample values"
            );

            // Write output
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // Remove existing file before (re)creating — HDF5 cannot acquire a lock
            // on a file that already exists on some filesystems (errno 35).
            if output_path.exists() {
                fs::remove_file(&output_path)?;
            }

            let write_start = Instant::now();
            write_timeseries_h5(
                &output_path,
                &cortical_raw,
                &subcortical_raw,
                // &cortical_detrended,
                // &subcortical_detrended,
                // &cortical_standardized,
                // &subcortical_standardized,
                // &cortical_detrended_standardized,
                // &subcortical_detrended_standardized,
            )?;
            let write_duration_ms = write_start.elapsed().as_millis();

            let total_duration_ms = file_start.elapsed().as_millis();
            processed_count += 1;

            // Wide event: one comprehensive log per subject file processed
            info!(
                subject_key = subject_key,
                subject_idx = subject_idx,
                total_subjects = total_subjects,
                task_name = task_name,
                input_file = %file_path.display(),
                output_file = %output_path.display(),
                cortical_rois = cortical_raw.shape()[0],
                subcortical_rois = subcortical_raw.shape()[0],
                n_timepoints = cortical_raw.shape()[1],
                cortical_duration_ms = cortical_duration_ms,
                subcortical_duration_ms = subcortical_duration_ms,
                write_duration_ms = write_duration_ms,
                total_duration_ms = total_duration_ms,
                outcome = "success",
                "subject processed"
            );
        }
    }

    let run_duration_ms = run_start.elapsed().as_millis();

    // Final summary wide event
    info!(
        total_subjects = total_subjects,
        processed_count = processed_count,
        skipped_count = skipped_count,
        error_count = error_count,
        total_duration_ms = run_duration_ms,
        output_dir = %cfg.output_dir.display(),
        outcome = if error_count == 0 { "success" } else { "completed_with_errors" },
        "fMRI preprocessing pipeline completed"
    );

    Ok(())
}

/// Write a single 2-D cortical/subcortical pair to an open HDF5 file.
fn write_array_pair(
    file: &hdf5::File,
    cortical: &Array2<f32>,
    cortical_name: &str,
    subcortical: &Array2<f32>,
    subcortical_name: &str,
) -> Result<()> {
    let c_shape = cortical.shape();
    let c_ds = file
        .new_dataset::<f32>()
        .shape([c_shape[0], c_shape[1]])
        .create(cortical_name)?;
    c_ds.write_raw(cortical.as_slice().unwrap())?;

    let s_shape = subcortical.shape();
    let s_ds = file
        .new_dataset::<f32>()
        .shape([s_shape[0], s_shape[1]])
        .create(subcortical_name)?;
    s_ds.write_raw(subcortical.as_slice().unwrap())?;

    Ok(())
}

/// Write all four preprocessing permutations to a single HDF5 file.
///
/// Dataset naming convention:
///   tcp_{cortical|subcortical}_raw                    — no detrending, no standardization
///   tcp_{cortical|subcortical}_detrended              — detrended only
///   tcp_{cortical|subcortical}_standardized           — z-score standardized only
///   tcp_{cortical|subcortical}_detrended_standardized — detrended then z-score standardized
#[allow(clippy::too_many_arguments)]
fn write_timeseries_h5(
    path: &Path,
    cortical_raw: &Array2<f32>,
    subcortical_raw: &Array2<f32>,
    // cortical_detrended: &Array2<f32>,
    // subcortical_detrended: &Array2<f32>,
    // cortical_standardized: &Array2<f32>,
    // subcortical_standardized: &Array2<f32>,
    // cortical_detrended_standardized: &Array2<f32>,
    // subcortical_detrended_standardized: &Array2<f32>,
) -> Result<()> {
    let file = hdf5::File::create(path)?;

    write_array_pair(
        &file,
        cortical_raw,
        "tcp_cortical_raw",
        subcortical_raw,
        "tcp_subcortical_raw",
    )?;

    let timeseries_raw = concatenate(Axis(0), &[cortical_raw.view(), subcortical_raw.view()])?;
    let ts_shape = timeseries_raw.shape();
    let ts_ds = file
        .new_dataset::<f32>()
        .shape([ts_shape[0], ts_shape[1]])
        .create("tcp_timeseries_raw")?;
    ts_ds.write_raw(timeseries_raw.as_slice().unwrap())?;

    // write_array_pair(
    //     &file,
    //     cortical_detrended,
    //     "tcp_cortical_detrended",
    //     subcortical_detrended,
    //     "tcp_subcortical_detrended",
    // )?;

    // write_array_pair(
    //     &file,
    //     cortical_standardized,
    //     "tcp_cortical_standardized",
    //     subcortical_standardized,
    //     "tcp_subcortical_standardized",
    // )?;

    // write_array_pair(
    //     &file,
    //     cortical_detrended_standardized,
    //     "tcp_cortical_detrended_standardized",
    //     subcortical_detrended_standardized,
    //     "tcp_subcortical_detrended_standardized",
    // )?;

    Ok(())
}
