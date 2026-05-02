mod nifti_masker;

use anyhow::Result;
use ndarray::{Array2, ArrayBase, ArrayView1, Axis, Dim, OwnedRepr, concatenate, s};
use nifti_masker::{LabelsMasker, MaskerSignalConfig, Standardize, preprocess_signals};
use polars::prelude::*;
use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, error, info, info_span, warn};
use utils::bids_filename::{BidsFilename, find_bids_files};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::polars_csv;

/// Tracks which root-level datasets are absent from an existing HDF5 file.
///
/// Three groups are tracked independently:
///   - raw: parcellated signal with no preprocessing
///   - std: post-parcellation z-score standardization (per-ROI temporal z-score)
///   - voxel_zscore: voxel-wise z-score normalization applied before parcellation
///
/// Partial files can be extended without re-running the NIfTI masker.
#[derive(Debug)]
struct MissingDatasets {
    // cortical_raw: bool,
    // subcortical_raw: bool,
    full_run_raw: bool,
    // cortical_std: bool,
    // subcortical_std: bool,
    full_run_std: bool,
    // cortical_voxel_zscore: bool,
    // subcortical_voxel_zscore: bool,
    // timeseries_voxel_zscore: bool,
}

impl MissingDatasets {
    /// All datasets need to be written (fresh file or --force).
    /// `voxelwise_zscore` controls whether the voxel-zscore group is included.
    fn all(voxelwise_zscore: bool) -> Self {
        Self {
            // cortical_raw: true,
            // subcortical_raw: true,
            full_run_raw: true,
            // cortical_std: true,
            // subcortical_std: true,
            full_run_std: true,
            // cortical_voxel_zscore: voxelwise_zscore,
            // subcortical_voxel_zscore: voxelwise_zscore,
            // timeseries_voxel_zscore: voxelwise_zscore,
        }
    }

    /// True when no dataset needs to be written.
    fn all_present(&self) -> bool {
        // !self.cortical_raw
        // && !self.subcortical_raw
        !self.full_run_raw
            // && !self.cortical_std
            // && !self.subcortical_std
            && !self.full_run_std
        // && !self.cortical_voxel_zscore
        // && !self.subcortical_voxel_zscore
        // && !self.timeseries_voxel_zscore
    }

    /// True when any raw dataset is absent (masker must be run).
    fn needs_masker(&self) -> bool {
        // self.cortical_raw || self.subcortical_raw ||
        self.full_run_raw
    }

    /// True when any post-parcellation standardized dataset is absent.
    fn needs_std(&self) -> bool {
        // self.cortical_std || self.subcortical_std ||
        self.full_run_std
    }

    // /// True when any voxel-wise z-score dataset is absent (masker must be re-run with voxelwise config).
    // fn needs_voxel_zscore(&self) -> bool {
    //     self.cortical_voxel_zscore || self.subcortical_voxel_zscore || self.timeseries_voxel_zscore
    // }
}

/// Inspect an existing HDF5 file and return which datasets are absent.
/// `voxelwise_zscore` controls whether the voxelzscore group is checked at all;
/// when false, those fields are always false so they never trigger computation.
fn check_missing_datasets(path: &Path, voxelwise_zscore: bool) -> MissingDatasets {
    match hdf5::File::open(path) {
        Ok(f) => {
            let parc = f.group("01fmri_parcellation").ok();
            MissingDatasets {
                full_run_raw: parc
                    .as_ref()
                    .map_or(true, |g| g.dataset("full_run_raw").is_err()),
                full_run_std: parc
                    .as_ref()
                    .map_or(true, |g| g.dataset("full_run_std").is_err()),
            }
        }
        Err(_) => MissingDatasets::all(voxelwise_zscore),
    }
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    // Check that the fmri dir is even present.
    // If not, fail gracefully and inform the user that the
    // disk might not be connected, or the network disk is not opened.
    let fmriprep_output_dir = &cfg.fmriprep_output_dir;
    match fs::read_dir(fmriprep_output_dir) {
        Ok(_) => { /* Process entries */ }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            error!(
                fmriprep_output_dir = %fmriprep_output_dir.display(),
                "Directory not found: {}. Make sure to have the disk connected, or connecting to the network drive", fmriprep_output_dir.display()
            );
            return Ok(());
        }
        Err(e) => panic!("Failed to read directory: {}", e),
    }

    let run_start = Instant::now();

    info!(
        fmriprep_output_dir = %cfg.fmriprep_output_dir.display(),
        filter_dir = %cfg.subject_filter_dir.display(),
        output_dir = %cfg.consolidated_data_dir.display(),
        cortical_atlas = %cfg.cortical_atlas.display(),
        subcortical_atlas = %cfg.subcortical_atlas.display(),
        force = cfg.force,
        voxelwise_zscore = cfg.parcellation.voxelwise_zscore,
        "starting fMRI preprocessing pipeline"
    );

    // Load subject filter files
    let csv_dir = &cfg.csv_output_dir;

    let filtered_subjects = [
        csv_dir.join("crate-00_filter-controls.csv"),
        csv_dir.join("crate-00_filter-anhedonic.csv"),
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
        .sort(["subjectkey"], Default::default()) // ascending
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

    std::fs::create_dir_all(&cfg.consolidated_data_dir)?;
    std::fs::create_dir_all(&cfg.csv_output_dir)?;

    let csv_crate_prefix = BidsFilename::new()
        .with_pair("crate", "01")
        .with_extension(".csv")
        .with_directory(&cfg.csv_output_dir);

    // Processing state
    let mut processed_count = 0usize;
    let mut skipped_count = 0usize;
    let error_count = 0usize;

    for (i, subject_key) in subject_keys.into_iter().flatten().enumerate() {
        let subject_idx = i + 1;
        let subject_id = BidsSubjectId::parse(subject_key);
        let dir_name = subject_id.clone().to_dir_name();
        let subject_dir = fmriprep_output_dir.join(&dir_name);

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

            let bids_filename =
                BidsFilename::parse(file_path.file_name().and_then(|n| n.to_str()).unwrap_or(""));
            let task_name = bids_filename.get("task").unwrap_or("unknown");
            let output_stem = bids_filename.to_stem();

            let output_h5_path = cfg
                .consolidated_data_dir
                .join(BidsSubjectId::parse(subject_key).to_dir_name())
                .join(format!("{}.h5", output_stem));

            let csv_subject_prefix = csv_crate_prefix
                .clone()
                .with_pair("sub", subject_id.as_bids_id())
                .with_pair("task", task_name)
                .with_pair("run", bids_filename.get("run").unwrap_or("unknown"));

            let csv_full_run_raw = csv_subject_prefix
                .clone()
                .with_pair("data", "timeseriesraw");
            let csv_full_run_std = csv_subject_prefix
                .clone()
                .with_pair("data", "timeseriesstd");
            // let csv_parcellation_output = cfg.csv_output_dir;

            // Determine which datasets still need to be written.
            //   --force  → treat all as missing (file will be removed and recreated)
            //   no file  → all missing (fresh create)
            //   file exists, no force → inspect per-dataset; only fill gaps
            let missing = if cfg.force || !output_h5_path.exists() {
                MissingDatasets::all(cfg.parcellation.voxelwise_zscore)
            } else {
                check_missing_datasets(&output_h5_path, cfg.parcellation.voxelwise_zscore)
            };

            if missing.all_present() {
                skipped_count += 1;
                info!(
                    subject_key = subject_key,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    task_name = task_name,
                    reason = "already_preprocessed",
                    output_file = %output_h5_path.display(),
                    "skipping file (all datasets present, use --force to reprocess)"
                );
                continue;
            }

            if output_h5_path.exists() {
                debug!(
                    subject_key = subject_key,
                    task_name = task_name,
                    output_file = %output_h5_path.display(),
                    // missing_cortical_raw = missing.cortical_raw,
                    // missing_subcortical_raw = missing.subcortical_raw,
                    missing_full_run_raw = missing.full_run_raw,
                    // missing_cortical_std = missing.cortical_std,
                    // missing_subcortical_std = missing.subcortical_std,
                    missing_full_run_std = missing.full_run_std,
                    "output file exists with missing datasets, appending"
                );
            }

            let file_start = Instant::now();

            debug!(
                subject_key = subject_key,
                task_name = task_name,
                file_path = %file_path.display(),
                run_masker = missing.needs_masker(),
                compute_std = missing.needs_std(),
                "starting parcellation"
            );

            // Run the NIfTI masker only when raw datasets need to be (re)written.
            // If only standardized datasets are missing we can read the raw arrays
            // directly from the existing HDF5, which is much faster.
            let mut cortical_duration_ms = 0u128;
            let mut subcortical_duration_ms = 0u128;

            let full_run_raw = if missing.needs_masker() {
                let raw_config = MaskerSignalConfig::default(); // detrend=false, standardize=None

                let cortical_start = Instant::now();
                let cortical_masker =
                    LabelsMasker::with_config(&cfg.cortical_atlas, raw_config.clone())?;
                let cortical_raw = cortical_masker.fit_transform(&file_path)?;
                cortical_duration_ms = cortical_start.elapsed().as_millis();

                debug!(
                    subject_key = subject_key,
                    atlas_type = "cortical",
                    n_rois = cortical_raw.shape()[0],
                    n_timepoints = cortical_raw.shape()[1],
                    duration_ms = cortical_duration_ms,
                    atlas_path = %cfg.cortical_atlas.display(),
                    "raw parcellation completed"
                );

                let subcortical_start = Instant::now();
                let subcortical_masker =
                    LabelsMasker::with_config(&cfg.subcortical_atlas, raw_config.clone())?;
                let subcortical_raw = subcortical_masker.fit_transform(&file_path)?;
                subcortical_duration_ms = subcortical_start.elapsed().as_millis();

                debug!(
                    subject_key = subject_key,
                    atlas_type = "subcortical",
                    n_rois = subcortical_raw.shape()[0],
                    n_timepoints = subcortical_raw.shape()[1],
                    duration_ms = subcortical_duration_ms,
                    atlas_path = %cfg.subcortical_atlas.display(),
                    "raw parcellation completed"
                );

                debug!(
                    subject_key = subject_key,
                    cortical_raw_first_roi_first_5_timepoints = ?cortical_raw.slice(s![0, ..5]),
                    subcortical_raw_first_roi_first_5_timepoints = ?subcortical_raw.slice(s![0, ..5]),
                    "raw timeseries sample values"
                );

                let concatenated_ts =
                    concatenate(Axis(0), &[cortical_raw.view(), subcortical_raw.view()])?;

                debug!("Cortical RAW: {:?}", cortical_raw.shape());
                debug!("Subcortical RAW: {:?}", subcortical_raw.shape());
                debug!("All ROIs RAW: {:?}", concatenated_ts.shape());

                concatenated_ts
                // (cortical_raw, subcortical_raw)
            } else {
                // Raw datasets are already in the file; read them to compute std.
                debug!(
                    subject_key = subject_key,
                    task_name = task_name,
                    "reading existing raw datasets from HDF5 to derive standardized"
                );
                let existing = hdf5::File::open(&output_h5_path)?;
                let parc = existing.group("01fmri_parcellation")?;
                let full_run_ts: Array2<f32> = parc.dataset("full_run_raw")?.read_2d()?;

                full_run_ts
            };

            // Compute post-parcellation z-score standardized variants only when needed.
            let full_run_std = if missing.needs_std() {
                let std_cfg = MaskerSignalConfig::default().standardize(Standardize::ZscoreSample);
                let std_ts = preprocess_signals(&full_run_raw, &std_cfg);
                Some(std_ts)
            } else {
                None
            };

            // Run the masker with voxel-wise z-score normalization (applied before parcellation).
            // let (cortical_voxel_zscore, subcortical_voxel_zscore) = if missing.needs_voxel_zscore()
            // {
            //     let vz_config = MaskerSignalConfig::default().voxelwise_zscore(true);

            //     let cortical_start = Instant::now();
            //     let cortical_masker =
            //         LabelsMasker::with_config(&cfg.cortical_atlas, vz_config.clone())?;
            //     let c = cortical_masker.fit_transform(&file_path)?;
            //     debug!(
            //         subject_key = subject_key,
            //         atlas_type = "cortical",
            //         n_rois = c.shape()[0],
            //         n_timepoints = c.shape()[1],
            //         duration_ms = cortical_start.elapsed().as_millis(),
            //         "voxel-wise z-score parcellation completed"
            //     );

            //     let subcortical_start = Instant::now();
            //     let subcortical_masker =
            //         LabelsMasker::with_config(&cfg.subcortical_atlas, vz_config)?;
            //     let s = subcortical_masker.fit_transform(&file_path)?;
            //     debug!(
            //         subject_key = subject_key,
            //         atlas_type = "subcortical",
            //         n_rois = s.shape()[0],
            //         n_timepoints = s.shape()[1],
            //         duration_ms = subcortical_start.elapsed().as_millis(),
            //         "voxel-wise z-score parcellation completed"
            //     );

            //     (Some(c), Some(s))
            // } else {
            //     (None, None)
            // };

            // Prepare output directory.
            if let Some(parent) = output_h5_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // --force: remove the existing file so we start from a clean slate.
            // For the append path the file is opened in RW mode below.
            if cfg.force && output_h5_path.exists() {
                fs::remove_file(&output_h5_path)?;
            }

            let write_start = Instant::now();

            // TODO: Uncomment this when implemented
            // if !csv_full_run_raw.exists() {
            //     save_timeseries_to_csv(
            //         csv_full_run_raw.to_path_buf(),
            //         cortical_raw.clone(),
            //         subcortical_raw.clone(),
            //     )?;
            // }

            // TODO: Uncomment this when implemented
            // if !csv_full_run_std.exists() {
            //     save_timeseries_to_csv(
            //         csv_full_run_std.to_path_buf(),
            //         cortical_std.clone(),
            //         subcortical_std.clone(),
            //     )?;
            // }

            append_missing_datasets(
                &output_h5_path,
                &missing,
                &full_run_raw,
                full_run_std.as_ref(),
                // cortical_voxel_zscore.as_ref(),
                // subcortical_voxel_zscore.as_ref(),
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
                output_file = %output_h5_path.display(),
                n_timepoints = full_run_raw.shape()[1],
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
        output_dir = %cfg.consolidated_data_dir.display(),
        outcome = if error_count == 0 { "success" } else { "completed_with_errors" },
        "fMRI preprocessing pipeline completed"
    );

    Ok(())
}

/// Open an existing HDF5 file for appending (RW) or create a new one.
/// Writes only the datasets flagged as missing in `missing`.
///
/// Dataset naming convention:
///   tcp_cortical_raw             — per-atlas parcellated signal, no preprocessing
///   tcp_subcortical_raw          — per-atlas parcellated signal, no preprocessing
///   tcp_full_run_raw           — cortical + subcortical concatenated, no preprocessing
///   tcp_cortical_standardized    — post-parcellation z-score (per-ROI temporal)
///   tcp_subcortical_standardized — post-parcellation z-score (per-ROI temporal)
///   tcp_timeseries_standardized  — cortical + subcortical concatenated, post-parcellation z-score
///   tcp_cortical_voxelzscore     — voxel-wise z-score applied before parcellation
///   tcp_subcortical_voxelzscore  — voxel-wise z-score applied before parcellation
///   tcp_timeseries_voxelzscore   — cortical + subcortical concatenated, voxel-wise z-score
fn append_missing_datasets(
    path: &Path,
    missing: &MissingDatasets,
    full_run_raw: &Array2<f32>,
    full_run_std: Option<&Array2<f32>>,
    // cortical_voxel_zscore: Option<&Array2<f32>>,
    // subcortical_voxel_zscore: Option<&Array2<f32>>,
) -> Result<()> {
    let file = if path.exists() {
        hdf5::File::open_rw(path)?
    } else {
        hdf5::File::create(path)?
    };

    let parc = match file.group("01fmri_parcellation") {
        Ok(g) => g,
        Err(_) => file.create_group("01fmri_parcellation")?,
    };

    if missing.full_run_raw {
        write_2d_dataset(&parc, &full_run_raw, "full_run_raw")?;
    }

    if missing.full_run_std {
        if let Some(ts) = full_run_std {
            write_2d_dataset(&parc, &ts, "full_run_std")?;
        }
    }

    // if missing.cortical_voxel_zscore {
    //     if let Some(c) = cortical_voxel_zscore {
    //         write_2d_dataset(&file, c, "tcp_cortical_voxelzscore")?;
    //     }
    // }
    // if missing.subcortical_voxel_zscore {
    //     if let Some(s) = subcortical_voxel_zscore {
    //         write_2d_dataset(&file, s, "tcp_subcortical_voxelzscore")?;
    //     }
    // }
    // if missing.timeseries_voxel_zscore {
    //     if let (Some(c), Some(s)) = (cortical_voxel_zscore, subcortical_voxel_zscore) {
    //         let ts = concatenate(Axis(0), &[c.view(), s.view()])?;
    //         write_2d_dataset(&file, &ts, "tcp_timeseries_voxelzscore")?;
    //     }
    // }

    Ok(())
}

/// Write a single 2-D array as a named dataset into an open HDF5 group.
fn write_2d_dataset(group: &hdf5::Group, data: &Array2<f32>, name: &str) -> Result<()> {
    let shape = data.shape();
    let ds = group
        .new_dataset::<f32>()
        .shape([shape[0], shape[1]])
        .create(name)?;
    ds.write_raw(data.as_slice().unwrap())?;
    Ok(())
}

type MyArray = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>;
fn save_timeseries_to_csv<P: AsRef<Path>>(
    path: P,
    part1: impl Into<Option<MyArray>>,
    part2: impl Into<Option<MyArray>>,
) -> PolarsResult<()> {
    // 1. Unwrap the inputs
    let a = part1
        .into()
        .ok_or_else(|| PolarsError::ComputeError("Part 1 is None".into()))?;
    let b = part2
        .into()
        .ok_or_else(|| PolarsError::ComputeError("Part 2 is None".into()))?;

    let mut columns = Vec::with_capacity(a.nrows() + b.nrows());

    // 2. Map Array A Channels to Columns
    for (i, row) in a.axis_iter(Axis(0)).enumerate() {
        // We explicitly cast the row to an ArrayView1<f32> to stop inference errors
        let row_view: ArrayView1<f32> = row;
        let s = Series::new(format!("CH_{}", i).into(), row_view.to_vec());
        columns.push(Column::from(s));
    }

    // 3. Map Array B Channels to Columns (continuing the index)
    let offset = a.nrows();
    for (i, row) in b.axis_iter(Axis(0)).enumerate() {
        let row_view: ArrayView1<f32> = row;
        let s = Series::new(format!("CH_{}", i + offset).into(), row_view.to_vec());
        columns.push(Column::from(s));
    }

    // 4. Create DataFrame and Write
    let mut df = DataFrame::new(columns)?;

    let file = File::create(path.as_ref())
        .map_err(|e| PolarsError::ComputeError(format!("IO Error: {}", e).into()))?;

    CsvWriter::new(file).include_header(true).finish(&mut df)?;

    Ok(())
}
