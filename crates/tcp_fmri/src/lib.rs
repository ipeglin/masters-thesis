use anyhow::Result;
use config::{TCPfMRIPreprocessConfig, polars_csv};
use ndarray::{Array2, Axis};
use nifti_masker::LabelsMasker;
use polars::prelude::*;
use std::path::Path;
use tracing::{info, warn};

pub fn run(cfg: &TCPfMRIPreprocessConfig) -> Result<()> {
    info!("{:?}", cfg);

    ///////////////////
    // Load subjects //
    ///////////////////

    let filter_dir = &cfg.filter_dir;

    // Filter files
    let filtered_subjects = [
        filter_dir.join("healthy_controls.csv"),
        filter_dir.join("low_anhedonic.csv"),
        filter_dir.join("high_anhedonic.csv"),
    ];

    // Load dataframes
    let dataframes: Vec<LazyFrame> = filtered_subjects
        .iter()
        .filter_map(|file| {
            polars_csv::read_dataframe(file)
                .map_err(|e| warn!("failed to read {}: {}", file.display(), e))
                .ok()
                .map(|df| df.lazy())
        })
        .collect();

    // Get subject key series
    let subjects = concat(dataframes, UnionArgs::default())?.collect()?;
    let subject_keys = subjects.column("subjectkey")?.str()?;

    //////////////////
    // Load atlases //
    //////////////////

    // Check if atlas files are present
    if !&cfg.cortical_atlas.exists() || !&cfg.subcortical_atlas.exists() {
        panic!("failed to locate atlases");
    }

    // Ensure output directory exists
    std::fs::create_dir_all(&cfg.output_dir)?;

    ///////////////////////////////
    // Process subject fMRI BOLD //
    ///////////////////////////////

    let mut unavailable_subjects: Vec<&str> = vec![];
    let total_subjects = subject_keys.len();

    // Iterate over subject keys
    for (i, subject_key) in subject_keys.into_iter().flatten().enumerate() {
        let current = i + 1;
        let dir_name = parse_subject_directory_name(subject_key);
        let subject_dir = &cfg.fmri_dir.join(&dir_name);

        if !subject_dir.is_dir() {
            unavailable_subjects.push(subject_key);
            warn!(
                "[{}/{}] subject missing fMRI data. Skipping {}",
                current, total_subjects, subject_key
            );
            continue;
        }

        // fMRI files for parcellation
        let mni_results_dir = subject_dir.join("MNINonLinear").join("Results");
        let files_to_preprocess = [mni_results_dir
            .join("task-hammerAP_run-01_bold")
            .join("task-hammerAP_run-01_bold.nii.gz")];

        for file_path in files_to_preprocess {
            // Check if file exists
            if !file_path.exists() {
                warn!(
                    "[{}/{}] subject fMRI file does not exist. Skipping {} {}",
                    current,
                    total_subjects,
                    subject_key,
                    file_path.display()
                );
                continue;
            }
            info!(
                "[{}/{}] processing file {} {}",
                current,
                total_subjects,
                subject_key,
                file_path.display()
            );

            // Label time series parcellations
            let cortical_masker = LabelsMasker::new(&cfg.cortical_atlas)?;
            let cortical_bold = cortical_masker.fit_transform(&file_path)?;
            info!(
                "[{}/{}] cortical parcellation: {} ROIs x {} timepoints",
                current,
                total_subjects,
                cortical_bold.shape()[0],
                cortical_bold.shape()[1]
            );

            let subcortical_masker = LabelsMasker::new(&cfg.subcortical_atlas)?;
            let subcortical_bold = subcortical_masker.fit_transform(&file_path)?;
            info!(
                "[{}/{}] subcortical parcellation: {} ROIs x {} timepoints",
                current,
                total_subjects,
                subcortical_bold.shape()[0],
                subcortical_bold.shape()[1]
            );

            // Concatenate cortical and subcortical timeseries
            let combined =
                ndarray::concatenate(Axis(0), &[cortical_bold.view(), subcortical_bold.view()])?;
            info!(
                "[{}/{}] combined parcellation: {} ROIs x {} timepoints",
                current,
                total_subjects,
                combined.shape()[0],
                combined.shape()[1]
            );

            // Write output to HDF5
            let task_name = file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let output_path = cfg
                .output_dir
                .join(format!("{}_{}.h5", subject_key, task_name));

            write_timeseries_h5(&output_path, &combined)?;
            info!(
                "[{}/{}] wrote timeseries to {}",
                current,
                total_subjects,
                output_path.display()
            );
        }
    }

    Ok(())
}

fn parse_subject_directory_name(key: &str) -> String {
    format!("sub-{}", key.replace("_", ""))
}

fn write_timeseries_h5(path: &Path, timeseries: &Array2<f32>) -> Result<()> {
    let file = hdf5::File::create(path)?;

    let shape = timeseries.shape();
    let ds = file
        .new_dataset::<f32>()
        .shape([shape[0], shape[1]])
        .create("timeseries")?;
    ds.write_raw(timeseries.as_slice().unwrap())?;

    Ok(())
}
