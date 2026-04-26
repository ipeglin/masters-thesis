mod feature_extractor;
mod models;
pub mod preprocessing;
pub mod strategies;

use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};

use anyhow::Result;
use tch::Tensor;
use tracing::{info, warn};
use utils::atlas::BrainAtlas;
use utils::bids_filename::BidsFilename;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::{AppConfig, RoiSet};

pub use feature_extractor::FeatureExtractor;
pub use strategies::{AnalysisCtx, FeatureSrc};

/// Run the DenseNet feature extraction pipeline over every subject HDF5 file
/// found in `cfg.consolidated_data_dir`. Per file, the BIDS task name selects
/// which analysis set runs (see `strategies::run_for_file`). All outputs are
/// written back into the same file under `features/<src>/<analysis>/...`.
pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        consolidated_data_dir = %cfg.consolidated_data_dir.display(),
        force = cfg.force,
        roi_set = ?cfg.feature_extraction.roi_set,
        image_fit = ?cfg.feature_extraction.image_fit,
        hht_log_amp = cfg.feature_extraction.hht_log_amp,
        "starting CNN feature extraction pipeline"
    );

    let weights_path = cfg.feature_extraction.cnn_weights_path.as_deref();
    let extractor = FeatureExtractor::new(weights_path, 1)?;
    match weights_path {
        Some(p) => info!(weights = %p.display(), "DenseNet-201 initialised with pretrained weights"),
        None => info!("DenseNet-201 initialised with random weights"),
    }

    let brain_atlas =
        BrainAtlas::from_lut_files(&cfg.cortical_atlas_lut, &cfg.subcortical_atlas_lut);
    let (roi_indices, roi_labels) = match cfg.feature_extraction.roi_set {
        RoiSet::Subset28 => {
            let pairs = brain_atlas.vpfc_mpfc_amy_ids();
            let idx: Vec<i64> = pairs.iter().map(|(i, _)| *i as i64).collect();
            let lab: Vec<String> = pairs.iter().map(|(_, l)| l.clone()).collect();
            (idx, lab)
        }
        RoiSet::All => {
            // Atlas does not enumerate all rows; let strategies infer count from data.
            // For "All" we generate placeholder indices later per-file. Bail early if
            // misconfigured — the rest of the pipeline assumes Subset28 labelling.
            anyhow::bail!(
                "RoiSet::All not yet supported in the feature extraction pipeline; \
                 set feature_extraction.roi_set = \"subset28\""
            );
        }
    };
    if roi_indices.is_empty() {
        anyhow::bail!(
            "no PFCv/PFCm/AMY ROIs matched in atlas — check LUT paths ({}, {})",
            cfg.cortical_atlas_lut.display(),
            cfg.subcortical_atlas_lut.display()
        );
    }
    let roi_index_tensor = Tensor::from_slice(&roi_indices);
    let labels_joined = roi_labels.join(",");
    info!(
        n_target_rois = roi_indices.len(),
        rois = ?roi_labels,
        "selected target ROIs (vPFC + mPFC + AMY)"
    );

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.consolidated_data_dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if !path.is_dir() {
                return None;
            }
            let id = path.file_name()?.to_str()?;
            let formatted = BidsSubjectId::parse(id).to_dir_name();
            Some((formatted, path))
        })
        .collect();

    let total_subjects = subjects.len();
    info!(num_subjects = total_subjects, "found subject directories");

    let mut subject_idx = 0;
    let mut error_count = 0;

    for (formatted_id, dir) in &subjects {
        subject_idx += 1;
        let _subject_span = tracing::info_span!(
            "subject",
            subject = %formatted_id,
            subject_idx,
            total_subjects
        )
        .entered();

        let files: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("h5"))
            .collect();

        info!(num_files = files.len(), "processing subject");

        for file_path in &files {
            let _file_span = tracing::info_span!("file", file = %file_path.display()).entered();

            let result: Result<()> = (|| {
                let bids =
                    BidsFilename::parse(match file_path.file_name().and_then(|n| n.to_str()) {
                        Some(name) => name,
                        None => return Ok(()),
                    });
                let task_name = bids.get("task").unwrap_or("unknown");

                let h5_file = utils::hdf5_io::open_or_create(file_path)?;
                let ctx = AnalysisCtx {
                    extractor: &extractor,
                    fit: cfg.feature_extraction.image_fit,
                    hht_log_amp: cfg.feature_extraction.hht_log_amp,
                    roi_indices: &roi_indices,
                    roi_index_tensor: &roi_index_tensor,
                    roi_labels_joined: &labels_joined,
                    roi_set: cfg.feature_extraction.roi_set,
                    force: cfg.force,
                    subject_id: formatted_id,
                    task_name,
                };
                strategies::run_for_file(&ctx, &h5_file)
            })();

            if let Err(e) = result {
                error_count += 1;
                warn!(file = %file_path.display(), error = %e, "skipping file due to error");
            }
        }
    }

    if error_count > 0 {
        warn!(error_count, "some files were skipped due to errors");
    }
    info!(
        error_count,
        total_duration_ms = run_start.elapsed().as_millis() as u64,
        "feature extraction pipeline complete"
    );
    Ok(())
}
