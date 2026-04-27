pub mod analyses;
pub mod classifiers;
pub mod dataset;
pub mod eval;
pub mod normalizer;
pub mod splits;

use crate::analyses::{
    baseline, baseline_averaging, block_ensemble, face_block_averaging, face_block_concatenation,
    subject_stratified, task_block,
};

use anyhow::Result;
use utils::config::AppConfig;

pub fn run(cfg: &AppConfig) -> Result<()> {
    baseline::run(cfg)?;
    baseline_averaging::run(cfg)?;
    face_block_concatenation::run(cfg)?;
    task_block::run(cfg)?;
    face_block_averaging::run(cfg)?;
    Ok(())
}
