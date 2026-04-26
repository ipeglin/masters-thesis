pub mod baseline;
pub mod baseline_averaging;
pub mod block_ensemble;
pub mod classifiers;
pub mod dataset;
pub mod eval;
pub mod face_block_averaging;
pub mod face_block_concatenation;
pub mod normalizer;
pub mod splits;
pub mod subject_stratified;
pub mod task_block;

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
