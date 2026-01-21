use anyhow::Result;
use config::TCPPreprocessConfig;
use config::annex;
use git2::Repository;
use std::path::PathBuf;
use thiserror::Error;
use tracing::info;

#[derive(Error, Debug)]
pub enum TCPPreprocessError {
    #[error("File already exists: {0}")]
    AlreadyExists(String),
    #[error("Required file missing: {0}")]
    RequiredFileMissing(String),
}

pub fn run(cfg: &TCPPreprocessConfig) -> Result<()> {
    info!("{:?}", cfg);

    ////////////////////////
    // Initialize dataset //
    ////////////////////////

    // Clone dataset
    if !cfg.tcp_dir.is_dir() {
        let dataset_url = "https://github.com/OpenNeuroDatasets/ds005237.git";
        let local_path = &cfg.tcp_dir;

        if !cfg.dry_run {
            if let Some(parent) = local_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            info!("Cloning {} into {}...", dataset_url, local_path.display());
            match Repository::clone(dataset_url, local_path) {
                Ok(_repo) => info!("Cloned successfully!"),
                Err(e) => panic!("failed to clone: {}", e),
            };
        } else {
            info!("Skipped cloning. Dry_run config detected")
        }
    }
    info!("TCP Dataset available on: {}", cfg.tcp_dir.display());

    // Validate and set annex remote
    if annex::validate_remote(&cfg.tcp_dir, &cfg.tcp_annex_remote).is_err() {
        annex::enable_remote(&cfg.tcp_dir, &cfg.tcp_annex_remote)?;
    }
    info!("Validated annex remote: {}", &cfg.tcp_annex_remote);

    // Validate dataset
    let phenotype_dir = &cfg.tcp_dir.join("phenotype");
    let required_files: Vec<PathBuf> = vec![
        phenotype_dir.join("demos.tsv"),
        phenotype_dir.join("shaps01.tsv"),
    ];
    required_files.iter().try_for_each(|file_path| {
        if !file_path.is_file() && !annex::is_broken_symlink(file_path) {
            return Err(TCPPreprocessError::RequiredFileMissing(format!(
                "{}",
                file_path.display()
            )));
        }
        // Continue to next iteration
        Ok(())
    })?;
    info!("All required files located: {:?}", required_files);

    ///////////////////
    // Apply filters //
    ///////////////////

    //////////////////////////////
    // Establish subject groups //
    //////////////////////////////

    Ok(())
}
