use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub tcp_preprocess: TCPPreprocessConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCPPreprocessConfig {
    pub fmri_dir: PathBuf,
    pub tcp_dir: PathBuf,
    pub tcp_annex_remote: PathBuf,
    pub output_dir: PathBuf,
    #[serde(default)]
    pub filters: Option<Vec<String>>,
    #[serde(default)]
    pub dry_run: bool,
}

impl Default for TCPPreprocessConfig {
    fn default() -> Self {
        Self {
            fmri_dir: PathBuf::from("/path/to/fmri"),
            tcp_dir: PathBuf::from("/path/to/tcp"),
            tcp_annex_remote: PathBuf::from("/path/to/tcp/annex"),
            output_dir: PathBuf::from("/path/to/output"),
            filters: None,
            dry_run: false,
        }
    }
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;

    let cfg: AppConfig =
        toml::from_str(&s).with_context(|| format!("Failed to parse TOML: {}", path.display()))?;

    Ok(cfg)
}
