pub mod annex;
pub mod polars_csv;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub tcp_subject_selection: TCPSubjectSelectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCPSubjectSelectionConfig {
    pub tcp_dir: PathBuf,
    pub tcp_annex_remote: String,
    pub output_dir: PathBuf,
    #[serde(default)]
    pub filters: Option<Vec<String>>,
    #[serde(default)]
    pub dry_run: bool,
}

impl Default for TCPSubjectSelectionConfig {
    fn default() -> Self {
        Self {
            tcp_dir: PathBuf::from("/path/to/tcp"),
            tcp_annex_remote: String::from(""),
            output_dir: PathBuf::from("/path/to/output"),
            filters: None,
            dry_run: false,
        }
    }
}

impl fmt::Display for TCPSubjectSelectionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC Preprocessing:")?;
        writeln!(f, "  TCP Dir: {}", self.tcp_dir.display())?;
        writeln!(f, "  Output Dir: {}", self.output_dir.display())?;

        // Handling the Option for cleaner output
        match &self.filters {
            Some(flts) => writeln!(f, "  Filters: {:?}", flts)?,
            None => writeln!(f, "  Filters: None")?,
        }

        write!(f, "  Dry run: {}", self.dry_run)
    }
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;

    let cfg: AppConfig =
        toml::from_str(&s).with_context(|| format!("Failed to parse TOML: {}", path.display()))?;

    Ok(cfg)
}
