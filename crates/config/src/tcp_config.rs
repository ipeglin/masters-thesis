use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

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
        writeln!(f, "TPC Subject Selection:")?;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCPfMRIPreprocessConfig {
    pub fmri_dir: PathBuf,
    pub filter_dir: PathBuf,
    pub output_dir: PathBuf,
    pub cortical_atlas: PathBuf,
    pub subcortical_atlas: PathBuf,
    #[serde(default)]
    pub dry_run: bool,
}

impl Default for TCPfMRIPreprocessConfig {
    fn default() -> Self {
        Self {
            fmri_dir: PathBuf::from("/path/to/raw_fmri_data"),
            filter_dir: PathBuf::from("/path/to/output"),
            output_dir: PathBuf::from("/path/to/output"),
            cortical_atlas: PathBuf::from("/path/to/atlas"),
            subcortical_atlas: PathBuf::from("/path/to/atlas"),
            dry_run: false,
        }
    }
}

impl fmt::Display for TCPfMRIPreprocessConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC fMRI Preprocessing:")?;
        writeln!(f, "  fMRI Dir: {}", self.fmri_dir.display())?;
        writeln!(f, "  Filter Dir: {}", self.filter_dir.display())?;
        writeln!(f, "  Output Dir: {}", self.output_dir.display())?;

        write!(f, "  Dry run: {}", self.dry_run)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCPfMRIProcessConfig {
    pub fmri_dir: PathBuf,
    pub output_dir: PathBuf,
    pub cortical_atlas_lut: PathBuf,
    pub subcortical_atlas_lut: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject_file: Option<PathBuf>,
    /// Force reprocessing of subjects that already exist in output files
    #[serde(default)]
    pub force: bool,
}

impl Default for TCPfMRIProcessConfig {
    fn default() -> Self {
        Self {
            fmri_dir: PathBuf::from("/path/to/raw_fmri_data"),
            output_dir: PathBuf::from("/path/to/output"),
            cortical_atlas_lut: PathBuf::from("/path/to/cortical_atlas_lut"),
            subcortical_atlas_lut: PathBuf::from("/path/to/subcortical_atlas_lut"),
            subject_file: None,
            force: false,
        }
    }
}

impl fmt::Display for TCPfMRIProcessConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC fMRI Preprocessing:")?;
        writeln!(f, "  fMRI Dir: {}", self.fmri_dir.display())?;
        writeln!(f, "  Output Dir: {}", self.output_dir.display())?;
        writeln!(
            f,
            "  Cortical Atlast LUT: {}",
            self.cortical_atlas_lut.display()
        )?;
        writeln!(
            f,
            "  Subcortical Atlast LUT: {}",
            self.subcortical_atlas_lut.display()
        )?;
        write!(f, "  Subjects: {:?}", self.subject_file)
    }
}
