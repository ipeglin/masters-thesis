use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::process::Command;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AnnexError {
    #[error("File already exists: {0}")]
    AlreadyExists(String),
    #[error("File not found locally: {0}")]
    FileNotFound(String),
    #[error("Unbroken symlink file: {0}")]
    UnbrokenSymlink(String),
    #[error("Broken symlink file: {0}")]
    BrokenSymlink(String),
    #[error("Missing target: {0}")]
    MissingTarget(String),
    #[error("Missing symlink metadata for file: {0}")]
    MissingSymlinkMetadata(String),
    #[error("Failed to get file from annex: {0}")]
    FetchError(String),
}

pub fn is_broken_symlink<P: AsRef<Path>>(path: P) -> bool {
    let path = path.as_ref();

    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                match fs::metadata(path) {
                    Ok(_) => false,
                    Err(err) => err.kind() == ErrorKind::NotFound,
                }
            } else {
                false
            }
        }
        Err(err) => err.kind() == ErrorKind::NotFound,
    }
}

pub fn get_file_from_annex<P: AsRef<Path>>(
    local_remote: P,
    symlink_path: P,
) -> Result<(), AnnexError> {
    let remote = local_remote.as_ref();
    let symlink = symlink_path.as_ref();

    if symlink.exists() {
        return Err(AnnexError::AlreadyExists(format!("{}", symlink.display())));
    }
    if !is_broken_symlink(symlink) {
        return Err(AnnexError::UnbrokenSymlink(format!(
            "{}",
            symlink.display()
        )));
    }

    let status = Command::new("git")
        .arg("-C")
        .arg(remote)
        .arg("annex")
        .arg("get")
        .arg(symlink)
        .status()
        .expect("failed to get file from annex");

    if !status.success() {
        return Err(AnnexError::FetchError(format!("{}", symlink.display())));
    }

    Ok(())
}

pub fn drop_file<P: AsRef<Path>>(local_remote: P, symlink_path: P) -> Result<(), AnnexError> {
    let remote = local_remote.as_ref();
    let symlink = symlink_path.as_ref();

    if !symlink.exists() {
        return Err(AnnexError::FileNotFound(format!("{}", symlink.display())));
    }
    if is_broken_symlink(symlink) {
        return Err(AnnexError::BrokenSymlink(format!("{}", symlink.display())));
    }

    let status = Command::new("git")
        .arg("-C")
        .arg(remote)
        .arg("annex")
        .arg("drop")
        .arg(symlink)
        .status()
        .expect("failed to get file from annex");

    if !status.success() {
        return Err(AnnexError::FetchError(format!("{}", symlink.display())));
    }

    Ok(())
}
