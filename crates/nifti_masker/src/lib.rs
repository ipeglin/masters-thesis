use anyhow::{Result, bail};
use nalgebra::Matrix4;
use ndarray::{Array2, Array3, Array4, ShapeBuilder};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, NiftiVolume, ReaderOptions};
use std::{collections::HashSet, path::PathBuf};
use tracing::{debug, trace};

/// Standardization strategy for masked signal preprocessing.
///
/// Based on nilearn's standardization options for NiftiLabelsMasker.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Standardize {
    /// No standardization
    #[default]
    None,
    /// Z-score using sample standard deviation (ddof=1)
    /// This is the recommended method.
    ZscoreSample,
    /// Percent signal change: (signal - mean) / |mean| * 100
    Psc,
}

/// Configuration for signal preprocessing in maskers.
///
/// Based on nilearn's NiftiLabelsMasker preprocessing options.
/// This controls how extracted time series are cleaned before being returned.
#[derive(Debug, Clone)]
pub struct MaskerSignalConfig {
    /// Whether to perform linear detrending
    pub detrend: bool,
    /// Standardization strategy applied to ROI timeseries after parcellation
    pub standardize: Standardize,
    /// Whether to apply voxel-wise z-score normalization before parcellation.
    ///
    /// Each voxel's timeseries is independently normalized by its own temporal
    /// mean and standard deviation before ROI averaging. This corresponds to
    /// "subject-level Z-score maps" as described in the neuroimaging literature:
    /// each voxel's amplitude is standardized across the full scan duration,
    /// making signal amplitude comparable across voxels and subjects prior to
    /// any spatial averaging.
    pub voxelwise_zscore: bool,
}

impl Default for MaskerSignalConfig {
    fn default() -> Self {
        Self {
            detrend: false,
            standardize: Standardize::None,
            voxelwise_zscore: false,
        }
    }
}

impl MaskerSignalConfig {
    /// Create a preprocessing config with detrending and z-score standardization.
    /// This matches nilearn's default behavior.
    pub fn with_defaults() -> Self {
        Self {
            detrend: true,
            standardize: Standardize::ZscoreSample,
            voxelwise_zscore: false,
        }
    }

    /// Builder method to set detrending
    pub fn detrend(mut self, detrend: bool) -> Self {
        self.detrend = detrend;
        self
    }

    /// Builder method to set standardization
    pub fn standardize(mut self, standardize: Standardize) -> Self {
        self.standardize = standardize;
        self
    }

    /// Builder method to enable voxel-wise z-score normalization before parcellation
    pub fn voxelwise_zscore(mut self, enabled: bool) -> Self {
        self.voxelwise_zscore = enabled;
        self
    }

    /// Check if any preprocessing is enabled
    pub fn is_enabled(&self) -> bool {
        self.detrend || self.standardize != Standardize::None || self.voxelwise_zscore
    }
}

/// Linear detrending of a 2D signal array.
///
/// Removes the mean and linear trend from each row (each ROI's time series).
/// Based on nilearn's `_detrend` function.
///
/// # Arguments
/// * `data` - Array2 of shape (n_rois, n_timepoints)
///
/// # Returns
/// Detrended signal with the same shape
fn detrend_signal(data: &Array2<f32>) -> Array2<f32> {
    let (n_rois, n_timepoints) = data.dim();

    if n_timepoints <= 1 {
        return data.clone();
    }

    let mut result = data.clone();

    // Create time regressor: [0, 1, 2, ..., n-1], then center and normalize
    let regressor_mean = (n_timepoints - 1) as f32 / 2.0;
    let mut regressor: Vec<f32> = (0..n_timepoints)
        .map(|i| i as f32 - regressor_mean)
        .collect();

    // Normalize regressor
    let regressor_std: f32 = regressor.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if regressor_std > f32::EPSILON {
        for r in &mut regressor {
            *r /= regressor_std;
        }
    }

    for roi_idx in 0..n_rois {
        // Step 1: Remove mean
        let mean: f32 = result.row(roi_idx).sum() / n_timepoints as f32;
        for t in 0..n_timepoints {
            result[[roi_idx, t]] -= mean;
        }

        // Step 2: Remove linear trend
        // Compute dot product of regressor with signal (already mean-centered)
        let dot: f32 = result
            .row(roi_idx)
            .iter()
            .zip(regressor.iter())
            .map(|(&val, &r)| val * r)
            .sum();

        // Subtract the linear component
        for (t, &r) in regressor.iter().enumerate() {
            result[[roi_idx, t]] -= dot * r;
        }
    }

    result
}

/// Standardize signal using the specified strategy.
///
/// Based on nilearn's `standardize_signal` function.
///
/// # Arguments
/// * `data` - Array2 of shape (n_rois, n_timepoints)
/// * `standardize` - Standardization strategy
/// * `already_detrended` - Whether the signal has already been detrended (mean removed)
///
/// # Returns
/// Standardized signal with the same shape
fn standardize_signal(
    data: &Array2<f32>,
    standardize: Standardize,
    already_detrended: bool,
) -> Array2<f32> {
    let (n_rois, n_timepoints) = data.dim();

    if n_timepoints <= 1 {
        return data.clone();
    }

    match standardize {
        Standardize::None => data.clone(),

        Standardize::ZscoreSample => {
            let mut result = data.clone();

            for roi_idx in 0..n_rois {
                // Remove mean if not already detrended
                if !already_detrended {
                    let mean: f32 = result.row(roi_idx).sum() / n_timepoints as f32;
                    for t in 0..n_timepoints {
                        result[[roi_idx, t]] -= mean;
                    }
                }

                // Compute sample standard deviation (ddof=1)
                let variance: f32 = result.row(roi_idx).iter().map(|&v| v * v).sum::<f32>()
                    / (n_timepoints - 1) as f32;
                let std = variance.sqrt();

                // Avoid division by zero
                let std = if std < f32::EPSILON { 1.0 } else { std };

                for t in 0..n_timepoints {
                    result[[roi_idx, t]] /= std;
                }
            }

            result
        }

        Standardize::Psc => {
            let mut result = data.clone();

            for roi_idx in 0..n_rois {
                let mean: f32 = data.row(roi_idx).sum() / n_timepoints as f32;
                let abs_mean = mean.abs();

                if abs_mean < f32::EPSILON {
                    // Mean is zero, PSC is meaningless - set to zero
                    for t in 0..n_timepoints {
                        result[[roi_idx, t]] = 0.0;
                    }
                } else {
                    for t in 0..n_timepoints {
                        result[[roi_idx, t]] = (data[[roi_idx, t]] - mean) / abs_mean * 100.0;
                    }
                }
            }

            result
        }
    }
}

/// Apply voxel-wise z-score normalization to a 4D BOLD array in-place.
///
/// Each voxel's timeseries `[x, y, z, :]` is independently normalized by its
/// own temporal mean and sample standard deviation (ddof=1). Voxels with near-
/// zero variance (e.g. outside the brain mask) are left unchanged.
fn voxelwise_zscore_bold(data: &mut Array4<f32>) {
    let shape = data.shape().to_vec();
    let (nx, ny, nz, nt) = (shape[0], shape[1], shape[2], shape[3]);

    if nt <= 1 {
        return;
    }

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let mean: f32 = (0..nt).map(|t| data[[x, y, z, t]]).sum::<f32>() / nt as f32;
                let variance: f32 = (0..nt)
                    .map(|t| {
                        let d = data[[x, y, z, t]] - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / (nt - 1) as f32;
                let std = variance.sqrt();

                if std < f32::EPSILON {
                    continue;
                }

                for t in 0..nt {
                    data[[x, y, z, t]] = (data[[x, y, z, t]] - mean) / std;
                }
            }
        }
    }
}

/// Apply preprocessing (detrend and/or standardize) to extracted time series.
///
/// Based on nilearn's signal cleaning pipeline.
///
/// # Arguments
/// * `data` - Array2 of shape (n_rois, n_timepoints)
/// * `config` - Preprocessing configuration
///
/// # Returns
/// Preprocessed signal with the same shape
pub fn preprocess_signals(data: &Array2<f32>, config: &MaskerSignalConfig) -> Array2<f32> {
    if !config.is_enabled() {
        return data.clone();
    }

    // Apply detrending first (order matters: detrend before standardize)
    let data = if config.detrend {
        detrend_signal(data)
    } else {
        data.clone()
    };

    // Apply standardization
    standardize_signal(&data, config.standardize, config.detrend)
}

pub struct LabelsMasker {
    /// The atlas with integer labels for each ROI
    labels_volume: Array3<f32>,
    /// Affine transformation matrix of the atlas (voxel to world)
    labels_affine: Matrix4<f64>,
    /// Unique labels in the atlas (excluding 0/background)
    unique_labels: Vec<i32>,
    /// Shape of the atlas volume
    atlas_shape: [usize; 3],
    /// Signal preprocessing configuration
    signal_config: MaskerSignalConfig,
}

impl LabelsMasker {
    /// Load and prepare the atlas
    pub fn new(atlas_path: &PathBuf) -> Result<Self> {
        Self::with_config(atlas_path, MaskerSignalConfig::default())
    }

    /// Load and prepare the atlas with preprocessing configuration
    pub fn with_config(atlas_path: &PathBuf, signal_config: MaskerSignalConfig) -> Result<Self> {
        debug!(
            atlas_path = %atlas_path.display(),
            detrend = signal_config.detrend,
            standardize = ?signal_config.standardize,
            "loading atlas"
        );

        let atlas_obj = ReaderOptions::new().read_file(atlas_path)?;
        let header = atlas_obj.header();

        let affine_source = if header.sform_code > 0 {
            "sform"
        } else if header.qform_code > 0 {
            "qform"
        } else {
            "pixdim"
        };

        let labels_affine = Self::get_affine_from_header(header);
        let labels_volume = Self::volume_to_array3(atlas_obj.into_volume())?;
        let atlas_shape = [
            labels_volume.shape()[0],
            labels_volume.shape()[1],
            labels_volume.shape()[2],
        ];

        // Find unique labels (excluding 0 which is background)
        let mut unique_labels: Vec<i32> = labels_volume
            .iter()
            .map(|&v| v.round() as i32)
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|&l| l != 0)
            .collect();
        unique_labels.sort();

        debug!(
            atlas_path = %atlas_path.display(),
            atlas_shape_x = atlas_shape[0],
            atlas_shape_y = atlas_shape[1],
            atlas_shape_z = atlas_shape[2],
            n_labels = unique_labels.len(),
            affine_source = affine_source,
            label_min = unique_labels.first().copied().unwrap_or(0),
            label_max = unique_labels.last().copied().unwrap_or(0),
            "atlas loaded"
        );

        Ok(Self {
            labels_volume,
            labels_affine,
            unique_labels,
            atlas_shape,
            signal_config,
        })
    }

    /// Get the number of unique labels (ROIs) in the atlas
    pub fn n_labels(&self) -> usize {
        self.unique_labels.len()
    }

    /// Get the signal preprocessing configuration
    pub fn signal_config(&self) -> &MaskerSignalConfig {
        &self.signal_config
    }

    /// Extract time series for each labeled region from BOLD data
    /// Returns Array2 of shape (n_labels, n_timepoints)
    ///
    /// If preprocessing is configured, detrending and/or standardization
    /// will be applied to the extracted time series.
    pub fn fit_transform(&self, bold_path: &PathBuf) -> Result<Array2<f32>> {
        debug!(
            bold_path = %bold_path.display(),
            n_atlas_labels = self.unique_labels.len(),
            "starting fit_transform"
        );

        let bold_obj = ReaderOptions::new().read_file(bold_path)?;
        let bold_header = bold_obj.header();
        let bold_affine = Self::get_affine_from_header(bold_header);
        let bold_data = Self::volume_to_array4(bold_obj.into_volume())?;

        let (bold_sx, bold_sy, bold_sz, n_timepoints) = {
            let s = bold_data.shape();
            (s[0], s[1], s[2], s[3])
        };

        debug!(
            bold_shape_x = bold_sx,
            bold_shape_y = bold_sy,
            bold_shape_z = bold_sz,
            n_timepoints = n_timepoints,
            bold_path = %bold_path.display(),
            "BOLD data loaded"
        );

        // Apply voxel-wise z-score normalization before parcellation if configured
        let mut bold_data = bold_data;
        if self.signal_config.voxelwise_zscore {
            debug!(
                bold_path = %bold_path.display(),
                n_timepoints = n_timepoints,
                "applying voxel-wise z-score normalization"
            );
            voxelwise_zscore_bold(&mut bold_data);
        }

        // Resample atlas to BOLD space
        let needs_resampling =
            self.labels_volume.shape() != &[bold_sx, bold_sy, bold_sz];

        debug!(
            needs_resampling = needs_resampling,
            atlas_shape = ?self.atlas_shape,
            bold_shape = ?[bold_sx, bold_sy, bold_sz],
            "checking resampling requirement"
        );

        let resampled_labels = self.resample_to_target(&bold_data, &bold_affine)?;

        // Extract time series for each label
        let n_labels = self.unique_labels.len();
        let mut result = Array2::<f32>::zeros((n_labels, n_timepoints));
        let mut empty_rois = 0usize;

        for (label_idx, &label) in self.unique_labels.iter().enumerate() {
            let mask: Vec<(usize, usize, usize)> = resampled_labels
                .indexed_iter()
                .filter_map(|((x, y, z), &val)| {
                    if val.round() as i32 == label {
                        Some((x, y, z))
                    } else {
                        None
                    }
                })
                .collect();

            if mask.is_empty() {
                empty_rois += 1;
                trace!(
                    label = label,
                    label_idx = label_idx,
                    "ROI has no voxels after resampling"
                );
                continue;
            }

            trace!(
                label = label,
                label_idx = label_idx,
                n_voxels = mask.len(),
                "extracting timeseries for ROI"
            );

            // Compute mean time series across all voxels in this ROI
            for t in 0..n_timepoints {
                let sum: f32 = mask.iter().map(|&(x, y, z)| bold_data[[x, y, z, t]]).sum();
                result[[label_idx, t]] = sum / mask.len() as f32;
            }
        }

        debug!(
            n_labels = n_labels,
            n_timepoints = n_timepoints,
            empty_rois = empty_rois,
            output_shape = ?[n_labels, n_timepoints],
            bold_path = %bold_path.display(),
            "extraction completed, applying preprocessing"
        );

        // Apply preprocessing if configured
        let result = preprocess_signals(&result, &self.signal_config);

        debug!(
            preprocessing_enabled = self.signal_config.is_enabled(),
            detrend = self.signal_config.detrend,
            standardize = ?self.signal_config.standardize,
            "fit_transform completed"
        );

        Ok(result)
    }

    /// Resample labels to match target BOLD data dimensions using nearest-neighbor interpolation
    fn resample_to_target(
        &self,
        bold_data: &Array4<f32>,
        bold_affine: &Matrix4<f64>,
    ) -> Result<Array3<f32>> {
        let target_shape = (
            bold_data.shape()[0],
            bold_data.shape()[1],
            bold_data.shape()[2],
        );

        // If dimensions already match, return as is
        if self.labels_volume.shape() == &[target_shape.0, target_shape.1, target_shape.2] {
            trace!(
                target_shape = ?target_shape,
                "dimensions match, skipping resampling"
            );
            return Ok(self.labels_volume.clone());
        }

        debug!(
            source_shape = ?self.atlas_shape,
            target_shape = ?target_shape,
            "resampling atlas to BOLD space"
        );

        let labels_affine_inv = self
            .labels_affine
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("Failed to invert atlas affine matrix"))?;
        let transform = labels_affine_inv * bold_affine;

        let mut resampled = Array3::<f32>::zeros(target_shape);
        let src_shape = self.labels_volume.shape();

        let mut in_bounds_count = 0usize;
        let mut out_of_bounds_count = 0usize;

        for x in 0..target_shape.0 {
            for y in 0..target_shape.1 {
                for z in 0..target_shape.2 {
                    let bold_voxel = nalgebra::Vector4::new(x as f64, y as f64, z as f64, 1.0);
                    let atlas_voxel = transform * bold_voxel;

                    let sx = atlas_voxel[0].round() as i64;
                    let sy = atlas_voxel[1].round() as i64;
                    let sz = atlas_voxel[2].round() as i64;

                    if sx >= 0
                        && sx < src_shape[0] as i64
                        && sy >= 0
                        && sy < src_shape[1] as i64
                        && sz >= 0
                        && sz < src_shape[2] as i64
                    {
                        resampled[[x, y, z]] =
                            self.labels_volume[[sx as usize, sy as usize, sz as usize]];
                        in_bounds_count += 1;
                    } else {
                        out_of_bounds_count += 1;
                    }
                }
            }
        }

        let total_voxels = target_shape.0 * target_shape.1 * target_shape.2;
        let coverage_pct = (in_bounds_count as f64 / total_voxels as f64) * 100.0;

        debug!(
            total_voxels = total_voxels,
            in_bounds = in_bounds_count,
            out_of_bounds = out_of_bounds_count,
            coverage_pct = format!("{:.1}", coverage_pct),
            "resampling completed"
        );

        Ok(resampled)
    }

    /// Extract affine matrix from NIfTI header (sform or qform)
    fn get_affine_from_header(header: &NiftiHeader) -> Matrix4<f64> {
        if header.sform_code > 0 {
            trace!(sform_code = header.sform_code, "using sform affine");
            Matrix4::new(
                header.srow_x[0] as f64,
                header.srow_x[1] as f64,
                header.srow_x[2] as f64,
                header.srow_x[3] as f64,
                header.srow_y[0] as f64,
                header.srow_y[1] as f64,
                header.srow_y[2] as f64,
                header.srow_y[3] as f64,
                header.srow_z[0] as f64,
                header.srow_z[1] as f64,
                header.srow_z[2] as f64,
                header.srow_z[3] as f64,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        } else if header.qform_code > 0 {
            trace!(qform_code = header.qform_code, "using qform affine");
            Self::qform_to_affine(header)
        } else {
            trace!(
                pixdim = ?[header.pixdim[1], header.pixdim[2], header.pixdim[3]],
                "using pixdim fallback for affine"
            );
            Matrix4::new(
                header.pixdim[1] as f64,
                0.0,
                0.0,
                0.0,
                0.0,
                header.pixdim[2] as f64,
                0.0,
                0.0,
                0.0,
                0.0,
                header.pixdim[3] as f64,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        }
    }

    /// Convert qform quaternion parameters to affine matrix
    fn qform_to_affine(header: &NiftiHeader) -> Matrix4<f64> {
        let b = header.quatern_b as f64;
        let c = header.quatern_c as f64;
        let d = header.quatern_d as f64;
        let a = (1.0 - b * b - c * c - d * d).max(0.0).sqrt();

        let r11 = a * a + b * b - c * c - d * d;
        let r12 = 2.0 * (b * c - a * d);
        let r13 = 2.0 * (b * d + a * c);
        let r21 = 2.0 * (b * c + a * d);
        let r22 = a * a + c * c - b * b - d * d;
        let r23 = 2.0 * (c * d - a * b);
        let r31 = 2.0 * (b * d - a * c);
        let r32 = 2.0 * (c * d + a * b);
        let r33 = a * a + d * d - b * b - c * c;

        let pi = header.pixdim[1] as f64;
        let pj = header.pixdim[2] as f64;
        let pk = header.pixdim[3] as f64;

        let qfac = if header.pixdim[0] < 0.0 { -1.0 } else { 1.0 };

        let qx = header.quatern_x as f64;
        let qy = header.quatern_y as f64;
        let qz = header.quatern_z as f64;

        trace!(
            quatern_b = b,
            quatern_c = c,
            quatern_d = d,
            quatern_a = a,
            qfac = qfac,
            offset = ?[qx, qy, qz],
            "qform parameters"
        );

        Matrix4::new(
            r11 * pi,
            r12 * pj,
            r13 * pk * qfac,
            qx,
            r21 * pi,
            r22 * pj,
            r23 * pk * qfac,
            qy,
            r31 * pi,
            r32 * pj,
            r33 * pk * qfac,
            qz,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Helper: Convert NiftiVolume to Array3
    ///
    /// Note: NIfTI data is stored in Fortran (column-major) order, so we must
    /// preserve this memory layout when converting to ndarray.
    fn volume_to_array3<V>(volume: V) -> Result<Array3<f32>>
    where
        V: NiftiVolume + IntoNdArray,
    {
        let array = volume.into_ndarray::<f32>()?;
        let ndim = array.ndim();
        let shape: Vec<usize> = array.shape().to_vec();

        trace!(
            ndim = ndim,
            shape = ?shape,
            "converting volume to Array3"
        );

        match ndim {
            3 => {
                // nifti-rs returns data in Fortran order, so we must use .f() to preserve it
                let (data, _offset) = array.into_raw_vec_and_offset();
                Ok(Array3::from_shape_vec(
                    (shape[0], shape[1], shape[2]).f(),
                    data,
                )?)
            }
            4 => {
                if shape[3] == 1 {
                    // nifti-rs returns data in Fortran order, so we must use .f() to preserve it
                    let (data, _offset) = array.into_raw_vec_and_offset();
                    Ok(Array3::from_shape_vec(
                        (shape[0], shape[1], shape[2]).f(),
                        data,
                    )?)
                } else {
                    bail!("Expected 3D volume for atlas, got 4D with multiple timepoints")
                }
            }
            _ => bail!("Unexpected number of dimensions: {}", ndim),
        }
    }

    /// Helper: Convert NiftiVolume to Array4 (for 4D BOLD data)
    ///
    /// Note: NIfTI data is stored in Fortran (column-major) order, so we must
    /// preserve this memory layout when converting to ndarray.
    fn volume_to_array4<V>(volume: V) -> Result<Array4<f32>>
    where
        V: NiftiVolume + IntoNdArray,
    {
        let array = volume.into_ndarray::<f32>()?;
        let ndim = array.ndim();
        let shape: Vec<usize> = array.shape().to_vec();

        trace!(
            ndim = ndim,
            shape = ?shape,
            "converting volume to Array4"
        );

        if ndim == 4 {
            // nifti-rs returns data in Fortran order, so we must use .f() to preserve it
            let (data, _offset) = array.into_raw_vec_and_offset();
            Ok(Array4::from_shape_vec(
                (shape[0], shape[1], shape[2], shape[3]).f(),
                data,
            )?)
        } else {
            bail!("Expected 4D BOLD volume, got {}D", ndim)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_detrend_removes_linear_trend() {
        // Create a signal with a known linear trend (2 ROIs, 5 timepoints)
        // ROI 0: y = 2*t -> [0, 2, 4, 6, 8]
        // ROI 1: y = t + 10 -> [10, 11, 12, 13, 14]
        let data = array![
            [0.0_f32, 2.0, 4.0, 6.0, 8.0],
            [10.0_f32, 11.0, 12.0, 13.0, 14.0]
        ];

        let detrended = detrend_signal(&data);

        // After detrending, the signal should be approximately zero
        for roi in 0..2 {
            for t in 0..5 {
                assert!(
                    detrended[[roi, t]].abs() < 1e-5,
                    "Detrended value [{}, {}] = {} should be near zero",
                    roi,
                    t,
                    detrended[[roi, t]]
                );
            }
        }
    }

    #[test]
    fn test_standardize_zscore() {
        // Create a signal with known mean and std (1 ROI, 5 timepoints)
        let data = array![[1.0_f32, 2.0, 3.0, 4.0, 5.0]];

        let standardized = standardize_signal(&data, Standardize::ZscoreSample, false);

        // Check mean is approximately 0
        let mean: f32 = standardized.row(0).sum() / 5.0;
        assert!(mean.abs() < 1e-5, "Mean {} should be near zero", mean);

        // Check std is approximately 1 (using ddof=1)
        let variance: f32 = standardized.row(0).iter().map(|&v| v * v).sum::<f32>() / 4.0;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 1e-5, "Std {} should be near 1.0", std);
    }

    #[test]
    fn test_standardize_psc() {
        // Create a signal with known mean (1 ROI, 5 timepoints)
        // Mean = 100
        let data = array![[100.0_f32, 110.0, 90.0, 105.0, 95.0]];

        let standardized = standardize_signal(&data, Standardize::Psc, false);

        // PSC[0] = (100 - 100) / 100 * 100 = 0
        // PSC[1] = (110 - 100) / 100 * 100 = 10
        // PSC[2] = (90 - 100) / 100 * 100 = -10
        assert!((standardized[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((standardized[[0, 1]] - 10.0).abs() < 1e-5);
        assert!((standardized[[0, 2]] - (-10.0)).abs() < 1e-5);
    }

    #[test]
    fn test_voxelwise_zscore_bold() {
        // 2x1x1 spatial, 5 timepoints
        // voxel (0,0,0): [1, 2, 3, 4, 5]  mean=3, std=sqrt(2.5)
        // voxel (1,0,0): [10, 10, 10, 10, 10]  constant -> should be left unchanged
        let mut data = Array4::<f32>::zeros((2, 1, 1, 5));
        for t in 0..5usize {
            data[[0, 0, 0, t]] = (t + 1) as f32;
            data[[1, 0, 0, t]] = 10.0;
        }

        voxelwise_zscore_bold(&mut data);

        // Voxel 0: mean should be ~0, std ~1
        let mean0: f32 = (0..5).map(|t| data[[0, 0, 0, t]]).sum::<f32>() / 5.0;
        assert!(mean0.abs() < 1e-5, "mean {} should be near zero", mean0);
        let var0: f32 = (0..5).map(|t| data[[0, 0, 0, t]].powi(2)).sum::<f32>() / 4.0;
        assert!((var0.sqrt() - 1.0).abs() < 1e-5, "std {} should be near 1", var0.sqrt());

        // Voxel 1: constant, left unchanged
        for t in 0..5 {
            assert_eq!(data[[1, 0, 0, t]], 10.0);
        }
    }

    #[test]
    fn test_masker_signal_config_builder() {
        let config = MaskerSignalConfig::default()
            .detrend(true)
            .standardize(Standardize::ZscoreSample);

        assert!(config.detrend);
        assert_eq!(config.standardize, Standardize::ZscoreSample);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_masker_signal_config_disabled() {
        let config = MaskerSignalConfig::default();

        assert!(!config.detrend);
        assert_eq!(config.standardize, Standardize::None);
        assert!(!config.is_enabled());
    }
}
