use anyhow::{Result, bail};
use nalgebra::Matrix4;
use ndarray::{Array2, Array3, Array4};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, NiftiVolume, ReaderOptions};
use std::{collections::HashSet, path::PathBuf};

pub struct LabelsMasker {
    /// The atlas with integer labels for each ROI
    labels_volume: Array3<f32>,
    /// Affine transformation matrix of the atlas (voxel to world)
    labels_affine: Matrix4<f64>,
    /// Unique labels in the atlas (excluding 0/background)
    unique_labels: Vec<i32>,
}

impl LabelsMasker {
    /// Load and prepare the atlas
    pub fn new(atlas_path: &PathBuf) -> Result<Self> {
        let atlas_obj = ReaderOptions::new().read_file(atlas_path)?;
        let header = atlas_obj.header();
        let labels_affine = Self::get_affine_from_header(header);
        let labels_volume = Self::volume_to_array3(atlas_obj.into_volume())?;

        // Find unique labels (excluding 0 which is background)
        let mut unique_labels: Vec<i32> = labels_volume
            .iter()
            .map(|&v| v.round() as i32)
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|&l| l != 0)
            .collect();
        unique_labels.sort();

        Ok(Self {
            labels_volume,
            labels_affine,
            unique_labels,
        })
    }

    /// Get the number of unique labels (ROIs) in the atlas
    pub fn n_labels(&self) -> usize {
        self.unique_labels.len()
    }

    /// Extract time series for each labeled region from BOLD data
    /// Returns Array2 of shape (n_labels, n_timepoints)
    pub fn fit_transform(&self, bold_path: &PathBuf) -> Result<Array2<f32>> {
        let bold_obj = ReaderOptions::new().read_file(bold_path)?;
        let bold_header = bold_obj.header();
        let bold_affine = Self::get_affine_from_header(bold_header);
        let bold_data = Self::volume_to_array4(bold_obj.into_volume())?;
        let n_timepoints = bold_data.shape()[3];

        // Resample atlas to BOLD space
        let resampled_labels = self.resample_to_target(&bold_data, &bold_affine)?;

        // Extract time series for each label
        let n_labels = self.unique_labels.len();
        let mut result = Array2::<f32>::zeros((n_labels, n_timepoints));

        for (label_idx, &label) in self.unique_labels.iter().enumerate() {
            // Find all voxels with this label
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
                continue;
            }

            // Compute mean time series across all voxels in this ROI
            for t in 0..n_timepoints {
                let sum: f32 = mask.iter().map(|&(x, y, z)| bold_data[[x, y, z, t]]).sum();
                result[[label_idx, t]] = sum / mask.len() as f32;
            }
        }

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
            return Ok(self.labels_volume.clone());
        }

        // Compute transformation from BOLD voxel space to atlas voxel space
        // bold_voxel -> world -> atlas_voxel
        // T = inv(labels_affine) @ bold_affine
        let labels_affine_inv = self
            .labels_affine
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("Failed to invert atlas affine matrix"))?;
        let transform = labels_affine_inv * bold_affine;

        let mut resampled = Array3::<f32>::zeros(target_shape);

        let src_shape = self.labels_volume.shape();

        for x in 0..target_shape.0 {
            for y in 0..target_shape.1 {
                for z in 0..target_shape.2 {
                    // Transform BOLD voxel coordinate to atlas voxel coordinate
                    let bold_voxel = nalgebra::Vector4::new(x as f64, y as f64, z as f64, 1.0);
                    let atlas_voxel = transform * bold_voxel;

                    // Nearest neighbor interpolation (appropriate for label data)
                    let sx = atlas_voxel[0].round() as i64;
                    let sy = atlas_voxel[1].round() as i64;
                    let sz = atlas_voxel[2].round() as i64;

                    // Check bounds
                    if sx >= 0
                        && sx < src_shape[0] as i64
                        && sy >= 0
                        && sy < src_shape[1] as i64
                        && sz >= 0
                        && sz < src_shape[2] as i64
                    {
                        resampled[[x, y, z]] =
                            self.labels_volume[[sx as usize, sy as usize, sz as usize]];
                    }
                }
            }
        }

        Ok(resampled)
    }

    /// Extract affine matrix from NIfTI header (sform or qform)
    fn get_affine_from_header(header: &NiftiHeader) -> Matrix4<f64> {
        // Prefer sform if available (sform_code > 0), otherwise use qform
        if header.sform_code > 0 {
            // sform is stored as srow_x, srow_y, srow_z (each is [f32; 4])
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
            // Build affine from quaternion parameters
            Self::qform_to_affine(header)
        } else {
            // Fallback to simple scaling from pixdim
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

        // Rotation matrix from quaternion
        let r11 = a * a + b * b - c * c - d * d;
        let r12 = 2.0 * (b * c - a * d);
        let r13 = 2.0 * (b * d + a * c);
        let r21 = 2.0 * (b * c + a * d);
        let r22 = a * a + c * c - b * b - d * d;
        let r23 = 2.0 * (c * d - a * b);
        let r31 = 2.0 * (b * d - a * c);
        let r32 = 2.0 * (c * d + a * b);
        let r33 = a * a + d * d - b * b - c * c;

        // Voxel dimensions
        let pi = header.pixdim[1] as f64;
        let pj = header.pixdim[2] as f64;
        let pk = header.pixdim[3] as f64;

        // qfac determines the sign of the third column
        let qfac = if header.pixdim[0] < 0.0 { -1.0 } else { 1.0 };

        // Translation (quaternion offsets)
        let qx = header.quatern_x as f64;
        let qy = header.quatern_y as f64;
        let qz = header.quatern_z as f64;

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
    fn volume_to_array3<V>(volume: V) -> Result<Array3<f32>>
    where
        V: NiftiVolume + IntoNdArray,
    {
        let array = volume.into_ndarray::<f32>()?;
        let ndim = array.ndim();
        let shape: Vec<usize> = array.shape().to_vec();

        match ndim {
            3 => {
                let (data, _offset) = array.into_raw_vec_and_offset();
                Ok(Array3::from_shape_vec(
                    (shape[0], shape[1], shape[2]),
                    data,
                )?)
            }
            4 => {
                if shape[3] == 1 {
                    let (data, _offset) = array.into_raw_vec_and_offset();
                    Ok(Array3::from_shape_vec(
                        (shape[0], shape[1], shape[2]),
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
    fn volume_to_array4<V>(volume: V) -> Result<Array4<f32>>
    where
        V: NiftiVolume + IntoNdArray,
    {
        let array = volume.into_ndarray::<f32>()?;
        let ndim = array.ndim();
        let shape: Vec<usize> = array.shape().to_vec();

        if ndim == 4 {
            let (data, _offset) = array.into_raw_vec_and_offset();
            Ok(Array4::from_shape_vec(
                (shape[0], shape[1], shape[2], shape[3]),
                data,
            )?)
        } else {
            bail!("Expected 4D BOLD volume, got {}D", ndim)
        }
    }
}
