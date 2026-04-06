use anyhow::Result;
use config::{bids_filename::BidsFilename, bids_subject_id::BidsSubjectId, tcp_config::CwtConfig};
use fastcwt::*;
use hdf5_io::{open_or_create, open_or_create_group, write_dataset};
use ndarray::{Array2, Axis, concatenate};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};
use tracing::{debug, info, warn};

/// Compute CWT independently for each channel of a 2D signal `(n_channels, n_timepoints)`.
///
/// Returns `(real, imag, [n_channels, output_len])` where `output_len` is the number of
/// complex coefficients returned by `fastcwt` per channel (equals `n_timepoints`).
/// Compute CWT independently for each channel of a 2D signal `(n_channels, n_timepoints)`.
///
/// `fastcwt` requires the input slice to be pre-padded to `n_timepoints.next_power_of_two()`;
/// we handle that here so callers pass the raw signal unchanged.
///
/// Returns `(real, imag, [n_channels, padded_len])` where `padded_len` is
/// `n_timepoints.next_power_of_two()`.
fn cwt_multichannel(
    transform: &mut FastCWT,
    signal: &Array2<f64>,
) -> (Vec<f64>, Vec<f64>, [usize; 2]) {
    let n_channels = signal.nrows();
    let n_timepoints = signal.ncols();
    let padded_len = n_timepoints.next_power_of_two();

    let mut real: Vec<f64> = Vec::with_capacity(n_channels * padded_len);
    let mut imag: Vec<f64> = Vec::with_capacity(n_channels * padded_len);

    let mut padded = vec![0.0f64; padded_len];
    for ch in 0..n_channels {
        let channel = signal.row(ch);
        let channel_slice = channel.as_slice().expect("channel row is not contiguous");
        padded[..n_timepoints].copy_from_slice(channel_slice);
        // zeros from n_timepoints..padded_len are already set from the previous iteration or init

        let result = transform.cwt(padded_len, &padded, make_scales());
        real.extend(result.iter().map(|c| c.re));
        imag.extend(result.iter().map(|c| c.im));
    }

    (real, imag, [n_channels, padded_len])
}

fn make_scales() -> Scales {
    let distribution_type = ScaleTypes::LinFreq;
    let repetition_time: f64 = 0.8; // 800 ms
    let analog_sampling_freq: usize = (1.0 / repetition_time).floor() as usize; // really 1.25 Hz
    let freq_range_start = 2.00;
    let freq_range_end = (analog_sampling_freq / 2) as f64;
    let num_wavelets = 1000;
    Scales::create(
        distribution_type,
        analog_sampling_freq,
        freq_range_start,
        freq_range_end,
        num_wavelets,
    )
}

pub fn run(cfg: &CwtConfig) -> Result<()> {
    let run_start = Instant::now();

    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    /////////////////////////
    // Create CWT Instance //
    /////////////////////////

    let wavelet = Wavelet::create(1.0);
    let mut transform = FastCWT::create(wavelet, true);

    info!(
        bold_ts_dir = %cfg.bold_ts_dir.display(),
        force = cfg.force,
        "starting fMRI CWT pipeline"
    );

    /////////////////////
    // Get Time Series //
    /////////////////////

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.bold_ts_dir)?
        .filter_map(|entry_result| entry_result.ok())
        .filter_map(|entry| {
            let path = entry.path();
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

        let available_timeseries: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .filter_map(|path| {
                let p = path.file_name()?.to_str()?.to_string();
                if p.contains(".h5") { Some(path) } else { None }
            })
            .collect();

        for file_path in &available_timeseries {
            let bids = BidsFilename::parse(match file_path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name,
                None => continue,
            });
            let task_name = bids.get("task").unwrap_or("unknown");

            let h5_file = open_or_create(&file_path)?;
            let cwt_group = open_or_create_group(&h5_file, "cwt", false)?;

            let wb_done = !cfg.force && cwt_group.group("whole-band").is_ok();

            //////////////////////////////////
            // Continuous Wavelet Transform //
            //////////////////////////////////

            // Whole-band
            if wb_done {
                debug!(
                    subject = formatted_id,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    task_name = task_name,
                    "whole-band CWT already computed, skipping (use --force to recompute)"
                );
            } else {
                let cwt_start = Instant::now();

                let dataset = h5_file.dataset("tcp_timeseries_raw")?;
                let data_f32: Array2<f32> = dataset.read_2d()?;
                let [n_channels, n_timepoints] = match data_f32.shape() {
                    &[r, c] => [r, c],
                    _ => anyhow::bail!("expected 2D timeseries, got shape {:?}", data_f32.shape()),
                };
                let data_f64 = data_f32.mapv(|val| val as f64);
                let (real, imag, shape) = cwt_multichannel(&mut transform, &data_f64);
                let cwt_duration_ms = cwt_start.elapsed().as_millis();

                let write_start = Instant::now();
                let wb_group = open_or_create_group(&cwt_group, "whole-band", cfg.force)?;
                write_dataset(&wb_group, "spectrogram_real", &real, &shape, None)?;
                write_dataset(&wb_group, "spectrogram_imag", &imag, &shape, None)?;
                let write_duration_ms = write_start.elapsed().as_millis();

                debug!(
                    subject = formatted_id,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    task_name = task_name,
                    n_channels = n_channels,
                    n_timepoints = n_timepoints,
                    output_shape = ?shape,
                    cwt_duration_ms = cwt_duration_ms,
                    write_duration_ms = write_duration_ms,
                    output_file = %file_path.display(),
                    "computed whole-band CWT"
                );
            }

            // Block-level
            let blocks_group = match h5_file.group("blocks") {
                Ok(g) => g,
                Err(_) => {
                    debug!(
                        subject = formatted_id,
                        task_name = task_name,
                        "no blocks group found, skipping block decomposition"
                    );
                    continue;
                }
            };

            let block_names: Vec<String> = blocks_group
                .member_names()?
                .into_iter()
                .filter(|n| n.starts_with("block_"))
                .collect();

            if block_names.is_empty() {
                debug!(
                    subject = formatted_id,
                    task_name = task_name,
                    "blocks group is empty, skipping block decomposition"
                );
                continue;
            }

            let cwt_blocks_group = open_or_create_group(&cwt_group, "blocks", cfg.force)?;
            for block_name in &block_names {
                if !cfg.force && cwt_blocks_group.group(block_name).is_ok() {
                    debug!(
                        subject = formatted_id,
                        task_name = task_name,
                        block = block_name,
                        "block CWT already computed, skipping (use --force to recompute)"
                    );
                    continue;
                }

                let block_group = blocks_group.group(block_name)?;

                let cortical: Array2<f32> = block_group.dataset("cortical_raw")?.read_2d()?;
                let subcortical: Array2<f32> = block_group.dataset("subcortical_raw")?.read_2d()?;

                let block_signal_f32 =
                    concatenate(Axis(0), &[cortical.view(), subcortical.view()])?;
                let [block_channels, block_timepoints] = match block_signal_f32.shape() {
                    &[r, c] => [r, c],
                    _ => {
                        error_count += 1;
                        warn!(
                            subject = formatted_id,
                            task_name = task_name,
                            block = block_name,
                            reason = "unexpected_block_shape",
                            shape = ?block_signal_f32.shape(),
                            "skipping block CWT due to unexpected signal shape"
                        );
                        continue;
                    }
                };
                let block_signal_f64 = block_signal_f32.mapv(|val| val as f64);

                let block_cwt_start = Instant::now();
                let (real, imag, shape) = cwt_multichannel(&mut transform, &block_signal_f64);
                let block_cwt_duration_ms = block_cwt_start.elapsed().as_millis();

                let block_write_start = Instant::now();
                let cwt_block_group =
                    open_or_create_group(&cwt_blocks_group, block_name, cfg.force)?;
                write_dataset(&cwt_block_group, "spectrogram_real", &real, &shape, None)?;
                write_dataset(&cwt_block_group, "spectrogram_imag", &imag, &shape, None)?;
                let block_write_duration_ms = block_write_start.elapsed().as_millis();

                debug!(
                    subject = formatted_id,
                    task_name = task_name,
                    block = block_name,
                    n_channels = block_channels,
                    n_timepoints = block_timepoints,
                    output_shape = ?shape,
                    cwt_duration_ms = block_cwt_duration_ms,
                    write_duration_ms = block_write_duration_ms,
                    "computed block CWT"
                );
            }

            debug!(
                subject = formatted_id,
                task_name = task_name,
                num_blocks = block_names.len(),
                "finished block CWT decompositions"
            );
        }
    }

    if error_count > 0 {
        warn!(
            error_count = error_count,
            "some blocks were skipped due to errors"
        );
    }

    let total_duration_ms = run_start.elapsed().as_millis();
    info!(
        total_subjects = total_subjects,
        error_count = error_count,
        total_duration_ms = total_duration_ms,
        "Continuous Wavelet Transform pipeline completed"
    );

    Ok(())
}
