//! Probabilistic KNN evaluator.
//!
//! Three-way stratified row-wise split, z-score using train stats, fit KNN
//! with the configured distance, then **emit per-sample probabilities** rather
//! than hard labels:
//!
//! * `p1_raw`  – raw KNN vote-share for class 1 (anhedonic).
//! * `p1_cal`  – Platt-scaled probability fit on the validation split and
//!               applied to test (and to val itself for diagnostics).
//!
//! For each split we compute Brier score, log loss, AUC-ROC, AUC-PR, expected
//! calibration error, a uniform-binned reliability table, a threshold sweep,
//! and the Youden-optimal threshold. We retain the legacy `accuracy /
//! sensitivity / specificity / confusion_matrix` block so existing notebooks
//! keep working — they're now reported at the Youden threshold rather than
//! 0.5, with a parallel `*_at_0_5` block for direct comparison to the
//! pre-refactor numbers.
//!
//! A sibling `*_subject_probs.csv` is dumped next to each JSON for downstream
//! python plotting (reliability diagram, subject-rank uncertainty plot).

use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::path::Path;
use tracing::info;
use utils::bids_filename::BidsFilename;

use crate::calibration::PlattScaler;
use crate::classifiers::{
    DistanceMetric, KNN, KnnConfig, accuracy, confusion_matrix_binary, sensitivity_from_cm,
    specificity_from_cm,
};
use crate::dataset::{FeatureSource, Label};
use crate::metrics::{
    CalibrationBin, ThresholdReport, auc_pr, auc_roc, brier_score, calibration_bins,
    expected_calibration_error, log_loss, threshold_sweep, youden_optimal_threshold,
};
use crate::normalizer::ZScoreNormalizer;
use crate::splits::split_rows_stratified_new;

const SEED: u64 = 42;
const SWEEP_THRESHOLDS: &[f32] = &[0.3, 0.4, 0.5, 0.6, 0.7];
const N_CALIBRATION_BINS: usize = 10;
const LOGLOSS_EPS: f32 = 1e-7;

#[derive(Debug, Serialize)]
struct HardReport {
    threshold: f32,

    accuracy: f32,
    sensitivity: f32,
    specificity: f32,

    precision: f32,
    npv: f32,
    f1_score: f32,
    mcc: f32,

    confusion_matrix: [[u32; 2]; 2],
}

#[derive(Debug, Serialize)]
struct ProbabilisticReport {
    brier: f32,
    log_loss: f32,
    auc_roc: f32,
    auc_pr: f32,
    expected_calibration_error: f32,
    calibration_bins: Vec<CalibrationBin>,
    threshold_sweep: Vec<ThresholdReport>,
    youden_threshold: f32,
}

#[derive(Debug, Serialize)]
struct SplitReport {
    n_samples: usize,
    /// Hard-decision metrics at the legacy 0.5 threshold.
    at_0_5: HardReport,
    /// Hard-decision metrics at the Youden-optimal threshold (estimated on
    /// this split's predictions).
    at_youden: HardReport,
    /// Same probabilistic block computed on raw KNN vote-share.
    raw: ProbabilisticReport,
    /// Probabilistic block after Platt scaling fit on val.
    calibrated: ProbabilisticReport,
}

#[derive(Debug, Serialize)]
struct SplitManifest {
    train: Vec<String>,
    test: Vec<String>,
    val: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PerSamplePrediction {
    subject: String,
    leaf: Option<String>,
    roi: Option<usize>,
    y_true: i32,
    p1_raw: f32,
    p1_calibrated: f32,
}

#[derive(Debug, Serialize)]
struct ClassificationReport {
    analysis: String,
    source: String,
    split_seed: u64,
    classifier: String,
    num_neighbors: usize,
    metric: String,
    distance_weighted: bool,
    n_train: usize,
    platt_a: f32,
    platt_b: f32,
    test: SplitReport,
    val: SplitReport,
    test_predictions: Vec<PerSamplePrediction>,
    val_predictions: Vec<PerSamplePrediction>,
    split_manifest: SplitManifest,
}

fn sorted_unique_subjects(indices: &[usize], groups: &[String]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    for &i in indices {
        seen.insert(parse_subject(&groups[i]));
    }
    seen.into_iter().collect()
}

/// Group strings produced by `dataset.rs` look like
/// `sub-NDARxxxx[_<leaf>]_roiNNN`. Subjects don't contain `_`, so the first
/// `_` (or `_roi` when there's no leaf) marks the subject boundary.
fn parse_group(g: &str) -> (String, Option<String>, Option<usize>) {
    let (prefix, roi) = match g.rfind("_roi") {
        Some(i) => {
            let roi = g[i + 4..].parse::<usize>().ok();
            (&g[..i], roi)
        }
        None => (g, None),
    };
    let (subject, leaf) = match prefix.find('_') {
        Some(i) => (prefix[..i].to_string(), Some(prefix[i + 1..].to_string())),
        None => (prefix.to_string(), None),
    };
    (subject, leaf, roi)
}

fn parse_subject(g: &str) -> String {
    parse_group(g).0
}

fn hard_report_at(y_true: &[i32], p1: &[f32], threshold: f32) -> HardReport {
    let preds: Vec<i32> = p1.iter().map(|&p| (p >= threshold) as i32).collect();
    let cm = confusion_matrix_binary(y_true, &preds);

    let tn_f = cm[0][0] as f32;
    let fp_f = cm[0][1] as f32;
    let fn_f = cm[1][0] as f32;
    let tp_f = cm[1][1] as f32;

    let precision = if tp_f + fp_f > 0.0 {
        tp_f / (tp_f + fp_f)
    } else {
        0.0
    };
    let sensitivity = if tp_f + fn_f > 0.0 {
        tp_f / (tp_f + fn_f)
    } else {
        0.0
    };
    let npv = if tn_f + fn_f > 0.0 {
        tn_f / (tn_f + fn_f)
    } else {
        0.0
    };

    let f1_score = if precision + sensitivity > 0.0 {
        2.0 * (precision * sensitivity) / (precision + sensitivity)
    } else {
        0.0
    };

    let mcc_denominator = ((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f)).sqrt();
    let mcc = if mcc_denominator > 0.0 {
        ((tp_f * tn_f) - (fp_f * fn_f)) / mcc_denominator
    } else {
        0.0
    };

    HardReport {
        threshold,
        accuracy: accuracy(y_true, &preds),
        sensitivity: sensitivity_from_cm(&cm),
        specificity: specificity_from_cm(&cm),
        precision,
        npv,
        f1_score,
        mcc,
        confusion_matrix: cm,
    }
}

fn prob_report(y_true: &[i32], p1: &[f32]) -> ProbabilisticReport {
    let bins = calibration_bins(y_true, p1, N_CALIBRATION_BINS);
    let ece = expected_calibration_error(&bins);
    ProbabilisticReport {
        brier: brier_score(y_true, p1),
        log_loss: log_loss(y_true, p1, LOGLOSS_EPS),
        auc_roc: auc_roc(y_true, p1),
        auc_pr: auc_pr(y_true, p1),
        expected_calibration_error: ece,
        calibration_bins: bins,
        threshold_sweep: threshold_sweep(y_true, p1, SWEEP_THRESHOLDS),
        youden_threshold: youden_optimal_threshold(y_true, p1),
    }
}

fn p1_index(classes: &[i32]) -> Option<usize> {
    classes.iter().position(|&c| c == 1)
}

fn build_predictions(
    parsed_groups: Vec<(String, Option<String>, Option<usize>)>,
    y_true: &[i32],
    p1_raw: &[f32],
    p1_cal: &[f32],
) -> Vec<PerSamplePrediction> {
    parsed_groups
        .into_iter()
        .enumerate()
        .map(|(j, (subject, leaf, roi))| PerSamplePrediction {
            subject,
            leaf,
            roi,
            y_true: y_true[j],
            p1_raw: p1_raw[j],
            p1_calibrated: p1_cal[j],
        })
        .collect()
}

fn write_subject_probs_csv<'a>(
    path: &Path,
    predictions: impl IntoIterator<Item = &'a PerSamplePrediction>,
) -> Result<()> {
    let mut out = String::from("subject,leaf,roi,y_true,p1_raw,p1_calibrated\n");
    for p in predictions {
        out.push_str(&format!(
            "{},{},{},{},{},{}\n",
            p.subject,
            p.leaf.as_deref().unwrap_or(""),
            p.roi.map(|r| r.to_string()).unwrap_or_default(),
            p.y_true,
            p.p1_raw,
            p.p1_calibrated,
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

/// Stratified row-wise split, train-fit z-score, K-NN with the supplied
/// distance metric. Computes raw and calibrated per-sample probabilities and
/// reports the full probabilistic metric suite for both test and val.
///
/// Takes `xs` and `ys` by value so the caller's row buffer can be released
/// as soon as we've drained it into train/test/val splits — for the larger
/// face-block analyses this is the difference between peak ~2× and peak ~4×
/// the dataset size.
pub fn eval_knn_three_way_split(
    xs: Vec<Vec<f32>>,
    ys: Vec<Label>,
    groups: &[String],
    num_neighbors: usize,
    metric: DistanceMetric,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
) -> Result<()> {
    let (train_idx, test_idx, val_idx) = split_rows_stratified_new(&ys, SEED);

    // Pre-compute everything we need from `groups` before we consume `xs`/`ys`,
    // so the per-row iteration that follows can move rows directly into the
    // split buffers without ever holding the original row twice.
    let train_subjects = sorted_unique_subjects(&train_idx, groups);
    let test_subjects = sorted_unique_subjects(&test_idx, groups);
    let val_subjects = sorted_unique_subjects(&val_idx, groups);
    let test_parsed: Vec<(String, Option<String>, Option<usize>)> =
        test_idx.iter().map(|&i| parse_group(&groups[i])).collect();
    let val_parsed: Vec<(String, Option<String>, Option<usize>)> =
        val_idx.iter().map(|&i| parse_group(&groups[i])).collect();

    // Build a per-row destination map: index → (split, slot in split). One
    // pass over `xs.into_iter()` then drains every row into exactly one
    // split (or drops it). Original `xs` Vec spine is freed at the end of
    // the loop; only the three split Vecs survive.
    #[derive(Clone, Copy)]
    enum Bucket {
        Train(usize),
        Test(usize),
        Val(usize),
        None,
    }
    let n = ys.len();
    let mut bucket = vec![Bucket::None; n];
    for (slot, &i) in train_idx.iter().enumerate() {
        bucket[i] = Bucket::Train(slot);
    }
    for (slot, &i) in test_idx.iter().enumerate() {
        bucket[i] = Bucket::Test(slot);
    }
    for (slot, &i) in val_idx.iter().enumerate() {
        bucket[i] = Bucket::Val(slot);
    }

    let placeholder_row: Vec<f32> = Vec::new();
    let mut x_train_n: Vec<Vec<f32>> = vec![placeholder_row.clone(); train_idx.len()];
    let mut x_test_n: Vec<Vec<f32>> = vec![placeholder_row.clone(); test_idx.len()];
    let mut x_val_n: Vec<Vec<f32>> = vec![placeholder_row; val_idx.len()];
    let mut y_train: Vec<i32> = vec![0; train_idx.len()];
    let mut y_test: Vec<i32> = vec![0; test_idx.len()];
    let mut y_val: Vec<i32> = vec![0; val_idx.len()];

    for (i, (row, label)) in xs.into_iter().zip(ys.into_iter()).enumerate() {
        match bucket[i] {
            Bucket::Train(s) => {
                x_train_n[s] = row;
                y_train[s] = label.as_i32();
            }
            Bucket::Test(s) => {
                x_test_n[s] = row;
                y_test[s] = label.as_i32();
            }
            Bucket::Val(s) => {
                x_val_n[s] = row;
                y_val[s] = label.as_i32();
            }
            Bucket::None => {}
        }
    }
    drop(bucket);

    // Fit then normalise in-place. f32-native — no f64 round-trip, no extra
    // full-dataset allocations. Combined with the move-into-splits above,
    // peak memory is ~2× xs (xs in caller is already freed by the time we
    // get here; the live data is the three split Vecs only).
    let normalizer = ZScoreNormalizer::fit_f32(&x_train_n);
    normalizer.transform_f32_inplace(&mut x_train_n);
    normalizer.transform_f32_inplace(&mut x_test_n);
    normalizer.transform_f32_inplace(&mut x_val_n);

    // Distance-weight votes so the raw probability is smoother than k-step
    // quantisation. This is the change that makes p1_raw a useful input to
    // Platt scaling on small calibration sets.
    let mut knn = KNN::new(KnnConfig {
        num_neighbors,
        metric,
        distance_weighted: true,
        mahalanobis_shrinkage: 0.0,
    });
    knn.fit(x_train_n, y_train)?;

    let classes = knn.classes().to_vec();
    let pos_idx = p1_index(&classes).ok_or_else(|| {
        anyhow::anyhow!("eval: positive class label `1` missing from training data")
    })?;

    // Forecasting model
    let p1 = |xs: &[Vec<f32>]| -> Result<Vec<f32>> {
        Ok(knn
            .predict_proba_batch(xs)?
            .into_iter()
            .map(|row| row[pos_idx])
            .collect())
    };

    let p1_test_raw = p1(&x_test_n)?;
    let p1_val_raw = p1(&x_val_n)?;

    drop(x_test_n);
    drop(x_val_n);

    // Platt: fit on val. Val is the calibration set; test is the held-out
    // evaluation set we must not touch during scaler fitting.
    let scaler = PlattScaler::fit(&p1_val_raw, &y_val).unwrap_or(PlattScaler::identity());
    let p1_test_cal = scaler.transform_slice(&p1_test_raw);
    let p1_val_cal = scaler.transform_slice(&p1_val_raw);

    let test_youden_t = youden_optimal_threshold(&y_test, &p1_test_cal);
    let val_youden_t = youden_optimal_threshold(&y_val, &p1_val_cal);
    let test_split = SplitReport {
        n_samples: test_idx.len(),
        at_0_5: hard_report_at(&y_test, &p1_test_cal, 0.5),
        at_youden: hard_report_at(&y_test, &p1_test_cal, test_youden_t),
        raw: prob_report(&y_test, &p1_test_raw),
        calibrated: prob_report(&y_test, &p1_test_cal),
    };
    let val_split = SplitReport {
        n_samples: val_idx.len(),
        at_0_5: hard_report_at(&y_val, &p1_val_cal, 0.5),
        at_youden: hard_report_at(&y_val, &p1_val_cal, val_youden_t),
        raw: prob_report(&y_val, &p1_val_raw),
        calibrated: prob_report(&y_val, &p1_val_cal),
    };

    info!(
        analysis,
        source = ?source,
        n_train = train_idx.len(),
        n_test = test_idx.len(),
        n_val = val_idx.len(),
        test_acc_0_5 = format!("{:.2}%", test_split.at_0_5.accuracy * 100.0),
        test_acc_youden = format!("{:.2}%", test_split.at_youden.accuracy * 100.0),
        test_brier_cal = format!("{:.4}", test_split.calibrated.brier),
        test_logloss_cal = format!("{:.4}", test_split.calibrated.log_loss),
        test_auc_roc_cal = format!("{:.4}", test_split.calibrated.auc_roc),
        test_auc_pr_cal = format!("{:.4}", test_split.calibrated.auc_pr),
        test_ece_cal = format!("{:.4}", test_split.calibrated.expected_calibration_error),
        test_youden_t = format!("{:.3}", test_split.at_youden.threshold),
        test_cm_youden = ?test_split.at_youden.confusion_matrix,
        platt_a = scaler.a,
        platt_b = scaler.b,
        "knn probabilistic results"
    );

    fs::create_dir_all(results_dir)?;
    let source_name = source.dir().to_string();
    let metric_name = metric.as_str().to_string();
    let test_predictions = build_predictions(test_parsed, &y_test, &p1_test_raw, &p1_test_cal);
    let val_predictions = build_predictions(val_parsed, &y_val, &p1_val_raw, &p1_val_cal);

    let report = ClassificationReport {
        analysis: analysis.to_string(),
        source: source_name.clone(),
        split_seed: SEED,
        classifier: "knn".to_string(),
        num_neighbors,
        metric: metric_name.clone(),
        distance_weighted: true,
        n_train: train_idx.len(),
        platt_a: scaler.a,
        platt_b: scaler.b,
        test: test_split,
        val: val_split,
        test_predictions,
        val_predictions,
        split_manifest: SplitManifest {
            train: train_subjects,
            test: test_subjects,
            val: val_subjects,
        },
    };

    let mut run_counter = 0;
    let (json_path, csv_path) = loop {
        let base = BidsFilename::new()
            .with_pair("analysis", analysis)
            .with_pair("source", source_name.as_str())
            .with_pair("classifier", "knn")
            .with_pair("k", num_neighbors.to_string())
            .with_pair("metric", metric_name.as_str())
            .with_pair("run", &format!("{:02}", run_counter));

        let json_filename = base
            .clone()
            .with_suffix("classification")
            .with_extension(".json")
            .to_filename();
        let csv_filename = base
            .with_suffix("subject_probs")
            .with_extension(".csv")
            .to_filename();

        let json_path = results_dir.join(json_filename);
        let csv_path = results_dir.join(csv_filename);
        if !json_path.exists() && !csv_path.exists() {
            break (json_path, csv_path);
        }
        run_counter += 1;
    };

    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&json_path, json)?;

    write_subject_probs_csv(
        &csv_path,
        report
            .test_predictions
            .iter()
            .chain(report.val_predictions.iter()),
    )?;

    info!(
        json = %json_path.display(),
        csv = %csv_path.display(),
        "wrote classification report"
    );

    Ok(())
}
