//! Platt scaling: a 1-D logistic regression that maps an uncalibrated score
//! `s` to a calibrated probability `σ(a · s + b)`.
//!
//! Platt's original 1999 trick is to fit on smoothed targets
//! `t+ = (N+ + 1) / (N+ + 2)`, `t- = 1 / (N- + 2)` rather than 0/1 — this
//! prevents the logits from blowing up when the input scores are perfectly
//! separable on the training set, which is common for high-capacity models
//! that overfit a small calibration set.
//!
//! Optimisation: Newton-Raphson on the negative log-likelihood, with a small
//! step-size cap and damping to keep the Hessian positive-definite. Converges
//! in <20 iterations on well-conditioned data.

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy)]
pub struct PlattScaler {
    pub a: f32,
    pub b: f32,
}

impl PlattScaler {
    /// Identity calibration: `σ(s)` (`a = 1, b = 0`). Useful as a fallback if
    /// the calibration fit fails.
    pub fn identity() -> Self {
        Self { a: 1.0, b: 0.0 }
    }

    /// Fit `(a, b)` so that `σ(a · score + b) ≈ P(y = 1 | score)`.
    ///
    /// `scores` should be the raw probability `p̂_1` from the underlying model
    /// (or a logit / decision function — the output is just `σ(a·s + b)`).
    /// Returns `Self::identity()` when the calibration set cannot support a
    /// meaningful logistic fit:
    ///   * length mismatch is an error,
    ///   * fewer than 4 samples, or
    ///   * a single class present, or
    ///   * non-finite scores anywhere, or
    ///   * score variance below `1e-12` (all scores effectively equal — Newton
    ///     has no signal to fit `a` against, and would previously bail with
    ///     `a = 0` after the first damped step failed).
    pub fn fit(scores: &[f32], y: &[i32]) -> Result<Self> {
        if scores.len() != y.len() {
            bail!(
                "platt fit: length mismatch ({} vs {})",
                scores.len(),
                y.len()
            );
        }
        let n = scores.len();
        if n < 4 {
            return Ok(Self::identity());
        }
        let n_pos = y.iter().filter(|&&v| v == 1).count();
        let n_neg = n - n_pos;
        if n_pos == 0 || n_neg == 0 {
            return Ok(Self::identity());
        }
        if scores.iter().any(|s| !s.is_finite()) {
            return Ok(Self::identity());
        }
        let mean: f64 = scores.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
        let var: f64 = scores
            .iter()
            .map(|&s| (s as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        if var < 1e-12 {
            return Ok(Self::identity());
        }

        // Platt's smoothed targets.
        let hi = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
        let lo = 1.0 / (n_neg as f64 + 2.0);
        let t: Vec<f64> = y.iter().map(|&v| if v == 1 { hi } else { lo }).collect();
        let s: Vec<f64> = scores.iter().map(|&v| v as f64).collect();

        let mut a = 0.0f64;
        let mut b = (n_neg as f64 + 1.0).ln() - (n_pos as f64 + 1.0).ln(); // bias prior
        let max_iter = 100;
        let lambda_init = 1e-3;
        let mut lambda = lambda_init;

        let nll = |a: f64, b: f64| -> f64 {
            let mut acc = 0.0;
            for i in 0..n {
                let f = a * s[i] + b;
                // log(1 + exp(f)) - t * f, numerically stable.
                let lse = if f >= 0.0 {
                    f + (1.0 + (-f).exp()).ln()
                } else {
                    (1.0 + f.exp()).ln()
                };
                acc += lse - t[i] * f;
            }
            acc
        };

        let mut prev_loss = nll(a, b);
        for _ in 0..max_iter {
            // Gradient and Hessian.
            let (mut g_a, mut g_b) = (0.0f64, 0.0f64);
            let (mut h_aa, mut h_bb, mut h_ab) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..n {
                let f = a * s[i] + b;
                let p = 1.0 / (1.0 + (-f).exp());
                let r = p * (1.0 - p);
                g_a += (p - t[i]) * s[i];
                g_b += p - t[i];
                h_aa += r * s[i] * s[i];
                h_bb += r;
                h_ab += r * s[i];
            }

            // Damped Newton step: solve (H + λI) Δ = -g.
            let mut accepted = false;
            for _ in 0..10 {
                let h_aa_d = h_aa + lambda;
                let h_bb_d = h_bb + lambda;
                let det = h_aa_d * h_bb_d - h_ab * h_ab;
                if det <= 0.0 || !det.is_finite() {
                    lambda *= 4.0;
                    continue;
                }
                let da = (-g_a * h_bb_d + g_b * h_ab) / det;
                let db = (g_a * h_ab - g_b * h_aa_d) / det;
                let new_a = a + da;
                let new_b = b + db;
                let new_loss = nll(new_a, new_b);
                if new_loss.is_finite() && new_loss < prev_loss {
                    a = new_a;
                    b = new_b;
                    prev_loss = new_loss;
                    lambda = (lambda * 0.5).max(1e-9);
                    accepted = true;
                    break;
                }
                lambda *= 4.0;
            }
            if !accepted {
                break;
            }
            if g_a.abs() < 1e-7 && g_b.abs() < 1e-7 {
                break;
            }
        }

        Ok(Self {
            a: a as f32,
            b: b as f32,
        })
    }

    pub fn transform(&self, score: f32) -> f32 {
        let f = self.a * score + self.b;
        sigmoid(f)
    }

    pub fn transform_slice(&self, scores: &[f32]) -> Vec<f32> {
        scores.iter().map(|&s| self.transform(s)).collect()
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_identity_like_when_scores_already_calibrated() {
        // Synthetic well-separated scores.
        let mut scores = Vec::new();
        let mut y = Vec::new();
        for i in 0..100 {
            let x = (i as f32) / 99.0;
            scores.push(x);
            y.push(if x > 0.5 { 1 } else { 0 });
        }
        let p = PlattScaler::fit(&scores, &y).unwrap();
        // Probability at boundary score should be near 0.5.
        let mid = p.transform(0.5);
        assert!((mid - 0.5).abs() < 0.15, "mid = {}", mid);
        // Far positive should be near 1.
        assert!(p.transform(0.99) > 0.7);
        // Far negative should be near 0.
        assert!(p.transform(0.01) < 0.3);
    }

    #[test]
    fn handles_single_class_safely() {
        let scores = vec![0.1, 0.2, 0.3];
        let y = vec![1, 1, 1];
        let p = PlattScaler::fit(&scores, &y).unwrap();
        // Identity fallback.
        assert!((p.a - 1.0).abs() < 1e-6);
        assert!((p.b - 0.0).abs() < 1e-6);
    }

    #[test]
    fn rescales_overconfident_scores() {
        // Model emits 0/1 deterministically but is wrong half the time;
        // calibration should pull probabilities toward the base rate.
        let scores = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let y = vec![0, 1, 0, 1, 1, 0, 1, 0];
        let p = PlattScaler::fit(&scores, &y).unwrap();
        let p0 = p.transform(0.0);
        let p1 = p.transform(1.0);
        assert!(p0 > 0.2 && p0 < 0.8);
        assert!(p1 > 0.2 && p1 < 0.8);
    }
}
