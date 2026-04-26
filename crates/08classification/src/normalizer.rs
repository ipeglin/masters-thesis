#[derive(Debug, Clone)]
pub struct ZScoreNormalizer {
    pub means: Vec<f64>,
    pub std_devs: Vec<f64>,
}

impl ZScoreNormalizer {
    pub fn fit(data: &[Vec<f64>]) -> Self {
        if data.is_empty() {
            return Self {
                means: Vec::new(),
                std_devs: Vec::new(),
            };
        }

        let n_features = data[0].len();
        let n_samples = data.len() as f64;
        let mut means = vec![0.0; n_features];
        
        for row in data {
            for (i, &val) in row.iter().enumerate() {
                means[i] += val;
            }
        }
        for m in &mut means {
            *m /= n_samples;
        }
        
        let mut variances = vec![0.0; n_features];
        for row in data {
            for (i, &val) in row.iter().enumerate() {
                let diff = val - means[i];
                variances[i] += diff * diff;
            }
        }
        let mut std_devs = vec![0.0; n_features];
        for (i, &var) in variances.iter().enumerate() {
            let std = (var / n_samples).sqrt();
            std_devs[i] = if std > 0.0 { std } else { 1.0 }; // Avoid div by zero
        }
        
        Self { means, std_devs }
    }

    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(i, &val)| (val - self.means[i]) / self.std_devs[i])
                    .collect()
            })
            .collect()
    }
}
