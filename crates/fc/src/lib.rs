use ndarray::Array2;
use polars::prelude::*;

#[derive(Clone)]
pub struct Pearson;
#[derive(Clone)]
pub struct FisherZ;

#[derive(Clone)]
pub struct ConnectivityMatrix<State> {
    pub data: DataFrame,
    pub labels: Vec<String>,
    _state: std::marker::PhantomData<State>,
}

impl<State> ConnectivityMatrix<State> {
    pub fn get_value(&self, row_roi: &str, col_roi: &str) -> Option<f64> {
        let row_idx = self.labels.iter().position(|r| r == row_roi)?;
        let series = self.data.column(col_roi).ok()?;
        series.get(row_idx).ok()?.try_extract::<f64>().ok()
    }

    pub fn get_values(&self) -> DataFrame {
        self.data.clone()
    }

    /// Converts the connectivity matrix into an ndarray Array2<f64>.
    /// Iterates column-by-column using the stored labels.
    pub fn to_ndarray(&self) -> PolarsResult<Array2<f64>> {
        let n = self.labels.len();
        let mut matrix = Array2::<f64>::zeros((n, n));
        for (i, label) in self.labels.iter().enumerate() {
            let col = self.data.column(label.as_str())?;
            let values = col.f64()?;
            for (j, val) in values.iter().enumerate() {
                matrix[[i, j]] = val.unwrap_or(f64::NAN);
            }
        }
        Ok(matrix)
    }
}

impl ConnectivityMatrix<Pearson> {
    pub fn new(df: DataFrame) -> PolarsResult<Self> {
        let labels: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let corr_df = compute_correlation_matrix(&df)?;

        Ok(Self {
            data: corr_df,
            labels,
            _state: std::marker::PhantomData,
        })
    }

    /// Consumes the Pearson matrix and returns a Fisher-transformed Z-matrix
    pub fn into_fisher_z(self) -> PolarsResult<ConnectivityMatrix<FisherZ>> {
        // Fisher Z-transform: arctanh(r)
        // Clip to avoid infinity at r = 1.0 or -1.0
        let transformed = self
            .data
            .lazy()
            .select([col("*").clip(lit(-0.9999), lit(0.9999)).arctanh()])
            .collect()?;

        Ok(ConnectivityMatrix {
            data: transformed,
            labels: self.labels,
            _state: std::marker::PhantomData,
        })
    }
}

impl ConnectivityMatrix<FisherZ> {
    /// Consumes the Z-matrix and returns it to Pearson correlation space
    pub fn into_pearson(self) -> PolarsResult<ConnectivityMatrix<Pearson>> {
        let transformed = self
            .data
            .lazy()
            .select([col("*").tanh()]) // tanh is the inverse Fisher transform
            .collect()?;

        Ok(ConnectivityMatrix {
            data: transformed,
            labels: self.labels,
            _state: std::marker::PhantomData,
        })
    }

    /// Zeroes out connections where the absolute Z-score is below the threshold.
    /// This is common for creating adjacency matrices for graph theory.
    pub fn threshold(self, min_z: f64) -> PolarsResult<Self> {
        let thresholded = self
            .data
            .lazy()
            .select([col("*").map(
                move |column| {
                    let ca = column.f64()?;
                    let out: Float64Chunked =
                        ca.apply(|opt_v| opt_v.map(|v| if v.abs() < min_z { 0.0 } else { v }));
                    Ok(out.into_column())
                },
                |_schema, field| Ok(field.clone()),
            )])
            .collect()?;

        Ok(Self {
            data: thresholded,
            labels: self.labels,
            _state: std::marker::PhantomData,
        })
    }
}

/// Computes a Pearson correlation matrix for all columns in the DataFrame.
/// Each column is treated as a variable, and rows are observations.
fn compute_correlation_matrix(df: &DataFrame) -> PolarsResult<DataFrame> {
    let col_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let n_cols = col_names.len();

    // Build correlation matrix column by column
    let mut columns: Vec<Column> = Vec::with_capacity(n_cols);

    for col_name in &col_names {
        let col_series = df.column(col_name)?;
        let mut corr_values: Vec<f64> = Vec::with_capacity(n_cols);

        for row_name in &col_names {
            let row_series = df.column(row_name)?;
            let corr = pearson_correlation(col_series, row_series)?;
            corr_values.push(corr);
        }

        let series = Series::new(col_name.as_str().into(), corr_values);
        columns.push(series.into());
    }

    DataFrame::new(columns)
}

/// Computes Pearson correlation between two series
fn pearson_correlation(a: &Column, b: &Column) -> PolarsResult<f64> {
    let a_f64 = a.cast(&DataType::Float64)?;
    let b_f64 = b.cast(&DataType::Float64)?;

    let a_ca = a_f64.f64()?;
    let b_ca = b_f64.f64()?;

    let n = a_ca.len() as f64;
    if n == 0.0 {
        return Ok(f64::NAN);
    }

    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    let mut sum_ab = 0.0;
    let mut sum_a2 = 0.0;
    let mut sum_b2 = 0.0;
    let mut count = 0.0;

    for (opt_a, opt_b) in a_ca.iter().zip(b_ca.iter()) {
        if let (Some(va), Some(vb)) = (opt_a, opt_b) {
            sum_a += va;
            sum_b += vb;
            sum_ab += va * vb;
            sum_a2 += va * va;
            sum_b2 += vb * vb;
            count += 1.0;
        }
    }

    if count == 0.0 {
        return Ok(f64::NAN);
    }

    let mean_a = sum_a / count;
    let mean_b = sum_b / count;

    let numerator = sum_ab - count * mean_a * mean_b;
    let denom_a = (sum_a2 - count * mean_a * mean_a).sqrt();
    let denom_b = (sum_b2 - count * mean_b * mean_b).sqrt();

    if denom_a == 0.0 || denom_b == 0.0 {
        return Ok(f64::NAN);
    }

    Ok(numerator / (denom_a * denom_b))
}
