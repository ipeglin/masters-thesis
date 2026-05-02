/// fMRI slow-band (Buzsáki) frequency ranges in Hz.
/// Intervals are [low, high) — a mode/frequency `f` falls in a band iff low <= f < high.
///
/// Used as the project-wide reference for the analysed BOLD frequency range.
/// CWT scale grids, MVMD initialisation/grid bounds, and HHT spectrum binning
/// all derive `f_min` / `f_max` from this table so every spectral representation
/// shares one consistent frequency window.
pub const SLOW_BANDS: &[(&str, f64, f64)] = &[
    ("slow_5_trunc", 0.005, 0.010),
    ("slow_5", 0.010, 0.027),
    ("slow_4", 0.027, 0.073),
    ("slow_3", 0.073, 0.198),
    ("slow_2_trunc", 0.198, 0.250),
    // ("slow_2", 0.198, 0.500),
];

/// Lowest frequency covered by `SLOW_BANDS` (inclusive lower bound of the lowest band).
pub fn f_min() -> f64 {
    SLOW_BANDS
        .iter()
        .map(|(_, lo, _)| *lo)
        .fold(f64::INFINITY, f64::min)
}

/// Highest frequency covered by `SLOW_BANDS` (exclusive upper bound of the highest band).
pub fn f_max() -> f64 {
    SLOW_BANDS
        .iter()
        .map(|(_, _, hi)| *hi)
        .fold(f64::NEG_INFINITY, f64::max)
}
