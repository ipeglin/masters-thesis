mod knn;
mod svm;

pub use knn::{
    DistanceMetric, KNN, KnnConfig, accuracy, confusion_matrix_binary, sensitivity_from_cm,
    specificity_from_cm,
};
