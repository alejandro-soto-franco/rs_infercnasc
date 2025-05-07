use crate::types::{MatrixView, BoolMatrix};
use ndarray::Axis;

pub fn binary_threshold(data: MatrixView<'_>, threshold: f32) -> BoolMatrix {
    let means = data.mean_axis(Axis(0)).unwrap();
    let stds = data.std_axis(Axis(0), 0.0);
    let mut result = BoolMatrix::default(data.raw_dim());

    for ((i, j), &val) in data.indexed_iter() {
        let z = (val - means[j]) / stds[j].max(1e-6);
        result[(i, j)] = z.abs() > threshold;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_zscore_threshold() {
        let data = Array2::from_shape_vec((4, 3), vec![
            2.0, 0.0, -2.0,
            2.0, 0.0, -2.0,
            2.0, 0.0, -2.0,
            8.0, 4.0, -8.0,
        ]).unwrap();

        let result = binary_threshold(data.view(), 1.5);
        
        // The last row should be marked as significant in all columns
        assert!(result[[3, 0]]);
        assert!(result[[3, 1]]);
        assert!(result[[3, 2]]);
        
        // Other rows should not be marked as significant
        assert!(!result[[0, 0]]);
        assert!(!result[[1, 1]]);
        assert!(!result[[2, 2]]);
    }
}