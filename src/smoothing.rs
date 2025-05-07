use crate::types::{Matrix, MatrixView};
use ndarray::{s, Axis};
use rayon::prelude::*;

pub fn moving_average(data: MatrixView<'_>, window: usize) -> Matrix {
    let (rows, cols) = data.dim();
    let mut result = Matrix::zeros((rows, cols));
    
    result.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
        for j in 0..cols {
            let start = j.saturating_sub(window);
            let end = (j + window + 1).min(cols);
            let row_slice = data.row(i);
            let window_data = row_slice.slice(s![start..end]);
            row[j] = window_data.mean().unwrap_or(0.0);
        }
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_moving_average() {
        let data = Array2::from_shape_vec((3, 5), vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            2.0, 4.0, 6.0, 8.0, 10.0,
            3.0, 6.0, 9.0, 12.0, 15.0,
        ]).unwrap();
        
        let smoothed = moving_average(data.view(), 1);
        assert_eq!(smoothed.shape(), &[3, 5]);
        
        // Test middle values which should be average of 3 numbers
        assert!((smoothed[[0, 2]] - 3.0).abs() < 1e-6);
        assert!((smoothed[[1, 2]] - 6.0).abs() < 1e-6);
        assert!((smoothed[[2, 2]] - 9.0).abs() < 1e-6);
    }
}