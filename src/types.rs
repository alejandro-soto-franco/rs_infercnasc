use ndarray::{Array2, ArrayView2, Data};

pub type Matrix = Array2<f32>;
pub type MatrixView<'a> = ArrayView2<'a, f32>;
pub type BoolMatrix = Array2<bool>;

pub fn ensure_finite<D>(matrix: &ndarray::ArrayBase<D, ndarray::Dim<[usize; 2]>>) -> bool 
where
    D: Data<Elem = f32>,
{
    matrix.iter().all(|x| x.is_finite())
}

pub fn safe_std(values: &[f32], mean: f32) -> f32 {
    let variance = values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    variance.sqrt()
}