use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

mod smoothing;
mod types;
mod zscore;

#[pyfunction]
fn smooth_matrix(
    py: Python<'_>,
    matrix: &PyArray2<f32>,
    window: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let readonly = matrix.readonly();
    let input = readonly.as_array();

    if !types::ensure_finite(&input) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input matrix contains non-finite values",
        ));
    }

    let output = smoothing::moving_average(input, window);
    Ok(output.into_pyarray(py).to_owned())
}

#[pyfunction]
fn zscore_threshold(
    py: Python<'_>,
    matrix: &PyArray2<f32>,
    threshold: f32,
) -> PyResult<Py<PyArray2<bool>>> {
    let readonly = matrix.readonly();
    let input = readonly.as_array();

    if !types::ensure_finite(&input) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input matrix contains non-finite values",
        ));
    }

    let result = zscore::binary_threshold(input, threshold);
    Ok(result.into_pyarray(py).to_owned())
}

#[pymodule]
fn infercnasc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smooth_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_threshold, m)?)?;
    Ok(())
}
