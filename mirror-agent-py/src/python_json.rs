//! JSON conversion at the Python tool boundary.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Map;
use serde_json::Value;

/// Conversion through Python's standard `json` module.
pub(crate) struct PythonJson;

impl PythonJson {
    pub(crate) fn to_dict(py: Python<'_>, value: &Map<String, Value>) -> PyResult<Py<PyDict>> {
        Ok(py
            .import("json")?
            .call_method1("loads", (Value::Object(value.clone()).to_string(),))?
            .cast_into::<PyDict>()?
            .unbind())
    }

    pub(crate) fn from_dict(
        py: Python<'_>,
        value: &Bound<'_, PyDict>,
    ) -> PyResult<Map<String, Value>> {
        let encoded: String = py
            .import("json")?
            .call_method1("dumps", (value,))?
            .extract()?;
        serde_json::from_str(&encoded)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }
}
