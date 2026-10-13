//! Python-defined tools executed by the Rust agent runtime.

use std::future::Future;
use std::pin::Pin;

use mirror_agent_core::tools::ToolDefinition;
use mirror_agent_core::tools::ToolOutput;
use mirror_agent_core::tools::{native_tool, AnyTool};
use mirror_agent_core::ExecutionContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::TaskLocals;
use serde_json::Map;
use serde_json::Value;

use crate::python_json::PythonJson;

/// A model-facing tool whose implementation is an async Python function.
#[pyclass(name = "Tool", frozen)]
pub struct PyTool {
    name: String,
    description: String,
    parameters: Map<String, Value>,
    needs_approval: bool,
    run: Py<PyAny>,
}

pub(crate) struct PythonTool {
    definition: ToolDefinition,
    needs_approval: bool,
    run: Py<PyAny>,
    locals: TaskLocals,
}

#[pymethods]
impl PyTool {
    /// Define a caller-owned async tool.
    #[new]
    #[pyo3(signature = (
        name,
        description,
        parameters: "dict[str, object]",
        run: "collections.abc.Callable[[dict[str, object]], collections.abc.Awaitable[str]]",
    ) -> "Tool")]
    fn new(
        py: Python<'_>,
        name: String,
        description: String,
        parameters: &Bound<'_, PyDict>,
        run: Py<PyAny>,
    ) -> PyResult<Self> {
        if !run.bind(py).is_callable() {
            return Err(PyTypeError::new_err("run must be callable"));
        }
        Ok(Self {
            name,
            description,
            parameters: PythonJson::from_dict(py, parameters)?,
            needs_approval: false,
            run,
        })
    }

    /// Use a native tool's contract with a caller-owned async implementation.
    #[staticmethod]
    #[pyo3(signature = (
        name,
        run: "collections.abc.Callable[[dict[str, object]], collections.abc.Awaitable[str]]",
    ) -> "Tool")]
    fn native(py: Python<'_>, name: &str, run: Py<PyAny>) -> PyResult<Self> {
        if !run.bind(py).is_callable() {
            return Err(PyTypeError::new_err("run must be callable"));
        }
        let tool = native_tool(name, std::path::Path::new("."))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let definition = tool.definition();
        Ok(Self {
            name: definition.name,
            description: definition.description,
            parameters: definition.parameters,
            needs_approval: tool.needs_approval(),
            run,
        })
    }
}

impl PyTool {
    pub(crate) fn runtime_tool(&self, py: Python<'_>, locals: TaskLocals) -> PythonTool {
        PythonTool {
            definition: ToolDefinition {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.parameters.clone(),
            },
            needs_approval: self.needs_approval,
            run: self.run.clone_ref(py),
            locals,
        }
    }
}

impl AnyTool for PythonTool {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn needs_approval(&self) -> bool {
        self.needs_approval
    }

    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn execute<'a>(
        &'a self,
        arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            match self.execute_callback(arguments).await {
                Ok(content) => ToolOutput::success(content),
                Err(error) => ToolOutput::error(error),
            }
        })
    }
}

impl PythonTool {
    async fn execute_callback(&self, arguments: &str) -> Result<String, String> {
        let arguments: Map<String, Value> = serde_json::from_str(if arguments.is_empty() {
            "{}"
        } else {
            arguments
        })
        .map_err(|error| format!("invalid tool arguments: {error}"))?;
        let future = Python::attach(|py| {
            let arguments = PythonJson::to_dict(py, &arguments)?;
            let awaitable = self.run.bind(py).call1((arguments,))?;
            pyo3_async_runtimes::into_future_with_locals(&self.locals, awaitable)
        })
        .map_err(|error| error.to_string())?;
        let output = future.await.map_err(|error| error.to_string())?;
        Python::attach(|py| output.extract::<String>(py)).map_err(|error| error.to_string())
    }
}
