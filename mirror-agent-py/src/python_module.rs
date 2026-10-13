//! Native `mirror_agent` Python module.

use pyo3::prelude::*;

/// Python bindings for the in-process Mirror Rust agent.
#[pymodule]
mod mirror_agent {
    #[pymodule_export]
    use crate::PyAgentConfig;
    #[pymodule_export]
    use crate::PyAgentService;
    #[pymodule_export]
    use crate::PyScriptedModel;
    #[pymodule_export]
    use crate::PySession;
    #[pymodule_export]
    use crate::PyTool;
}
