//! Normalized model-facing tool metadata.

use serde_json::Map;
use serde_json::Value;

/// A function tool advertised to a model.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Map<String, Value>,
}
