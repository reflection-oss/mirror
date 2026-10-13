use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use rmcp::model::CallToolResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{valid_mcp_name, valid_mcp_server_name, McpRouter};
use crate::tools::{AnyTool, ToolDefinition, ToolError, ToolOutput};
use crate::ExecutionContext;

/// One discovered MCP tool bound to a runtime-owned router.
pub struct McpTool {
    binding: McpToolDefinition,
    router: Arc<dyn McpRouter>,
    validator: Option<jsonschema::Validator>,
}

/// A serializable MCP binding; connection names are resolved by the owning runtime.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct McpToolDefinition {
    pub definition: ToolDefinition,
    pub connection: String,
    pub tool: String,
    pub needs_approval: bool,
}

impl McpTool {
    /// Bind a server's live schemas; only explicitly trusted hints bypass approval.
    pub async fn discover(
        router: Arc<dyn McpRouter>,
        server: &str,
        trust_read_only: bool,
    ) -> Result<Vec<Arc<McpTool>>, ToolError> {
        let mut names = HashSet::new();
        let mut tools = Vec::new();
        for tool in router.list_tools(server).await? {
            let Some(name) = qualified_mcp_name(server, &tool.name) else {
                continue;
            };
            if !names.insert(name.clone()) {
                continue;
            }
            let binding = McpToolDefinition {
                definition: ToolDefinition {
                    name,
                    description: tool.description.as_deref().unwrap_or_default().to_owned(),
                    parameters: tool.input_schema.as_ref().clone(),
                },
                connection: server.to_owned(),
                tool: tool.name.into_owned(),
                needs_approval: !(trust_read_only
                    && tool
                        .annotations
                        .as_ref()
                        .and_then(|annotations| annotations.read_only_hint)
                        .unwrap_or(false)),
            };
            tools.push(Arc::new(Self::new(binding, router.clone())));
        }
        Ok(tools)
    }

    pub(super) fn new(binding: McpToolDefinition, router: Arc<dyn McpRouter>) -> Self {
        // Unsupported schemas remain the server's responsibility; fetching refs is disabled.
        let validator =
            jsonschema::validator_for(&Value::Object(binding.definition.parameters.clone())).ok();
        Self {
            binding,
            router,
            validator,
        }
    }

    pub(super) fn binding(&self) -> &McpToolDefinition {
        &self.binding
    }

    pub(super) fn validate(&self, arguments: &Value) -> Result<(), String> {
        let object = arguments
            .as_object()
            .ok_or("Tool arguments must be an object")?;
        if let Some(required) = self
            .binding
            .definition
            .parameters
            .get("required")
            .and_then(Value::as_array)
        {
            for name in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(name) {
                    return Err(format!("Missing required argument: {name}"));
                }
            }
        }
        if let Some(validator) = &self.validator {
            validator
                .validate(arguments)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

impl AnyTool for McpTool {
    fn name(&self) -> &str {
        &self.binding.definition.name
    }
    fn needs_approval(&self) -> bool {
        self.binding.needs_approval
    }
    fn definition(&self) -> ToolDefinition {
        self.binding.definition.clone()
    }

    fn execute<'a>(
        &'a self,
        arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let value: Value = match serde_json::from_str(arguments) {
                Ok(value) => value,
                Err(error) => return ToolOutput::error(format!("Invalid arguments: {error}")),
            };
            let Value::Object(arguments) = value else {
                return ToolOutput::error("Tool arguments must be an object");
            };
            match self
                .router
                .call_tool(&self.binding.connection, &self.binding.tool, arguments)
                .await
            {
                Ok(result) => output(result),
                Err(error) => ToolOutput::error(error.to_string()),
            }
        })
    }
}

/// Form an unambiguous, model-compatible name from a server and an upstream tool.
fn qualified_mcp_name(server: &str, tool: &str) -> Option<String> {
    let name = format!("mcp__{server}__{tool}");
    (valid_mcp_server_name(server) && valid_mcp_name(tool) && name.len() <= 64).then_some(name)
}

fn output(result: CallToolResult) -> ToolOutput {
    let mut text = result
        .content
        .iter()
        .map(|content| match content.as_text() {
            Some(text) => text.text.clone(),
            None => serde_json::to_string(content).unwrap_or_default(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        text = result
            .structured_content
            .map(|content| content.to_string())
            .unwrap_or_default();
    }
    if result.is_error == Some(true) {
        ToolOutput::failure(text)
    } else {
        ToolOutput::success(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use rmcp::model::ContentBlock;
    use serde_json::json;

    #[test]
    fn results_preserve_text_structured_content_and_failure_status() {
        assert_eq!(
            output(CallToolResult::success(vec![
                ContentBlock::text("hello"),
                ContentBlock::text("world")
            ])),
            ToolOutput::success("hello\nworld")
        );
        assert_eq!(
            output(CallToolResult::error(vec![ContentBlock::text("rejected")])),
            ToolOutput::failure("rejected")
        );
        let structured =
            serde_json::from_value(json!({"content": [], "structuredContent": {"answer": 42}}))
                .expect("structured result");
        assert_eq!(output(structured), ToolOutput::success(r#"{"answer":42}"#));
        let image = json!({"type": "image", "data": "synthetic", "mimeType": "image/png"});
        let result = serde_json::from_value(json!({"content": [image]})).expect("image result");
        let result = output(result);
        assert!(result.success);
        assert_eq!(
            serde_json::from_str::<Value>(&result.content).expect("image JSON"),
            image
        );
    }
}
