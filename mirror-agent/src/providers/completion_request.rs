//! Normalized provider-neutral model input.

use mirror_protocol::chat_record;

use crate::tools::ToolDefinition;

/// Input for one model request.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CompletionRequest {
    #[serde(with = "crate::providers::protobuf_serde::chat_messages")]
    pub messages: Vec<chat_record::Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: Option<u32>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::providers::protobuf_serde::inference_config"
    )]
    pub inference: Option<mirror_protocol::InferenceConfig>,
}
