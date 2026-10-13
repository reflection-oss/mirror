//! Normalized completed model output.

use mirror_protocol::AssistantMessage;
use mirror_protocol::Usage;

/// The completed output of one model request.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Completion {
    #[serde(with = "crate::providers::protobuf_serde::assistant_message")]
    pub message: AssistantMessage,
    #[serde(with = "crate::providers::protobuf_serde::usage")]
    pub usage: Option<Usage>,
    pub truncated: bool,
}
