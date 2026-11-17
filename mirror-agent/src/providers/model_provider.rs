//! Runtime-selected model provider composition.

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use crate::providers::CompletionRequest;
use async_stream::try_stream;
use futures_core::Stream;
use futures_util::pin_mut;
use futures_util::StreamExt;
use mirror_protocol::InferenceConfig;
use reqwest::Client;
use thiserror::Error;

use super::omit_images;
use super::AnthropicMessages;
use super::AnthropicMessagesConfig;
use super::ChatCompletions;
use super::ChatCompletionsConfig;
use super::CodexResponses;
use super::CodexResponsesConfig;
use super::Llm;
use super::LlmError;
use super::ModelRequestGuard;
use super::Reflection;
use super::ResponsesClient;
use super::ResponsesConfig;
use super::ResponsesWebSocket;
use super::StreamEvent;

/// Immutable provider defaults shared by inference and metadata requests.
#[derive(Clone)]
pub struct ModelProvider {
    default_config: ModelProviderConfig,
    client: Client,
    request_guard: Option<Arc<dyn ModelRequestGuard>>,
    responses_websocket: ResponsesWebSocket,
}

/// Configuration shared by root agents and subagents.
#[derive(Clone)]
pub struct ModelProviderConfig {
    pub api: ModelApi,
    pub model: String,
    pub context_window: Option<usize>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub auth_path: Option<PathBuf>,
    pub prompt_cache_key: String,
    /// The longest one model request may take. `None` leaves it unbounded, so
    /// a long generation is never cut off and resampled; use it only where a
    /// turn deadline bounds the request instead.
    pub request_timeout: Option<Duration>,
}

impl std::fmt::Debug for ModelProviderConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelProviderConfig")
            .field("api", &self.api)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// Model providers and wire protocols supported by the Rust harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelApi {
    CodexResponses,
    Responses,
    ChatCompletions,
    Reflection,
    /// Gemini Developer API, using Google's OpenAI-compatible endpoint.
    Gemini,
    /// Vercel AI Gateway, using its OpenAI-compatible Chat Completions endpoint.
    Vercel,
    AnthropicMessages,
    ChatCompletionsNonStreaming,
}

/// An unsupported model protocol name.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("unsupported model API {0:?}")]
pub struct ModelApiParseError(String);

pub(super) const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/openai";
pub(super) const VERCEL_BASE_URL: &str = "https://ai-gateway.vercel.sh/v1";

impl ModelProvider {
    /// Construct reusable defaults without requiring credentials or a live endpoint.
    pub async fn new(
        config: ModelProviderConfig,
        request_guard: Option<Arc<dyn ModelRequestGuard>>,
    ) -> Result<Self, LlmError> {
        if config
            .context_window
            .is_some_and(|window| window == 0 || window > usize::MAX / 100)
        {
            return Err(LlmError::Request(
                "context-window must be a positive token limit".into(),
            ));
        }
        let client = super::http_client(config.request_timeout)?;
        Ok(Self {
            responses_websocket: ResponsesWebSocket::spawn(config.request_timeout)?,
            client,
            default_config: config,
            request_guard,
        })
    }
}

impl ModelProviderConfig {
    /// Resolve the same endpoint and credentials for inference and metadata.
    pub(super) async fn connection(&self) -> Result<(String, String), LlmError> {
        let (base_env, base_default, key_env) = match self.api {
            ModelApi::Gemini => {
                if self.model != "gemini-4-argon" {
                    return Err(LlmError::Request(
                        "gemini supports gemini-4-argon only; select /model gemini".into(),
                    ));
                }
                if self
                    .base_url
                    .as_deref()
                    .is_some_and(|url| url.trim_end_matches('/') != GEMINI_BASE_URL)
                {
                    return Err(LlmError::Request(
                    "gemini uses the Gemini Developer API, not Vertex AI or custom endpoints; use openai-chat-completions for a custom endpoint".into(),
                ));
                }
                let key = self.api_key.clone()
                .or_else(|| std::env::var("GEMINI_API_KEY").ok())
                .filter(|key| !key.trim().is_empty())
                .ok_or_else(|| LlmError::Request(
                    "set GEMINI_API_KEY from Google AI Studio and restart Mirror; Gemini 4 Argon requires paid API access".into(),
                ))?;
                return Ok((GEMINI_BASE_URL.into(), key));
            }
            ModelApi::Vercel => {
                if self
                    .base_url
                    .as_deref()
                    .is_some_and(|url| url.trim_end_matches('/') != VERCEL_BASE_URL)
                {
                    return Err(LlmError::Request(
                    "vercel uses the Vercel AI Gateway endpoint, not custom endpoints; use openai-chat-completions for a custom endpoint".into(),
                ));
                }
                let key = self.api_key.clone()
                .or_else(|| std::env::var("AI_GATEWAY_API_KEY").ok())
                .filter(|key| !key.trim().is_empty())
                .ok_or_else(|| LlmError::Request(
                    "set AI_GATEWAY_API_KEY to a Vercel AI Gateway API key and restart Mirror".into(),
                ))?;
                return Ok((VERCEL_BASE_URL.into(), key));
            }
            ModelApi::CodexResponses => (
                "MIRROR_CODEX_BASE_URL",
                super::codex_responses::DEFAULT_BASE_URL,
                None,
            ),
            ModelApi::Responses => (
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
                Some("OPENAI_API_KEY"),
            ),
            ModelApi::ChatCompletions | ModelApi::ChatCompletionsNonStreaming => (
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
                Some("OPENROUTER_API_KEY"),
            ),
            ModelApi::AnthropicMessages => (
                "ANTHROPIC_BASE_URL",
                "https://api.anthropic.com",
                Some("ANTHROPIC_API_KEY"),
            ),
            ModelApi::Reflection => {
                return Ok((
                    super::REFLECTION_BASE_URL.into(),
                    super::reflection::reflection_key(self).await?,
                ))
            }
        };
        let base_url = self
            .base_url
            .clone()
            .or_else(|| std::env::var(base_env).ok())
            .unwrap_or_else(|| base_default.into());
        let mut key = self
            .api_key
            .clone()
            .or_else(|| key_env.and_then(|name| std::env::var(name).ok()))
            .filter(|key| self.api != ModelApi::AnthropicMessages || !key.trim().is_empty());
        if key.is_none()
            && self.api == ModelApi::AnthropicMessages
            && base_url.trim_end_matches('/') == "https://api.anthropic.com"
        {
            if let Some(home) = std::env::var_os("HOME")
                .map(PathBuf::from)
                .filter(|home| home.is_absolute())
            {
                match tokio::fs::read_to_string(home.join(".mirror/anthropic_api_key")).await {
                    Ok(saved) if !saved.trim().is_empty() => key = Some(saved.trim().to_owned()),
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(_) => {
                        return Err(LlmError::Request(
                            "could not read saved Anthropic API key".into(),
                        ))
                    }
                }
            }
        }
        Ok((base_url, key.unwrap_or_else(|| "EMPTY".into())))
    }

    /// Whether the selected model reads images.
    ///
    /// Reflection's models read only text, on the Reflection API and under
    /// their `reflection/` name on a gateway. Any other model is assumed to
    /// read images, and its provider rejects them when it cannot.
    pub fn accepts_images(&self) -> bool {
        self.api != ModelApi::Reflection && !self.model.starts_with("reflection/")
    }

    pub fn for_inference(&self, inference: &InferenceConfig) -> Result<Self, LlmError> {
        let mut config = self.clone();
        if !inference.model.is_empty() {
            config.model = inference.model.clone();
        }
        if let Some(api) = &inference.api {
            if !api.is_empty() {
                config.api = api
                    .parse()
                    .map_err(|error: ModelApiParseError| LlmError::Request(error.to_string()))?;
            }
        }
        if config.api != self.api {
            config.api_key = None;
            config.base_url = None;
        }
        if let Some(base_url) = &inference.base_url {
            config.base_url = (!base_url.is_empty()).then(|| base_url.clone());
        }
        if config.base_url != self.base_url {
            config.api_key = None;
        }
        if config.model != self.model || config.api != self.api || config.base_url != self.base_url
        {
            config.context_window = None;
        }
        if let Some(window) = inference.context_window {
            config.context_window = Some(
                usize::try_from(window)
                    .ok()
                    .filter(|window| *window > 0 && *window <= usize::MAX / 100)
                    .ok_or_else(|| {
                        LlmError::Request("context-window must be a positive token limit".into())
                    })?,
            );
        }
        Ok(config)
    }
}

impl ModelProvider {
    pub(super) async fn verify_request(&self) -> Result<(), LlmError> {
        if let Some(guard) = &self.request_guard {
            guard.verify().await?;
        }
        Ok(())
    }

    /// Resolve the runtime selection for inference and the agent's metadata cache.
    pub(crate) fn config_for(
        &self,
        inference: Option<&InferenceConfig>,
    ) -> Result<ModelProviderConfig, LlmError> {
        match inference {
            Some(inference) => self.default_config.for_inference(inference),
            None => Ok(self.default_config.clone()),
        }
    }
}

impl Llm for ModelProvider {
    fn model(&self) -> &str {
        &self.default_config.model
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> {
        let config = self.config_for(request.inference.as_ref());
        let client = self.client.clone();
        let request_guard = self.request_guard.clone();
        let responses_websocket = self.responses_websocket.clone();
        try_stream! {
            let config = config?;
            let mut request = request;
            if !config.accepts_images() {
                omit_images(&mut request.messages);
            }
            let (base_url, api_key) = config.connection().await?;
            let websocket = ResponsesWebSocket::serves(&base_url).then_some(responses_websocket);
            match config.api {
                ModelApi::CodexResponses => {
                    let provider = CodexResponses::new(
                        CodexResponsesConfig {
                            model: config.model,
                            base_url: Some(base_url),
                            auth_path: config.auth_path,
                            prompt_cache_key: config.prompt_cache_key,
                            websocket,
                        },
                        client,
                    )?;
                    let stream = provider.stream_guarded(request, request_guard.as_deref());
                    pin_mut!(stream);
                    while let Some(event) = stream.next().await {
                        yield event?;
                    }
                }
                ModelApi::Responses => {
                    let provider = ResponsesClient::new(
                        ResponsesConfig {
                            model: config.model,
                            api_key,
                            base_url,
                            prompt_cache_key: Some(config.prompt_cache_key),
                            websocket,
                        },
                        client,
                    );
                    let stream = provider.stream_guarded(request, request_guard.as_deref());
                    pin_mut!(stream);
                    while let Some(event) = stream.next().await {
                        yield event?;
                    }
                }
                ModelApi::Reflection => {
                    let provider = Reflection::new(config.model, api_key, client);
                    let stream = provider.stream_guarded(request, request_guard.as_deref());
                    pin_mut!(stream);
                    while let Some(event) = stream.next().await {
                        yield event?;
                    }
                }
                ModelApi::ChatCompletions
                | ModelApi::ChatCompletionsNonStreaming
                | ModelApi::Gemini
                | ModelApi::Vercel => {
                    if config.api == ModelApi::Gemini {
                        validate_gemini_request(&request)?;
                    }
                    let provider = ChatCompletions::new(
                        ChatCompletionsConfig {
                            model: config.model,
                            api_key,
                            base_url,
                        },
                        client,
                    );
                    let stream = provider.stream_guarded(
                        request,
                        request_guard.as_deref(),
                        config.api != ModelApi::ChatCompletionsNonStreaming,
                    );
                    pin_mut!(stream);
                    while let Some(event) = stream.next().await {
                        yield event?;
                    }
                }
                ModelApi::AnthropicMessages => {
                    let provider = AnthropicMessages::new(
                        AnthropicMessagesConfig {
                            model: config.model,
                            api_key,
                            base_url,
                        },
                        client,
                    );
                    let stream = provider.stream_guarded(request, request_guard.as_deref());
                    pin_mut!(stream);
                    while let Some(event) = stream.next().await {
                        yield event?;
                    }
                }
            }
        }
    }

    fn prewarm(&self, request: CompletionRequest) {
        let Ok(config) = self.config_for(request.inference.as_ref()) else {
            return;
        };
        if !matches!(config.api, ModelApi::Responses | ModelApi::CodexResponses) {
            return;
        }
        let client = self.client.clone();
        let request_guard = self.request_guard.clone();
        let websocket = self.responses_websocket.clone();
        tokio::spawn(async move {
            let Ok((base_url, api_key)) = config.connection().await else {
                return;
            };
            if !ResponsesWebSocket::serves(&base_url) {
                return;
            }
            let guard = request_guard.as_deref();
            let _ = match config.api {
                ModelApi::CodexResponses => {
                    let provider = CodexResponses::new(
                        CodexResponsesConfig {
                            model: config.model,
                            base_url: Some(base_url),
                            auth_path: config.auth_path,
                            prompt_cache_key: config.prompt_cache_key,
                            websocket: Some(websocket),
                        },
                        client,
                    );
                    match provider {
                        Ok(provider) => provider.prewarm(request, guard).await,
                        Err(error) => Err(error),
                    }
                }
                _ => {
                    ResponsesClient::new(
                        ResponsesConfig {
                            model: config.model,
                            api_key,
                            base_url,
                            prompt_cache_key: Some(config.prompt_cache_key),
                            websocket: Some(websocket),
                        },
                        client,
                    )
                    .prewarm(request, guard)
                    .await
                }
            };
        });
    }
}

impl ModelApi {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CodexResponses => "openai-codex-responses",
            Self::Responses => "openai-responses",
            Self::ChatCompletions => "openai-chat-completions",
            Self::Reflection => "reflection",
            Self::Gemini => "gemini",
            Self::Vercel => "vercel",
            Self::AnthropicMessages => "anthropic-messages",
            Self::ChatCompletionsNonStreaming => "openai-chat-completions-non-streaming",
        }
    }

    /// Validate sampling controls against the selected inference protocol.
    pub fn validate_sampling(self, inference: &InferenceConfig) -> Result<(), LlmError> {
        if !matches!(
            self,
            Self::ChatCompletions
                | Self::ChatCompletionsNonStreaming
                | Self::Reflection
                | Self::Vercel
        ) && (inference.temperature.is_some()
            || inference.top_p.is_some()
            || inference.seed.is_some())
        {
            return Err(LlmError::Request(
                "sampling controls require openai-chat-completions".into(),
            ));
        }
        if inference
            .temperature
            .is_some_and(|value| !(0.0..=2.0).contains(&value))
        {
            return Err(LlmError::Request(
                "temperature must be between 0 and 2".into(),
            ));
        }
        if inference
            .top_p
            .is_some_and(|value| !(value > 0.0 && value <= 1.0))
        {
            return Err(LlmError::Request(
                "top_p must be greater than 0 and at most 1".into(),
            ));
        }
        Ok(())
    }
}

impl FromStr for ModelApi {
    type Err = ModelApiParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "openai-codex-responses" => Ok(Self::CodexResponses),
            "openai-responses" => Ok(Self::Responses),
            "reflection" => Ok(Self::Reflection),
            "gemini" => Ok(Self::Gemini),
            "vercel" => Ok(Self::Vercel),
            "openai-chat-completions" | "openrouter" => Ok(Self::ChatCompletions),
            "anthropic-messages" | "anthropic" => Ok(Self::AnthropicMessages),
            "openai-chat-completions-non-streaming" => Ok(Self::ChatCompletionsNonStreaming),
            value => Err(ModelApiParseError(value.to_string())),
        }
    }
}

fn validate_gemini_request(request: &CompletionRequest) -> Result<(), LlmError> {
    if request
        .max_tokens
        .is_some_and(|tokens| tokens == 0 || tokens > 65_536)
    {
        return Err(LlmError::Request(
            "Gemini 4 Argon supports 1..65536 output tokens per stream; long decode is not supported by the OpenAI-compatible API".into(),
        ));
    }
    if let Some(inference) = &request.inference {
        ModelApi::Gemini.validate_sampling(inference)?;
        if inference
            .reasoning_effort
            .as_deref()
            .is_some_and(|effort| !matches!(effort, "none" | "low" | "medium" | "high"))
        {
            return Err(LlmError::Request(
                "Gemini 4 Argon reasoning effort must be none (provider default), low, medium, or high".into(),
            ));
        }
        if inference
            .extra_body
            .as_deref()
            .is_some_and(|extra| !extra.trim().is_empty())
        {
            return Err(LlmError::Request(
                "gemini does not support extra_body overrides; use openai-chat-completions for custom requests".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use futures_util::TryStreamExt;
    use mirror_protocol::chat_record;
    use mirror_protocol::ImageContent;
    use mirror_protocol::UserMessage;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;
    use crate::tools::web_test_server::TestResponse;
    use crate::tools::web_test_server::TestServer;

    #[tokio::test]
    async fn text_only_model_gets_a_note_where_another_model_gets_the_image() {
        let reply = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let mut server = TestServer::start(vec![
            TestResponse::text(200, reply),
            TestResponse::text(200, reply),
        ])
        .await;
        let config = ModelProviderConfig {
            api: ModelApi::ChatCompletions,
            model: "reflection/beam".into(),
            api_key: Some("synthetic-key".into()),
            base_url: Some(server.base_url.clone()),
            context_window: None,
            auth_path: None,
            prompt_cache_key: "session".into(),
            request_timeout: None,
        };
        let provider = ModelProvider::new(config.clone(), None)
            .await
            .expect("provider");
        let text_only = CompletionRequest {
            messages: vec![chat_record::Message::User(UserMessage {
                content: "[Image #1] what is this?".into(),
                checkpoint: None,
                images: vec![ImageContent {
                    name: "[Image #1]".into(),
                    data: b"\x89PNG\r\n\x1a\npixels".to_vec(),
                }],
            })],
            tools: vec![],
            max_tokens: None,
            inference: None,
        };
        let reads_images = CompletionRequest {
            inference: Some(InferenceConfig {
                model: "vision-model".into(),
                ..Default::default()
            }),
            ..text_only.clone()
        };

        provider
            .stream(text_only)
            .try_collect::<Vec<_>>()
            .await
            .expect("text-only request");
        provider
            .stream(reads_images)
            .try_collect::<Vec<_>>()
            .await
            .expect("image request");

        assert_eq!(
            server.request().await.body["messages"][0]["content"],
            json!(
                "<image name=[Image #1]>image content omitted because you do not support image \
                 input</image>\n[Image #1] what is this?"
            )
        );
        assert_eq!(
            server.request().await.body["messages"][0]["content"][1]["type"],
            json!("image_url")
        );
        let reflection_api = ModelProviderConfig {
            api: ModelApi::Reflection,
            model: "Beam-1.0-501B-A23B".into(),
            ..config
        };
        assert!(!reflection_api.accepts_images());
    }

    #[tokio::test]
    async fn gemini_selection_resolves_its_own_endpoint_and_drops_previous_credentials() {
        let original = ModelProviderConfig {
            api: ModelApi::Responses,
            model: "other-model".into(),
            api_key: Some("synthetic-other-key".into()),
            base_url: Some("http://other.invalid".into()),
            context_window: None,
            auth_path: None,
            prompt_cache_key: "session".into(),
            request_timeout: None,
        };
        let mut selected = original
            .for_inference(&InferenceConfig {
                api: Some("gemini".into()),
                model: "gemini-4-argon".into(),
                ..Default::default()
            })
            .expect("selection");
        assert_eq!(selected.api_key, None);
        assert_eq!(selected.base_url, None);
        selected.api_key = Some("synthetic-gemini-key".into());
        let provider = ModelProvider::new(selected.clone(), None)
            .await
            .expect("provider");
        assert_eq!(
            super::super::model_context_window(&provider, &selected).await,
            1_048_576
        );
        assert_eq!(
            selected.connection().await.expect("connection"),
            (
                "https://generativelanguage.googleapis.com/v1beta/openai".into(),
                "synthetic-gemini-key".into(),
            )
        );
        selected.base_url = Some("https://other.invalid".into());
        assert_matches!(selected.connection().await, Err(LlmError::Request(error)) if error.contains("not Vertex AI"));
        selected.base_url = None;
        selected.model = "unverified-model".into();
        assert_matches!(selected.connection().await, Err(LlmError::Request(error)) if error.contains("gemini-4-argon only"));
    }

    #[tokio::test]
    async fn vercel_selection_resolves_its_own_endpoint_and_drops_previous_credentials() {
        let original = ModelProviderConfig {
            api: ModelApi::ChatCompletions,
            model: "other-model".into(),
            api_key: Some("synthetic-other-key".into()),
            base_url: Some("http://other.invalid".into()),
            context_window: None,
            auth_path: None,
            prompt_cache_key: "session".into(),
            request_timeout: None,
        };
        let mut selected = original
            .for_inference(&InferenceConfig {
                api: Some("vercel".into()),
                model: "reflection/beam".into(),
                ..Default::default()
            })
            .expect("selection");
        assert_eq!(selected.api, ModelApi::Vercel);
        assert_eq!(selected.api_key, None);
        assert_eq!(selected.base_url, None);
        selected.api_key = Some("synthetic-gateway-key".into());
        assert_eq!(
            selected.connection().await.expect("connection"),
            (
                "https://ai-gateway.vercel.sh/v1".into(),
                "synthetic-gateway-key".into(),
            )
        );
        selected.base_url = Some("https://ai-gateway.vercel.sh/v1/".into());
        assert_matches!(selected.connection().await, Ok(_));
        selected.base_url = Some("https://other.invalid".into());
        assert_matches!(selected.connection().await, Err(LlmError::Request(error)) if error.contains("not custom endpoints"));
        selected.base_url = None;
        selected.api_key = Some("  ".into());
        assert_matches!(selected.connection().await, Err(LlmError::Request(error)) if error.contains("AI_GATEWAY_API_KEY"));
    }

    #[test]
    fn vercel_accepts_chat_completions_sampling_controls() {
        assert_matches!(
            ModelApi::Vercel.validate_sampling(&InferenceConfig {
                temperature: Some(0.5),
                top_p: Some(0.9),
                seed: Some(7),
                ..Default::default()
            }),
            Ok(())
        );
    }

    #[test]
    fn gemini_rejects_unsupported_controls_before_inference() {
        let mut request = CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: Some(65_536),
            inference: Some(InferenceConfig {
                reasoning_effort: Some("medium".into()),
                ..Default::default()
            }),
        };
        assert_matches!(validate_gemini_request(&request), Ok(()));
        request.max_tokens = Some(65_537);
        assert_matches!(validate_gemini_request(&request), Err(LlmError::Request(error)) if error.contains("long decode"));
        request.max_tokens = None;
        request.inference = Some(InferenceConfig {
            reasoning_effort: Some("xhigh".into()),
            ..Default::default()
        });
        assert_matches!(validate_gemini_request(&request), Err(LlmError::Request(error)) if error.contains("reasoning effort"));
        request.inference = Some(InferenceConfig {
            temperature: Some(0.5),
            ..Default::default()
        });
        assert_matches!(validate_gemini_request(&request), Err(LlmError::Request(_)));
        request.inference = Some(InferenceConfig {
            extra_body: Some("{}".into()),
            ..Default::default()
        });
        assert_matches!(validate_gemini_request(&request), Err(LlmError::Request(error)) if error.contains("extra_body"));
    }

    #[test]
    fn parses_supported_api_names() {
        assert_eq!(
            "openai-codex-responses".parse(),
            Ok(ModelApi::CodexResponses)
        );
        assert_eq!("openai-responses".parse(), Ok(ModelApi::Responses));
        assert_eq!(
            "openai-chat-completions".parse(),
            Ok(ModelApi::ChatCompletions)
        );
        assert_eq!("openrouter".parse(), Ok(ModelApi::ChatCompletions));
        assert_eq!("vercel".parse(), Ok(ModelApi::Vercel));
        assert_eq!(
            "anthropic-messages".parse(),
            Ok(ModelApi::AnthropicMessages)
        );
        assert_eq!("anthropic".parse(), Ok(ModelApi::AnthropicMessages));
        assert_eq!(
            "openai-chat-completions-non-streaming".parse(),
            Ok(ModelApi::ChatCompletionsNonStreaming)
        );
    }

    #[test]
    fn api_names_parse_back_to_their_api() {
        for api in [
            ModelApi::CodexResponses,
            ModelApi::Responses,
            ModelApi::ChatCompletions,
            ModelApi::Reflection,
            ModelApi::Gemini,
            ModelApi::Vercel,
            ModelApi::AnthropicMessages,
            ModelApi::ChatCompletionsNonStreaming,
        ] {
            assert_eq!(api.as_str().parse(), Ok(api));
        }
    }

    #[tokio::test]
    async fn reflection_connection_uses_public_production_endpoint() {
        let config = ModelProviderConfig {
            context_window: Some(super::super::DEFAULT_CONTEXT_WINDOW),
            api: ModelApi::Reflection,
            model: "Beam-501B-A23B".into(),
            api_key: Some("synthetic-reflection-key".into()),
            base_url: None,
            auth_path: None,
            prompt_cache_key: "session".into(),
            request_timeout: None,
        };

        let (base_url, _) = config.connection().await.expect("Reflection connection");

        assert_eq!(base_url, "https://api.reflection.ai/openai/v1");
    }

    #[tokio::test]
    async fn reflection_switch_does_not_reuse_other_provider_credentials() {
        assert_eq!("reflection".parse(), Ok(ModelApi::Reflection));
        let config = ModelProviderConfig {
            context_window: Some(super::super::DEFAULT_CONTEXT_WINDOW),
            api: ModelApi::ChatCompletions,
            model: "original".into(),
            api_key: Some("synthetic-private-key".into()),
            base_url: Some("https://other.example/v1".into()),
            auth_path: None,
            prompt_cache_key: "session".into(),
            request_timeout: None,
        };
        assert!(!format!("{config:?}").contains("synthetic-private-key"));
        let provider = ModelProvider::new(config, None).await.expect("provider");
        let selected = provider
            .default_config
            .for_inference(&InferenceConfig {
                api: Some("reflection".into()),
                ..Default::default()
            })
            .expect("selection");
        assert_eq!(selected.api, ModelApi::Reflection);
        assert_eq!(selected.api_key, None);
        assert_eq!(selected.base_url, None);
    }

    #[tokio::test]
    async fn switching_away_from_reflection_drops_its_key() {
        let provider = ModelProvider::new(
            ModelProviderConfig {
                context_window: Some(super::super::DEFAULT_CONTEXT_WINDOW),
                api: ModelApi::Reflection,
                model: "Beam-501B-A23B".into(),
                api_key: Some("synthetic-reflection-key".into()),
                base_url: None,
                auth_path: None,
                prompt_cache_key: "session".into(),
                request_timeout: None,
            },
            None,
        )
        .await
        .expect("provider");
        for api in [
            "openai-responses",
            "openai-chat-completions",
            "anthropic-messages",
            "vercel",
        ] {
            let config = provider
                .default_config
                .for_inference(&InferenceConfig {
                    api: Some(api.into()),
                    ..Default::default()
                })
                .expect("selection");
            assert_eq!(config.api_key, None);
        }
    }

    #[tokio::test]
    async fn reflection_rejects_custom_endpoints_before_sending_credentials() {
        let provider = ModelProvider::new(
            ModelProviderConfig {
                context_window: Some(super::super::DEFAULT_CONTEXT_WINDOW),
                api: ModelApi::Reflection,
                model: "Beam-501B-A23B".into(),
                api_key: Some("synthetic-reflection-key".into()),
                base_url: Some("http://127.0.0.1:1/v1".into()),
                auth_path: None,
                prompt_cache_key: "session".into(),
                request_timeout: None,
            },
            None,
        )
        .await
        .expect("provider");
        let stream = provider.stream(CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            inference: None,
        });
        pin_mut!(stream);
        assert_matches!(stream.next().await, Some(Err(LlmError::Request(message)))
            if message.contains("does not accept a custom base URL"));
    }

    #[test]
    fn rejects_unsupported_api_name() {
        let result = "unknown".parse::<ModelApi>();

        assert_matches!(result, Err(ModelApiParseError(name)) if name == "unknown");
    }

    #[test]
    fn sampling_validation_accepts_boundaries_and_rejects_invalid_values() {
        assert_matches!(
            ModelApi::ChatCompletions.validate_sampling(&InferenceConfig {
                temperature: Some(2.0),
                top_p: Some(1.0),
                seed: Some(i64::MIN),
                ..Default::default()
            }),
            Ok(())
        );
        assert_matches!(ModelApi::ChatCompletions.validate_sampling(&InferenceConfig {
            temperature: Some(-0.1),
            ..Default::default()
        }), Err(LlmError::Request(message)) if message == "temperature must be between 0 and 2");
        assert_matches!(ModelApi::ChatCompletions.validate_sampling(&InferenceConfig {
            top_p: Some(0.0),
            ..Default::default()
        }), Err(LlmError::Request(message)) if message == "top_p must be greater than 0 and at most 1");
        assert_matches!(ModelApi::ChatCompletions.validate_sampling(&InferenceConfig {
            top_p: Some(f64::INFINITY),
            ..Default::default()
        }), Err(LlmError::Request(message)) if message == "top_p must be greater than 0 and at most 1");
        assert_matches!(ModelApi::ChatCompletions.validate_sampling(&InferenceConfig {
            top_p: Some(f64::NAN),
            ..Default::default()
        }), Err(LlmError::Request(message)) if message == "top_p must be greater than 0 and at most 1");
    }

    #[tokio::test]
    async fn selected_provider_rejects_unsupported_sampling() {
        let provider = ModelProvider::new(
            ModelProviderConfig {
                context_window: Some(super::super::DEFAULT_CONTEXT_WINDOW),
                api: ModelApi::ChatCompletions,
                model: "test-model".into(),
                api_key: Some("test-key".into()),
                base_url: Some("http://test/v1".into()),
                auth_path: None,
                prompt_cache_key: "session".into(),
                request_timeout: None,
            },
            None,
        )
        .await
        .expect("provider");
        let stream = provider.stream(CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            inference: Some(InferenceConfig {
                api: Some("openai-responses".into()),
                seed: Some(0),
                ..Default::default()
            }),
        });
        pin_mut!(stream);
        assert_matches!(stream.next().await, Some(Err(LlmError::Request(message)))
            if message == "sampling controls require openai-chat-completions");
    }

    #[tokio::test]
    async fn rejects_missing_codex_login_at_inference() {
        let auth_path = std::env::temp_dir().join("mirror-model-provider-missing-auth.json");
        let _ = std::fs::remove_file(&auth_path);

        let provider = ModelProvider::new(
            ModelProviderConfig {
                context_window: Some(super::super::DEFAULT_CONTEXT_WINDOW),
                api: ModelApi::CodexResponses,
                model: "gpt-test".to_string(),
                api_key: None,
                base_url: None,
                auth_path: Some(auth_path),
                prompt_cache_key: "session".to_string(),
                request_timeout: None,
            },
            None,
        )
        .await
        .expect("missing credentials must not prevent construction");
        let stream = provider.stream(CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            inference: None,
        });
        pin_mut!(stream);
        assert_matches!(stream.next().await, Some(Err(LlmError::Request(message)))
            if message.contains("codex login --device-auth"));
    }

    #[tokio::test]
    async fn empty_inference_fields_preserve_the_default_config() {
        let provider = ModelProvider::new(
            ModelProviderConfig {
                context_window: Some(super::super::DEFAULT_CONTEXT_WINDOW),
                api: ModelApi::ChatCompletions,
                model: "default-model".to_string(),
                api_key: Some("test-key".to_string()),
                base_url: Some("https://default.example.test/v1".to_string()),
                auth_path: None,
                prompt_cache_key: "session".to_string(),
                request_timeout: None,
            },
            None,
        )
        .await
        .expect("provider");
        let inference = InferenceConfig {
            model: String::new(),
            api: None,
            base_url: None,
            reasoning_effort: Some("high".to_string()),
            ..Default::default()
        };

        let selected = provider
            .default_config
            .for_inference(&inference)
            .expect("selection");

        assert_eq!(selected.api, ModelApi::ChatCompletions);
        assert_eq!(selected.model, "default-model");
        assert_eq!(
            selected.context_window,
            Some(super::super::DEFAULT_CONTEXT_WINDOW)
        );
        assert_eq!(
            selected.base_url,
            Some("https://default.example.test/v1".to_string())
        );
    }

    #[tokio::test]
    async fn request_inference_overrides_the_default_config() {
        let provider = ModelProvider::new(
            ModelProviderConfig {
                context_window: Some(super::super::DEFAULT_CONTEXT_WINDOW),
                api: ModelApi::ChatCompletions,
                model: "default-model".to_string(),
                api_key: Some("test-key".to_string()),
                base_url: Some("https://default.example.test/v1".to_string()),
                auth_path: None,
                prompt_cache_key: "session".to_string(),
                request_timeout: None,
            },
            None,
        )
        .await
        .expect("provider");
        let inference = InferenceConfig {
            model: "request-model".to_string(),
            api: Some("openai-responses".to_string()),
            base_url: Some("https://request.example.test/v1".to_string()),
            reasoning_effort: Some("high".to_string()),
            ..Default::default()
        };

        let selected = provider
            .default_config
            .for_inference(&inference)
            .expect("selection");

        assert_eq!(selected.api, ModelApi::Responses);
        assert_eq!(selected.model, "request-model");
        assert_eq!(selected.context_window, None);
        assert_eq!(
            selected.base_url,
            Some("https://request.example.test/v1".to_string())
        );
        assert_eq!(provider.model(), "default-model");
    }

    #[test]
    fn request_inference_names_the_context_window_of_its_model() {
        let config = ModelProviderConfig {
            context_window: Some(272_000),
            api: ModelApi::Responses,
            model: "default-model".to_string(),
            api_key: None,
            base_url: None,
            auth_path: None,
            prompt_cache_key: "session".to_string(),
            request_timeout: None,
        };
        let selected = config
            .for_inference(&InferenceConfig {
                model: "request-model".to_string(),
                api: Some("anthropic-messages".to_string()),
                context_window: Some(200_000),
                ..Default::default()
            })
            .expect("selection");

        assert_eq!(selected.api, ModelApi::AnthropicMessages);
        assert_eq!(selected.context_window, Some(200_000));
        assert_matches!(
            config.for_inference(&InferenceConfig {
                model: "request-model".to_string(),
                context_window: Some(0),
                ..Default::default()
            }),
            Err(LlmError::Request(error)) if error.contains("context-window")
        );
    }
}
