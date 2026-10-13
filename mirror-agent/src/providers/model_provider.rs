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
use super::StreamEvent;

/// Immutable provider defaults shared by inference and metadata requests.
#[derive(Clone)]
pub struct ModelProvider {
    default_config: ModelProviderConfig,
    client: Client,
    request_guard: Option<Arc<dyn ModelRequestGuard>>,
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
    AnthropicMessages,
    ChatCompletionsNonStreaming,
}

/// An unsupported model protocol name.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("unsupported model API {0:?}")]
pub struct ModelApiParseError(String);

impl ModelProvider {
    /// Construct reusable provider defaults and validate explicit configuration.
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
        if config.api == ModelApi::CodexResponses {
            CodexResponses::validate_login(config.auth_path.as_deref())?;
        }
        Ok(Self {
            client: super::http_client(config.request_timeout)?,
            default_config: config,
            request_guard,
        })
    }
}

impl ModelProviderConfig {
    /// Resolve the same endpoint and credentials for inference and metadata.
    pub(super) async fn connection(&self) -> Result<(String, String), LlmError> {
        let (base_env, base_default, key_env) = match self.api {
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
        try_stream! {
            let config = config?;
            let (base_url, api_key) = config.connection().await?;
            match config.api {
                ModelApi::CodexResponses => {
                    let provider = CodexResponses::new(
                        CodexResponsesConfig {
                            model: config.model,
                            base_url: Some(base_url),
                            auth_path: config.auth_path,
                            prompt_cache_key: config.prompt_cache_key,
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
                ModelApi::ChatCompletions | ModelApi::ChatCompletionsNonStreaming => {
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
                        config.api == ModelApi::ChatCompletions,
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
}

impl ModelApi {
    /// Validate sampling controls against the selected inference protocol.
    pub fn validate_sampling(self, inference: &InferenceConfig) -> Result<(), LlmError> {
        if !matches!(
            self,
            Self::ChatCompletions | Self::ChatCompletionsNonStreaming | Self::Reflection
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
            "openai-chat-completions" | "openrouter" => Ok(Self::ChatCompletions),
            "anthropic-messages" | "anthropic" => Ok(Self::AnthropicMessages),
            "openai-chat-completions-non-streaming" => Ok(Self::ChatCompletionsNonStreaming),
            value => Err(ModelApiParseError(value.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

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
    async fn rejects_missing_codex_login_at_construction() {
        let auth_path = std::env::temp_dir().join("mirror-model-provider-missing-auth.json");
        let _ = std::fs::remove_file(&auth_path);

        let error = ModelProvider::new(
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
        .err()
        .expect("missing Codex login fails at construction");

        assert_matches!(error, LlmError::Request(message) if message.contains("codex login --device-auth"));
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
        assert_eq!(
            selected.base_url,
            Some("https://request.example.test/v1".to_string())
        );
        assert_eq!(provider.model(), "default-model");
    }
}
