//! Keyless, direct-loopback inference through a user-owned llama.cpp server.

use std::time::Duration;

use mirror_protocol::local_models::{LocalModelEntry, LocalModelInfo};
use reqwest::{Client, Url};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{
    build_http_client, http_client_builder, llama_thinking_field, model_metadata,
    read_limited_json, usable_window, ChatCompletions, ChatCompletionsConfig, LlmError, ModelApi,
    ModelEvents, REASONING_EFFORT_LEVELS,
};

/// A client for a loopback llama-server, which is a router or serves one model.
#[derive(Debug)]
pub struct LlamaCpp {
    base_url: String,
    /// `base_url` without its `/v1`, where llama.cpp's native endpoints live.
    root: String,
    client: Client,
    poll: Duration,
    /// How long a load may take; `LOAD_WAIT` outside tests.
    load_wait: Duration,
}

/// Whether loading a model may unload another that the router holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eviction {
    Allowed,
    /// Automatic restoration uses this, so it never replaces the user's model.
    Forbidden,
}

#[derive(Debug, Deserialize)]
struct RouterProps {
    #[serde(default)]
    role: String,
    #[serde(default)]
    models_autoload: bool,
}

/// How a llama-server presents its models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Loads, unloads and downloads models on request.
    Router,
    /// Serves the one model it was started with.
    Single,
}

/// What a server reports about itself and the models it can serve.
struct Served {
    mode: Mode,
    props: Value,
    models: Vec<InventoryModel>,
}

#[derive(Debug, Deserialize)]
struct SingleCatalog {
    data: Vec<SingleModel>,
}

#[derive(Debug, Deserialize)]
struct SingleModel {
    id: String,
}

#[derive(Debug, Deserialize)]
struct Inventory {
    // Entries are read one by one, so one that llama.cpp reports in a shape we do
    // not know is skipped instead of hiding every other model.
    data: Vec<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct InventoryModel {
    id: String,
    #[serde(default)]
    source: String,
    status: ModelStatus,
    meta: Option<ModelMeta>,
    /// Where the file lives, for entries llama.cpp has on disk.
    #[serde(default)]
    path: Option<String>,
    /// Reported by newer llama.cpp releases; absent from older ones.
    #[serde(default)]
    memory: Option<ModelMemory>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelMemory {
    #[serde(default)]
    status: String,
    max: Option<ModelMeta>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelStatus {
    value: String,
    #[serde(default)]
    preset: String,
    #[serde(default)]
    failed: bool,
    exit_code: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelMeta {
    n_ctx: Option<u64>,
}

impl ModelMeta {
    fn window(&self) -> Option<usize> {
        self.n_ctx.and_then(usable_window)
    }
}

#[derive(Debug, Deserialize)]
struct ModelProps {
    default_generation_settings: ModelMeta,
    /// Missing when a release stops reporting it; the model is then not second-guessed.
    chat_template_caps: Option<TemplateCaps>,
}

#[derive(Debug, Default, Deserialize)]
struct TemplateCaps {
    #[serde(default)]
    supports_tools: bool,
    #[serde(default)]
    supports_tool_calls: bool,
    #[serde(default)]
    supports_reasoning_effort: bool,
}

#[derive(Debug, Deserialize)]
struct TemplatePrompt {
    prompt: String,
}

/// How long a request waits for a model that is still loading.
const LOAD_WAIT: Duration = Duration::from_secs(600);

/// What llama.cpp answers when a request names a model it does not hold.
pub(super) const NOT_LOADED: &str = "model is not loaded";

/// A reasoning effort no template knows, to tell a level the template reads from one it ignores.
const UNKNOWN_EFFORT: &str = "mirror-unsupported";

/// The effort a model starts with: thinking stays off until the user picks a setting.
const DEFAULT_EFFORT: &str = "none";

impl LlamaCpp {
    /// Resolve literal loopback endpoints without cloud configuration.
    pub fn base_url(value: Option<&str>) -> Result<String, LlmError> {
        let configured = std::env::var("MIRROR_LLAMA_CPP_BASE_URL").ok();
        let value = value
            .filter(|value| !value.is_empty())
            .or(configured.as_deref().filter(|value| !value.is_empty()))
            .unwrap_or("http://127.0.0.1:8080/v1");
        let invalid = || {
            local_error("requires http://127.0.0.1:<port>/v1 or http://[::1]:<port>/v1 without credentials, query or fragment")
        };
        let mut url = Url::parse(value).map_err(|_| invalid())?;
        if url.scheme() != "http"
            || !matches!(url.host_str(), Some("127.0.0.1" | "[::1]" | "localhost"))
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || url.path().trim_end_matches('/') != "/v1"
        {
            return Err(invalid());
        }
        if url.host_str() == Some("localhost") {
            url.set_host(Some("127.0.0.1")).map_err(|_| invalid())?;
        }
        Ok(url.as_str().trim_end_matches('/').to_owned())
    }

    /// Never inherit proxy settings, provider credentials or redirects.
    pub fn new(base_url: &str, timeout: Option<Duration>) -> Result<Self, LlmError> {
        Self::with_client(base_url, local_model_client(timeout)?)
    }

    /// Like `new`, sharing `client`, which must come from `local_model_client`.
    pub(super) fn with_client(base_url: &str, client: Client) -> Result<Self, LlmError> {
        let base_url = Self::base_url(Some(base_url))?;
        Ok(Self {
            root: base_url.trim_end_matches("/v1").to_owned(),
            base_url,
            client,
            poll: Duration::from_millis(500),
            load_wait: LOAD_WAIT,
        })
    }

    async fn metadata<T: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<T, LlmError> {
        let request = self
            .client
            .get(format!("{}{path}", self.root))
            .query(query)
            .timeout(Duration::from_secs(3));
        decode(model_metadata(request).await.map_err(local_error)?)
    }

    /// Check that a server is reachable and does not load models behind the selector's back.
    pub async fn check_server(&self) -> Result<(), LlmError> {
        self.server().await.map(|_| ())
    }

    async fn server(&self) -> Result<(Mode, Value), LlmError> {
        let props: Value = self.metadata("/props", &[]).await?;
        let router: RouterProps = decode(props.clone())?;
        if router.role != "router" {
            return Ok((Mode::Single, props));
        }
        if router.models_autoload {
            return Err(local_error(
                "use a llama-server router with --no-models-autoload",
            ));
        }
        Ok((Mode::Router, props))
    }

    async fn served(&self) -> Result<Served, LlmError> {
        let (mode, props) = self.server().await?;
        let models = match mode {
            Mode::Router => self
                .metadata::<Inventory>("/models", &[])
                .await?
                .data
                .into_iter()
                .filter_map(|entry| serde_json::from_value(entry).ok())
                .collect(),
            Mode::Single => {
                // Its one model is resident, with the context the server was started with.
                let n_ctx = props["default_generation_settings"]["n_ctx"].as_u64();
                self.metadata::<SingleCatalog>("/v1/models", &[])
                    .await?
                    .data
                    .into_iter()
                    .map(|model| InventoryModel {
                        id: model.id,
                        source: "external".into(),
                        status: ModelStatus {
                            value: "loaded".into(),
                            ..Default::default()
                        },
                        meta: Some(ModelMeta { n_ctx }),
                        ..Default::default()
                    })
                    .collect()
            }
        };
        Ok(Served {
            mode,
            props,
            models,
        })
    }

    /// Model management needs a router; a single-model server cannot load or download.
    async fn require_router(&self) -> Result<(), LlmError> {
        if self.server().await?.0 != Mode::Router {
            return Err(local_error(
                "this llama-server serves one fixed model; start a router to load or download models",
            ));
        }
        Ok(())
    }

    /// The allocated context and template capabilities of a model that is ready to serve.
    async fn properties(&self, model: &str) -> Result<(usize, TemplateCaps), LlmError> {
        let served = self.served().await?;
        let ready = served.models.iter().any(|item| {
            item.id == model
                // A model that slept after idling reloads itself on its next request.
                && matches!(item.status.value.as_str(), "loaded" | "sleeping")
                && item.meta.as_ref().and_then(ModelMeta::window).is_some()
        });
        if !ready {
            return Err(LlmError::ModelNotLoaded);
        }
        let props: ModelProps = match served.mode {
            Mode::Router => {
                self.metadata("/props", &[("model", model), ("autoload", "false")])
                    .await?
            }
            Mode::Single => decode(served.props)?,
        };
        let caps = props.chat_template_caps;
        if caps
            .as_ref()
            .is_some_and(|caps| !caps.supports_tools || !caps.supports_tool_calls)
        {
            return Err(local_error(
                "this model's chat template has no tool-call support, which Mirror needs",
            ));
        }
        let context = props
            .default_generation_settings
            .window()
            .ok_or_else(|| local_error("model did not report an allocated context"))?;
        Ok((context, caps.unwrap_or_default()))
    }

    /// Require an awake tool model, and return its allocated context.
    pub async fn check_model(&self, model: &str) -> Result<usize, LlmError> {
        Ok(self.properties(model).await?.0)
    }

    /// Wait, up to ten minutes, for a model that is still loading.
    pub async fn wait_until_loaded(&self, model: &str) -> Result<(), LlmError> {
        let deadline = tokio::time::Instant::now() + self.load_wait;
        while tokio::time::Instant::now() < deadline && self.is_loading(model).await? {
            tokio::time::sleep(self.poll).await;
        }
        Ok(())
    }

    async fn is_loading(&self, model: &str) -> Result<bool, LlmError> {
        Ok(self
            .served()
            .await?
            .models
            .iter()
            .any(|item| item.id == model && item.status.value == "loading"))
    }

    /// Require that the model's template demonstrably changes with `effort`.
    pub async fn check_effort(&self, model: &str, effort: &str) -> Result<(), LlmError> {
        if effort == "none" {
            return Ok(());
        }
        // A thinking toggle needs no capability report; a named level does.
        let caps = if matches!(effort, "on" | "off") {
            TemplateCaps::default()
        } else {
            self.properties(model).await?.1
        };
        // A level the template renders exactly as no setting is its default, or one it ignores;
        // either way the request behaves as the provider default.
        let toggle = matches!(effort, "on" | "off");
        if !self.supports_effort(model, effort, &caps).await?
            && (toggle || !self.renders_as_default(model, effort).await?)
        {
            return Err(local_error(
                "the model's chat template does not support that thinking setting",
            ));
        }
        Ok(())
    }

    /// Return only controls whose template rendering demonstrably changes.
    pub async fn info(&self, model: &str) -> Result<LocalModelInfo, LlmError> {
        let (context, caps) = self.properties(model).await?;
        let mut efforts = vec![DEFAULT_EFFORT.into()];
        if self.supports_effort(model, "on", &caps).await? {
            efforts.extend(["on".into(), "off".into()]);
        }
        if caps.supports_reasoning_effort {
            let ignored = self.template_prompt(model, UNKNOWN_EFFORT).await?;
            for effort in REASONING_EFFORT_LEVELS
                .into_iter()
                .filter(|level| *level != "none")
            {
                let prompt = self.template_prompt(model, effort).await?;
                if prompt.is_some() && prompt != ignored {
                    efforts.push(effort.into());
                }
            }
        }
        Ok(LocalModelInfo {
            context: context as u64,
            efforts,
            default_effort: DEFAULT_EFFORT.into(),
        })
    }

    /// Inspect cached artifacts without waking unloaded model processes.
    pub async fn inventory(&self) -> Result<Vec<LocalModelEntry>, LlmError> {
        let mut entries = Vec::new();
        for item in self.served().await?.models {
            // Speculative-decoding drafters are cached beside models but are not chat models.
            let path = item.path.as_deref().map(std::path::Path::new);
            if crate::local_models::is_drafter(path, &item.id).await {
                continue;
            }
            entries.push(LocalModelEntry {
                name: item.id,
                status: item.status.value,
                source: item.source,
                context: item.meta.and_then(|meta| meta.window()).unwrap_or(0) as u64,
                fit_context: item
                    .memory
                    .filter(|memory| memory.status == "measured")
                    .and_then(|memory| memory.max)
                    .and_then(|max| max.window())
                    .unwrap_or(0) as u64,
            });
        }
        Ok(entries)
    }

    async fn command(&self, path: &str, body: Value) -> Result<(), LlmError> {
        let request = self
            .client
            .post(format!("{}{path}", self.root))
            .json(&body)
            .timeout(Duration::from_secs(300));
        let result = model_metadata(request).await.map_err(local_error)?;
        if result["success"] != true {
            return Err(local_error("router did not accept the operation"));
        }
        Ok(())
    }

    /// Load only complete, offline artifacts; automatic restoration never evicts.
    pub async fn load(&self, model: &str, eviction: Eviction) -> Result<LocalModelInfo, LlmError> {
        let served = self.served().await?;
        if served.mode == Mode::Single {
            // Nothing to load: its model is resident, or it does not serve this one.
            return self.info(model).await;
        }
        let models = served.models;
        let selected = models.iter().find(|item| item.id == model)
            .filter(|item| matches!(item.source.as_str(), "cache" | "models_dir" | "preset")
                && matches!(item.status.value.as_str(), "unloaded" | "loading" | "loaded" | "sleeping"))
            .ok_or_else(|| local_error("Only completed cache/directory models can be loaded; restoration never downloads weights"))?;
        match selected.status.value.as_str() {
            // A sleeping model wakes on its next request, so it needs no load.
            "loaded" | "sleeping" => {}
            "loading" => self.wait_for_load(model).await?,
            _ => {
                // Only a model in memory would be evicted; a download holds no model slot.
                if eviction == Eviction::Forbidden
                    && models.iter().any(|item| {
                        item.id != model
                            && matches!(
                                item.status.value.as_str(),
                                "loading" | "loaded" | "sleeping"
                            )
                    })
                {
                    return Err(local_error("Another model is active; startup will not evict it, so it must be replaced by an explicit selection"));
                }
                if !offline(&selected.status.preset) {
                    return Err(local_error("Loading requires an offline child preset. Restart with --offline --no-models-autoload"));
                }
                self.command("/models/load", json!({"model": model}))
                    .await?;
                self.wait_for_load(model).await?;
            }
        }
        self.info(model).await
    }

    /// Wait, up to ten minutes, for a load to finish.
    async fn wait_for_load(&self, model: &str) -> Result<(), LlmError> {
        let deadline = tokio::time::Instant::now() + self.load_wait;
        loop {
            let models = self.served().await?.models;
            let item = models.iter().find(|item| item.id == model);
            if item.is_some_and(|item| item.status.value == "loaded") {
                return Ok(());
            }
            if item.is_none_or(|item| item.status.value != "loading") {
                let code = item
                    .filter(|item| item.status.failed)
                    .and_then(|item| item.status.exit_code.as_ref())
                    .and_then(Value::as_i64);
                let code = code.map_or_else(|| "not reported".into(), |code| code.to_string());
                return Err(local_error(&format!("failed to load (exit code {code}). Check that your build supports this model and enough RAM/VRAM is available. Restart after replacing the executable. See the llama-server log (~/.mirror/llama.cpp/logs/llama-server.log for a Mirror-managed server)")));
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(local_error(&format!(
                    "{model} did not finish loading within {}s",
                    self.load_wait.as_secs()
                )));
            }
            tokio::time::sleep(self.poll).await;
        }
    }

    /// Model state changes, until the stream ends.
    pub async fn model_events(&self) -> Result<ModelEvents, LlmError> {
        Ok(ModelEvents::new(self.events().await?))
    }

    /// The router's `/models/sse` stream; a download subscribes before it starts, so an
    /// already cached model's completion cannot be missed.
    pub async fn events(&self) -> Result<reqwest::Response, LlmError> {
        self.require_router().await?;
        self.client
            .get(format!("{}/models/sse", self.root))
            .timeout(Duration::from_secs(86400))
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|_| local_error("could not subscribe to the router's events"))
    }

    /// Download the explicitly reviewed variant, without sidecars: Mirror is text-only.
    pub async fn download(&self, model: &str) -> Result<(), LlmError> {
        self.require_router().await?;
        self.command("/models", json!({"model": model, "sidecars": []}))
            .await
    }

    /// Pause a transfer without deleting partial data or unloading an active model.
    pub async fn pause(&self, model: &str) -> Result<(), LlmError> {
        if self
            .served()
            .await?
            .models
            .iter()
            .any(|item| item.id == model && item.status.value == "downloading")
        {
            self.command("/models/unload", json!({"model": model}))
                .await?;
        }
        Ok(())
    }

    async fn supports_effort(
        &self,
        model: &str,
        effort: &str,
        caps: &TemplateCaps,
    ) -> Result<bool, LlmError> {
        if !ModelApi::LlamaCpp.accepts_reasoning_effort(effort) {
            return Ok(false);
        }
        let (candidate, other) = match effort {
            "on" | "off" => ("on", "off"),
            _ if caps.supports_reasoning_effort => (effort, UNKNOWN_EFFORT),
            _ => return Ok(false),
        };
        let prompt = self.template_prompt(model, candidate).await?;
        let comparison = self.template_prompt(model, other).await?;
        Ok(prompt.is_some()
            && prompt != comparison
            && (!matches!(effort, "on" | "off") || comparison.is_some()))
    }

    /// The prompt the model's template renders with `effort`, or `None` if it refuses it.
    ///
    /// A template refuses a value it does not support with a bad request, or, as Qwen's do for
    /// an unknown reasoning effort, by raising, which llama.cpp reports as a server error. Either
    /// counts as a refusal only once the template renders without a thinking setting.
    async fn template_prompt(&self, model: &str, effort: &str) -> Result<Option<String>, LlmError> {
        let response = self.render_template(model, Some(effort)).await?;
        if !response.status().is_success() {
            self.default_prompt(model).await?;
            return Ok(None);
        }
        let prompt: TemplatePrompt =
            decode(read_limited_json(response).await.map_err(local_error)?)?;
        Ok(Some(prompt.prompt))
    }

    /// Whether the template renders `effort` exactly as it renders no setting.
    async fn renders_as_default(&self, model: &str, effort: &str) -> Result<bool, LlmError> {
        let prompt = self.template_prompt(model, effort).await?;
        Ok(prompt.is_some() && prompt == Some(self.default_prompt(model).await?))
    }

    /// The prompt the template renders without a thinking setting; a template that cannot
    /// render it is broken, not refusing a setting.
    async fn default_prompt(&self, model: &str) -> Result<String, LlmError> {
        let response = self.render_template(model, None).await?;
        if !response.status().is_success() {
            let body = read_limited_json(response).await.unwrap_or_default();
            if body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains(NOT_LOADED))
            {
                return Err(LlmError::ModelNotLoaded);
            }
            return Err(local_error("the model's chat template failed to render"));
        }
        let prompt: TemplatePrompt =
            decode(read_limited_json(response).await.map_err(local_error)?)?;
        Ok(prompt.prompt)
    }

    async fn render_template(
        &self,
        model: &str,
        effort: Option<&str>,
    ) -> Result<reqwest::Response, LlmError> {
        let mut body = json!({"model": model, "messages": [{"role": "user", "content": "Hello"}]});
        if let Some(effort) = effort {
            let (field, value) = llama_thinking_field(effort);
            body[field] = value;
        }
        self.client
            .post(format!("{}/apply-template?autoload=false", self.root))
            .timeout(Duration::from_secs(3))
            .json(&body)
            .send()
            .await
            .map_err(|_| local_error("template probe failed"))
    }

    /// Reuse Chat Completions with llama.cpp's thinking fields and no key.
    pub(super) fn chat_completions(&self, model: String) -> ChatCompletions {
        ChatCompletions::new_llama_cpp(
            ChatCompletionsConfig {
                model,
                api_key: String::new(),
                base_url: self.base_url.clone(),
            },
            self.client.clone(),
        )
    }
}

/// Keyless model-management HTTP, including public metadata: no proxies or redirects.
pub fn local_model_client(timeout: Option<Duration>) -> Result<Client, LlmError> {
    build_http_client(
        http_client_builder(timeout)
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(2)),
    )
}

fn decode<T: DeserializeOwned>(value: Value) -> Result<T, LlmError> {
    serde_json::from_value(value).map_err(|_| local_error("invalid router/model metadata"))
}

fn offline(preset: &str) -> bool {
    let mut sections = 0;
    let mut offline = None;
    for line in preset
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with(['#', ';']))
    {
        if line.starts_with('[') && line.ends_with(']') {
            sections += 1;
        } else if let Some((key, value)) = line.split_once('=') {
            if sections != 1 {
                return false;
            }
            if key.trim().eq_ignore_ascii_case("offline") {
                if offline.is_some() {
                    return false;
                }
                offline = Some(matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "true" | "1" | "yes" | "on"
                ));
            }
        } else {
            return false;
        }
    }
    sections == 1 && offline == Some(true)
}

fn local_error(message: &str) -> LlmError {
    LlmError::Request(format!("llama.cpp: {message}"))
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use futures_util::TryStreamExt;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::providers::{
        CompletionRequest, Llm, ModelApi, ModelProvider, ModelProviderConfig, StreamEvent,
    };
    use crate::tools::web_test_server::{TestResponse, TestServer};

    fn router() -> TestResponse {
        TestResponse::json(200, json!({"role": "router", "models_autoload": false}))
    }

    fn listing(status: &str) -> TestResponse {
        TestResponse::json(
            200,
            json!({"data": [{"id": "synthetic", "status": {"value": status}, "meta": {"n_ctx": 8192}}]}),
        )
    }

    fn props(extra: Value) -> TestResponse {
        let mut props = json!({"default_generation_settings": {"n_ctx": 8192}});
        props
            .as_object_mut()
            .expect("object")
            .extend(extra.as_object().expect("object").clone());
        TestResponse::json(200, props)
    }

    fn tool_caps() -> Value {
        json!({"chat_template_caps": {"supports_tools": true, "supports_tool_calls": true, "supports_reasoning_effort": true}})
    }

    fn client(server: &TestServer) -> LlamaCpp {
        LlamaCpp::new(&format!("{}/v1", server.base_url), None).expect("loopback server")
    }

    #[test]
    fn base_url_restricts_destinations() {
        assert_eq!(
            LlamaCpp::base_url(Some("http://localhost:1234/v1/")).expect("loopback"),
            "http://127.0.0.1:1234/v1"
        );
        assert_eq!(
            LlamaCpp::base_url(Some("http://[::1]:1234/v1")).expect("IPv6"),
            "http://[::1]:1234/v1"
        );
        assert_matches!(LlamaCpp::base_url(Some("http://192.0.2.1/v1")), Err(_));
        assert_matches!(
            LlamaCpp::base_url(Some("http://user:synthetic@127.0.0.1/v1")),
            Err(_)
        );
        assert_matches!(
            LlamaCpp::base_url(Some("http://127.0.0.1/v1?key=synthetic")),
            Err(_)
        );
    }

    #[tokio::test]
    async fn requests_carry_no_key_and_use_deepseek_reasoning_without_autoload() {
        let mut server = TestServer::start(vec![TestResponse::text(
            200,
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"plan\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        )])
        .await;
        let config = ModelProviderConfig {
            api: ModelApi::LlamaCpp,
            model: "synthetic".into(),
            context_window: None,
            api_key: Some("synthetic-cloud-key-must-not-be-sent".into()),
            base_url: Some(format!("{}/v1", server.base_url)),
            auth_path: None,
            prompt_cache_key: "test".into(),
            request_timeout: None,
            managed_reference: None,
        };
        let provider = ModelProvider::new(config, None).await.expect("provider");
        let request = CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: Some(100),
            inference: None,
        };

        let events: Vec<_> = provider.stream(request).try_collect().await.expect("turn");

        assert_matches!(
            events.last(),
            Some(StreamEvent::Final(completion))
                if completion.message.reasoning_content.as_deref() == Some("plan")
        );
        let sent = server.request().await;
        assert_eq!(sent.path, "/v1/chat/completions?autoload=false");
        assert_eq!(sent.authorization, None);
        assert_eq!(sent.body["reasoning_format"], "deepseek");
    }

    #[tokio::test]
    async fn a_thinking_setting_must_change_the_rendered_template() {
        let mut server = TestServer::start(vec![
            TestResponse::json(200, json!({"prompt": "<think>"})),
            TestResponse::json(200, json!({"prompt": "<think></think>"})),
            router(),
            listing("loaded"),
            props(tool_caps()),
            TestResponse::json(400, json!({"error": "unsupported"})),
            TestResponse::json(200, json!({"prompt": "default"})),
            TestResponse::json(200, json!({"prompt": "default"})),
            TestResponse::json(400, json!({"error": "unsupported"})),
            TestResponse::json(200, json!({"prompt": "default"})),
        ])
        .await;
        let local = client(&server);

        local.check_effort("synthetic", "on").await.expect("toggle");
        let probe = server.request().await;
        assert_eq!(probe.path, "/apply-template?autoload=false");
        assert_eq!(
            probe.body["chat_template_kwargs"],
            json!({"enable_thinking": true})
        );
        assert_matches!(
            local.check_effort("synthetic", "high").await,
            Err(LlmError::Request(message)) if message.contains("thinking setting")
        );
    }

    #[tokio::test]
    async fn a_level_the_template_renders_as_its_default_is_accepted() {
        let server = TestServer::start(vec![
            router(),
            listing("loaded"),
            props(tool_caps()),
            TestResponse::json(200, json!({"prompt": "default"})),
            TestResponse::json(200, json!({"prompt": "default"})),
            TestResponse::json(200, json!({"prompt": "default"})),
            TestResponse::json(200, json!({"prompt": "default"})),
        ])
        .await;

        client(&server)
            .check_effort("synthetic", "medium")
            .await
            .expect("the template's own default level");
    }

    #[tokio::test]
    async fn a_template_that_raises_for_the_unsupported_sentinel_still_supports_the_level() {
        let server = TestServer::start(vec![
            router(),
            listing("loaded"),
            props(tool_caps()),
            TestResponse::json(200, json!({"prompt": "high effort"})),
            TestResponse::json(
                500,
                json!({"error": {"code": 500, "type": "server_error",
                    "message": "Jinja Exception: Unexpected reasoning effort mirror-unsupported"}}),
            ),
            TestResponse::json(200, json!({"prompt": "default"})),
        ])
        .await;

        client(&server)
            .check_effort("synthetic", "high")
            .await
            .expect("a template that renders the level and raises for the sentinel supports it");
    }

    #[tokio::test]
    async fn a_template_that_fails_without_a_thinking_setting_is_broken_not_refusing() {
        let raised = || {
            TestResponse::json(
                500,
                json!({"error": {"message": "Jinja Exception: unknown filter"}}),
            )
        };
        let server = TestServer::start(vec![raised(), raised()]).await;

        assert_matches!(
            client(&server).check_effort("synthetic", "on").await,
            Err(LlmError::Request(message)) if message.contains("failed to render")
        );
    }

    #[tokio::test]
    async fn a_model_the_router_unloaded_is_not_loaded_rather_than_refusing() {
        let unloaded = || {
            TestResponse::json(
                400,
                json!({"error": {"code": 400, "message": "model is not loaded"}}),
            )
        };
        let server = TestServer::start(vec![unloaded(), unloaded()]).await;

        assert_eq!(
            client(&server).check_effort("synthetic", "on").await,
            Err(LlmError::ModelNotLoaded)
        );
    }

    #[tokio::test]
    async fn a_router_that_autoloads_is_rejected() {
        let server = TestServer::start(vec![TestResponse::json(
            200,
            json!({"role": "router", "models_autoload": true}),
        )])
        .await;

        assert_matches!(
            client(&server).check_model("synthetic").await,
            Err(LlmError::Request(message)) if message.contains("--no-models-autoload")
        );
    }

    #[tokio::test]
    async fn a_load_that_stays_loading_gives_up_after_the_load_wait() {
        let server = TestServer::start(vec![router(), listing("loading")]).await;
        let mut local = client(&server);
        local.load_wait = Duration::ZERO;

        assert_matches!(
            local.wait_for_load("synthetic").await,
            Err(LlmError::Request(message)) if message.contains("did not finish loading")
        );
    }

    #[tokio::test]
    async fn restoration_evicts_no_model_in_memory_but_ignores_downloads() {
        let beside = |status: &str| {
            TestResponse::json(
                200,
                json!({"data": [
                    {"id": "synthetic", "source": "cache", "status": {"value": "unloaded"}},
                    {"id": "other", "source": "cache", "status": {"value": status}},
                ]}),
            )
        };
        let server = TestServer::start(vec![
            router(),
            beside("loaded"),
            router(),
            beside("downloading"),
        ])
        .await;
        let local = client(&server);

        assert_matches!(
            local.load("synthetic", Eviction::Forbidden).await,
            Err(LlmError::Request(error)) if error.contains("Another model is active")
        );
        // Past the eviction check, this entry fails only for lacking an offline preset.
        assert_matches!(
            local.load("synthetic", Eviction::Forbidden).await,
            Err(LlmError::Request(error)) if error.contains("offline child preset")
        );
    }

    #[tokio::test]
    async fn load_rejects_offline_outside_the_model_section() {
        let server = TestServer::start(vec![
            router(),
            TestResponse::json(200, json!({"data": [{"id": "synthetic", "source": "cache", "status": {"value": "unloaded", "preset": "offline = true\n[synthetic]\n"}}]})),
        ]).await;

        assert_matches!(
            client(&server).load("synthetic", Eviction::Allowed).await,
            Err(LlmError::Request(error)) if error.contains("offline child preset")
        );
    }

    #[tokio::test]
    async fn a_sleeping_model_without_reported_template_capabilities_is_usable() {
        let server = TestServer::start(vec![router(), listing("sleeping"), props(json!({}))]).await;

        assert_eq!(client(&server).check_model("synthetic").await, Ok(8192));
    }

    #[tokio::test]
    async fn a_request_waits_for_a_model_that_is_still_loading() {
        let server = TestServer::start(vec![
            router(),
            listing("loading"),
            router(),
            listing("loaded"),
            router(),
            listing("loaded"),
            props(json!({})),
        ])
        .await;
        let mut local = client(&server);
        local.poll = Duration::ZERO;

        local.wait_until_loaded("synthetic").await.expect("waited");

        assert_eq!(local.check_model("synthetic").await, Ok(8192));
    }

    #[tokio::test]
    async fn a_single_model_server_serves_only_its_resident_model() {
        let props = || {
            TestResponse::json(
                200,
                json!({"role": "model", "default_generation_settings": {"n_ctx": 16384}}),
            )
        };
        let models = || TestResponse::json(200, json!({"data": [{"id": "synthetic"}]}));
        let server = TestServer::start(vec![
            props(),
            models(),
            props(),
            models(),
            props(),
            models(),
        ])
        .await;
        let local = client(&server);

        assert_eq!(local.check_model("synthetic").await, Ok(16384));
        assert_eq!(
            local.check_model("other").await,
            Err(LlmError::ModelNotLoaded)
        );
        assert_eq!(
            local.inventory().await.expect("inventory"),
            vec![LocalModelEntry {
                name: "synthetic".into(),
                status: "loaded".into(),
                source: "external".into(),
                context: 16384,
                fit_context: 0,
            }]
        );
    }

    #[tokio::test]
    async fn a_single_model_server_cannot_download_models() {
        let server =
            TestServer::start(vec![TestResponse::json(200, json!({"role": "model"}))]).await;

        assert_matches!(
            client(&server).download("repo:Q4_K_M").await,
            Err(LlmError::Request(message)) if message.contains("one fixed model")
        );
    }

    #[tokio::test]
    async fn the_inventory_skips_unreadable_entries_and_drafters_and_reads_measured_fit() {
        let entry = |id: &str| {
            json!({"id": id, "status": {"value": "unloaded", "preset": "[x]\noffline = 1\n"},
                "source": "cache", "context_length": 262144})
        };
        let mut measured = entry("measured:Q4");
        measured["memory"] = json!({"status": "measured", "max": {"n_ctx": 131072}});
        let mut measuring = entry("measuring:Q4");
        measuring["memory"] = json!({"status": "measuring", "max": {"n_ctx": 4096}});
        let server = TestServer::start(vec![
            router(),
            TestResponse::json(
                200,
                json!({"data": [measured, measuring, entry("org/draft-dflash:Q8_0"), {"id": "no-status"}]}),
            ),
        ])
        .await;

        let inventory = client(&server).inventory().await.expect("inventory");

        assert_eq!(
            inventory
                .iter()
                .map(|item| (item.name.as_str(), item.fit_context))
                .collect::<Vec<_>>(),
            vec![("measured:Q4", 131072), ("measuring:Q4", 0)]
        );
    }
}
