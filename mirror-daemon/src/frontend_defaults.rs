//! Host-owned startup configuration; only a credential-free projection reaches frontends.

use mirror_protocol::daemon::{
    AgentConfiguration, ConfigureDaemonRequest, GetFrontendDefaultsResponse, WorkspacePreferences,
};
use mirror_protocol::mirror::AgentDefinition;
use tonic::Status;

/// Ephemeral provider inputs registered by the CLI running on this daemon's host.
#[derive(Default)]
pub struct FrontendDefaults {
    configured: Option<ConfigureDaemonRequest>,
}

impl FrontendDefaults {
    pub fn configure(&mut self, request: ConfigureDaemonRequest) -> Result<(), Status> {
        if request.configurations.is_empty() || request.configurations.len() > 128 {
            return Err(Status::invalid_argument(
                "expected 1 to 128 daemon configurations",
            ));
        }
        for configuration in &request.configurations {
            if configuration.inference.as_ref().is_none_or(|inference| {
                inference.model.is_empty() || inference.api.as_ref().is_none_or(String::is_empty)
            }) || !configuration.host_tools.is_empty()
            {
                return Err(Status::invalid_argument(
                    "daemon defaults require explicit inference and no frontend host tools",
                ));
            }
        }
        self.configured = Some(request);
        Ok(())
    }

    pub fn public(&self) -> Result<GetFrontendDefaultsResponse, Status> {
        let configured = self.configured()?;
        let configuration = &configured.configurations[0];
        let inference = configuration.inference.clone().unwrap_or_default();
        Ok(GetFrontendDefaultsResponse {
            directory: configured.directory.clone(),
            preferences: Some(WorkspacePreferences {
                approval_mode: Some(configuration.approval_mode),
                definition: Some(AgentDefinition {
                    model: Some(inference.model),
                    model_api: inference.api,
                    model_base_url: inference.base_url,
                    reasoning_effort: inference.reasoning_effort,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            models: configured
                .configurations
                .iter()
                .filter_map(|value| value.inference.clone())
                .collect(),
            sandbox: configured.sandbox.clone(),
        })
    }

    pub fn configuration(
        &self,
        preferences: Option<&WorkspacePreferences>,
    ) -> Result<AgentConfiguration, Status> {
        let configured = self.configured()?;
        let default = &configured.configurations[0];
        let mut inference = default.inference.clone().unwrap_or_default();
        let definition = preferences.and_then(|preferences| preferences.definition.as_ref());
        if let Some(definition) = definition {
            if let Some(model) = &definition.model {
                inference.model = model.clone();
            }
            if let Some(api) = &definition.model_api {
                inference.api = Some(api.clone());
            }
            if let Some(effort) = &definition.reasoning_effort {
                inference.reasoning_effort = Some(effort.clone());
            }
        }
        // An empty saved URL means the host's default for the requested API.
        let base_url = definition
            .and_then(|definition| definition.model_base_url.as_deref())
            .filter(|base_url| !base_url.is_empty());
        // Credentials are bound to their host-configured protocol and endpoint, never a new URL.
        let mut configuration = configured
            .configurations
            .iter()
            .find(|candidate| {
                candidate.inference.as_ref().is_some_and(|candidate| {
                    candidate.api == inference.api
                        && base_url
                            .is_none_or(|base_url| candidate.base_url.as_deref() == Some(base_url))
                })
            })
            .cloned()
            .ok_or_else(|| {
                Status::failed_precondition("provider is not configured on the daemon host")
            })?;
        inference.base_url = configuration
            .inference
            .as_ref()
            .and_then(|candidate| candidate.base_url.clone());
        configuration.inference = Some(inference);
        if let Some(mode) = preferences.and_then(|preferences| preferences.approval_mode) {
            configuration.approval_mode = mode;
        }
        if let Some(definition) = definition {
            if let Some(prompt) = &definition.system_prompt {
                configuration.system_prompt = Some(prompt.clone());
            }
            if let Some(tools) = &definition.tools_allowlist {
                configuration.tools = Some(tools.clone());
            }
        }
        Ok(configuration)
    }

    fn configured(&self) -> Result<&ConfigureDaemonRequest, Status> {
        self.configured.as_ref().ok_or_else(|| {
            Status::failed_precondition(
                "daemon defaults are unavailable; run mirror daemon address on its host",
            )
        })
    }
}
