//! Explicit web-provider selection and adapter construction.

use std::str::FromStr;

use serde::Deserialize;

use super::{
    FirecrawlClient, FirecrawlError, TavilyClient, TavilyError, WebError, WebProvider, YouClient,
    YouError,
};

/// User-selected retrieval backend, independent of the model provider.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum WebProviderConfig {
    #[default]
    You,
    Tavily,
    Firecrawl,
}

/// Request-time adapter construction supplied by local or hosted tool assembly.
pub type WebProviderFactory =
    dyn Fn(Option<WebProviderConfig>) -> Result<Box<dyn WebProvider>, WebError> + Send + Sync;

impl WebProviderConfig {
    /// Environment setting read by launchers, never by workspace tools.
    pub fn key_env(self) -> &'static str {
        match self {
            Self::You => "YOU_API_KEY",
            Self::Tavily => "TAVILY_API_KEY",
            Self::Firecrawl => "FIRECRAWL_API_KEY",
        }
    }
}

impl FromStr for WebProviderConfig {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "you" => Ok(Self::You),
            "tavily" => Ok(Self::Tavily),
            "firecrawl" => Ok(Self::Firecrawl),
            _ => Err("web provider must be you, tavily, or firecrawl"),
        }
    }
}

/// Construct a backend; omission preserves legacy You.com missing-key behavior.
pub fn web_provider_from_config(
    selection: Option<WebProviderConfig>,
    api_key: Option<String>,
) -> Result<Box<dyn WebProvider>, WebError> {
    match selection {
        None => Ok(Box::new(YouClient::from_api_key(api_key)?)),
        Some(WebProviderConfig::You) => {
            let client = YouClient::from_api_key(api_key)?;
            if !client.is_configured() {
                return Err(YouError::MissingApiKey.into());
            }
            Ok(Box::new(client))
        }
        Some(WebProviderConfig::Tavily) => {
            let client = TavilyClient::from_api_key(api_key)?;
            if !client.is_configured() {
                return Err(TavilyError::MissingApiKey.into());
            }
            Ok(Box::new(client))
        }
        Some(WebProviderConfig::Firecrawl) => {
            let client = FirecrawlClient::from_api_key(api_key)?;
            if !client.is_configured() {
                return Err(FirecrawlError::MissingApiKey.into());
            }
            Ok(Box::new(client))
        }
    }
}
