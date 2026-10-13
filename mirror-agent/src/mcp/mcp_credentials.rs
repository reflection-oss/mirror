use std::borrow::Cow;

use rmcp::transport::auth::{AuthError, AuthorizationManager, CredentialStore, OAuthClientConfig};

use super::valid_mcp_url;

/// Resolves authentication without exposing runtime-specific credential storage.
#[async_trait::async_trait]
pub trait McpCredentials: Send + Sync {
    /// Runtime-specific credential lookup failure.
    type Error;

    /// Check access and resolve authentication for a configured endpoint.
    async fn authorize(&self, url: &str) -> Result<McpAuthorization, Self::Error>;
}

/// Resolved authentication material, never serialized or included in diagnostics.
#[non_exhaustive]
pub enum McpAuthorization {
    None,
    Bearer(String),
    OAuth(Box<AuthorizationManager>),
}

impl McpAuthorization {
    /// Load OAuth authentication from a runtime-owned credential store.
    pub async fn oauth(
        url: &str,
        store: impl CredentialStore + 'static,
        config: OAuthClientConfig,
    ) -> Result<Self, AuthError> {
        if !valid_mcp_url(url) {
            return Err(AuthError::AuthorizationFailed("Invalid MCP URL".into()));
        }
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut manager = AuthorizationManager::new(url).await?;
        manager.set_credential_store(store);
        if !manager.initialize_from_store().await? {
            return Err(AuthError::AuthorizationRequired);
        }
        manager.configure_client(config)?;
        Ok(Self::OAuth(Box::new(manager)))
    }

    /// Identify the current credential for connection reuse, not for diagnostics.
    pub async fn fingerprint(&self) -> Result<Option<[u8; 32]>, AuthError> {
        let token = match self {
            Self::None => return Ok(None),
            Self::Bearer(token) => Cow::Borrowed(token.as_str()),
            Self::OAuth(manager) => Cow::Owned(manager.get_access_token().await?),
        };
        let digest = ring::digest::digest(&ring::digest::SHA256, token.as_bytes());
        let mut fingerprint = [0; 32];
        fingerprint.copy_from_slice(digest.as_ref());
        Ok(Some(fingerprint))
    }
}
