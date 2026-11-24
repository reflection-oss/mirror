//! Ephemeral, isolated execution of untrusted OCI images over stdio.

use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncRead, AsyncWrite};
use uuid::Uuid;

use crate::SandboxProviderError;

/// Providers must keep the image stopped until `connect` and enforce isolation before starting it.
#[async_trait]
pub trait ImageSandboxProvider: Send + Sync {
    async fn prepare(
        &self,
        id: Uuid,
        request: &ImageSandboxRequest,
    ) -> Result<(), SandboxProviderError>;
    async fn connect(
        &self,
        id: Uuid,
        request: &ImageSandboxRequest,
    ) -> Result<ImageSandboxStream, SandboxProviderError>;
    /// Revoke proxy capabilities when the operation is deleted or no longer executing.
    async fn active(&self, id: Uuid) -> Result<bool, SandboxProviderError>;
    /// Delete by the caller's stable ID even after an ambiguous `prepare` failure.
    async fn delete(&self, id: Uuid) -> Result<(), SandboxProviderError>;
}

#[derive(Clone, Debug)]
pub struct ImageSandboxRequest {
    pub image: String,
    /// Empty uses the image's entrypoint and arguments.
    pub command: Vec<String>,
    pub lifetime: Duration,
    /// Publication-only capability for the deployment-controlled proxy.
    pub egress: Option<ImageSandboxEgress>,
}

#[derive(Clone)]
pub struct ImageSandboxEgress {
    pub token: String,
}

impl std::fmt::Debug for ImageSandboxEgress {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImageSandboxEgress")
            .finish_non_exhaustive()
    }
}

pub const MAX_IMAGE_SANDBOX_LIFETIME: Duration = Duration::from_secs(900);

pub type ImageSandboxStream = (
    Box<dyn AsyncRead + Send + Unpin>,
    Box<dyn AsyncWrite + Send + Unpin>,
);

/// Require an explicit public registry and immutable digest; never ask the node to pull arbitrary URLs.
pub fn validate_sandbox_image(image: &str) -> Result<(), &'static str> {
    validate_oci_image(image)?;
    if !image
        .split_once('/')
        .is_some_and(|(registry, _)| matches!(registry, "ghcr.io" | "docker.io"))
    {
        return Err(
            "use a public image on ghcr.io or docker.io, or an owned private image built by Mirror",
        );
    }
    Ok(())
}

/// Validate syntax without granting permission to pull from a registry.
pub fn validate_oci_image(image: &str) -> Result<(), &'static str> {
    let (repository, digest) = image
        .split_once("@sha256:")
        .ok_or("image must use an OCI sha256 digest")?;
    if image.len() > 512
        || digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("image must use a lowercase 64-digit sha256 digest");
    }
    let (registry, path) = repository
        .split_once('/')
        .ok_or("image must include its registry and repository")?;
    let (host, port) = registry.split_once(':').unwrap_or((registry, "443"));
    if host.split('.').any(|label| {
        label.is_empty()
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            || label.starts_with('-')
            || label.ends_with('-')
    }) || port.parse::<u16>().map_or(true, |port| port == 0)
    {
        return Err("image registry must be a DNS name or IPv4 address with an optional port");
    }
    if path.is_empty()
        || path.split('/').any(|part| {
            part.is_empty()
                || !part.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'_' | b'-' | b'.')
                })
                || !part.as_bytes()[0].is_ascii_alphanumeric()
                || !part.as_bytes()[part.len() - 1].is_ascii_alphanumeric()
        })
    {
        return Err("image repository must be a lowercase OCI repository without a tag");
    }
    Ok(())
}
