//! The check before each llama.cpp request that its server is still the one selected.

use futures_util::future::BoxFuture;

use super::{LlmError, ModelRequestGuard};

/// Verifies, before each request, that the managed server is still at `endpoint`.
pub struct LocalRequestGuard<'a> {
    /// The selection's `base_url`, which may name the managed server.
    pub base: Option<&'a str>,
    /// The managed server's reference the worker started with.
    pub reference: Option<&'a str>,
    /// Where the request is about to go.
    pub endpoint: &'a str,
    pub upstream: Option<&'a dyn ModelRequestGuard>,
}

impl ModelRequestGuard for LocalRequestGuard<'_> {
    fn verify(&self) -> BoxFuture<'_, Result<(), LlmError>> {
        Box::pin(async {
            if let Some(upstream) = self.upstream {
                upstream.verify().await?;
            }
            // A restarted server listens on another port, which another process may now hold.
            if crate::local_models::resolve_endpoint(self.base, self.reference).await?
                != self.endpoint
            {
                return Err(LlmError::Request(
                    "llama.cpp: the server restarted during the request; send it again".into(),
                ));
            }
            Ok(())
        })
    }
}
