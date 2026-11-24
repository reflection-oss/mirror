//! Resolve daemon-bound references at the native inference boundary.

use std::time::Duration;

use mirror_protocol::local_models::{execute_request::Operation, ExecuteRequest};
use tonic::Status;
use uuid::Uuid;

use super::host::connect;
use super::host_state::{MANAGED, MANAGED_SCHEME};
use crate::providers::{LlamaCpp, LlmError};

/// The llama-server endpoint `base` names.
///
/// `managed` is reached through `reference`, the managed server's reference the worker
/// started with, by asking the daemon that issued it; anything else is a loopback address.
pub async fn resolve_endpoint(
    base: Option<&str>,
    reference: Option<&str>,
) -> Result<String, LlmError> {
    if base != Some(MANAGED) {
        return LlamaCpp::base_url(base);
    }
    let reference = reference.ok_or_else(|| {
        LlmError::Request("Mirror's llama.cpp server is not available to this session".into())
    })?;
    let url = reqwest::Url::parse(reference)
        .map_err(|_| LlmError::Request("Invalid managed model reference".into()))?;
    if !reference.starts_with(MANAGED_SCHEME)
        || url.host_str() != Some("127.0.0.1")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || Uuid::parse_str(url.path().trim_start_matches('/')).is_err()
    {
        return Err(LlmError::Request("Invalid managed model reference".into()));
    }
    let port = url
        .port()
        .ok_or_else(|| LlmError::Request("Missing managed control port".into()))?;
    let resolve = async {
        let mut client = connect(port).await?;
        let response = client
            .execute(ExecuteRequest {
                base_url: reference.into(),
                operation: Some(Operation::Status(())),
            })
            .await?
            .into_inner();
        if !response.error.is_empty() {
            return Err(Status::failed_precondition(response.error));
        }
        // An empty endpoint would select the default llama.cpp address instead.
        if response.endpoint.is_empty() {
            return Err(Status::failed_precondition(
                "The managed server reported no endpoint",
            ));
        }
        LlamaCpp::base_url(Some(&response.endpoint))
            .map_err(|error| Status::failed_precondition(error.to_string()))
    };
    tokio::time::timeout(Duration::from_secs(5), resolve)
        .await
        .map_err(|_| LlmError::Request("Local runtime timed out; no fallback".into()))?
        .map_err(|error| {
            LlmError::Request(format!(
                "{} ({reference}); no fallback was selected",
                error.message()
            ))
        })
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;

    use assert_matches::assert_matches;
    use futures_core::Stream;
    use mirror_protocol::local_models::local_models_service_server::{
        LocalModelsService, LocalModelsServiceServer,
    };
    use mirror_protocol::local_models::{ExecuteResponse, WatchRequest, WatchResponse};
    use tonic::{Request, Response};

    use super::*;

    /// A control service that answers every request with an empty reply.
    struct Silent;

    #[tonic::async_trait]
    impl LocalModelsService for Silent {
        type WatchStream = Pin<Box<dyn Stream<Item = Result<WatchResponse, Status>> + Send>>;

        async fn execute(
            &self,
            _request: Request<ExecuteRequest>,
        ) -> Result<Response<ExecuteResponse>, Status> {
            Ok(Response::new(ExecuteResponse::default()))
        }

        async fn watch(
            &self,
            _request: Request<WatchRequest>,
        ) -> Result<Response<Self::WatchStream>, Status> {
            Err(Status::unimplemented("watch"))
        }
    }

    #[tokio::test]
    async fn a_reply_without_an_endpoint_never_falls_back_to_the_default_server() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("control port");
        let port = listener.local_addr().expect("address").port();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(LocalModelsServiceServer::new(Silent))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        let reference = format!("managed://127.0.0.1:{port}/{}", Uuid::new_v4().simple());

        let endpoint = resolve_endpoint(Some(MANAGED), Some(&reference)).await;

        assert_matches!(
            endpoint,
            Err(LlmError::Request(message)) if message.starts_with("The managed server reported no endpoint")
        );
    }
}
