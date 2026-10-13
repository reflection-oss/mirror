//! In-process AgentRPC frontend using the same session handler as gRPC.

use futures_util::StreamExt;
use mirror_protocol::ClientMessage;
use mirror_protocol::SessionMessage;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;

use super::AgentRpc;
use crate::ReactAgentServiceHandle;

/// An in-process AgentRPC frontend for a ReACT service.
pub struct AgentInProcessRpc {
    service: ReactAgentServiceHandle,
}

/// One connection to an in-process AgentRPC frontend.
pub struct AgentInProcessRpcConnection {
    rpc: AgentRpc,
    requests: Option<mpsc::Sender<Result<ClientMessage, Status>>>,
    responses: ReceiverStream<Result<SessionMessage, Status>>,
}

impl AgentInProcessRpc {
    /// Create a frontend without starting a network transport.
    pub fn new(service: ReactAgentServiceHandle) -> Self {
        Self { service }
    }

    /// Attach one protobuf connection to the service.
    pub async fn connect(&self) -> Result<AgentInProcessRpcConnection, Status> {
        let rpc = AgentRpc::new(self.service.clone());
        let (requests, receiver) = mpsc::channel(32);
        // Half-closing input must not cancel an active turn's output stream.
        let input = ReceiverStream::new(receiver).chain(futures_util::stream::pending());
        let responses = rpc.open_session(input).await?;
        Ok(AgentInProcessRpcConnection {
            rpc,
            requests: Some(requests),
            responses,
        })
    }
}

impl AgentInProcessRpcConnection {
    /// Send one protobuf client message.
    pub async fn send(&mut self, message: ClientMessage) -> Result<(), Status> {
        self.requests
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("agent session input closed"))?
            .send(Ok(message))
            .await
            .map_err(|_| Status::unavailable("agent session closed"))
    }

    /// Receive the next protobuf session message.
    pub async fn recv(&mut self) -> Result<Option<SessionMessage>, Status> {
        self.responses.next().await.transpose()
    }

    /// Half-close client input while allowing remaining output to drain.
    pub fn close_input(&mut self) {
        self.requests = None;
    }
}

impl Drop for AgentInProcessRpcConnection {
    fn drop(&mut self) {
        self.rpc.shutdown();
    }
}

#[cfg(test)]
#[path = "agent_in_process_rpc_test.rs"]
mod tests;
