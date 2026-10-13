use crate::execution_runtime::ExecutionCommand;
use mirror_sandbox::{egress_update, EgressEvent, EgressUpdate};
use mirror_sandbox::{
    sandboxd_server::Sandboxd, CancelRequest, CancelResponse, ExecuteEvent, ExecuteRequest,
};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status};

#[derive(Clone)]
pub(crate) struct SandboxdService {
    pub(crate) commands: mpsc::Sender<ExecutionCommand>,
    pub(crate) egress: crate::egress_runtime::GuestEgress,
    pub(crate) closing: tokio::sync::watch::Receiver<bool>,
}
#[tonic::async_trait]
impl Sandboxd for SandboxdService {
    type EgressStream = ReceiverStream<Result<EgressEvent, Status>>;

    async fn egress(
        &self,
        request: Request<tonic::Streaming<EgressUpdate>>,
    ) -> Result<Response<Self::EgressStream>, Status> {
        let mut updates = request.into_inner();
        let update = tokio::time::timeout(std::time::Duration::from_secs(5), updates.message())
            .await
            .map_err(|_| Status::deadline_exceeded("egress configuration timed out"))??;
        let Some(egress_update::Update::Configure(config)) =
            update.and_then(|update| update.update)
        else {
            return Err(Status::invalid_argument(
                "egress requires initial configuration",
            ));
        };
        let connector = Arc::new(
            crate::egress_connector::EgressConnector::start(
                config,
                std::path::Path::new(crate::java_truststore::JAVA_TRUSTSTORE),
            )
            .await
            .map_err(|_| Status::invalid_argument("invalid egress transport configuration"))?,
        );
        self.egress
            .set(connector.clone())
            .map_err(|_| Status::already_exists("egress is already configured"))?;
        let (events, receiver) = mpsc::channel(2);
        tokio::spawn(crate::egress_runtime::renew(
            updates,
            events,
            connector,
            self.closing.clone(),
        ));
        Ok(Response::new(ReceiverStream::new(receiver)))
    }

    type ExecuteStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<ExecuteEvent, Status>> + Send>>;
    async fn execute(
        &self,
        request: Request<ExecuteRequest>,
    ) -> Result<Response<Self::ExecuteStream>, Status> {
        let (events, receiver) = mpsc::channel(2);
        self.commands
            .send(ExecutionCommand::Execute {
                request: request.into_inner(),
                events,
            })
            .await
            .map_err(|_| Status::unavailable("sandboxd execution stopped"))?;
        Ok(Response::new(Box::pin(
            ReceiverStream::new(receiver).map(Ok),
        )))
    }
    async fn cancel(
        &self,
        request: Request<CancelRequest>,
    ) -> Result<Response<CancelResponse>, Status> {
        self.commands
            .send(ExecutionCommand::Cancel(request.into_inner().call_id))
            .await
            .map_err(|_| Status::unavailable("sandboxd execution stopped"))?;
        Ok(Response::new(CancelResponse {}))
    }
}
