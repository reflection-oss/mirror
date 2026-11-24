//! RPC adapter and native startup access to the host-local model actor.

use std::pin::Pin;
use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt;
use mirror_protocol::local_models::{
    execute_request::Operation, local_models_service_client::LocalModelsServiceClient,
    local_models_service_server::LocalModelsService, ExecuteRequest, ExecuteResponse,
    LocalModelServerPhase, LocalModelState, WatchRequest, WatchResponse,
};
use mirror_protocol::InferenceConfig;
use tokio::sync::{mpsc, oneshot};
use tonic::{Request, Response, Status};

use super::host_state::{Access, ServerRef, MANAGED};
use super::reads::Reads;
use super::runtime::{Command, CommandRequest};
use super::LocalModelError;
use crate::providers::Eviction;

/// Cloneable service access; dropping a client never stops the host actor.
#[derive(Clone, Debug)]
pub struct Service {
    backend: Backend,
}

#[derive(Clone, Debug)]
enum Backend {
    Local(Local),
    Peer(LocalModelsServiceClient<tonic::transport::Channel>),
}

/// The host actor and the readers beside it, as one caller sees them.
#[derive(Clone, Debug)]
struct Local {
    commands: mpsc::Sender<CommandRequest>,
    reads: Reads,
    access: Access,
}

type StateStream = Pin<Box<dyn Stream<Item = Result<WatchResponse, Status>> + Send>>;

/// How long preparing waits for a server that is starting or restarting: the longest delay
/// before a restart, then a start.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(120);

impl Service {
    pub(super) fn local(
        commands: mpsc::Sender<CommandRequest>,
        reads: Reads,
        access: Access,
    ) -> Self {
        Self {
            backend: Backend::Local(Local {
                commands,
                reads,
                access,
            }),
        }
    }

    pub(super) fn peer(client: LocalModelsServiceClient<tonic::transport::Channel>) -> Self {
        Self {
            backend: Backend::Peer(client),
        }
    }

    /// The managed server's reference for the workers this daemon starts; empty when the
    /// owner cannot be reached.
    pub async fn reference(&self) -> String {
        let status = ExecuteRequest {
            operation: Some(Operation::Status(())),
            ..Default::default()
        };
        self.call(status)
            .await
            .ok()
            .and_then(|reply| reply.state)
            .map(|state| state.managed_reference)
            .unwrap_or_default()
    }

    /// Start the managed server for a new native worker that selected it, unless it is
    /// already running, and wait for a server on its way up, so the worker finds its endpoint.
    ///
    /// Returns the selected model when the server is not already loading or serving it. The
    /// caller loads it with `restore_in_background` once the worker is up, so opening a
    /// session never waits for a model to fill memory, nor for a load or restore in progress.
    pub async fn prepare(
        &self,
        inference: &InferenceConfig,
    ) -> Result<Option<String>, LocalModelError> {
        if inference.api.as_deref() != Some("llama_cpp")
            || inference.base_url.as_deref() != Some(MANAGED)
        {
            return Ok(None);
        }
        let mut state = self.settled().await?;
        if state.phase == i32::from(LocalModelServerPhase::Stopped) {
            state = self
                .request(Operation::Start(()))
                .await?
                .state
                .unwrap_or_default();
        }
        Ok((state.active_model != inference.model).then(|| inference.model.clone()))
    }

    /// The managed server's state once it is neither starting nor restarting.
    ///
    /// The state is watched beside the actor, so this never waits behind a running command.
    async fn settled(&self) -> Result<LocalModelState, LocalModelError> {
        let mut states = self
            .watch(Request::new(WatchRequest {}))
            .await?
            .into_inner();
        let settled = async {
            while let Some(response) = states.next().await {
                let state = response?.state.unwrap_or_default();
                if !matches!(
                    LocalModelServerPhase::try_from(state.phase),
                    Ok(LocalModelServerPhase::Starting | LocalModelServerPhase::Restarting)
                ) {
                    return Ok(state);
                }
            }
            Err(LocalModelError::Stopped)
        };
        tokio::time::timeout(SETTLE_TIMEOUT, settled)
            .await
            .map_err(|_| LocalModelError::StartTimedOut)?
    }

    /// Load a saved model without holding anything up; a failure leaves the model unloaded
    /// and is published as the state's notice.
    pub fn restore_in_background(&self, model: String) {
        let service = self.clone();
        tokio::spawn(async move {
            let _ = service.request(Operation::Restore(model)).await;
        });
    }

    /// Run `operation` on the managed server, failing with the error the reply names.
    async fn request(&self, operation: Operation) -> Result<ExecuteResponse, LocalModelError> {
        let reply = self
            .call(ExecuteRequest {
                operation: Some(operation),
                ..Default::default()
            })
            .await?;
        if reply.error.is_empty() {
            Ok(reply)
        } else {
            Err(LocalModelError::Rejected(reply.error))
        }
    }

    async fn call(&self, request: ExecuteRequest) -> Result<ExecuteResponse, Status> {
        match &self.backend {
            Backend::Local(local) => Ok(local.execute(request).await),
            Backend::Peer(client) => Ok(client.clone().execute(request).await?.into_inner()),
        }
    }
}

impl Local {
    /// Answer reads here and send everything else to the actor; the reply always has the state.
    async fn execute(&self, request: ExecuteRequest) -> ExecuteResponse {
        let outcome = match self.dispatch(request).await {
            Ok(reply) => reply,
            Err(error) => ExecuteResponse {
                error: error.to_string(),
                ..Default::default()
            },
        };
        ExecuteResponse {
            state: Some(self.state()),
            ..outcome
        }
    }

    fn state(&self) -> LocalModelState {
        self.reads.state().borrow().view(self.access)
    }

    async fn dispatch(&self, request: ExecuteRequest) -> Result<ExecuteResponse, LocalModelError> {
        let reference = ServerRef::parse(&request.base_url);
        let operation = request.operation.ok_or(LocalModelError::MissingOperation)?;
        let command = match operation {
            Operation::Status(()) => return Ok(self.reads.status(reference, self.access)),
            Operation::Inventory(()) => {
                let models = self.reads.inventory(reference).await?;
                return Ok(ExecuteResponse {
                    models,
                    ..Default::default()
                });
            }
            Operation::Info(model) => {
                let info = Some(self.reads.info(reference, &model).await?);
                return Ok(ExecuteResponse {
                    info,
                    ..Default::default()
                });
            }
            Operation::Catalog(query) => {
                let candidates = self.reads.search(&query, reference).await?;
                return Ok(ExecuteResponse {
                    candidates,
                    ..Default::default()
                });
            }
            Operation::Variants(repository) => {
                let candidates = self.reads.variants(&repository, reference).await?;
                return Ok(ExecuteResponse {
                    candidates,
                    ..Default::default()
                });
            }
            Operation::Review(model) => {
                let download = Some(self.reads.review(&model).await?);
                return Ok(ExecuteResponse {
                    download,
                    ..Default::default()
                });
            }
            Operation::Start(()) => Command::Start,
            Operation::Stop(()) => Command::Stop,
            Operation::Restart(()) => Command::Restart,
            Operation::Context(context) => Command::Context(context),
            Operation::SleepIdle(seconds) => Command::SleepIdle(seconds),
            Operation::CancelDownload(name) => Command::CancelDownload(name),
            Operation::Load(load) => Command::Load {
                model: load.model,
                eviction: if load.allow_eviction {
                    Eviction::Allowed
                } else {
                    Eviction::Forbidden
                },
                base_url: request.base_url.clone(),
            },
            Operation::Restore(model) => Command::Restore(model),
            Operation::Download(review) => Command::Download {
                review,
                base_url: request.base_url.clone(),
            },
        };
        if matches!(command, Command::Restore(_)) && reference != ServerRef::Managed {
            return Err(LocalModelError::RestoreElsewhere);
        }
        if command.owner_only() && self.access == Access::Peer {
            return Err(LocalModelError::OwnerOnly);
        }
        self.run(command).await
    }

    async fn run(&self, command: Command) -> Result<ExecuteResponse, LocalModelError> {
        let (result, outcome) = oneshot::channel();
        self.commands
            .send(CommandRequest { command, result })
            .await
            .map_err(|_| LocalModelError::Stopped)?;
        let info = outcome.await.map_err(|_| LocalModelError::Stopped)??;
        Ok(ExecuteResponse {
            info,
            ..Default::default()
        })
    }
}

#[tonic::async_trait]
impl LocalModelsService for Service {
    type WatchStream = StateStream;

    async fn execute(
        &self,
        request: Request<ExecuteRequest>,
    ) -> Result<Response<ExecuteResponse>, Status> {
        Ok(Response::new(self.call(request.into_inner()).await?))
    }

    async fn watch(&self, request: Request<WatchRequest>) -> Result<Response<StateStream>, Status> {
        match &self.backend {
            Backend::Peer(client) => Ok(Response::new(Box::pin(
                client.clone().watch(request).await?.into_inner(),
            ))),
            Backend::Local(local) => {
                let mut state = local.reads.state().clone();
                let access = local.access;
                let stream = async_stream::try_stream! {
                    loop {
                        let snapshot = state.borrow_and_update().clone();
                        let mut alive = snapshot.alive.clone();
                        let running = *alive.borrow_and_update();
                        let running = running && alive.has_changed().is_ok();
                        yield WatchResponse { state: Some(snapshot.view(access)) };
                        tokio::select! {
                            changed = state.changed() => if changed.is_err() { break; },
                            _ = alive.changed(), if running => {},
                        }
                    }
                };
                Ok(Response::new(Box::pin(stream)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_models::host_state::HostState;
    use crate::local_models::resolve_endpoint;
    use crate::local_models::Catalog;
    use crate::providers::{LlmError, LocalRequestGuard, ModelRequestGuard};
    use assert_matches::assert_matches;
    use mirror_protocol::local_models::local_models_service_server::LocalModelsServiceServer;
    use pretty_assertions::assert_eq;
    use std::sync::Arc;
    use tokio::sync::watch;

    /// The senders behind a test service's server: keep them alive.
    struct Server {
        _alive: watch::Sender<bool>,
        state: watch::Sender<HostState>,
    }

    /// A service whose server is up with `state`.
    fn service(
        access: Access,
        state: LocalModelState,
    ) -> (Service, mpsc::Receiver<CommandRequest>, Server) {
        let (commands, requests) = mpsc::channel(4);
        let (alive, lifetime) = watch::channel(true);
        let (sender, state) = watch::channel(HostState {
            state,
            endpoint: "http://127.0.0.1:1/v1".into(),
            alive: lifetime,
        });
        let reads = Reads::new(Arc::new(Catalog::new().expect("catalog")), state);
        let server = Server {
            _alive: alive,
            state: sender,
        };
        (Service::local(commands, reads, access), requests, server)
    }

    fn managed() -> InferenceConfig {
        InferenceConfig {
            model: "synthetic".into(),
            api: Some("llama_cpp".into()),
            base_url: Some(MANAGED.into()),
            ..Default::default()
        }
    }

    /// Answer the next command as the actor would; the state in the reply is the service's own.
    async fn answer(
        requests: &mut mpsc::Receiver<CommandRequest>,
        outcome: Result<(), LocalModelError>,
    ) -> Command {
        let request = requests.recv().await.expect("command");
        request.result.send(outcome.map(|()| None)).expect("reply");
        request.command
    }

    fn phase(phase: LocalModelServerPhase, active_model: &str) -> LocalModelState {
        LocalModelState {
            phase: phase.into(),
            active_model: active_model.into(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn preparing_starts_a_stopped_server_and_names_the_model_to_restore() {
        let stopped = phase(LocalModelServerPhase::Stopped, "");
        let (service, mut requests, _server) = service(Access::Owner, stopped);
        let inference = managed();

        let (prepared, command) =
            tokio::join!(service.prepare(&inference), answer(&mut requests, Ok(())));

        assert_matches!(command, Command::Start);
        assert_eq!(prepared.expect("prepared"), Some("synthetic".to_string()));
    }

    #[tokio::test]
    async fn a_server_already_serving_the_model_needs_no_start_or_restore() {
        let serving = phase(LocalModelServerPhase::Serving, "synthetic");
        let (service, mut requests, _server) = service(Access::Owner, serving);

        let prepared = service.prepare(&managed()).await;

        assert_matches!(prepared, Ok(None));
        assert_matches!(requests.try_recv(), Err(_), "nothing reached the actor");
    }

    #[tokio::test]
    async fn a_load_in_progress_does_not_hold_up_preparing() {
        // Nothing answers the actor's channel, as while it runs another model's load.
        let loading = phase(LocalModelServerPhase::Loading, "other/m:Q4");
        let (service, mut requests, _server) = service(Access::Owner, loading);

        let prepared = service.prepare(&managed()).await;

        assert_matches!(prepared, Ok(Some(model)) if model == "synthetic");
        assert_matches!(requests.try_recv(), Err(_), "nothing reached the actor");
    }

    #[tokio::test]
    async fn preparing_waits_for_a_restarting_server_to_come_back() {
        let restarting = phase(LocalModelServerPhase::Restarting, "synthetic");
        let (service, mut requests, server) = service(Access::Owner, restarting);
        let inference = managed();
        let prepared = service.prepare(&inference);
        tokio::pin!(prepared);

        assert_matches!(futures_util::poll!(&mut prepared), std::task::Poll::Pending);
        server.state.send_modify(|host| {
            host.state = phase(LocalModelServerPhase::Serving, "synthetic");
        });

        assert_matches!(prepared.await, Ok(None));
        assert_matches!(requests.try_recv(), Err(_), "nothing reached the actor");
    }

    #[tokio::test]
    async fn a_failed_server_start_fails_preparing() {
        let stopped = phase(LocalModelServerPhase::Stopped, "");
        let (service, mut requests, _server) = service(Access::Owner, stopped);
        let inference = managed();

        let (result, _) = tokio::join!(
            service.prepare(&inference),
            answer(&mut requests, Err(LocalModelError::NotInstalled))
        );

        assert_matches!(
            result,
            Err(LocalModelError::Rejected(error)) if error == LocalModelError::NotInstalled.to_string()
        );
    }

    #[tokio::test]
    async fn a_saved_managed_selection_resolves_through_the_reference_the_worker_started_with() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("control port");
        let port = listener.local_addr().expect("address").port();
        let reference = format!(
            "managed://127.0.0.1:{port}/{}",
            uuid::Uuid::new_v4().simple()
        );
        let state = LocalModelState {
            managed_reference: reference.clone(),
            ..Default::default()
        };
        let (service, _requests, _server) = service(Access::Peer, state);
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(LocalModelsServiceServer::new(service))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        let saved = managed();

        let endpoint = resolve_endpoint(saved.base_url.as_deref(), Some(&reference)).await;

        assert_eq!(endpoint.expect("endpoint"), "http://127.0.0.1:1/v1");
    }

    #[tokio::test]
    async fn a_request_guard_fails_once_its_reference_resolves_to_another_endpoint() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("control port");
        let port = listener.local_addr().expect("address").port();
        let reference = format!(
            "managed://127.0.0.1:{port}/{}",
            uuid::Uuid::new_v4().simple()
        );
        let (_running, alive) = watch::channel(true);
        let (published, state) = watch::channel(HostState {
            state: LocalModelState {
                managed_reference: reference.clone(),
                ..Default::default()
            },
            endpoint: "http://127.0.0.1:1/v1".into(),
            alive,
        });
        let catalog = Arc::new(Catalog::new().expect("catalog"));
        let service = Service::local(mpsc::channel(1).0, Reads::new(catalog, state), Access::Peer);
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(LocalModelsServiceServer::new(service))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        let guard = LocalRequestGuard {
            base: Some(MANAGED),
            reference: Some(&reference),
            endpoint: "http://127.0.0.1:1/v1",
            upstream: None,
        };
        guard.verify().await.expect("the server has not moved");

        published.send_modify(|host| host.endpoint = "http://127.0.0.1:2/v1".into());

        assert_matches!(
            guard.verify().await,
            Err(LlmError::Request(message)) if message.contains("restarted during the request")
        );
    }

    #[tokio::test]
    async fn the_saved_model_is_restored_in_the_background() {
        let (service, mut requests, _server) = service(Access::Owner, LocalModelState::default());

        service.restore_in_background("synthetic".into());

        assert_matches!(answer(&mut requests, Ok(())).await, Command::Restore(model) if model == "synthetic");
    }

    #[tokio::test]
    async fn restoring_acts_on_the_managed_server_only() {
        let (service, mut requests, _server) = service(Access::Owner, LocalModelState::default());

        let elsewhere = service
            .call(ExecuteRequest {
                base_url: "http://127.0.0.1:9".into(),
                operation: Some(Operation::Restore("org/m:Q4".into())),
            })
            .await
            .expect("reply");

        assert_eq!(
            elsewhere.error,
            LocalModelError::RestoreElsewhere.to_string()
        );
        assert_matches!(requests.try_recv(), Err(_), "nothing reached the actor");
    }

    #[tokio::test]
    async fn a_load_may_name_another_loopback_server() {
        let (service, mut requests, _server) = service(Access::Owner, LocalModelState::default());
        let load = service.call(ExecuteRequest {
            base_url: "http://127.0.0.1:9/v1".into(),
            operation: Some(Operation::Load(
                mirror_protocol::local_models::LocalModelLoad {
                    model: "org/m:Q4".into(),
                    allow_eviction: false,
                },
            )),
        });

        let (reply, command) = tokio::join!(load, answer(&mut requests, Ok(())));

        assert_eq!(reply.expect("reply").error, "");
        assert_matches!(command, Command::Load { base_url, .. } if base_url == "http://127.0.0.1:9/v1");
    }
}
