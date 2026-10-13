//! Python session using the existing AgentRPC protobuf envelopes.

use std::collections::VecDeque;

use mirror_agent_core::rpc::AgentInProcessRpcConnection;
use mirror_protocol::ClientMessage;
use prost::Message;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_async_runtimes::tokio::future_into_py;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// One attached in-process AgentRPC session.
#[pyclass(name = "Session")]
pub struct PySession {
    commands: SessionSender,
}

pub(crate) type SessionSender = mpsc::UnboundedSender<SessionCommand>;
type SessionResult<T> = Result<T, String>;

pub(crate) enum SessionCommand {
    Send {
        message: Box<ClientMessage>,
        result: oneshot::Sender<SessionResult<()>>,
    },
    Receive(oneshot::Sender<SessionResult<Vec<u8>>>),
    Close(Option<oneshot::Sender<SessionResult<()>>>),
}

struct SessionActor {
    session: AgentInProcessRpcConnection,
    commands: mpsc::UnboundedReceiver<SessionCommand>,
    queued: VecDeque<Vec<u8>>,
    receivers: VecDeque<oneshot::Sender<SessionResult<Vec<u8>>>>,
}

impl PySession {
    pub(crate) fn new(commands: SessionSender) -> Self {
        Self { commands }
    }
}

#[pymethods]
impl PySession {
    /// Send one encoded `ClientMessage`.
    #[pyo3(signature = (message: "bytes") -> "collections.abc.Awaitable[None]")]
    fn send<'py>(
        &self,
        py: Python<'py>,
        message: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let message = ClientMessage::decode(message.as_bytes())
            .map_err(|error| PyValueError::new_err(format!("invalid ClientMessage: {error}")))?;
        let commands = self.commands.clone();
        future_into_py(py, async move {
            let (result, receiver) = oneshot::channel();
            send_command(
                &commands,
                SessionCommand::Send {
                    message: Box::new(message),
                    result,
                },
            )?;
            receive(receiver).await
        })
    }

    /// Receive one encoded `SessionMessage`.
    #[pyo3(signature = () -> "collections.abc.Awaitable[bytes]")]
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let commands = self.commands.clone();
        future_into_py(py, async move {
            let (result, receiver) = oneshot::channel();
            send_command(&commands, SessionCommand::Receive(result))?;
            let message = receive(receiver).await?;
            Ok(Python::attach(|py| PyBytes::new(py, &message).unbind()))
        })
    }

    /// Half-close client input while allowing remaining output to drain.
    #[pyo3(signature = () -> "collections.abc.Awaitable[None]")]
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let commands = self.commands.clone();
        future_into_py(py, async move {
            let (result, receiver) = oneshot::channel();
            if commands.send(SessionCommand::Close(Some(result))).is_err() {
                return Ok(());
            }
            receive(receiver).await
        })
    }
}

impl Drop for PySession {
    fn drop(&mut self) {
        let _ = self.commands.send(SessionCommand::Close(None));
    }
}

impl SessionActor {
    async fn run(mut self) {
        let mut commands_open = true;
        let mut output_open = true;
        loop {
            if !commands_open || (!output_open && self.queued.is_empty()) {
                break;
            }
            tokio::select! {
                command = self.commands.recv(), if commands_open => {
                    match command {
                        Some(command) => self.handle(command).await,
                        None => {
                            commands_open = false;
                            self.session.close_input();
                        }
                    }
                }
                output = self.session.recv(), if output_open => {
                    match output {
                        Ok(Some(message)) => self.deliver(message.encode_to_vec()),
                        Ok(None) => {
                            output_open = false;
                            self.fail_receivers("agent session closed".to_string());
                        }
                        Err(error) => {
                            self.fail_receivers(error.to_string());
                            return;
                        }
                    }
                }
            }
        }
        self.fail_receivers("agent session closed".to_string());
    }

    async fn handle(&mut self, command: SessionCommand) {
        match command {
            SessionCommand::Send { message, result } => {
                let response = self
                    .session
                    .send(*message)
                    .await
                    .map_err(|error| error.to_string());
                let _ = result.send(response);
            }
            SessionCommand::Receive(result) => {
                if let Some(message) = self.queued.pop_front() {
                    let _ = result.send(Ok(message));
                } else {
                    self.receivers.push_back(result);
                }
            }
            SessionCommand::Close(result) => {
                self.session.close_input();
                if let Some(result) = result {
                    let _ = result.send(Ok(()));
                }
            }
        }
    }

    fn deliver(&mut self, message: Vec<u8>) {
        let mut message = Some(message);
        while let Some(receiver) = self.receivers.pop_front() {
            let Some(next) = message.take() else {
                return;
            };
            match receiver.send(Ok(next)) {
                Ok(()) => return,
                Err(Ok(rejected)) => message = Some(rejected),
                Err(Err(_)) => return,
            }
        }
        if let Some(message) = message {
            self.queued.push_back(message);
        }
    }

    fn fail_receivers(&mut self, error: String) {
        for receiver in self.receivers.drain(..) {
            let _ = receiver.send(Err(error.clone()));
        }
    }
}

pub(crate) fn start_session(session: AgentInProcessRpcConnection) -> SessionSender {
    let (commands, receiver) = mpsc::unbounded_channel();
    tokio::task::spawn_local(
        SessionActor {
            session,
            commands: receiver,
            queued: VecDeque::new(),
            receivers: VecDeque::new(),
        }
        .run(),
    );
    commands
}

fn send_command(commands: &SessionSender, command: SessionCommand) -> PyResult<()> {
    commands
        .send(command)
        .map_err(|_| PyRuntimeError::new_err("agent session closed"))
}

async fn receive<T>(receiver: oneshot::Receiver<SessionResult<T>>) -> PyResult<T> {
    receiver
        .await
        .map_err(|_| PyRuntimeError::new_err("agent session closed"))?
        .map_err(PyRuntimeError::new_err)
}
