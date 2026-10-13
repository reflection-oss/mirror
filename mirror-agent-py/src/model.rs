//! Scripted Python-configured model for deterministic in-process replay.

use std::collections::VecDeque;
use std::sync::mpsc;

use futures_util::stream;
use mirror_agent_core::providers::{Completion, CompletionRequest};
use mirror_agent_core::providers::{Llm, LlmError, StreamEvent};
use mirror_protocol::AssistantMessage;
use prost::Message;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use tokio::sync::oneshot;

use crate::python_json::PythonJson;

/// A deterministic model that returns pre-recorded assistant messages.
#[pyclass(name = "ScriptedModel", frozen)]
pub struct PyScriptedModel {
    model: ScriptedModel,
}

#[derive(Clone)]
pub(crate) struct ScriptedModel {
    name: String,
    commands: mpsc::Sender<ScriptCommand>,
}

enum ScriptCommand {
    Complete {
        request: Box<CompletionRequest>,
        reply: oneshot::Sender<Result<AssistantMessage, LlmError>>,
    },
    Snapshot(mpsc::SyncSender<Vec<CompletionRequest>>),
}

struct ScriptState {
    responses: VecDeque<AssistantMessage>,
    request_deltas: Vec<CompletionRequest>,
    request_messages: Vec<mirror_protocol::chat_record::Message>,
}

#[pymethods]
impl PyScriptedModel {
    #[new]
    #[pyo3(signature = (
        model_name,
        responses: "collections.abc.Sequence[bytes]",
    ) -> "ScriptedModel")]
    fn new(model_name: String, responses: Vec<Vec<u8>>) -> PyResult<Self> {
        let responses = responses
            .into_iter()
            .map(|encoded| {
                AssistantMessage::decode(encoded.as_slice())
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .collect::<PyResult<VecDeque<_>>>()?;
        let (commands, receiver) = mpsc::channel();
        // Python can construct a script before the service's Tokio runtime exists.
        std::thread::Builder::new()
            .name("mirror-replay-model".to_string())
            .spawn(move || {
                ScriptState {
                    responses,
                    request_deltas: Vec::new(),
                    request_messages: Vec::new(),
                }
                .run(receiver);
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            model: ScriptedModel {
                name: model_name,
                commands,
            },
        })
    }

    /// Return each request's suffix relative to the preceding request.
    #[pyo3(signature = () -> "list[dict[str, object]]")]
    fn request_deltas(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let (reply, response) = mpsc::sync_channel(1);
        self.model
            .commands
            .send(ScriptCommand::Snapshot(reply))
            .map_err(|_| PyRuntimeError::new_err("scripted model closed"))?;
        response
            .recv()
            .map_err(|_| PyRuntimeError::new_err("scripted model closed"))?
            .iter()
            .map(|request| {
                let Value::Object(value) = serde_json::to_value(request)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
                else {
                    return Err(PyRuntimeError::new_err(
                        "completion request did not serialize as an object",
                    ));
                };
                PythonJson::to_dict(py, &value)
            })
            .collect()
    }
}

impl PyScriptedModel {
    pub(crate) fn runtime_model(&self) -> ScriptedModel {
        self.model.clone()
    }
}

impl ScriptState {
    fn run(mut self, commands: mpsc::Receiver<ScriptCommand>) {
        for command in commands {
            match command {
                ScriptCommand::Complete { request, reply } => {
                    let _ = reply.send(self.complete(*request));
                }
                ScriptCommand::Snapshot(reply) => {
                    let _ = reply.send(self.request_deltas.clone());
                }
            }
        }
    }

    fn complete(&mut self, mut request: CompletionRequest) -> Result<AssistantMessage, LlmError> {
        if !request.messages.starts_with(&self.request_messages) {
            return Err(LlmError::Request(
                "scripted model request rewrote its message prefix".to_string(),
            ));
        }
        let message_count = self.request_messages.len();
        self.request_messages = request.messages.clone();
        request.messages = request.messages.split_off(message_count);
        self.request_deltas.push(request);
        self.responses
            .pop_front()
            .ok_or_else(|| LlmError::Request("scripted model is exhausted".to_string()))
    }
}

impl Llm for ScriptedModel {
    fn model(&self) -> &str {
        &self.name
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        let (reply, response) = oneshot::channel();
        let _ = self.commands.send(ScriptCommand::Complete {
            request: Box::new(request),
            reply,
        });
        stream::once(async move {
            let message = response
                .await
                .map_err(|_| LlmError::Request("scripted model closed".to_string()))??;
            Ok(StreamEvent::Final(Completion {
                message,
                usage: None,
                truncated: false,
            }))
        })
    }
}
