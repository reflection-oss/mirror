use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use assert_matches::assert_matches;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::prelude::{Engine, BASE64_STANDARD};
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use pretty_assertions::assert_eq;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

struct Fixture {
    workspace: tempfile::TempDir,
    output: PathBuf,
    model_url: String,
    requests: mpsc::UnboundedReceiver<Value>,
    model: JoinHandle<()>,
}

struct Server {
    url: String,
    client: reqwest::Client,
    child: Child,
}

impl Fixture {
    async fn start() -> Self {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let workspace = tempfile::tempdir().expect("workspace");
        let (requests, received) = mpsc::unbounded_channel();
        let app = Router::new()
            .route("/models", get(models))
            .route("/{prefix}/models", get(models))
            .route("/chat/completions", post(completion))
            .route("/remote/chat/completions", post(remote_completion))
            .route(
                "/remote-read/chat/completions",
                post(remote_read_completion),
            )
            .route("/read_command", post(remote_read_command))
            .route("/send_command", post(remote_command))
            .route("/pending_command", post(pending_remote_command))
            .route(
                "/non-streaming/chat/completions",
                post(non_streaming_completion),
            )
            .route("/recover/chat/completions", post(recover_completion))
            .route("/empty/chat/completions", post(empty_completion))
            .route(
                "/empty-recover/chat/completions",
                post(empty_recovery_completion),
            )
            .route("/default-tools/chat/completions", post(default_tools))
            .route("/pending/chat/completions", post(pending))
            .route(
                "/pending-after-tool/chat/completions",
                post(pending_after_tool),
            )
            .route(
                "/failed/chat/completions",
                post(|| async { (StatusCode::BAD_REQUEST, "model failed") }),
            )
            .route("/slow-tool/chat/completions", post(slow_tool))
            .route("/compaction/chat/completions", post(compaction))
            .with_state(requests);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("model listener");
        let model_url = format!("http://{}", listener.local_addr().expect("address"));
        let model = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("model server");
        });
        Self {
            workspace,
            output: "logs/trace.jsonl".into(),
            model_url,
            requests: received,
            model,
        }
    }

    fn command(&self, endpoint: &str) -> Command {
        let binary = std::env::var_os("MIRROR_AGENT_BIN")
            .map(PathBuf::from)
            .or_else(|| option_env!("CARGO_BIN_EXE_mirror-agent").map(PathBuf::from))
            .expect("agent binary");
        let mut command = Command::new(binary.canonicalize().expect("binary path"));
        command
            .current_dir(self.workspace.path())
            .arg("serve")
            .args([
                "--model",
                "test-model",
                "--base-url",
                &format!("{}{endpoint}", self.model_url),
            ])
            .arg("--workspace")
            .arg(self.workspace.path())
            .arg("--output")
            .arg(&self.output)
            .args(["--max-tokens", "128"])
            .env("HOME", self.workspace.path().join("home"))
            .env("OPENAI_API_KEY", "secret-model-key")
            .kill_on_drop(true);
        if let Some(rg) = std::env::var_os("MIRROR_RG_PATH") {
            command.env(
                "MIRROR_RG_PATH",
                PathBuf::from(rg).canonicalize().expect("ripgrep path"),
            );
        }
        command
    }

    async fn serve(&self, endpoint: &str, args: &[&str]) -> Server {
        let mut child = self
            .command(endpoint)
            .args(["--bind", "127.0.0.1:0"])
            .args(args)
            .env_remove("MIRROR_SERVER_TOKEN")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start HTTP server");
        let mut lines = BufReader::new(child.stdout.take().expect("stdout")).lines();
        let line = tokio::time::timeout(Duration::from_secs(10), lines.next_line())
            .await
            .expect("server startup deadline")
            .expect("read address")
            .expect("server published address");
        Server {
            url: line
                .strip_prefix("MIRROR_HTTP_ADDRESS=")
                .expect("address prefix")
                .into(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("client"),
            child,
        }
    }

    async fn request(&mut self) -> Value {
        tokio::time::timeout(Duration::from_secs(5), self.requests.recv())
            .await
            .expect("inference starts")
            .expect("request")
    }

    fn add_prompt_context(&self) {
        let root = self.workspace.path();
        std::fs::write(root.join("AGENTS.md"), "Local project instruction.").expect("instructions");
        std::fs::create_dir_all(root.join(".agents/skills/local")).expect("local skills");
        std::fs::write(
            root.join(".agents/skills/local/SKILL.md"),
            "---\nname: local\ndescription: Local eval skill.\n---\nRead files.",
        )
        .expect("local skill");
        let global = root.join("home/.mirror");
        std::fs::create_dir_all(global.join("skills/global")).expect("global skills");
        std::fs::write(global.join("AGENTS.md"), "Global instruction.")
            .expect("global instructions");
        std::fs::write(
            global.join("skills/global/SKILL.md"),
            "---\nname: global\ndescription: Global eval skill.\n---\nRead files.",
        )
        .expect("global skill");
    }

    fn model_calls(&self) -> Vec<Value> {
        std::fs::read_to_string(
            self.workspace
                .path()
                .join(self.output.with_extension("model-calls.jsonl")),
        )
        .expect("model-call trace file")
        .lines()
        .map(|line| serde_json::from_str(line).expect("model-call record"))
        .collect()
    }

    fn trace(&self) -> Vec<Value> {
        std::fs::read_to_string(self.workspace.path().join(&self.output))
            .expect("trace file")
            .lines()
            .map(|line| serde_json::from_str(line).expect("native JSON record"))
            .collect()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.model.abort();
    }
}

impl Server {
    async fn prompt(&self) -> reqwest::Response {
        self.client
            .post(format!("{}/message", self.url))
            .header("content-type", "text/plain")
            .body("Write result.txt and finish.")
            .send()
            .await
            .expect("prompt")
    }

    async fn stop(mut self) {
        kill(
            Pid::from_raw(self.child.id().expect("PID") as i32),
            Signal::SIGTERM,
        )
        .expect("SIGTERM");
        let status = tokio::time::timeout(Duration::from_secs(5), self.child.wait())
            .await
            .expect("shutdown deadline")
            .expect("wait");
        assert!(status.success(), "server exited with {status}");
    }
}

async fn completion(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    Json(body): Json<Value>,
) -> impl axum::response::IntoResponse {
    requests.send(body.clone()).expect("record request");
    let done = body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .any(|message| message["role"] == "tool");
    let delta = if done {
        json!({"content":"Task complete"})
    } else {
        json!({"tool_calls":[{"index":0,"id":"call-1","function":{"name":"bash","arguments":"{\"command\":\"printf done > result.txt\"}"}}]})
    };
    stream(delta, if done { "stop" } else { "tool_calls" })
}

async fn recover_completion(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    requests.send(body.clone()).expect("record request");
    let results = body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .filter(|message| message["role"] == "tool")
        .count();
    match results {
        0 => stream(
            json!({"tool_calls":[{"index":0,"id":"bad","function":{
                "name":"bash", "arguments": r#"{"__parse_error__":"missing closing parameter tag","__raw__":"printf unsafe > rejected.txt"}"#
            }}]}),
            "tool_calls",
        ),
        1 => stream(
            json!({"tool_calls":[{"index":0,"id":"good","function":{
                "name":"bash", "arguments": r#"{"command":"printf recovered > result.txt"}"#
            }}]}),
            "tool_calls",
        ),
        _ => stream(json!({"content":"Task complete"}), "stop"),
    }
}

async fn empty_completion(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    requests.send(body).expect("record request");
    stream(json!({"reasoning_content":"Still thinking."}), "stop")
}

async fn empty_recovery_completion(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    requests.send(body.clone()).expect("record request");
    let retried = body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .any(|message| message["role"] == "assistant");
    stream(
        if retried {
            json!({"content":"Recovered"})
        } else {
            json!({"content":""})
        },
        "stop",
    )
}

#[tokio::test]
async fn http_empty_completion_can_recover() {
    let mut fixture = Fixture::start().await;
    let server = fixture.serve("/empty-recover", &["--tools", "bash"]).await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let _ = fixture.request().await;
    let _ = fixture.request().await;
    assert!(fixture.requests.try_recv().is_err());
    assert_eq!(
        fixture.model_calls().last().expect("final record")["content"],
        "Recovered"
    );
    server.stop().await;
}

#[tokio::test]
async fn http_builtin_bash_recovers_from_rejected_arguments() {
    let mut fixture = Fixture::start().await;
    let server = fixture
        .serve("/recover", &["--tools", "bash", "--use-native-bash"])
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let first = fixture.request().await;
    assert_eq!(
        first["chat_template_kwargs"]["builtin_tools"],
        json!(["bash"])
    );
    let tools = first["tools"].as_array().expect("tool list");
    assert_eq!(tools.len(), 1, "the eval invocation advertises bash alone");
    assert_eq!(tools[0]["function"]["name"], "bash");
    let retry = fixture.request().await;
    assert_eq!(retry["messages"][2]["tool_call_id"], "bad");
    assert!(retry["messages"][2]["content"]
        .as_str()
        .expect("error observation")
        .starts_with("Invalid arguments for tool bash: "));
    assert!(!fixture.workspace.path().join("rejected.txt").exists());
    assert_eq!(
        std::fs::read_to_string(fixture.workspace.path().join("result.txt"))
            .expect("corrected call ran"),
        "recovered"
    );
    let final_request = fixture.request().await;
    assert_eq!(final_request["messages"][4]["tool_call_id"], "good");
    assert_eq!(
        fixture.model_calls().last().expect("final record")["content"],
        "Task complete"
    );
    server.stop().await;
}

#[tokio::test]
async fn http_reasoning_only_completions_have_bounded_retries() {
    let mut fixture = Fixture::start().await;
    let server = fixture.serve("/empty", &["--tools", "bash"]).await;
    assert!(server.prompt().await.status().is_server_error());
    let _ = fixture.request().await;
    let _ = fixture.request().await;
    let _ = fixture.request().await;
    assert!(fixture.requests.try_recv().is_err());
    let calls = fixture.model_calls();
    assert_eq!(
        calls
            .iter()
            .filter(|call| call["role"] == "assistant")
            .count(),
        3
    );
    server.stop().await;
}

async fn remote_completion(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    requests.send(body.clone()).expect("record request");
    let done = body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .any(|message| message["role"] == "tool");
    let delta = if done {
        json!({"content": "Task complete"})
    } else {
        json!({"tool_calls":[{"index":0,"id":"remote-1","function":{"name":"bash","arguments":"{\"command\":\"printf done > result.txt\"}"}}]})
    };
    stream(delta, if done { "stop" } else { "tool_calls" })
}

async fn remote_command(
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
    headers: axum::http::HeaderMap,
    body: String,
) -> Json<Value> {
    let request = capture_request(&body);
    assert_eq!(request["command"], "printf done > result.txt");
    assert!(!request.to_string().contains("secret-model-key"));
    assert_eq!(query.get("timeout").map(String::as_str), Some("630"));
    assert!(!headers.contains_key("authorization"));
    let result = json!({
        "stdout": BASE64_STANDARD.encode("remote output"),
        "stderr": BASE64_STANDARD.encode("remote error"),
        "return_code": 3, "timed_out": false, "elapsed_seconds": 0.125,
    });
    Json(json!({"output": result.to_string(), "exit_code": 0, "alerts": []}))
}

#[tokio::test]
async fn http_remote_bash_has_no_local_execution_or_project_context() {
    let mut fixture = Fixture::start().await;
    fixture.add_prompt_context();
    let endpoint = format!("{}/send_command", fixture.model_url);
    let server = fixture
        .serve(
            "/remote",
            &[
                "--tools",
                "bash",
                "--remote",
                &endpoint,
                "--include-project-instructions",
                "--include-skills",
            ],
        )
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    assert!(!fixture.workspace.path().join("result.txt").exists());
    let request = fixture.request().await;
    assert_eq!(request["tools"].as_array().expect("tools").len(), 1);
    assert_eq!(request["tools"][0]["function"]["name"], "bash");
    assert!(!request["messages"]
        .to_string()
        .contains("Local project instruction"));
    assert!(!request["messages"].to_string().contains("Local eval skill"));
    assert!(!request["messages"]
        .to_string()
        .contains("Primary working directory"));
    let next = fixture.request().await;
    let result = next["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("tool result");
    assert!(result["content"]
        .as_str()
        .expect("content")
        .contains("remote output"));
    assert!(result["content"]
        .as_str()
        .expect("content")
        .contains("Exit code: 3"));
    server.stop().await;
}

#[tokio::test]
async fn http_remote_bash_uses_local_stderr_formatting() {
    let mut fixture = Fixture::start().await;
    let endpoint = format!("{}/send_command", fixture.model_url);
    let server = fixture.serve("/remote", &["--remote", &endpoint]).await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let _ = fixture.request().await;
    let next = fixture.request().await;
    let result = next["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("tool result");
    assert_eq!(
        result["content"],
        "Exit code: 3\nremote output\nstderr:\nremote error\n[Completed in 0.1s]"
    );
    server.stop().await;
}

fn capture_request(body: &str) -> Value {
    assert!(body.starts_with("python3 -I -c "));
    let input = body
        .rsplit_once("MIRROR_ENVIRONMENT_INPUT'\n")
        .expect("encoded request")
        .1;
    let input = input
        .strip_suffix("\nMIRROR_ENVIRONMENT_INPUT")
        .expect("request end");
    serde_json::from_slice(&BASE64_STANDARD.decode(input).expect("base64 request"))
        .expect("JSON request")
}

async fn pending_remote_command(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    body: String,
) -> StatusCode {
    requests
        .send(json!({"remote_command": capture_request(&body)["command"]}))
        .expect("record remote command");
    std::future::pending().await
}

#[tokio::test]
async fn http_turn_timeout_interrupts_pending_remote_bash() {
    let mut fixture = Fixture::start().await;
    let endpoint = format!("{}/pending_command", fixture.model_url);
    let server = fixture
        .serve(
            "/remote",
            &[
                "--tools",
                "bash",
                "--remote",
                &endpoint,
                "--turn-timeout-seconds",
                "1",
            ],
        )
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::GATEWAY_TIMEOUT);
    let _ = fixture.request().await;
    assert_eq!(
        fixture.request().await["remote_command"],
        "printf done > result.txt"
    );
    assert!(!fixture.workspace.path().join("result.txt").exists());
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    server.stop().await;
}

async fn remote_read_completion(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    requests.send(body.clone()).expect("record request");
    let done = body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .any(|message| message["role"] == "tool");
    let delta = if done {
        json!({"content":"done"})
    } else {
        json!({"tool_calls":[{"index":0,"id":"read-1","function":{"name":"read","arguments":"{\"file_path\":\"data\"}"}}]})
    };
    stream(delta, if done { "stop" } else { "tool_calls" })
}

async fn remote_read_command(body: String) -> Json<Value> {
    let request = capture_request(&body);
    assert_eq!(request["operation"], "read_file");
    assert_eq!(request["args"]["file_path"], "data");
    let output = json!({"content": BASE64_STANDARD.encode("remote contents"), "exceeded_byte_cap":false, "root":"/workspace", "path":"/workspace/data"});
    Json(json!({"output":output.to_string(), "exit_code":0}))
}

#[tokio::test]
async fn http_remote_read_returns_endpoint_content_not_host_content() {
    let mut fixture = Fixture::start().await;
    std::fs::write(fixture.workspace.path().join("data"), "HOST_SECRET_CANARY").expect("host file");
    let endpoint = format!("{}/read_command", fixture.model_url);
    let server = fixture
        .serve("/remote-read", &["--tools", "read", "--remote", &endpoint])
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let _ = fixture.request().await;
    let request = fixture.request().await;
    let result = request["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("tool result");
    assert_eq!(result["content"], "remote contents");
    assert!(!request.to_string().contains("HOST_SECRET_CANARY"));
    server.stop().await;
}

#[tokio::test]
async fn http_remote_catalog_includes_all_workspace_tools() {
    let mut fixture = Fixture::start().await;
    let endpoint = format!("{}/read_command", fixture.model_url);
    let server = fixture
        .serve(
            "/remote-read",
            &[
                "--tools",
                "bash,read,ls,find,grep,write,edit",
                "--remote",
                &endpoint,
            ],
        )
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let request = fixture.request().await;
    assert_eq!(request["tools"].as_array().expect("tools").len(), 7);
    let result = fixture.request().await;
    assert!(result.to_string().contains("remote contents"));
    server.stop().await;
}

#[tokio::test]
async fn cli_rejects_host_memory_for_remote_workspaces() {
    let fixture = Fixture::start().await;
    let memory = fixture.workspace.path().join("memory");
    let result = fixture
        .command("")
        .args(["--remote", "http://localhost:1/exec", "--memory-dir"])
        .arg(&memory)
        .output()
        .await
        .expect("CLI");
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr)
        .contains("--memory-dir needs the workspace on this machine"));
    assert!(!memory.exists());
}

async fn models(headers: axum::http::HeaderMap) -> Json<Value> {
    assert_eq!(headers["authorization"], "Bearer secret-model-key");
    Json(json!({"data": [{"id": "test-model", "max_model_len": 4096}]}))
}

async fn non_streaming_completion(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Json<Value> {
    assert_eq!(headers["accept"], "application/json");
    assert_eq!(headers["authorization"], "Bearer secret-model-key");
    assert_eq!(body["stream"], false);
    assert!(body.get("stream_options").is_none());
    let done = body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .any(|message| message["role"] == "tool");
    requests.send(body).expect("record request");
    let message = if done {
        json!({"role":"assistant", "content":"Task complete"})
    } else {
        json!({
            "role":"assistant", "content":null, "reasoning_content":"Write both files.",
            "reasoning_details":[{"type":"reasoning.encrypted", "data":"opaque", "id":"reasoning-1"}],
            "tool_calls":[
                {"id":"call-1", "type":"function", "function":{"name":"write", "arguments":"{\"file_path\":\"result.txt\",\"content\":\"done\"}"}},
                {"id":"call-2", "type":"function", "function":{"name":"write", "arguments":"{\"file_path\":\"second.txt\",\"content\":\"other\"}"}}
            ]
        })
    };
    Json(json!({
        "choices":[{"message":message, "finish_reason":if done { "stop" } else { "tool_calls" }}],
        "usage":{"prompt_tokens":10, "completion_tokens":5, "prompt_tokens_details":{"cached_tokens":2}}
    }))
}

#[tokio::test]
async fn http_non_streaming_completions_execute_tools_and_preserve_history() {
    let mut fixture = Fixture::start().await;
    let server = fixture
        .serve(
            "/non-streaming",
            &[
                "--api",
                "openai-chat-completions-non-streaming",
                "--tools",
                "write",
                "--temperature",
                "0.2",
                "--top-p",
                "0.9",
                "--seed",
                "42",
            ],
        )
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        std::fs::read_to_string(fixture.workspace.path().join("result.txt")).expect("first file"),
        "done"
    );
    assert_eq!(
        std::fs::read_to_string(fixture.workspace.path().join("second.txt")).expect("second file"),
        "other"
    );
    let first = fixture.request().await;
    assert_eq!(first["temperature"], 0.2);
    assert_eq!(first["top_p"], 0.9);
    assert_eq!(first["seed"], 42);
    assert_eq!(first["max_tokens"], 128);
    assert_eq!(first["tools"][0]["function"]["name"], "write");
    let next = fixture.request().await;
    let assistant = &next["messages"][1];
    assert_eq!(assistant["reasoning_content"], "Write both files.");
    assert_eq!(assistant["reasoning_details"][0]["data"], "opaque");
    assert_eq!(assistant["tool_calls"][0]["id"], "call-1");
    assert_eq!(assistant["tool_calls"][1]["id"], "call-2");
    let trace = fixture.trace();
    assert_eq!(trace.len(), 10);
    assert_eq!(
        trace[2]["inference"]["api"],
        "openai-chat-completions-non-streaming"
    );
    assert_eq!(trace[9]["chat"]["assistant"]["content"], "Task complete");
    let calls = fixture.model_calls();
    assert_eq!(calls.len(), 5);
    assert_eq!(calls[1]["role"], "assistant");
    assert_eq!(calls[1]["reasoning_content"], "Write both files.");
    assert_eq!(calls[1]["reasoning_details"][0]["data"], "opaque");
    assert_eq!(
        calls[1]["tool_calls"].as_array().expect("tool calls").len(),
        2
    );
    assert_eq!(calls[2]["role"], "tool");
    assert_eq!(calls[3]["role"], "tool");
    assert_eq!(calls[4]["role"], "assistant");
    assert_eq!(calls[4]["usage"]["cached_tokens"], 2);
    server.stop().await;
}

async fn default_tools(Json(body): Json<Value>) -> impl IntoResponse {
    let calls = [
        (
            "write",
            json!({"file_path":"example.txt", "content":"before\n"}),
        ),
        ("read", json!({"file_path":"example.txt"})),
        (
            "edit",
            json!({"file_path":"example.txt", "old_string":"before", "new_string":"after"}),
        ),
        ("glob", json!({"pattern":"example.txt"})),
        (
            "grep",
            json!({"pattern":"after", "path":"example.txt", "output_mode":"content"}),
        ),
        ("ls", json!({"path":"."})),
        ("bash", json!({"command":"printf shell"})),
    ];
    let step = body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .filter(|message| message["role"] == "tool")
        .count();
    match calls.get(step) {
        Some((name, arguments)) => stream(
            json!({"tool_calls":[{"index":0,"id":format!("call-{step}"),"function":{"name":name,"arguments":arguments.to_string()}}]}),
            "tool_calls",
        ),
        None => stream(json!({"content":"Task complete"}), "stop"),
    }
}

#[tokio::test]
async fn http_all_tools_work() {
    let fixture = Fixture::start().await;
    let server = fixture
        .serve(
            "/default-tools",
            &[
                "--system-prompt",
                "Exercise the default tools.",
                "--tools",
                "bash,read,write,edit,grep,find,ls",
            ],
        )
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let trace = fixture.trace();
    let replies: Vec<_> = trace
        .iter()
        .filter(|record| record["chat"]["tool"].is_object())
        .collect();
    assert_eq!(replies.len(), 7);
    assert!(
        replies
            .iter()
            .all(|record| record["chat"]["success"] == true),
        "{replies:?}"
    );
    assert!(replies[1]["chat"]["tool"]["content"]
        .as_str()
        .expect("read output")
        .contains("before"));
    assert!(replies[3]["chat"]["tool"]["content"]
        .as_str()
        .expect("find output")
        .contains("example.txt"));
    assert!(replies[4]["chat"]["tool"]["content"]
        .as_str()
        .expect("grep output")
        .contains("after"));
    assert!(replies[5]["chat"]["tool"]["content"]
        .as_str()
        .expect("ls output")
        .contains("example.txt"));
    assert!(replies[6]["chat"]["tool"]["content"]
        .as_str()
        .expect("bash output")
        .contains("shell"));
    assert_eq!(
        std::fs::read_to_string(fixture.workspace.path().join("example.txt")).expect("edited file"),
        "after\n"
    );
    server.stop().await;
}

fn stream(delta: Value, finish: &str) -> impl axum::response::IntoResponse {
    let event = json!({"choices":[{"delta":delta,"finish_reason":finish}],"usage":{"prompt_tokens":100,"completion_tokens":10}});
    (
        [("content-type", "text/event-stream")],
        format!("data: {event}\n\ndata: [DONE]\n\n"),
    )
}

async fn pending(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    Json(body): Json<Value>,
) -> StatusCode {
    let _ = requests.send(body);
    std::future::pending().await
}

async fn pending_after_tool(
    requests: State<mpsc::UnboundedSender<Value>>,
    body: Json<Value>,
) -> axum::response::Response {
    if body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .any(|message| message["role"] == "tool")
    {
        pending(requests, body).await.into_response()
    } else {
        completion(requests, body).await.into_response()
    }
}

async fn compaction(
    State(requests): State<mpsc::UnboundedSender<Value>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let content = if body["tools"].is_null() {
        "Compacted history"
    } else {
        "Task complete"
    };
    requests.send(body).expect("record request");
    stream(json!({"content": content}), "stop")
}

#[tokio::test]
async fn http_model_calls_capture_compaction_and_rebuilt_context() {
    let mut fixture = Fixture::start().await;
    let server = fixture
        .serve(
            "/compaction",
            &[
                "--system-prompt",
                "Actual system prompt.",
                "--tools",
                "bash",
                "--compaction-threshold",
                "1000",
            ],
        )
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let initial_request = fixture.request().await;
    let response = server
        .client
        .post(format!("{}/message", server.url))
        .header("content-type", "text/plain")
        .body("Long task instructions. ".repeat(1000))
        .send()
        .await
        .expect("prompt");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    let summary_request = fixture.request().await;
    let next_request = fixture.request().await;
    let calls = fixture.model_calls();
    assert_eq!(calls.len(), 11);
    assert_eq!(calls[0]["role"], "system");
    assert_eq!(
        calls[0]["content"],
        initial_request["messages"][0]["content"]
    );
    assert_eq!(calls[3]["event"], "context_reset");
    assert_eq!(calls[4]["role"], "system");
    assert_eq!(
        calls[4]["content"],
        summary_request["messages"][0]["content"]
    );
    assert_eq!(calls[4]["tools"], json!([]));
    assert_eq!(calls[6]["role"], "assistant");
    assert_eq!(calls[6]["content"], "Compacted history");
    assert_eq!(calls[7]["event"], "context_reset");
    assert_eq!(calls[8]["tools"], calls[0]["tools"]);
    assert_eq!(calls[8]["tools"][0], next_request["tools"][0]["function"]);
    assert_eq!(calls[8]["max_tokens"], next_request["max_tokens"]);
    assert_eq!(calls[8]["inference"]["model"], "test-model");
    assert_eq!(calls[8]["role"], "system");
    assert_eq!(calls[8]["content"], "Actual system prompt.");
    assert_eq!(calls[9]["role"], "user");
    assert_eq!(calls[9]["content"], next_request["messages"][1]["content"]);
    assert!(next_request["messages"][1]["content"]
        .as_str()
        .expect("compacted context")
        .contains("Compacted history"));
    assert_eq!(calls[10]["role"], "assistant");
    assert_eq!(calls[10]["content"], "Task complete");
    server.stop().await;
}

async fn slow_tool() -> impl axum::response::IntoResponse {
    stream(
        json!({"tool_calls":[{"index":0,"id":"slow","function":{"name":"bash","arguments":"{\"command\":\"printf '%s' $$ > process.pid; exec sleep 60\"}"}}]}),
        "tool_calls",
    )
}

#[tokio::test]
async fn http_uses_cli_configuration_and_native_tools() {
    let mut fixture = Fixture::start().await;
    fixture.add_prompt_context();
    let server = fixture
        .serve(
            "",
            &[
                "--tools",
                "bash",
                "--system-prompt",
                "Custom eval prompt.",
                "--temperature",
                "0",
                "--top-p",
                "0.9",
                "--seed",
                "123",
                "--reasoning-effort",
                "high",
                "--extra-body",
                r#"{"chat_template_kwargs": {"enable_thinking": false}}"#,
            ],
        )
        .await;
    let response = server.prompt().await;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert_eq!(response.headers()["cache-control"], "no-store");
    assert_eq!(response.text().await.expect("body"), "");
    let trace = fixture.trace();
    assert_eq!(trace.len(), 7);
    assert_eq!(
        trace[3]["toolApproval"],
        json!({
            "callId": "call-1", "toolName": "bash", "outcome": "TOOL_APPROVAL_OUTCOME_REQUESTED",
        })
    );
    assert_eq!(
        trace[4]["toolApproval"],
        json!({
            "callId": "call-1", "toolName": "bash", "outcome": "TOOL_APPROVAL_OUTCOME_APPROVED",
            "source": "APPROVAL_SOURCE_AUTOMATIC",
        })
    );
    assert_eq!(trace[2]["inference"]["temperature"], 0.0);
    assert_eq!(trace[2]["inference"]["topP"], 0.9);
    assert_eq!(trace[2]["inference"]["seed"], "123");
    assert!(!serde_json::to_string(&trace)
        .expect("trace JSON")
        .contains("secret-model-key"));
    assert_eq!(
        std::fs::read_to_string(fixture.workspace.path().join("result.txt")).expect("tool effect"),
        "done"
    );
    let request = fixture.request().await;
    assert_eq!(
        request["messages"][0],
        json!({"role":"system","content":"Custom eval prompt."})
    );
    assert_eq!(request["temperature"], 0.0);
    assert_eq!(request["top_p"], 0.9);
    assert_eq!(request["seed"], 123);
    assert_eq!(request["max_tokens"], 128);
    assert_eq!(request["reasoning_effort"], "high");
    assert_eq!(
        request["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
    assert_eq!(request["tools"].as_array().expect("tools").len(), 1);
    assert!(request["tools"][0]["function"]["parameters"]["properties"]["detached"].is_null());
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    assert_eq!(fixture.trace().len(), 9);
    server.stop().await;
}

#[tokio::test]
async fn http_model_calls_capture_prompt_tools_history_and_responses() {
    let mut fixture = Fixture::start().await;
    let server = fixture
        .serve(
            "",
            &[
                "--system-prompt",
                "Actual system prompt.",
                "--tools",
                "bash",
                "--temperature",
                "0.5",
            ],
        )
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let first = fixture.request().await;
    let second = fixture.request().await;
    let calls = fixture.model_calls();
    assert_eq!(calls.len(), 5);
    assert_eq!(calls[0]["role"], "system");
    assert_eq!(calls[0]["content"], first["messages"][0]["content"]);
    assert_eq!(calls[1]["role"], "user");
    assert_eq!(calls[1]["content"], first["messages"][1]["content"]);
    assert_eq!(calls[0]["tools"][0], first["tools"][0]["function"]);
    assert_eq!(calls[0]["max_tokens"], first["max_tokens"]);
    assert_eq!(calls[0]["inference"]["temperature"], first["temperature"]);
    assert_eq!(calls[2]["role"], "assistant");
    assert_eq!(calls[2]["tool_calls"][0]["function"]["name"], "bash");
    assert_eq!(calls[3]["role"], "tool");
    assert_eq!(calls[3]["content"], second["messages"][3]["content"]);
    assert_eq!(calls[3]["tool_call_id"], calls[2]["tool_calls"][0]["id"]);
    assert_eq!(calls[4]["role"], "assistant");
    assert_eq!(calls[4]["content"], "Task complete");
    assert_eq!(calls[4]["usage"]["completion_tokens"], 10);
    assert_eq!(calls[4]["truncated"], false);
    assert!(calls[4]["timestamp_ms"].is_u64());
    let raw = std::fs::read_to_string(
        fixture
            .workspace
            .path()
            .join(fixture.output.with_extension("model-calls.jsonl")),
    )
    .expect("trace");
    assert!(raw.starts_with(r#"{"role":"system","content":"Actual system prompt.","tools":["#));
    assert!(raw.contains(r#"{"role":"assistant","content":"Task complete","usage":{"completion_tokens":10,"prompt_tokens":100},"truncated":false,"timestamp_ms":"#));
    assert!(!raw.contains("\"call_id\""));
    assert!(!raw.contains("\"event\""));
    assert!(!serde_json::to_string(&calls)
        .expect("JSON")
        .contains("secret-model-key"));
    assert_eq!(fixture.trace().len(), 7);
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let calls = fixture.model_calls();
    assert_eq!(calls.len(), 7);
    assert_eq!(calls[5]["role"], "user");
    assert!(calls[5].get("tools").is_none());
    assert_eq!(calls[5]["content"], first["messages"][1]["content"]);
    assert_eq!(calls[6]["role"], "assistant");
    assert_eq!(calls[6]["content"], "Task complete");
    server.stop().await;
}

#[tokio::test]
async fn http_writes_to_a_bare_output_filename_before_acknowledging() {
    let mut fixture = Fixture::start().await;
    fixture.output = "result.jsonl".into();
    let server = fixture.serve("", &[]).await;
    assert_eq!(fixture.trace().len(), 1);
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    assert_eq!(fixture.trace().len(), 7);
    server.stop().await;
}

#[tokio::test]
async fn http_default_has_no_system_prompt_even_with_project_context() {
    let mut fixture = Fixture::start().await;
    fixture.add_prompt_context();
    let server = fixture.serve("", &[]).await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let request = fixture.request().await;
    assert_eq!(
        request["messages"],
        json!([{"role": "user", "content": "Write result.txt and finish."}])
    );
    assert_eq!(request["tools"].as_array().expect("tools").len(), 1);
    assert_eq!(request["tools"][0]["function"]["name"], "bash");
    assert_eq!(
        fixture.model_calls()[0]["tools"][0],
        request["tools"][0]["function"]
    );
    assert_eq!(fixture.model_calls()[0]["role"], "user");
    let next = fixture.request().await;
    assert!(next["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .all(|message| message["role"] != "system"));
    server.stop().await;
}

#[tokio::test]
async fn http_empty_explicit_prompt_sends_no_system_message() {
    let mut fixture = Fixture::start().await;
    fixture.add_prompt_context();
    let server = fixture.serve("", &["--system-prompt", ""]).await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        fixture.request().await["messages"],
        json!([{"role": "user", "content": "Write result.txt and finish."}])
    );
    server.stop().await;
}

#[tokio::test]
async fn http_whitespace_explicit_prompt_sends_no_system_message() {
    let mut fixture = Fixture::start().await;
    fixture.add_prompt_context();
    let server = fixture
        .serve("", &["--system-prompt", " \t\r\n\u{2003}"])
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        fixture.request().await["messages"],
        json!([{"role": "user", "content": "Write result.txt and finish."}])
    );
    server.stop().await;
}

async fn prompt_with_context(args: &[&str]) -> String {
    let mut fixture = Fixture::start().await;
    fixture.add_prompt_context();
    let server = fixture.serve("", args).await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let request = fixture.request().await;
    assert_eq!(request["messages"][0]["role"], "system");
    let prompt = request["messages"][0]["content"]
        .as_str()
        .expect("system prompt")
        .to_string();
    server.stop().await;
    prompt
}

#[tokio::test]
async fn http_project_instructions_are_opt_in() {
    let prompt = prompt_with_context(&["--include-project-instructions"]).await;

    assert!(prompt.contains("Local project instruction."));
    assert!(prompt.contains("Global instruction."));
    assert!(!prompt.contains("Local eval skill."));
    assert!(!prompt.contains("Global eval skill."));
}

#[tokio::test]
async fn http_skills_are_opt_in() {
    let prompt = prompt_with_context(&["--include-skills"]).await;

    assert!(prompt.contains("Local eval skill."));
    assert!(prompt.contains("Global eval skill."));
    assert!(!prompt.contains("Local project instruction."));
    assert!(!prompt.contains("Global instruction."));
}

#[tokio::test]
async fn http_opts_in_to_both_prompt_sources_but_never_an_environment_section() {
    let prompt = prompt_with_context(&["--include-project-instructions", "--include-skills"]).await;

    assert!(prompt.contains("Local project instruction."));
    assert!(prompt.contains("Local eval skill."));
    assert!(!prompt.contains("# Environment"));
}

#[tokio::test]
async fn http_accepts_unauthenticated_plain_text_and_rejects_invalid_requests() {
    let fixture = Fixture::start().await;
    let server = fixture.serve("", &[]).await;
    let url = format!("{}/message", server.url);
    assert_eq!(
        server
            .client
            .post(&url)
            .header("content-type", "text/plain")
            .body("hello")
            .send()
            .await
            .expect("no auth")
            .status(),
        StatusCode::NO_CONTENT
    );
    assert_eq!(
        server
            .client
            .post(&url)
            .header("origin", "https://untrusted.example")
            .header("content-type", "text/plain")
            .body("hello")
            .send()
            .await
            .expect("browser")
            .status(),
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        server
            .client
            .post(&url)
            .json(&json!({"content":"hello", "temperature":0}))
            .send()
            .await
            .expect("JSON envelope is unsupported")
            .status(),
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
    assert_eq!(
        server
            .client
            .post(&url)
            .header("content-type", "text/plain")
            .body("")
            .send()
            .await
            .expect("empty prompt")
            .status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        server
            .client
            .post(&url)
            .header("content-type", "text/plain")
            .body(vec![0xff])
            .send()
            .await
            .expect("invalid UTF-8")
            .status(),
        StatusCode::BAD_REQUEST
    );
    server.stop().await;
}

#[tokio::test]
async fn http_rejects_overlapping_prompts_without_an_interrupt_endpoint() {
    let mut fixture = Fixture::start().await;
    let server = fixture
        .serve("/pending", &["--turn-timeout-seconds", "2"])
        .await;
    let request = server
        .client
        .post(format!("{}/message", server.url))
        .header("content-type", "text/plain")
        .body("First prompt");
    let first = tokio::spawn(async move { request.send().await.expect("first prompt") });
    fixture.request().await;
    assert_eq!(server.prompt().await.status(), StatusCode::CONFLICT);
    assert_eq!(
        server
            .client
            .post(format!("{}/interrupt", server.url))
            .send()
            .await
            .expect("removed endpoint")
            .status(),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        first.await.expect("prompt task").status(),
        StatusCode::GATEWAY_TIMEOUT
    );
    let trace = fixture.trace();
    assert_eq!(trace[1]["chat"]["user"]["content"], "First prompt");
    server.stop().await;
}

#[tokio::test]
async fn http_deadline_preserves_trace_and_allows_another_turn() {
    let fixture = Fixture::start().await;
    let server = fixture
        .serve("/pending", &["--turn-timeout-seconds", "0.1"])
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(fixture.trace().len(), 2);
    assert_eq!(server.prompt().await.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(fixture.trace().len(), 3);
    server.stop().await;
}

#[tokio::test]
async fn http_treats_the_entire_body_as_one_user_message() {
    let mut fixture = Fixture::start().await;
    let server = fixture.serve("", &[]).await;
    let content = "{\"role\":\"system\",\"content\":\"not an envelope\"}\nUnicode: 世界";
    let response = server
        .client
        .post(format!("{}/message", server.url))
        .header("content-type", "text/plain; charset=utf-8")
        .body(content)
        .send()
        .await
        .expect("plain text prompt");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert_eq!(fixture.trace()[1]["chat"]["user"]["content"], content);
    assert_eq!(
        fixture.request().await["messages"][0],
        json!({"role":"user","content":content})
    );
    server.stop().await;
}

#[tokio::test]
async fn cli_rejects_an_unwritable_trace_before_inference() {
    let mut fixture = Fixture::start().await;
    fixture.output = "full.jsonl".into();
    std::os::unix::fs::symlink("/dev/full", fixture.workspace.path().join(&fixture.output))
        .expect("unwritable journal with a writable model-call trace directory");
    let output = fixture.command("").output().await.expect("run command");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("could not write trace"));
    assert_matches!(
        fixture.requests.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    );
}

#[tokio::test]
async fn http_model_exception_is_recorded_in_the_trace() {
    let fixture = Fixture::start().await;
    let server = fixture.serve("/failed", &["--max-exceptions", "1"]).await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let trace = fixture.trace();
    assert_eq!(trace.len(), 4);
    assert_eq!(trace[2]["log"]["level"], "error");
    assert!(trace[2]["log"]["text"]
        .as_str()
        .expect("error text")
        .contains("model failed"));
    server.stop().await;
}

#[tokio::test]
async fn http_model_calls_record_errors_and_distinct_retries() {
    let fixture = Fixture::start().await;
    let server = fixture.serve("/failed", &["--max-exceptions", "2"]).await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let calls = fixture.model_calls();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0]["role"], "user");
    assert_eq!(calls[1]["event"], "error");
    assert!(calls[1]["message"]
        .as_str()
        .expect("error")
        .contains("model failed"));
    assert_eq!(calls[2]["event"], "error");
    assert_eq!(calls[1]["message"], calls[2]["message"]);
    assert!(calls[2]["timestamp_ms"].is_u64());
    server.stop().await;
}

#[tokio::test]
async fn http_unfinished_trace_survives_sigkill() {
    let mut fixture = Fixture::start().await;
    let mut server = fixture.serve("/pending-after-tool", &[]).await;
    let request = server
        .client
        .post(format!("{}/message", server.url))
        .header("content-type", "text/plain")
        .body("Write result.txt and finish.");
    let response = tokio::spawn(async move { request.send().await });
    fixture.request().await;
    fixture.request().await;
    let trace = fixture.trace();
    assert_eq!(trace.len(), 6);
    assert_eq!(
        trace[1]["chat"]["user"]["content"],
        "Write result.txt and finish."
    );
    assert_eq!(
        trace[2]["chat"]["assistant"]["toolCalls"][0]["id"],
        "call-1"
    );
    assert_eq!(trace[5]["chat"]["tool"]["toolCallId"], "call-1");
    server.child.kill().await.expect("SIGKILL");
    assert_matches!(response.await.expect("request task"), Err(_));
    assert_eq!(fixture.trace(), trace);
}

#[tokio::test]
async fn http_unfinished_trace_survives_sigterm() {
    let mut fixture = Fixture::start().await;
    let server = fixture.serve("/pending-after-tool", &[]).await;
    let request = server
        .client
        .post(format!("{}/message", server.url))
        .header("content-type", "text/plain")
        .body("Write result.txt and finish.");
    let response = tokio::spawn(async move { request.send().await.expect("prompt") });
    fixture.request().await;
    fixture.request().await;
    let trace = fixture.trace();
    assert_eq!(trace.len(), 6);
    server.stop().await;
    assert!(!response.await.expect("request task").status().is_success());
    assert_eq!(fixture.trace(), trace);
}

#[tokio::test]
async fn cli_rejects_invalid_sampling_before_inference() {
    let fixture = Fixture::start().await;
    let output = fixture
        .command("")
        .args(["--temperature", "3"])
        .output()
        .await
        .expect("run command");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("temperature must be between 0 and 2"));
    assert!(!fixture.workspace.path().join("result.txt").exists());
}

#[tokio::test]
async fn cli_rejects_sampling_for_unsupported_api_before_inference() {
    let fixture = Fixture::start().await;
    let output = fixture
        .command("")
        .args(["--api", "openai-responses", "--seed", "0"])
        .output()
        .await
        .expect("run command");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("sampling controls require openai-chat-completions"));
    assert!(!fixture.workspace.path().join("result.txt").exists());
}

#[tokio::test]
async fn http_timeout_kills_the_native_shell_process_and_repairs_history() {
    let fixture = Fixture::start().await;
    let server = fixture
        .serve("/slow-tool", &["--turn-timeout-seconds", "1"])
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::GATEWAY_TIMEOUT);
    let pid: i32 = std::fs::read_to_string(fixture.workspace.path().join("process.pid"))
        .expect("tool started")
        .parse()
        .expect("PID");
    tokio::time::timeout(Duration::from_secs(5), async {
        while kill(Pid::from_raw(pid), None).is_ok() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|error| {
        panic!(
            "tool process did not exit: {error}; status: {:?}",
            std::fs::read_to_string(format!("/proc/{pid}/status"))
        )
    });
    assert_matches!(
        kill(Pid::from_raw(pid), None),
        Err(nix::errno::Errno::ESRCH)
    );
    assert_eq!(fixture.trace()[5]["chat"]["tool"]["toolCallId"], "slow");
    server.stop().await;
}

#[tokio::test]
async fn memory_import_reports_provider_totals_and_only_actionable_errors() {
    let home = tempfile::tempdir().expect("home");
    let claude = home.path().join(".claude/projects/project/memory");
    let duplicate = home.path().join(".claude/projects/project-copy/memory");
    let codex = home.path().join(".codex/memories");
    std::fs::create_dir_all(&claude).expect("Claude source");
    std::fs::create_dir_all(&duplicate).expect("duplicate source");
    std::fs::create_dir_all(&codex).expect("Codex source");
    std::fs::write(claude.join("style.md"), "Use concise responses.").expect("Claude note");
    std::fs::write(duplicate.join("style.md"), "Use concise responses.").expect("duplicate note");
    std::fs::write(
        codex.join("memory_summary.md"),
        "## General Tips\n- Run focused tests.\n",
    )
    .expect("Codex note");
    let binary = std::env::var_os("MIRROR_AGENT_BIN")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_mirror-agent").map(PathBuf::from))
        .expect("agent binary");
    let mut command = Command::new(binary);
    command.env("HOME", home.path()).args(["memory", "import"]);
    let output = command.output().await.expect("import");
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).expect("stdout"),
        "imported 2 memories · 1 skipped\n\
         \x20 Claude Code: imported 1 memory · 1 skipped\n\
         \x20 Codex: imported 1 memory\n"
    );

    let output = command.output().await.expect("repeat import");
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).expect("stdout"),
        "nothing new to import · 2 already there · 1 skipped\n\
         \x20 Claude Code: nothing new to import · 1 already there · 1 skipped\n\
         \x20 Codex: nothing new to import · 1 already there\n"
    );

    std::fs::write(codex.join("memory_summary.md"), "").expect("empty Codex source");
    let output = command.output().await.expect("import after removal");
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).expect("stdout"),
        "nothing new to import · 1 retired · 1 already there · 1 skipped\n\
         \x20 Claude Code: nothing new to import · 1 already there · 1 skipped\n\
         \x20 Codex: nothing new to import · 1 retired\n"
    );

    let output = command
        .arg("codex=~/missing\nsource")
        .output()
        .await
        .expect("unreadable source");
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8(output.stdout).expect("stdout");
    assert!(
        text.starts_with(
            "nothing new to import · 1 skipped\n\
         \x20 Codex: nothing new to import · 1 skipped\n\
         \x20 skipped Codex ~/missing\\nsource: could not read: "
        ),
        "{text}"
    );
    assert!(text.contains("no Codex memory files found"), "{text}");
    assert_eq!(text.lines().count(), 3);
}

#[tokio::test]
async fn http_custom_prompt_composes_with_opted_in_context() {
    let prompt = prompt_with_context(&[
        "--system-prompt",
        "Custom instructions.",
        "--include-project-instructions",
        "--include-skills",
    ])
    .await;
    assert!(prompt.starts_with("Custom instructions.\n\n"));
    assert!(prompt.contains("Local project instruction."));
    assert!(prompt.contains("Global instruction."));
    assert!(prompt.contains("Local eval skill."));
    assert!(prompt.contains("Global eval skill."));
}

#[tokio::test]
async fn http_empty_custom_prompt_keeps_opted_in_context() {
    let prompt = prompt_with_context(&[
        "--system-prompt",
        "",
        "--include-project-instructions",
        "--include-skills",
    ])
    .await;
    assert!(prompt.contains("Local project instruction."));
    assert!(prompt.contains("Local eval skill."));
}
