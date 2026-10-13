use std::collections::VecDeque;

use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

pub(crate) struct TestServer {
    pub base_url: String,
    requests: mpsc::UnboundedReceiver<TestRequest>,
    task: tokio::task::JoinHandle<()>,
}

impl TestServer {
    pub async fn start(responses: Vec<TestResponse>) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server listener");
        let base_url = format!("http://{}", listener.local_addr().expect("address"));
        let (requests, request_receiver) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let mut responses = VecDeque::from(responses);
            while let Some(response) = responses.pop_front() {
                let (mut stream, _) = listener.accept().await.expect("accept");
                let request = read_request(&mut stream).await;
                requests.send(request).expect("request receiver");
                write_response(&mut stream, response).await;
            }
        });
        Self {
            base_url,
            requests: request_receiver,
            task,
        }
    }

    pub async fn shell(workspace: &std::path::Path) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("shell listener");
        let base_url = format!("http://{}", listener.local_addr().expect("address"));
        let workspace = workspace.to_path_buf();
        let path = std::env::var_os("MIRROR_TEST_PYTHON")
            .map(|python| {
                let python = std::path::PathBuf::from(python)
                    .canonicalize()
                    .expect("Python interpreter");
                format!(
                    "{}:/usr/bin:/bin",
                    python.parent().expect("Python bin directory").display()
                )
            })
            .unwrap_or_else(|| std::env::var("PATH").expect("PATH"));
        let rg = std::env::var_os("MIRROR_RG_PATH").map(|rg| {
            std::path::PathBuf::from(rg)
                .canonicalize()
                .expect("ripgrep")
        });
        let path = rg.map_or(path.clone(), |rg| {
            format!(
                "{}:{path}",
                rg.parent().expect("ripgrep directory").display()
            )
        });
        let (requests, request_receiver) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.expect("accept shell request");
                let request = read_request(&mut stream).await;
                let command = request
                    .body
                    .as_str()
                    .expect("raw shell command")
                    .to_string();
                requests.send(request).expect("record shell request");
                let output = tokio::process::Command::new("/bin/bash")
                    .args(["-c", &command])
                    .current_dir(&workspace)
                    .env_clear()
                    .env("PATH", &path)
                    .env("OPENAI_API_KEY", "synthetic-remote-model-key")
                    .kill_on_drop(true)
                    .output()
                    .await
                    .expect("execute shell request");
                assert!(
                    output.status.success(),
                    "synthetic wrapper failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let response = TestResponse::json(
                    200,
                    serde_json::json!({
                        "output": format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr)),
                        "exit_code": output.status.code().unwrap_or(-1),
                    }),
                );
                write_response(&mut stream, response).await;
            }
        });
        Self {
            base_url,
            requests: request_receiver,
            task,
        }
    }

    pub async fn request(&mut self) -> TestRequest {
        self.requests.recv().await.expect("request")
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

pub(crate) struct TestResponse {
    status: u16,
    body: String,
}

impl TestResponse {
    pub fn json(status: u16, body: serde_json::Value) -> Self {
        Self {
            status,
            body: body.to_string(),
        }
    }

    pub fn text(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
        }
    }
}

pub(crate) struct TestRequest {
    pub path: String,
    pub authorization: Option<String>,
    pub body: serde_json::Value,
}

async fn read_request(stream: &mut tokio::net::TcpStream) -> TestRequest {
    let mut bytes = Vec::new();
    let mut buffer = [0; 4096];
    let header_end = loop {
        let read = stream.read(&mut buffer).await.expect("read request");
        assert!(read > 0, "request closed before headers");
        bytes.extend_from_slice(&buffer[..read]);
        if let Some(position) = find_bytes(&bytes, b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers = String::from_utf8(bytes[..header_end].to_vec()).expect("headers are UTF-8");
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().expect("content length"))
        })
        .unwrap_or(0);
    while bytes.len() - header_end < content_length {
        let read = stream.read(&mut buffer).await.expect("read body");
        assert!(read > 0, "request closed before body");
        bytes.extend_from_slice(&buffer[..read]);
    }
    let mut lines = headers.lines();
    let path = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .expect("request path")
        .to_string();
    let authorization = lines.find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("authorization")
            .then(|| value.trim().to_string())
    });
    let body_bytes = &bytes[header_end..header_end + content_length];
    let body = if content_length == 0 {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(body_bytes).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8(body_bytes.to_vec()).expect("UTF-8 body"))
        })
    };
    TestRequest {
        path,
        authorization,
        body,
    }
}

async fn write_response(stream: &mut tokio::net::TcpStream, response: TestResponse) {
    let reason = match response.status {
        200 => "OK",
        401 => "Unauthorized",
        429 => "Too Many Requests",
        503 => "Service Unavailable",
        _ => "Error",
    };
    let headers = format!(
        "HTTP/1.1 {} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        response.body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .await
        .expect("write headers");
    stream
        .write_all(response.body.as_bytes())
        .await
        .expect("write body");
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// A host-writing fixture deliberately lacking remote support.
pub(crate) struct LocalOnlyTool(pub std::path::PathBuf);

impl super::Tool for LocalOnlyTool {
    type Args = serde_json::Value;
    fn name(&self) -> &str {
        "local_only"
    }
    fn description(&self) -> &str {
        "Local-only test fixture."
    }
    fn parameters(&self) -> serde_json::Map<String, serde_json::Value> {
        super::object_schema(serde_json::Map::new(), &[])
    }
    async fn run(
        &self,
        _args: Self::Args,
        _context: &crate::ExecutionContext,
    ) -> Result<super::ToolOutput, super::ToolError> {
        std::fs::write(&self.0, "unexpected local execution").map_err(super::execution_error)?;
        Ok(super::ToolOutput::success("local"))
    }
}
