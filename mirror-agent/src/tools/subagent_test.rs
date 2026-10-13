use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::AssistantMessage;
use mirror_protocol::ChatRecord;
use pretty_assertions::assert_eq;

use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures_util::future::BoxFuture;
use mirror_protocol::InferenceConfig;
use serde_json::json;

use super::*;
use crate::providers::LlmError;
use crate::providers::ModelApi;
use crate::system_prompt::EnvironmentInfo;
use crate::task_manager::TaskEvent;
use crate::task_manager::TaskManager;
use crate::tools::web_test_server::TestResponse;
use crate::tools::web_test_server::TestServer;
use crate::AgentApprovalMode;

#[test]
fn final_answer_returns_last_assistant_content() {
    let entries = vec![AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::Assistant(AssistantMessage {
                content: Some("done".to_string()),
                ..AssistantMessage::default()
            })),
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }];

    assert_eq!(final_answer(&entries).as_deref(), Some("done"));
}

#[test]
fn summary_collapses_whitespace() {
    assert_eq!(summary("one\n  two"), "one two");
}

struct Authority {
    active: AtomicBool,
    rejected: AtomicUsize,
}

impl ModelRequestGuard for Authority {
    fn verify(&self) -> BoxFuture<'_, Result<(), LlmError>> {
        Box::pin(async move {
            if self.active.load(Ordering::SeqCst) {
                Ok(())
            } else {
                self.rejected.fetch_add(1, Ordering::SeqCst);
                Err(LlmError::Request("lease revoked".to_string()))
            }
        })
    }
}

struct Revoke(Arc<Authority>);

impl Tool for Revoke {
    type Args = Value;

    fn name(&self) -> &str {
        "revoke"
    }

    fn description(&self) -> &str {
        "Revoke model authority."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(Map::new(), &[])
    }

    async fn run(
        &self,
        _args: Value,
        _context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        self.0.active.store(false, Ordering::SeqCst);
        // Finish the initial HTTP exchange before fast-forwarding retry backoff.
        tokio::time::pause();
        Ok(ToolOutput::success("revoked"))
    }
}

#[tokio::test]
async fn detached_subagent_checks_authority_before_its_next_model_request() {
    let first_response = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices": [{
            "delta": {"tool_calls": [{
                "index": 0, "id": "revoke-1", "type": "function",
                "function": {"name": "revoke", "arguments": "{}"}
            }]},
            "finish_reason": "tool_calls"
        }]})
    );
    let mut server = TestServer::start(vec![
        TestResponse::text(200, first_response),
        TestResponse::text(200, "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"),
    ]).await;
    let workspace = tempfile::tempdir().expect("workspace");
    std::fs::write(
        workspace.path().join("AGENTS.md"),
        "HOST GUIDANCE MUST NOT LEAK",
    )
    .expect("host guidance");
    let authority = Arc::new(Authority {
        active: AtomicBool::new(true),
        rejected: AtomicUsize::new(0),
    });
    let tool = SubagentTool::new(
        workspace.path(),
        "parent-session",
        ModelProviderConfig {
            context_window: Some(crate::providers::DEFAULT_CONTEXT_WINDOW),
            api: ModelApi::ChatCompletions,
            model: "test-model".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(server.base_url.clone()),
            auth_path: None,
            prompt_cache_key: "parent-session".to_string(),
            request_timeout: None,
        },
        ReactAgentConfig::default(),
        Some(EnvironmentInfo {
            working_directory: "/workspace/repository".into(),
            is_git_repo: true,
            platform: "Linux".into(),
            os_version: "Linux 6.1.0".into(),
            shell: "bash 5.2.21".into(),
        }),
        vec![Arc::new(Revoke(authority.clone()))],
    )
    .expect("subagent tool")
    .with_resolved_context(crate::system_prompt::ResolvedInstructions {
        instructions: vec![(
            "Hosted instructions".into(),
            "HOSTED REPOSITORY GUIDANCE".into(),
        )],
        skills: vec![],
    })
    .with_request_guard(authority.clone());
    let (manager, tasks, mut events, runner) = TaskManager::new();
    let manager = tokio::spawn(manager.run());
    let runner = tokio::spawn(runner);
    let context = ExecutionContext::new(
        AgentApprovalMode::Auto,
        None,
        tasks.clone(),
        None,
        Some("Inherited instructions.\n\n".into()),
    );
    context.input().set_inference_config(Some(InferenceConfig {
        model: "test-model".to_string(),
        ..Default::default()
    }));
    let started = tool
        .run(
            SubagentArgs {
                prompt: "Call revoke, then answer.".to_string(),
            },
            &context,
        )
        .await
        .expect("start subagent");
    assert!(started.success);
    // The default exception limit allows four retries with 10 seconds of backoff.
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if let TaskEvent::Finished(_, result) = events.recv().await.expect("task event") {
                break result;
            }
        }
    })
    .await
    .expect("subagent finishes");
    assert!(
        !authority.active.load(Ordering::SeqCst),
        "the first tool call revoked authority"
    );
    assert!(
        authority.rejected.load(Ordering::SeqCst) > 0,
        "no follow-up model request was rejected: {result:?}"
    );
    let request = server.request().await;
    assert_eq!(request.path, "/chat/completions");
    let body = request.body.to_string();
    assert!(body.contains("HOSTED REPOSITORY GUIDANCE"));
    let system_prompt = request.body["messages"][0]["content"]
        .as_str()
        .expect("system prompt");
    assert!(system_prompt.starts_with(
        "Inherited instructions.\n\n# Environment\n- Working directory: /workspace/repository\n"
    ));
    assert!(!body.contains("HOST GUIDANCE MUST NOT LEAK"));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), server.request())
            .await
            .is_err()
    );
    tasks.close().await.expect("close tasks");
    manager.await.expect("join manager");
    runner.abort();
}

struct GatedTool(Arc<AtomicUsize>);

impl Tool for GatedTool {
    type Args = Value;

    fn name(&self) -> &str {
        "gated"
    }
    fn description(&self) -> &str {
        "Record an approved execution."
    }
    fn parameters(&self) -> Map<String, Value> {
        object_schema(Map::new(), &[])
    }
    fn needs_approval(&self) -> bool {
        true
    }

    async fn run(
        &self,
        _args: Value,
        _context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ToolOutput::success("executed"))
    }
}

async fn check_subagent_approval(approved: bool) {
    let response = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices": [{
            "delta": {"tool_calls": [{
                "index": 0, "id": "gated-1", "type": "function",
                "function": {"name": "gated", "arguments": "{}"}
            }]},
            "finish_reason": "tool_calls"
        }]})
    );
    let server = TestServer::start(vec![
        TestResponse::text(200, response),
        TestResponse::text(200, "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"),
    ]).await;
    let workspace = tempfile::tempdir().expect("workspace");
    let executions = Arc::new(AtomicUsize::new(0));
    let tool = SubagentTool::new(
        workspace.path(),
        "parent",
        ModelProviderConfig {
            api: ModelApi::ChatCompletions,
            model: "test-model".into(),
            api_key: Some("test-key".into()),
            base_url: Some(server.base_url.clone()),
            auth_path: None,
            prompt_cache_key: "parent".into(),
            context_window: Some(crate::providers::DEFAULT_CONTEXT_WINDOW),
            request_timeout: None,
        },
        ReactAgentConfig::default(),
        None,
        vec![Arc::new(GatedTool(executions.clone()))],
    )
    .expect("subagent tool");
    let (manager, tasks, mut events, runner) = TaskManager::new();
    let manager = tokio::spawn(manager.run());
    let runner = tokio::spawn(runner);
    let (hook, mut approvals) = crate::ApprovalHook::channel();
    let context = ExecutionContext::new(
        AgentApprovalMode::Hitl,
        Some(hook),
        tasks.clone(),
        None,
        None,
    );
    context.input().set_inference_config(Some(InferenceConfig {
        model: "test-model".into(),
        ..Default::default()
    }));
    let started = tool
        .run(
            SubagentArgs {
                prompt: "Call gated.".into(),
            },
            &context,
        )
        .await
        .expect("start child");
    assert!(started.success);
    let request = tokio::time::timeout(Duration::from_secs(5), approvals.recv())
        .await
        .expect("child requests parent approval")
        .expect("approval channel open");
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    assert_eq!(request.call.function.expect("function").name, "gated");
    request
        .response
        .send(crate::ApprovalDecision::new(
            approved,
            mirror_protocol::ApprovalSource::Manual,
        ))
        .expect("resolve approval");
    let result = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let TaskEvent::Finished(_, result) = events.recv().await.expect("task event") {
                break result;
            }
        }
    })
    .await
    .expect("child finishes");
    assert!(result.success, "{result:?}");
    assert_eq!(executions.load(Ordering::SeqCst), usize::from(approved));
    tasks.close().await.expect("close tasks");
    manager.await.expect("join manager");
    runner.abort();
}

#[tokio::test]
async fn subagent_discovers_instructions_through_the_workspace_path_it_was_given() {
    let mut server = TestServer::start(vec![TestResponse::text(
        200,
        "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    )])
    .await;
    let directory = tempfile::tempdir().expect("directory");
    let checkout = directory.path().join("checkout");
    std::fs::create_dir(&checkout).expect("checkout");
    std::fs::write(checkout.join("AGENTS.md"), "Linked guidance.").expect("instructions");
    let linked = directory.path().join("linked");
    std::os::unix::fs::symlink(&checkout, &linked).expect("workspace link");
    let tool = SubagentTool::new(
        &linked,
        "parent",
        ModelProviderConfig {
            api: ModelApi::ChatCompletions,
            model: "test-model".into(),
            api_key: Some("test-key".into()),
            base_url: Some(server.base_url.clone()),
            auth_path: None,
            prompt_cache_key: "parent".into(),
            context_window: Some(crate::providers::DEFAULT_CONTEXT_WINDOW),
            request_timeout: None,
        },
        ReactAgentConfig::default(),
        None,
        Vec::new(),
    )
    .expect("subagent tool");
    let (manager, tasks, mut events, runner) = TaskManager::new();
    let manager = tokio::spawn(manager.run());
    let runner = tokio::spawn(runner);
    let context = ExecutionContext::new(AgentApprovalMode::Auto, None, tasks.clone(), None, None);
    context.input().set_inference_config(Some(InferenceConfig {
        model: "test-model".into(),
        ..Default::default()
    }));

    let started = tool
        .run(
            SubagentArgs {
                prompt: "Answer.".into(),
            },
            &context,
        )
        .await
        .expect("start child");
    assert!(started.success);
    tokio::time::timeout(Duration::from_secs(5), async {
        while !matches!(
            events.recv().await.expect("task event"),
            TaskEvent::Finished(..)
        ) {}
    })
    .await
    .expect("child finishes");

    let request = server.request().await;
    let system_prompt = request.body["messages"][0]["content"]
        .as_str()
        .expect("system prompt");
    assert!(
        system_prompt.contains(&format!(
            "Instructions from: {}\nLinked guidance.",
            linked.join("AGENTS.md").display()
        )),
        "{system_prompt}"
    );
    tasks.close().await.expect("close tasks");
    manager.await.expect("join manager");
    runner.abort();
}

#[tokio::test]
async fn subagent_waits_for_parent_approval() {
    check_subagent_approval(true).await;
}

#[tokio::test]
async fn subagent_does_not_execute_denied_tool() {
    check_subagent_approval(false).await;
}

async fn descendant_model(
    axum::extract::State(requests): axum::extract::State<tokio::sync::mpsc::UnboundedSender<Value>>,
    axum::Json(body): axum::Json<Value>,
) -> String {
    requests.send(body.clone()).expect("record model request");
    let messages = body["messages"].as_array().expect("messages");
    let tool = messages
        .iter()
        .rev()
        .find(|message| message["role"] == "tool");
    let (name, arguments) = match tool {
        None => {
            let prompt = messages.iter().find(|message| message["role"] == "user").expect("prompt")["content"].as_str().expect("text");
            if prompt == "leaf" {
                ("bash", json!({"command": "printf remote > descendant; printf remote"}))
            } else {
                ("subagent", json!({"prompt": "leaf"}))
            }
        }
        Some(tool) if tool["tool_call_id"] == "call-subagent" => {
            let content = tool["content"].as_str().expect("subagent result");
            let id = content.strip_prefix("Subagent task ").expect("detached task").split_whitespace().next().expect("task id");
            ("task", json!({"action":"wait", "task_id":id}))
        }
        Some(tool) if tool["tool_call_id"] == "call-bash" => ("read", json!({"file_path":"host-only"})),
        _ => return "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into(),
    };
    format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":format!("call-{name}"),"type":"function","function":{"name":name,"arguments":arguments.to_string()}}]},"finish_reason":"tool_calls"}]})
    )
}

#[tokio::test]
async fn remote_subagent_and_grandchild_retain_target_and_exclude_host_context() {
    use crate::tools::{BashTool, ReadTool, Task};

    let host = tempfile::tempdir().expect("host workspace");
    let remote_workspace = tempfile::tempdir().expect("remote workspace");
    std::fs::write(host.path().join("AGENTS.md"), "HOST_CONTEXT_CANARY").expect("instructions");
    std::fs::write(host.path().join("host-only"), "HOST_SECRET_CANARY").expect("host data");
    let mut shell = TestServer::shell(remote_workspace.path()).await;
    let (requests, mut received) = tokio::sync::mpsc::unbounded_channel();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("model listener");
    let url = format!("http://{}", listener.local_addr().expect("address"));
    let app = axum::Router::new()
        .route("/chat/completions", axum::routing::post(descendant_model))
        .with_state(requests);
    let model_server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("model server");
    });
    let provider = ModelProviderConfig {
        api: ModelApi::ChatCompletions,
        model: "test-model".into(),
        api_key: Some("synthetic-model-secret".into()),
        base_url: Some(url),
        auth_path: None,
        prompt_cache_key: "parent".into(),
        context_window: Some(crate::providers::DEFAULT_CONTEXT_WINDOW),
        request_timeout: None,
    };
    let grandchild = SubagentTool::new(
        host.path(),
        "child",
        provider.clone(),
        ReactAgentConfig::default(),
        None,
        vec![
            Arc::new(BashTool::new(host.path()).expect("bash")),
            Arc::new(ReadTool::remote()),
        ],
    )
    .expect("grandchild");
    let child = SubagentTool::new(
        host.path(),
        "parent",
        provider,
        ReactAgentConfig::default(),
        None,
        vec![Arc::new(grandchild), Arc::new(Task)],
    )
    .expect("child")
    .with_resolved_context(ResolvedInstructions {
        instructions: vec![("host".into(), "HOST_CONTEXT_CANARY".into())],
        skills: vec![],
    });
    let executor = RemoteToolExecutor::new(
        vec![Arc::new(child)],
        reqwest::Url::parse(&shell.base_url).expect("URL"),
    )
    .expect("executor");
    let (manager, tasks, mut events, runner) = TaskManager::new();
    let manager = tokio::spawn(manager.run());
    let runner = tokio::spawn(runner);
    let context = ExecutionContext::new(
        AgentApprovalMode::Auto,
        None,
        tasks.clone(),
        None,
        Some("INHERITED_SAFETY_POLICY".into()),
    );
    context.input().set_inference_config(Some(InferenceConfig {
        model: "test-model".into(),
        ..Default::default()
    }));
    let started = executor
        .call_tool(
            "subagent".into(),
            json!({"prompt":"branch"}).to_string(),
            context,
        )
        .await
        .expect("dispatch");
    assert!(started.success, "{started:?}");
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if let TaskEvent::Finished(_, result) = events.recv().await.expect("task event") {
                break result;
            }
        }
    })
    .await
    .expect("descendants finish");
    assert!(result.success, "{result:?}");
    assert_eq!(result.content, "done");
    assert!(!host.path().join("descendant").exists());
    assert_eq!(
        std::fs::read_to_string(remote_workspace.path().join("descendant")).expect("remote write"),
        "remote"
    );
    let shell_request = shell.request().await;
    assert_eq!(shell_request.authorization, None);
    assert!(!shell_request
        .body
        .to_string()
        .contains("synthetic-model-secret"));
    let mut count = 0;
    let mut remote_read_rejected = false;
    while let Ok(request) = received.try_recv() {
        count += 1;
        let text = request.to_string();
        assert!(!text.contains("HOST_CONTEXT_CANARY"));
        assert!(!text.contains("HOST_SECRET_CANARY"));
        assert!(text.contains("INHERITED_SAFETY_POLICY"));
        remote_read_rejected |= text.contains("No such file or directory");
    }
    assert!(
        count >= 6,
        "both generations must finish their tool exchanges"
    );
    assert!(
        remote_read_rejected,
        "grandchild read must fail rather than read host files"
    );
    tasks.close().await.expect("close");
    manager.await.expect("manager");
    runner.abort();
    model_server.abort();
}

#[test]
fn remote_executor_rejects_unsupported_grandchild_tools_up_front() {
    let workspace = tempfile::tempdir().expect("workspace");
    let subagent = |tools| {
        SubagentTool::new(
            workspace.path(),
            "parent",
            ModelProviderConfig {
                api: ModelApi::ChatCompletions,
                model: "test-model".into(),
                api_key: None,
                base_url: Some("http://localhost:1".into()),
                auth_path: None,
                prompt_cache_key: "parent".into(),
                context_window: Some(crate::providers::DEFAULT_CONTEXT_WINDOW),
                request_timeout: None,
            },
            ReactAgentConfig::default(),
            None,
            tools,
        )
        .expect("subagent")
    };
    let grandchild = subagent(vec![Arc::new(
        crate::tools::web_test_server::LocalOnlyTool(workspace.path().join("file")),
    )]);
    let child = subagent(vec![Arc::new(grandchild)]);
    let error = RemoteToolExecutor::new(
        vec![Arc::new(child)],
        reqwest::Url::parse("http://localhost:1/exec").expect("URL"),
    )
    .expect_err("unsupported descendant");
    assert!(error
        .to_string()
        .contains("descendants do not support remote execution"));
}
