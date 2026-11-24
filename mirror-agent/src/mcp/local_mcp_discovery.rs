use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use rmcp::model::{CallToolResult, JsonObject, Tool};
use tokio::sync::{mpsc, oneshot, watch};

use super::{LocalMcpConnection, McpClient, McpDiscovery, McpRouter, McpTool};
use crate::tools::ToolError;

/// Owns live host declarations and a bounded, lazily connected MCP session cache.
pub struct LocalMcpDiscovery {
    connections: watch::Sender<Arc<Vec<LocalMcpConnection>>>,
    requests: mpsc::Sender<oneshot::Sender<Result<Vec<Arc<McpTool>>, ToolError>>>,
    http_only: bool,
}

// A pending approval keeps its original route, but cannot dispatch after replacement.
struct LocalMcpRoute {
    client: Arc<McpClient>,
    selected: Arc<Vec<LocalMcpConnection>>,
    current: watch::Receiver<Arc<Vec<LocalMcpConnection>>>,
}

impl LocalMcpDiscovery {
    pub fn new(connections: Vec<LocalMcpConnection>, http_only: bool) -> Result<Self, ToolError> {
        let (sender, receiver) = watch::channel(Arc::new(Vec::new()));
        let (requests, mut pending) =
            mpsc::channel::<oneshot::Sender<Result<Vec<Arc<McpTool>>, ToolError>>>(8);
        let source = Self {
            connections: sender,
            requests,
            http_only,
        };
        source.update(connections, true)?;
        tokio::spawn(async move {
            let mut current = receiver;
            let mut clients = HashMap::new();
            loop {
                tokio::select! {
                    changed = current.changed() => {
                        if changed.is_err() { break; }
                        clients.clear();
                    }
                    request = pending.recv() => {
                        let Some(request) = request else { break; };
                        if request.is_closed() { continue; }
                        if current.has_changed().unwrap_or(false) { clients.clear(); }
                        let selected = current.borrow_and_update().clone();
                        let result = tokio::time::timeout(Duration::from_secs(25), discover(&selected, &current, &mut clients)).await;
                        let result = if Arc::ptr_eq(&selected, &current.borrow()) {
                            result.map_err(|_| ToolError::Execution("MCP discovery timed out".into()))
                        } else {
                            clients.clear();
                            Err(changed())
                        };
                        let _ = request.send(result);
                    }
                }
            }
        });
        Ok(source)
    }

    /// Replace ephemeral inputs without restarting tools already executing.
    pub fn update(
        &self,
        connections: Vec<LocalMcpConnection>,
        reconnect: bool,
    ) -> Result<(), ToolError> {
        if connections.len() > 128 {
            return Err(ToolError::InvalidArguments(
                "Too many MCP connections".into(),
            ));
        }
        let mut names = HashSet::new();
        for connection in &connections {
            connection.timeouts()?;
            if !names.insert(&connection.name) || (self.http_only && !connection.is_http()) {
                return Err(ToolError::InvalidArguments(
                    "Duplicate or unsupported MCP connection".into(),
                ));
            }
        }
        if reconnect || self.connections.borrow().as_ref() != &connections {
            self.connections.send_replace(Arc::new(connections));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpDiscovery for LocalMcpDiscovery {
    async fn discover(&self) -> Result<Vec<Arc<McpTool>>, ToolError> {
        let (response, receiver) = oneshot::channel();
        tokio::time::timeout(Duration::from_secs(30), async {
            self.requests
                .send(response)
                .await
                .map_err(|_| unavailable())?;
            receiver.await.map_err(|_| unavailable())?
        })
        .await
        .map_err(|_| ToolError::Execution("MCP discovery timed out".into()))?
    }
}

async fn discover(
    connections: &Arc<Vec<LocalMcpConnection>>,
    current: &watch::Receiver<Arc<Vec<LocalMcpConnection>>>,
    clients: &mut HashMap<String, Arc<McpClient>>,
) -> Vec<Arc<McpTool>> {
    let discoveries = connections.iter().map(|connection| {
        let client = clients
            .get(&connection.name)
            .filter(|client| !client.is_closed())
            .cloned();
        async move {
            let (startup, tool) = connection.timeouts()?;
            tokio::time::timeout(startup, async {
                let client = match client {
                    Some(client) => client,
                    None => Arc::new(connection.connect(startup, tool).await?),
                };
                let route = Arc::new(LocalMcpRoute {
                    client: client.clone(),
                    selected: connections.clone(),
                    current: current.clone(),
                });
                let mut tools = McpTool::discover(route, &connection.name, false).await?;
                tools.retain(|tool| connection.allows(tool));
                Ok::<_, ToolError>((client, tools))
            })
            .await
            .map_err(|_| {
                ToolError::Execution(format!("MCP discovery timed out for {}", connection.name))
            })?
        }
    });
    let results = futures_util::future::join_all(discoveries).await;
    let mut tools = Vec::new();
    for (connection, result) in connections.iter().zip(results) {
        match result {
            Ok((client, discovered)) => {
                clients.insert(connection.name.clone(), client);
                tools.extend(discovered);
            }
            Err(error) => {
                clients.remove(&connection.name);
                eprintln!("{error}");
            }
        }
    }
    tools
}

impl McpRouter for LocalMcpRoute {
    fn list_tools<'a>(
        &'a self,
        server: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Tool>, ToolError>> + Send + 'a>> {
        McpRouter::list_tools(self.client.as_ref(), server)
    }

    fn call_tool<'a>(
        &'a self,
        server: &'a str,
        tool: &'a str,
        arguments: JsonObject,
        tool_call_id: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<CallToolResult, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            if !Arc::ptr_eq(&self.selected, &self.current.borrow()) {
                return Err(changed());
            }
            McpRouter::call_tool(self.client.as_ref(), server, tool, arguments, tool_call_id).await
        })
    }
}

fn changed() -> ToolError {
    ToolError::Execution(
        "MCP connections changed; search and describe the current tools before calling again"
            .into(),
    )
}

fn unavailable() -> ToolError {
    ToolError::Execution("MCP discovery is unavailable".into())
}
