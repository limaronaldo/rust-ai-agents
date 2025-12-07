//! MCP Transport implementations
//!
//! Supports STDIO and SSE transports as per the MCP specification.

use async_trait::async_trait;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, error, warn};

use crate::error::McpError;
use crate::protocol::{JsonRpcRequest, JsonRpcResponse, RequestId};

/// Transport trait for MCP communication
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a request and wait for response
    async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse, McpError>;

    /// Send a notification (no response expected)
    async fn notify(&self, method: &str, params: Option<serde_json::Value>)
        -> Result<(), McpError>;

    /// Close the transport
    async fn close(&self) -> Result<(), McpError>;
}

// =============================================================================
// STDIO Transport
// =============================================================================

/// STDIO transport - launches MCP server as subprocess
pub struct StdioTransport {
    sender: mpsc::Sender<OutgoingMessage>,
    pending: Arc<Mutex<HashMap<RequestId, oneshot::Sender<JsonRpcResponse>>>>,
    request_id: AtomicI64,
    _child: Arc<Mutex<Child>>,
}

enum OutgoingMessage {
    Request(JsonRpcRequest),
    Notification(String),
}

impl StdioTransport {
    /// Spawn a new MCP server process
    ///
    /// # Arguments
    /// * `command` - The command to execute (e.g., "npx", "python")
    /// * `args` - Arguments to pass to the command
    ///
    /// # Example
    /// ```rust,ignore
    /// let transport = StdioTransport::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]).await?;
    /// ```
    pub async fn spawn(command: &str, args: &[&str]) -> Result<Self, McpError> {
        Self::spawn_with_env(command, args, HashMap::new()).await
    }

    /// Spawn with environment variables
    pub async fn spawn_with_env(
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
    ) -> Result<Self, McpError> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Transport(e.to_string()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("Failed to open stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("Failed to open stdout".to_string()))?;
        let stderr = child.stderr.take();

        let pending: Arc<Mutex<HashMap<RequestId, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let (tx, mut rx) = mpsc::channel::<OutgoingMessage>(100);

        // Writer task
        let mut stdin = stdin;
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let json = match &msg {
                    OutgoingMessage::Request(req) => serde_json::to_string(req),
                    OutgoingMessage::Notification(n) => Ok(n.clone()),
                };

                match json {
                    Ok(json) => {
                        let line = format!("{}\n", json);
                        if let Err(e) = stdin.write_all(line.as_bytes()).await {
                            error!("Failed to write to MCP server: {}", e);
                            break;
                        }
                        if let Err(e) = stdin.flush().await {
                            error!("Failed to flush to MCP server: {}", e);
                            break;
                        }
                        debug!("Sent to MCP: {}", json);
                    }
                    Err(e) => {
                        error!("Failed to serialize message: {}", e);
                    }
                }
            }
        });

        // Reader task
        let pending_clone = pending.clone();
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                debug!("Received from MCP: {}", line);

                match serde_json::from_str::<JsonRpcResponse>(&line) {
                    Ok(response) => {
                        let mut pending = pending_clone.lock().await;
                        if let Some(sender) = pending.remove(&response.id) {
                            let _ = sender.send(response);
                        }
                    }
                    Err(e) => {
                        // Might be a notification, try parsing as generic JSON
                        debug!("Failed to parse as response (might be notification): {}", e);
                    }
                }
            }
        });

        // Stderr reader task
        if let Some(stderr) = stderr {
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();

                while let Ok(Some(line)) = lines.next_line().await {
                    warn!("MCP server stderr: {}", line);
                }
            });
        }

        Ok(Self {
            sender: tx,
            pending,
            request_id: AtomicI64::new(1),
            _child: Arc::new(Mutex::new(child)),
        })
    }

    fn next_id(&self) -> RequestId {
        RequestId::Number(self.request_id.fetch_add(1, Ordering::SeqCst))
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn request(&self, mut request: JsonRpcRequest) -> Result<JsonRpcResponse, McpError> {
        // Assign ID if not set
        if matches!(request.id, RequestId::Number(0)) {
            request.id = self.next_id();
        }

        let (tx, rx) = oneshot::channel();

        {
            let mut pending = self.pending.lock().await;
            pending.insert(request.id.clone(), tx);
        }

        self.sender
            .send(OutgoingMessage::Request(request.clone()))
            .await
            .map_err(|_| McpError::ConnectionClosed)?;

        // Wait for response with timeout
        match tokio::time::timeout(std::time::Duration::from_secs(30), rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(McpError::ConnectionClosed),
            Err(_) => {
                // Remove pending request
                let mut pending = self.pending.lock().await;
                pending.remove(&request.id);
                Err(McpError::Timeout)
            }
        }
    }

    async fn notify(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), McpError> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });

        self.sender
            .send(OutgoingMessage::Notification(notification.to_string()))
            .await
            .map_err(|_| McpError::ConnectionClosed)
    }

    async fn close(&self) -> Result<(), McpError> {
        let mut child = self._child.lock().await;
        child
            .kill()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))
    }
}

// =============================================================================
// SSE Transport
// =============================================================================

/// SSE transport - connects to HTTP-based MCP servers
pub struct SseTransport {
    #[allow(dead_code)]
    base_url: String,
    post_endpoint: Arc<Mutex<Option<String>>>,
    client: reqwest::Client,
    request_id: AtomicI64,
}

impl SseTransport {
    /// Connect to an SSE-based MCP server
    ///
    /// # Arguments
    /// * `url` - The SSE endpoint URL
    ///
    /// # Example
    /// ```rust,ignore
    /// let transport = SseTransport::connect("http://localhost:3000/sse").await?;
    /// ```
    pub async fn connect(url: &str) -> Result<Self, McpError> {
        let client = reqwest::Client::new();

        // Connect to SSE endpoint to get the POST endpoint
        let response = client
            .get(url)
            .header("Accept", "text/event-stream")
            .send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        if !response.status().is_success() {
            return Err(McpError::Transport(format!(
                "SSE connection failed: {}",
                response.status()
            )));
        }

        // For now, assume POST endpoint is at the same base URL + /message
        let base_url = url.trim_end_matches("/sse").to_string();
        let post_endpoint = format!("{}/message", base_url);

        Ok(Self {
            base_url,
            post_endpoint: Arc::new(Mutex::new(Some(post_endpoint))),
            client,
            request_id: AtomicI64::new(1),
        })
    }

    fn next_id(&self) -> RequestId {
        RequestId::Number(self.request_id.fetch_add(1, Ordering::SeqCst))
    }
}

#[async_trait]
impl McpTransport for SseTransport {
    async fn request(&self, mut request: JsonRpcRequest) -> Result<JsonRpcResponse, McpError> {
        if matches!(request.id, RequestId::Number(0)) {
            request.id = self.next_id();
        }

        let endpoint = self.post_endpoint.lock().await;
        let url = endpoint
            .as_ref()
            .ok_or_else(|| McpError::Transport("POST endpoint not available".to_string()))?;

        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        if !response.status().is_success() {
            return Err(McpError::Transport(format!(
                "Request failed: {}",
                response.status()
            )));
        }

        let json_response: JsonRpcResponse = response
            .json()
            .await
            .map_err(|e| McpError::InvalidResponse(e.to_string()))?;

        Ok(json_response)
    }

    async fn notify(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), McpError> {
        let endpoint = self.post_endpoint.lock().await;
        let url = endpoint
            .as_ref()
            .ok_or_else(|| McpError::Transport("POST endpoint not available".to_string()))?;

        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });

        self.client
            .post(url)
            .json(&notification)
            .send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        Ok(())
    }

    async fn close(&self) -> Result<(), McpError> {
        // SSE connections are stateless, nothing to close
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_generation() {
        let id1 = RequestId::from(1i64);
        let id2 = RequestId::from("test".to_string());

        assert_eq!(id1, RequestId::Number(1));
        assert_eq!(id2, RequestId::String("test".to_string()));
    }
}
