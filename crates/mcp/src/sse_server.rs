//! SSE Server Transport for MCP
//!
//! Implements HTTP-based MCP server transport using Server-Sent Events (SSE)
//! for server-to-client communication and POST requests for client-to-server.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     SSE Server Transport                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  GET /sse ─────────────► SSE Stream (server→client)         │
//! │                          - Sends endpoint event first       │
//! │                          - Then streams responses           │
//! │                                                              │
//! │  POST /message ────────► JSON-RPC Request (client→server)   │
//! │                          - Returns immediately              │
//! │                          - Response sent via SSE            │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_mcp::{McpServer, SseServerConfig};
//!
//! let server = McpServer::builder()
//!     .name("my-server")
//!     .add_tool(my_tool)
//!     .build();
//!
//! // Run with SSE transport on port 3000
//! server.run_sse(SseServerConfig::default()).await?;
//! ```

use axum::{
    extract::State,
    http::{header, Method},
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use http::StatusCode;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::error::McpError;
use crate::protocol::JsonRpcResponse;
use crate::server::McpServer;

// =============================================================================
// SSE Server Configuration
// =============================================================================

/// Configuration for the SSE server transport
#[derive(Debug, Clone)]
pub struct SseServerConfig {
    /// Host to bind to
    pub host: String,
    /// Port to bind to
    pub port: u16,
    /// Path for SSE endpoint
    pub sse_path: String,
    /// Path for message endpoint
    pub message_path: String,
    /// Enable CORS
    pub enable_cors: bool,
    /// Keep-alive interval in seconds
    pub keep_alive_secs: u64,
}

impl Default for SseServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            sse_path: "/sse".to_string(),
            message_path: "/message".to_string(),
            enable_cors: true,
            keep_alive_secs: 30,
        }
    }
}

impl SseServerConfig {
    /// Create config for localhost on specified port
    pub fn localhost(port: u16) -> Self {
        Self {
            port,
            ..Default::default()
        }
    }

    /// Create config that binds to all interfaces
    pub fn public(port: u16) -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port,
            ..Default::default()
        }
    }
}

// =============================================================================
// SSE Server State
// =============================================================================

/// Internal state for SSE server
struct SseServerState {
    /// Reference to the MCP server
    mcp_server: Arc<McpServer>,
    /// Configuration
    config: SseServerConfig,
    /// Active SSE sessions: session_id -> response sender
    sessions: RwLock<HashMap<String, mpsc::Sender<JsonRpcResponse>>>,
    /// Broadcast channel for shutdown
    shutdown_tx: broadcast::Sender<()>,
}

impl SseServerState {
    fn new(
        mcp_server: Arc<McpServer>,
        config: SseServerConfig,
        shutdown_tx: broadcast::Sender<()>,
    ) -> Self {
        Self {
            mcp_server,
            config,
            sessions: RwLock::new(HashMap::new()),
            shutdown_tx,
        }
    }

    fn register_session(&self, session_id: String, sender: mpsc::Sender<JsonRpcResponse>) {
        self.sessions.write().insert(session_id, sender);
    }

    fn unregister_session(&self, session_id: &str) {
        self.sessions.write().remove(session_id);
    }

    fn get_session_sender(&self, session_id: &str) -> Option<mpsc::Sender<JsonRpcResponse>> {
        self.sessions.read().get(session_id).cloned()
    }
}

// =============================================================================
// SSE Server Extension for McpServer
// =============================================================================

impl McpServer {
    /// Run the server with SSE transport
    ///
    /// This starts an HTTP server with:
    /// - GET /sse - SSE endpoint for server-to-client streaming
    /// - POST /message - Endpoint for client-to-server requests
    pub async fn run_sse(self: Arc<Self>, config: SseServerConfig) -> Result<(), McpError> {
        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        let state = Arc::new(SseServerState::new(
            self.clone(),
            config.clone(),
            shutdown_tx,
        ));

        let mut app = Router::new()
            .route(&config.sse_path, get(handle_sse))
            .route(&config.message_path, post(handle_message))
            .with_state(state.clone());

        if config.enable_cors {
            let cors = CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::CONTENT_TYPE, header::ACCEPT]);
            app = app.layer(cors);
        }

        let addr: SocketAddr = format!("{}:{}", config.host, config.port)
            .parse()
            .map_err(|e| McpError::Transport(format!("Invalid address: {}", e)))?;

        info!(
            "Starting MCP SSE server on http://{}{}",
            addr, config.sse_path
        );
        info!("Message endpoint: http://{}{}", addr, config.message_path);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| McpError::Transport(format!("Failed to bind: {}", e)))?;

        axum::serve(listener, app)
            .await
            .map_err(|e| McpError::Transport(format!("Server error: {}", e)))?;

        Ok(())
    }

    /// Run the server with SSE transport and return the router
    /// (for embedding in existing Axum applications)
    pub fn sse_router(self: Arc<Self>, config: SseServerConfig) -> Router {
        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        let state = Arc::new(SseServerState::new(
            self.clone(),
            config.clone(),
            shutdown_tx,
        ));

        let mut router = Router::new()
            .route(&config.sse_path, get(handle_sse))
            .route(&config.message_path, post(handle_message))
            .with_state(state);

        if config.enable_cors {
            let cors = CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::CONTENT_TYPE, header::ACCEPT]);
            router = router.layer(cors);
        }

        router
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

/// SSE endpoint event - tells client where to POST messages
#[derive(Debug, serde::Serialize)]
struct EndpointEvent {
    endpoint: String,
}

// =============================================================================
// HTTP Handlers
// =============================================================================

/// Handle SSE connection
async fn handle_sse(
    State(state): State<Arc<SseServerState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, mut rx) = mpsc::channel::<JsonRpcResponse>(100);

    state.register_session(session_id.clone(), tx);

    let config = state.config.clone();
    let state_clone = state.clone();
    let session_id_clone = session_id.clone();

    info!("New SSE session: {}", session_id);

    let stream = async_stream::stream! {
        // First, send the endpoint event
        let endpoint = EndpointEvent {
            endpoint: format!("{}?sessionId={}", config.message_path, session_id_clone),
        };
        let endpoint_json = serde_json::to_string(&endpoint).unwrap();
        yield Ok(Event::default().event("endpoint").data(endpoint_json));

        debug!("Sent endpoint event for session {}", session_id_clone);

        // Then stream responses
        let mut shutdown_rx = state_clone.shutdown_tx.subscribe();
        loop {
            tokio::select! {
                Some(response) = rx.recv() => {
                    match serde_json::to_string(&response) {
                        Ok(json) => {
                            debug!("Sending SSE message: {}", json);
                            yield Ok(Event::default().event("message").data(json));
                        }
                        Err(e) => {
                            error!("Failed to serialize response: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("SSE session {} shutting down", session_id_clone);
                    break;
                }
            }
        }

        state_clone.unregister_session(&session_id_clone);
        info!("SSE session {} closed", session_id_clone);
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(state.config.keep_alive_secs))
            .text("ping"),
    )
}

/// Query parameters for message endpoint
#[derive(Debug, Default, serde::Deserialize)]
struct MessageQuery {
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
}

/// Handle incoming JSON-RPC message
async fn handle_message(
    State(state): State<Arc<SseServerState>>,
    axum::extract::Query(query): axum::extract::Query<MessageQuery>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = query.session_id;

    debug!(
        "Received message for session {:?}: {}",
        session_id,
        serde_json::to_string_pretty(&body).unwrap_or_default()
    );

    // Parse as JSON-RPC request
    let request = match serde_json::from_value::<crate::protocol::JsonRpcRequest>(body.clone()) {
        Ok(req) => req,
        Err(e) => {
            error!("Failed to parse JSON-RPC request: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {}", e)
                    }
                })),
            );
        }
    };

    // Handle the request
    let response = state.mcp_server.handle_request(request).await;

    // If we have a session, send via SSE; otherwise return directly
    if let Some(ref sid) = session_id {
        if let Some(sender) = state.get_session_sender(sid) {
            if sender.send(response.clone()).await.is_ok() {
                // Return accepted - response will come via SSE
                return (
                    StatusCode::ACCEPTED,
                    Json(serde_json::json!({"status": "accepted"})),
                );
            }
        }
    }

    // No session or send failed - return response directly
    let response_json = serde_json::to_value(&response).unwrap_or_default();
    (StatusCode::OK, Json(response_json))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::FnTool;
    use serde_json::json;

    fn create_test_server() -> Arc<McpServer> {
        McpServer::builder()
            .name("test-sse-server")
            .version("1.0.0")
            .add_tool(FnTool::new(
                "echo",
                "Echoes input",
                json!({
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    }
                }),
                |args| {
                    let msg = args["message"].as_str().unwrap_or("no message");
                    Ok(json!({"echoed": msg}))
                },
            ))
            .build()
    }

    #[test]
    fn test_sse_config_default() {
        let config = SseServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 3000);
        assert_eq!(config.sse_path, "/sse");
        assert_eq!(config.message_path, "/message");
        assert!(config.enable_cors);
    }

    #[test]
    fn test_sse_config_localhost() {
        let config = SseServerConfig::localhost(8080);
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_sse_config_public() {
        let config = SseServerConfig::public(9000);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 9000);
    }

    #[tokio::test]
    async fn test_sse_router_creation() {
        let server = create_test_server();
        let config = SseServerConfig::default();
        let _router = server.sse_router(config);
        // Router created successfully
    }

    #[tokio::test]
    async fn test_session_registration() {
        let server = create_test_server();
        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        let state = SseServerState::new(server, SseServerConfig::default(), shutdown_tx);

        let (tx, _rx) = mpsc::channel::<JsonRpcResponse>(10);
        state.register_session("test-session".to_string(), tx);

        assert!(state.get_session_sender("test-session").is_some());
        assert!(state.get_session_sender("nonexistent").is_none());

        state.unregister_session("test-session");
        assert!(state.get_session_sender("test-session").is_none());
    }
}
