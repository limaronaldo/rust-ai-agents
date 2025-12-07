//! MCP Error types

use thiserror::Error;

/// MCP-specific errors
#[derive(Debug, Error)]
pub enum McpError {
    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("JSON-RPC error {code}: {message}")]
    JsonRpc { code: i32, message: String },

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Timeout waiting for response")]
    Timeout,

    #[error("Version mismatch: client={client}, server={server}")]
    VersionMismatch { client: String, server: String },

    #[error("Capability not supported: {0}")]
    CapabilityNotSupported(String),

    #[error("Method not found: {0}")]
    MethodNotFound(String),

    #[error("Invalid params: {0}")]
    InvalidParams(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
