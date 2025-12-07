//! # Model Context Protocol (MCP) Support
//!
//! Implementation of the Model Context Protocol for rust-ai-agents.

#![allow(clippy::type_complexity)]
#![allow(clippy::await_holding_lock)]
//! MCP provides a standardized way to connect AI agents to external tools and data sources.
//!
//! ## Features
//!
//! - **STDIO Transport**: Launch MCP servers as subprocesses
//! - **SSE Transport**: Connect to HTTP-based MCP servers
//! - **Tool Integration**: Automatic conversion between MCP tools and agent tools
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_mcp::{McpClient, StdioTransport};
//!
//! // Connect to an MCP server via stdio
//! let transport = StdioTransport::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]).await?;
//! let client = McpClient::new(transport).await?;
//!
//! // List available tools
//! let tools = client.list_tools().await?;
//!
//! // Call a tool
//! let result = client.call_tool("read_file", json!({"path": "/tmp/test.txt"})).await?;
//! ```

pub mod bridge;
pub mod client;
pub mod error;
pub mod protocol;
pub mod server;
pub mod transport;

pub use bridge::McpToolBridge;
pub use client::McpClient;
pub use error::McpError;
pub use protocol::*;
pub use server::{
    AsyncFnTool, FnTool, McpServer, McpServerBuilder, PromptContent, PromptHandler, PromptMessage,
    ResourceHandler, ServerConfig, ToolHandler,
};
pub use transport::{McpTransport, SseTransport, StdioTransport};
