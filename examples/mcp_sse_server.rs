//! MCP SSE Server Example
//!
//! This example demonstrates how to create an MCP server that exposes
//! tools via HTTP/SSE transport instead of STDIO.
//!
//! # Running the Example
//!
//! ```bash
//! cargo run --example mcp_sse_server
//! ```
//!
//! The server will start on http://localhost:3000
//!
//! # Endpoints
//!
//! - `GET /sse` - SSE endpoint for server-to-client streaming
//! - `POST /message?sessionId=xxx` - JSON-RPC requests from client
//!
//! # Testing
//!
//! 1. Connect to SSE endpoint:
//!    ```bash
//!    curl -N http://localhost:3000/sse
//!    ```
//!    You'll receive an "endpoint" event with the POST URL.
//!
//! 2. Send a request (replace SESSION_ID with actual value):
//!    ```bash
//!    curl -X POST "http://localhost:3000/message?sessionId=SESSION_ID" \
//!      -H "Content-Type: application/json" \
//!      -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
//!    ```
//!
//! # What This Example Does
//!
//! Creates an MCP server with demo tools accessible via HTTP:
//! - `echo`: Simple echo tool
//! - `calculate`: Basic calculator
//! - `get_time`: Returns current time

use async_trait::async_trait;
use rust_ai_agents_mcp::{
    CallToolResult, McpError, McpServer, McpTool, SseServerConfig, ToolContent, ToolHandler,
};
use serde_json::json;
use tracing::info;

// =============================================================================
// Tool Implementations
// =============================================================================

/// Echo tool - returns the input message
struct EchoTool;

#[async_trait]
impl ToolHandler for EchoTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "echo".to_string(),
            description: Some("Echoes back the provided message".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo back"
                    }
                },
                "required": ["message"]
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
        let message = arguments
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("No message provided");

        Ok(CallToolResult {
            content: vec![ToolContent::text(format!("Echo: {}", message))],
            is_error: false,
        })
    }
}

/// Calculator tool - performs basic math operations
struct CalculatorTool;

#[async_trait]
impl ToolHandler for CalculatorTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "calculate".to_string(),
            description: Some("Performs basic mathematical operations".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First operand"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second operand"
                    }
                },
                "required": ["operation", "a", "b"]
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
        let operation = arguments
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("add");

        let a = arguments.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b = arguments.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let result = match operation {
            "add" => Ok(a + b),
            "subtract" => Ok(a - b),
            "multiply" => Ok(a * b),
            "divide" => {
                if b == 0.0 {
                    Err("Division by zero")
                } else {
                    Ok(a / b)
                }
            }
            _ => Err("Unknown operation"),
        };

        match result {
            Ok(value) => Ok(CallToolResult {
                content: vec![ToolContent::text(format!(
                    "{} {} {} = {}",
                    a, operation, b, value
                ))],
                is_error: false,
            }),
            Err(e) => Ok(CallToolResult {
                content: vec![ToolContent::text(e.to_string())],
                is_error: true,
            }),
        }
    }
}

/// Time tool - returns the current time
struct TimeTool;

#[async_trait]
impl ToolHandler for TimeTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "get_time".to_string(),
            description: Some("Returns the current date and time".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["iso", "human", "unix"],
                        "description": "Output format (default: human)",
                        "default": "human"
                    }
                }
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
        let format = arguments
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("human");

        let now = chrono::Utc::now();

        let formatted = match format {
            "iso" => now.to_rfc3339(),
            "unix" => now.timestamp().to_string(),
            _ => now.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        };

        Ok(CallToolResult {
            content: vec![ToolContent::text(formatted)],
            is_error: false,
        })
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting rust-ai-agents MCP SSE Server");

    // Build the MCP server with our tools
    let server = McpServer::builder()
        .name("rust-ai-agents-sse-demo")
        .version(env!("CARGO_PKG_VERSION"))
        .instructions("Demo MCP server over HTTP/SSE. Tools: echo, calculate, get_time")
        .add_tool(EchoTool)
        .add_tool(CalculatorTool)
        .add_tool(TimeTool)
        .build();

    // Configure SSE server
    let config = SseServerConfig {
        host: "127.0.0.1".to_string(),
        port: 3000,
        sse_path: "/sse".to_string(),
        message_path: "/message".to_string(),
        enable_cors: true,
        keep_alive_secs: 30,
    };

    info!("Server configured with 3 tools");
    info!(
        "SSE endpoint: http://{}:{}{}",
        config.host, config.port, config.sse_path
    );
    info!(
        "Message endpoint: http://{}:{}{}",
        config.host, config.port, config.message_path
    );

    // Run the server (blocks until shutdown)
    server.run_sse(config).await?;

    info!("Server shutting down");
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo_tool() {
        let tool = EchoTool;
        let result = tool.execute(json!({"message": "Hello SSE"})).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content[0].as_text().unwrap(), "Echo: Hello SSE");
    }

    #[tokio::test]
    async fn test_calculator_multiply() {
        let tool = CalculatorTool;
        let result = tool
            .execute(json!({
                "operation": "multiply",
                "a": 7,
                "b": 6
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content[0].as_text().unwrap().contains("42"));
    }

    #[tokio::test]
    async fn test_time_tool() {
        let tool = TimeTool;
        let result = tool.execute(json!({"format": "unix"})).await.unwrap();
        assert!(!result.is_error);
        // Should be a number (Unix timestamp)
        let text = result.content[0].as_text().unwrap();
        assert!(text.parse::<i64>().is_ok());
    }
}
