//! MCP Server Example
//!
//! This example demonstrates how to create an MCP server that exposes
//! tools to MCP clients like Claude Desktop.
//!
//! # Running the Example
//!
//! ```bash
//! cargo run --example mcp_server
//! ```
//!
//! # Using with Claude Desktop
//!
//! Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "rust-ai-agents": {
//!       "command": "/path/to/rust-ai-agents/target/release/examples/mcp_server"
//!     }
//!   }
//! }
//! ```
//!
//! # What This Example Does
//!
//! Creates an MCP server with several demo tools:
//! - `echo`: Simple echo tool
//! - `calculate`: Basic calculator
//! - `get_time`: Returns current time
//! - `random_number`: Generates random numbers
//! - `word_count`: Counts words in text

use async_trait::async_trait;
use rust_ai_agents_mcp::{CallToolResult, McpError, McpServer, McpTool, ToolContent, ToolHandler};
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

/// Random number generator tool
struct RandomTool;

#[async_trait]
impl ToolHandler for RandomTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "random_number".to_string(),
            description: Some("Generates a random number within a range".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "min": {
                        "type": "integer",
                        "description": "Minimum value (inclusive)",
                        "default": 1
                    },
                    "max": {
                        "type": "integer",
                        "description": "Maximum value (inclusive)",
                        "default": 100
                    }
                }
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
        use rand::Rng;

        let min = arguments.get("min").and_then(|v| v.as_i64()).unwrap_or(1);

        let max = arguments.get("max").and_then(|v| v.as_i64()).unwrap_or(100);

        if min > max {
            return Ok(CallToolResult {
                content: vec![ToolContent::text("min must be less than or equal to max")],
                is_error: true,
            });
        }

        let mut rng = rand::thread_rng();
        let number = rng.gen_range(min..=max);

        Ok(CallToolResult {
            content: vec![ToolContent::text(format!(
                "Random number between {} and {}: {}",
                min, max, number
            ))],
            is_error: false,
        })
    }
}

/// Word count tool
struct WordCountTool;

#[async_trait]
impl ToolHandler for WordCountTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "word_count".to_string(),
            description: Some("Counts words, characters, and lines in text".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
        let text = arguments.get("text").and_then(|v| v.as_str()).unwrap_or("");

        let words = text.split_whitespace().count();
        let chars = text.chars().count();
        let chars_no_spaces = text.chars().filter(|c| !c.is_whitespace()).count();
        let lines = text.lines().count();

        let result = json!({
            "words": words,
            "characters": chars,
            "characters_no_spaces": chars_no_spaces,
            "lines": lines
        });

        Ok(CallToolResult {
            content: vec![ToolContent::text(
                serde_json::to_string_pretty(&result).unwrap(),
            )],
            is_error: false,
        })
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging to stderr (stdout is used for MCP protocol)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting rust-ai-agents MCP Server");

    // Build the MCP server with our tools
    let server = McpServer::builder()
        .name("rust-ai-agents-demo")
        .version(env!("CARGO_PKG_VERSION"))
        .instructions("This is a demo MCP server with utility tools. Available tools: echo, calculate, get_time, random_number, word_count")
        .add_tool(EchoTool)
        .add_tool(CalculatorTool)
        .add_tool(TimeTool)
        .add_tool(RandomTool)
        .add_tool(WordCountTool)
        .build();

    info!("Server configured with {} tools", 5);
    info!("Listening on STDIO...");

    // Run the server (blocks until stdin is closed)
    server.run_stdio().await?;

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
        let result = tool.execute(json!({"message": "Hello"})).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content[0].as_text().unwrap(), "Echo: Hello");
    }

    #[tokio::test]
    async fn test_calculator_add() {
        let tool = CalculatorTool;
        let result = tool
            .execute(json!({
                "operation": "add",
                "a": 5,
                "b": 3
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content[0].as_text().unwrap().contains("8"));
    }

    #[tokio::test]
    async fn test_calculator_divide_by_zero() {
        let tool = CalculatorTool;
        let result = tool
            .execute(json!({
                "operation": "divide",
                "a": 10,
                "b": 0
            }))
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn test_word_count() {
        let tool = WordCountTool;
        let result = tool
            .execute(json!({
                "text": "Hello world\nThis is a test"
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        let text = result.content[0].as_text().unwrap();
        assert!(text.contains("\"words\": 5"));
    }
}
