//! Bridge between MCP tools and rust-ai-agents Tool trait
//!
//! This module provides `McpToolBridge` which wraps an MCP client
//! and exposes its tools as native rust-ai-agents tools.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

use rust_ai_agents_core::errors::ToolError;
use rust_ai_agents_core::tool::{ExecutionContext, Tool, ToolRegistry, ToolSchema};

use crate::client::McpClient;
use crate::error::McpError;
use crate::protocol::{CallToolResult, McpTool, ToolContent};
use crate::transport::McpTransport;

/// Bridge that exposes MCP server tools as rust-ai-agents tools
pub struct McpToolBridge<T: McpTransport> {
    client: Arc<RwLock<McpClient<T>>>,
    server_name: String,
}

impl<T: McpTransport + 'static> McpToolBridge<T> {
    /// Create a new MCP tool bridge
    pub fn new(client: McpClient<T>) -> Self {
        let server_name = client
            .server_info()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "mcp-server".to_string());

        Self {
            client: Arc::new(RwLock::new(client)),
            server_name,
        }
    }

    /// Get the server name
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// List available MCP tools
    pub async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        let mut client = self.client.write().await;
        client.list_tools().await
    }

    /// Register all MCP tools into a tool registry
    pub async fn register_tools(&self, registry: &mut ToolRegistry) -> Result<usize, McpError> {
        let tools = self.list_tools().await?;
        let count = tools.len();

        for mcp_tool in tools {
            let wrapper = McpToolWrapper {
                client: self.client.clone(),
                mcp_tool: mcp_tool.clone(),
                server_name: self.server_name.clone(),
            };
            registry.register(Arc::new(wrapper));
            debug!(
                "Registered MCP tool: {} from {}",
                mcp_tool.name, self.server_name
            );
        }

        info!(
            "Registered {} tools from MCP server '{}'",
            count, self.server_name
        );
        Ok(count)
    }

    /// Create tool wrappers for all MCP tools
    pub async fn create_tools(&self) -> Result<Vec<Arc<dyn Tool>>, McpError> {
        let tools = self.list_tools().await?;
        let mut wrappers: Vec<Arc<dyn Tool>> = Vec::with_capacity(tools.len());

        for mcp_tool in tools {
            let wrapper = McpToolWrapper {
                client: self.client.clone(),
                mcp_tool,
                server_name: self.server_name.clone(),
            };
            wrappers.push(Arc::new(wrapper));
        }

        Ok(wrappers)
    }

    /// Get a single tool by name
    pub async fn get_tool(&self, name: &str) -> Result<Arc<dyn Tool>, McpError> {
        let tools = self.list_tools().await?;

        for mcp_tool in tools {
            if mcp_tool.name == name {
                let wrapper = McpToolWrapper {
                    client: self.client.clone(),
                    mcp_tool,
                    server_name: self.server_name.clone(),
                };
                return Ok(Arc::new(wrapper));
            }
        }

        Err(McpError::ToolNotFound(name.to_string()))
    }

    /// Close the MCP connection
    pub async fn close(self) -> Result<(), McpError> {
        let client = Arc::try_unwrap(self.client)
            .map_err(|_| McpError::Transport("Cannot close: client still in use".to_string()))?
            .into_inner();
        client.close().await
    }
}

/// Wrapper that implements Tool trait for an MCP tool
struct McpToolWrapper<T: McpTransport> {
    client: Arc<RwLock<McpClient<T>>>,
    mcp_tool: McpTool,
    server_name: String,
}

impl<T: McpTransport + 'static> McpToolWrapper<T> {
    /// Convert MCP tool to ToolSchema
    fn to_schema(&self) -> ToolSchema {
        let mut schema = ToolSchema::new(
            &self.mcp_tool.name,
            self.mcp_tool
                .description
                .as_deref()
                .unwrap_or("MCP tool"),
        );

        // Convert MCP input_schema to our parameters format
        schema = schema.with_parameters(self.mcp_tool.input_schema.clone());

        // Add metadata about source
        schema = schema.add_metadata("mcp_server", serde_json::json!(self.server_name));
        schema = schema.add_metadata("mcp_tool", serde_json::json!(true));

        schema
    }

    /// Convert CallToolResult to JSON value
    fn result_to_json(result: CallToolResult) -> serde_json::Value {
        if result.is_error {
            // Extract error message from content
            let error_msg = result
                .content
                .iter()
                .filter_map(|c| c.as_text())
                .collect::<Vec<_>>()
                .join("\n");

            serde_json::json!({
                "error": true,
                "message": error_msg
            })
        } else {
            // Convert content to JSON
            let contents: Vec<serde_json::Value> = result
                .content
                .into_iter()
                .map(|c| match c {
                    ToolContent::Text { text } => serde_json::json!({
                        "type": "text",
                        "text": text
                    }),
                    ToolContent::Image { data, mime_type } => serde_json::json!({
                        "type": "image",
                        "data": data,
                        "mimeType": mime_type
                    }),
                    ToolContent::Resource { resource } => serde_json::json!({
                        "type": "resource",
                        "uri": resource.uri,
                        "mimeType": resource.mime_type,
                        "text": resource.text
                    }),
                })
                .collect();

            // If single text result, return just the text for convenience
            if contents.len() == 1 {
                if let Some(text) = contents[0].get("text") {
                    // Try to parse as JSON, otherwise return as-is
                    if let Some(text_str) = text.as_str() {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text_str) {
                            return parsed;
                        }
                        return serde_json::json!(text_str);
                    }
                }
            }

            serde_json::json!({
                "content": contents
            })
        }
    }
}

#[async_trait]
impl<T: McpTransport + 'static> Tool for McpToolWrapper<T> {
    fn schema(&self) -> ToolSchema {
        self.to_schema()
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        debug!(
            "Executing MCP tool '{}' from '{}' with args: {:?}",
            self.mcp_tool.name, self.server_name, arguments
        );

        let client = self.client.read().await;

        let result = client
            .call_tool(&self.mcp_tool.name, arguments)
            .await
            .map_err(|e| {
                error!("MCP tool execution failed: {}", e);
                ToolError::ExecutionFailed(format!("MCP error: {}", e))
            })?;

        if result.is_error {
            let error_msg = result
                .content
                .iter()
                .filter_map(|c| c.as_text())
                .collect::<Vec<_>>()
                .join("\n");

            return Err(ToolError::ExecutionFailed(error_msg));
        }

        Ok(Self::result_to_json(result))
    }

    fn validate(&self, arguments: &serde_json::Value) -> Result<(), ToolError> {
        // Basic validation: ensure it's an object if input_schema expects object
        if let Some(schema_type) = self.mcp_tool.input_schema.get("type") {
            if schema_type == "object" && !arguments.is_object() {
                return Err(ToolError::InvalidArguments(
                    "Expected object arguments".to_string(),
                ));
            }
        }

        // Check required fields
        if let Some(required) = self.mcp_tool.input_schema.get("required") {
            if let Some(required_arr) = required.as_array() {
                for req in required_arr {
                    if let Some(field_name) = req.as_str() {
                        if arguments.get(field_name).is_none() {
                            return Err(ToolError::InvalidArguments(format!(
                                "Missing required field: {}",
                                field_name
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// Ensure McpToolWrapper is Send + Sync
unsafe impl<T: McpTransport> Send for McpToolWrapper<T> {}
unsafe impl<T: McpTransport> Sync for McpToolWrapper<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_to_json_text() {
        let result = CallToolResult {
            content: vec![ToolContent::Text {
                text: "Hello, world!".to_string(),
            }],
            is_error: false,
        };

        let json = McpToolWrapper::<crate::transport::SseTransport>::result_to_json(result);
        assert_eq!(json, serde_json::json!("Hello, world!"));
    }

    #[test]
    fn test_result_to_json_error() {
        let result = CallToolResult {
            content: vec![ToolContent::Text {
                text: "Something went wrong".to_string(),
            }],
            is_error: true,
        };

        let json = McpToolWrapper::<crate::transport::SseTransport>::result_to_json(result);
        assert!(json.get("error").is_some());
        assert_eq!(json.get("message").unwrap(), "Something went wrong");
    }

    #[test]
    fn test_result_to_json_parsed() {
        let result = CallToolResult {
            content: vec![ToolContent::Text {
                text: r#"{"key": "value", "num": 42}"#.to_string(),
            }],
            is_error: false,
        };

        let json = McpToolWrapper::<crate::transport::SseTransport>::result_to_json(result);
        assert_eq!(json.get("key").unwrap(), "value");
        assert_eq!(json.get("num").unwrap(), 42);
    }
}
