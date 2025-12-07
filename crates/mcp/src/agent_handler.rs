//! Agent-as-MCP-Tool
//!
//! Exposes rust-ai-agents agents as MCP tools, allowing external MCP clients
//! (like Claude Desktop) to invoke agents directly.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     MCP Client (Claude)                     │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                    tools/call "agent_xxx"
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     McpServer                               │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              AgentMcpHandler                         │   │
//! │  │                                                      │   │
//! │  │  - Wraps AgentTool as MCP ToolHandler               │   │
//! │  │  - Converts MCP params to AgentToolInput            │   │
//! │  │  - Converts AgentToolOutput to MCP CallToolResult   │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     AgentTool / AgentEngine                 │
//! │  - Executes agent logic                                     │
//! │  - Returns structured response                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_mcp::{McpServer, AgentMcpHandler};
//! use rust_ai_agents_agents::agent_tool::AgentTool;
//!
//! // Create an agent tool
//! let research_agent = AgentTool::builder("research_agent")
//!     .description("Researches topics and provides summaries")
//!     .handler(|input| async move {
//!         // Agent logic here
//!         Ok(AgentToolOutput::success("Research results..."))
//!     });
//!
//! // Wrap as MCP handler
//! let mcp_handler = AgentMcpHandler::from_agent_tool(research_agent);
//!
//! // Add to MCP server
//! let server = McpServer::builder()
//!     .name("agent-server")
//!     .add_tool(mcp_handler)
//!     .build();
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::error::McpError;
use crate::protocol::{CallToolResult, McpTool, ToolContent};
use crate::server::ToolHandler;

/// Input schema for agent MCP tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMcpInput {
    /// The query or task for the agent
    pub query: String,
    /// Additional context as key-value pairs
    #[serde(default)]
    pub context: HashMap<String, String>,
    /// Conversation history (optional)
    #[serde(default)]
    pub history: Vec<String>,
    /// Maximum tokens for response (optional hint)
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

/// Output structure for agent responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMcpOutput {
    /// The response content
    pub content: String,
    /// Whether the agent successfully completed the task
    pub success: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Time taken in milliseconds
    pub duration_ms: u64,
    /// Tools used by the agent
    #[serde(default)]
    pub tools_used: Vec<String>,
}

/// Configuration for an agent MCP handler
#[derive(Debug, Clone)]
pub struct AgentMcpConfig {
    /// Name prefix for the MCP tool (e.g., "agent_")
    pub name_prefix: String,
    /// Whether to include metadata in response
    pub include_metadata: bool,
    /// Whether to include tools used in response
    pub include_tools_used: bool,
}

impl Default for AgentMcpConfig {
    fn default() -> Self {
        Self {
            name_prefix: "agent_".to_string(),
            include_metadata: true,
            include_tools_used: true,
        }
    }
}

/// Handler type for agent execution
pub type AgentHandlerFn = Arc<
    dyn Fn(
            AgentMcpInput,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<AgentMcpOutput, String>> + Send>,
        > + Send
        + Sync,
>;

/// MCP ToolHandler that wraps an agent
pub struct AgentMcpHandler {
    /// Tool name (as exposed in MCP)
    name: String,
    /// Tool description
    description: String,
    /// Agent capabilities/tags
    capabilities: Vec<String>,
    /// Handler function
    handler: AgentHandlerFn,
    /// Configuration
    config: AgentMcpConfig,
}

impl AgentMcpHandler {
    /// Create a new handler with a custom async function
    pub fn new<F, Fut>(name: impl Into<String>, description: impl Into<String>, handler: F) -> Self
    where
        F: Fn(AgentMcpInput) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<AgentMcpOutput, String>> + Send + 'static,
    {
        let config = AgentMcpConfig::default();
        let name_str = name.into();
        let tool_name = format!("{}{}", config.name_prefix, name_str);

        Self {
            name: tool_name,
            description: description.into(),
            capabilities: Vec::new(),
            handler: Arc::new(move |input| Box::pin(handler(input))),
            config,
        }
    }

    /// Create with custom configuration
    pub fn with_config(mut self, config: AgentMcpConfig) -> Self {
        // Update name with new prefix
        let base_name = self
            .name
            .strip_prefix(&self.config.name_prefix)
            .unwrap_or(&self.name)
            .to_string();
        self.name = format!("{}{}", config.name_prefix, base_name);
        self.config = config;
        self
    }

    /// Add a capability tag
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    /// Add multiple capabilities
    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.capabilities.extend(capabilities);
        self
    }

    /// Create a builder for fluent construction
    pub fn builder(name: impl Into<String>) -> AgentMcpHandlerBuilder {
        AgentMcpHandlerBuilder::new(name)
    }

    /// Get the tool name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the capabilities
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }
}

#[async_trait]
impl ToolHandler for AgentMcpHandler {
    fn definition(&self) -> McpTool {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query or task for the agent"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context as key-value pairs",
                    "additionalProperties": { "type": "string" }
                },
                "history": {
                    "type": "array",
                    "description": "Conversation history (optional)",
                    "items": { "type": "string" }
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for response (optional hint)"
                }
            },
            "required": ["query"]
        });

        // Add capabilities to description if present
        let description = if self.capabilities.is_empty() {
            self.description.clone()
        } else {
            format!(
                "{}\n\nCapabilities: {}",
                self.description,
                self.capabilities.join(", ")
            )
        };

        McpTool {
            name: self.name.clone(),
            description: Some(description),
            input_schema: schema,
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
        debug!(tool = %self.name, "Executing agent MCP handler");

        // Parse input
        let input: AgentMcpInput = serde_json::from_value(arguments.clone())
            .map_err(|e| McpError::InvalidParams(format!("Invalid input: {}", e)))?;

        info!(
            tool = %self.name,
            query = %input.query,
            context_keys = ?input.context.keys().collect::<Vec<_>>(),
            "Agent executing query"
        );

        // Execute agent
        let result = (self.handler)(input).await;

        match result {
            Ok(output) => {
                let mut response_parts = vec![output.content.clone()];

                // Add metadata if configured
                if self.config.include_metadata && !output.metadata.is_empty() {
                    let metadata_str = output
                        .metadata
                        .iter()
                        .map(|(k, v)| format!("  {}: {}", k, v))
                        .collect::<Vec<_>>()
                        .join("\n");
                    response_parts.push(format!("\n\nMetadata:\n{}", metadata_str));
                }

                // Add tools used if configured
                if self.config.include_tools_used && !output.tools_used.is_empty() {
                    response_parts
                        .push(format!("\n\nTools used: {}", output.tools_used.join(", ")));
                }

                let response_text = response_parts.join("");

                // Add structured data as additional content
                let structured_output = json!({
                    "success": output.success,
                    "confidence": output.confidence,
                    "duration_ms": output.duration_ms,
                    "metadata": output.metadata,
                    "tools_used": output.tools_used
                });

                Ok(CallToolResult {
                    content: vec![
                        ToolContent::text(response_text),
                        ToolContent::text(format!(
                            "\n---\nStructured output: {}",
                            serde_json::to_string_pretty(&structured_output).unwrap_or_default()
                        )),
                    ],
                    is_error: !output.success,
                })
            }
            Err(e) => Ok(CallToolResult {
                content: vec![ToolContent::text(format!("Agent error: {}", e))],
                is_error: true,
            }),
        }
    }
}

/// Builder for AgentMcpHandler
pub struct AgentMcpHandlerBuilder {
    name: String,
    description: String,
    capabilities: Vec<String>,
    config: AgentMcpConfig,
}

impl AgentMcpHandlerBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            capabilities: Vec::new(),
            config: AgentMcpConfig::default(),
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    pub fn capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.capabilities.extend(capabilities);
        self
    }

    pub fn config(mut self, config: AgentMcpConfig) -> Self {
        self.config = config;
        self
    }

    pub fn name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.name_prefix = prefix.into();
        self
    }

    pub fn include_metadata(mut self, include: bool) -> Self {
        self.config.include_metadata = include;
        self
    }

    pub fn include_tools_used(mut self, include: bool) -> Self {
        self.config.include_tools_used = include;
        self
    }

    /// Build with a handler function
    pub fn handler<F, Fut>(self, handler: F) -> AgentMcpHandler
    where
        F: Fn(AgentMcpInput) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<AgentMcpOutput, String>> + Send + 'static,
    {
        let tool_name = format!("{}{}", self.config.name_prefix, self.name);

        AgentMcpHandler {
            name: tool_name,
            description: self.description,
            capabilities: self.capabilities,
            handler: Arc::new(move |input| Box::pin(handler(input))),
            config: self.config,
        }
    }
}

/// Helper to create a simple agent handler from a closure
pub fn simple_agent<F, Fut>(
    name: impl Into<String>,
    description: impl Into<String>,
    handler: F,
) -> AgentMcpHandler
where
    F: Fn(String) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<String, String>> + Send + 'static,
{
    let handler = Arc::new(handler);
    AgentMcpHandler::builder(name)
        .description(description)
        .handler(move |input: AgentMcpInput| {
            let h = handler.clone();
            async move {
                let start = std::time::Instant::now();
                match h(input.query).await {
                    Ok(content) => Ok(AgentMcpOutput {
                        content,
                        success: true,
                        confidence: 1.0,
                        metadata: HashMap::new(),
                        duration_ms: start.elapsed().as_millis() as u64,
                        tools_used: Vec::new(),
                    }),
                    Err(e) => Err(e),
                }
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_mcp_handler_basic() {
        let handler = AgentMcpHandler::builder("test_agent")
            .description("A test agent")
            .handler(|input: AgentMcpInput| async move {
                Ok(AgentMcpOutput {
                    content: format!("Processed: {}", input.query),
                    success: true,
                    confidence: 0.95,
                    metadata: HashMap::new(),
                    duration_ms: 100,
                    tools_used: vec!["tool1".to_string()],
                })
            });

        let def = handler.definition();
        assert_eq!(def.name, "agent_test_agent");
        assert!(def.description.unwrap().contains("test agent"));

        let result = handler
            .execute(json!({"query": "Hello world"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content[0]
            .as_text()
            .unwrap()
            .contains("Processed: Hello world"));
    }

    #[tokio::test]
    async fn test_agent_mcp_handler_with_context() {
        let handler = AgentMcpHandler::builder("context_agent")
            .description("Agent that uses context")
            .handler(|input: AgentMcpInput| async move {
                let name = input.context.get("name").cloned().unwrap_or_default();
                Ok(AgentMcpOutput {
                    content: format!("Hello, {}!", name),
                    success: true,
                    confidence: 1.0,
                    metadata: HashMap::new(),
                    duration_ms: 50,
                    tools_used: Vec::new(),
                })
            });

        let result = handler
            .execute(json!({
                "query": "greet",
                "context": {"name": "World"}
            }))
            .await
            .unwrap();

        assert!(result.content[0]
            .as_text()
            .unwrap()
            .contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_agent_mcp_handler_error() {
        let handler = AgentMcpHandler::builder("failing_agent")
            .description("Agent that fails")
            .handler(|_: AgentMcpInput| async move { Err("Intentional failure".to_string()) });

        let result = handler.execute(json!({"query": "test"})).await.unwrap();

        assert!(result.is_error);
        assert!(result.content[0].as_text().unwrap().contains("Agent error"));
    }

    #[tokio::test]
    async fn test_agent_mcp_handler_capabilities() {
        let handler = AgentMcpHandler::builder("capable_agent")
            .description("Agent with capabilities")
            .capability("math")
            .capability("science")
            .handler(|_: AgentMcpInput| async move {
                Ok(AgentMcpOutput {
                    content: "OK".to_string(),
                    success: true,
                    confidence: 1.0,
                    metadata: HashMap::new(),
                    duration_ms: 10,
                    tools_used: Vec::new(),
                })
            });

        let def = handler.definition();
        let desc = def.description.unwrap();
        assert!(desc.contains("math"));
        assert!(desc.contains("science"));
    }

    #[tokio::test]
    async fn test_simple_agent_helper() {
        let handler = simple_agent("simple", "A simple agent", |query: String| async move {
            Ok(format!("Echo: {}", query))
        });

        let result = handler
            .execute(json!({"query": "test message"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content[0]
            .as_text()
            .unwrap()
            .contains("Echo: test message"));
    }

    #[tokio::test]
    async fn test_agent_mcp_handler_custom_prefix() {
        let handler = AgentMcpHandler::builder("custom")
            .description("Custom prefix agent")
            .name_prefix("ai_")
            .handler(|_: AgentMcpInput| async move {
                Ok(AgentMcpOutput {
                    content: "OK".to_string(),
                    success: true,
                    confidence: 1.0,
                    metadata: HashMap::new(),
                    duration_ms: 10,
                    tools_used: Vec::new(),
                })
            });

        let def = handler.definition();
        assert_eq!(def.name, "ai_custom");
    }

    #[tokio::test]
    async fn test_agent_mcp_handler_metadata_output() {
        let handler = AgentMcpHandler::builder("metadata_agent")
            .description("Agent with metadata")
            .include_metadata(true)
            .handler(|_: AgentMcpInput| async move {
                let mut metadata = HashMap::new();
                metadata.insert("source".to_string(), "database".to_string());
                metadata.insert("version".to_string(), "1.0".to_string());

                Ok(AgentMcpOutput {
                    content: "Result with metadata".to_string(),
                    success: true,
                    confidence: 0.9,
                    metadata,
                    duration_ms: 200,
                    tools_used: vec!["db_query".to_string()],
                })
            });

        let result = handler.execute(json!({"query": "test"})).await.unwrap();

        let text = result.content[0].as_text().unwrap();
        assert!(text.contains("Result with metadata"));
        assert!(text.contains("source: database"));
    }

    #[test]
    fn test_agent_mcp_input_deserialization() {
        let json = json!({
            "query": "What is 2+2?",
            "context": {"mode": "math"},
            "history": ["previous query"],
            "max_tokens": 100
        });

        let input: AgentMcpInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.query, "What is 2+2?");
        assert_eq!(input.context.get("mode").unwrap(), "math");
        assert_eq!(input.history.len(), 1);
        assert_eq!(input.max_tokens, Some(100));
    }

    #[test]
    fn test_agent_mcp_input_minimal() {
        let json = json!({"query": "simple query"});
        let input: AgentMcpInput = serde_json::from_value(json).unwrap();

        assert_eq!(input.query, "simple query");
        assert!(input.context.is_empty());
        assert!(input.history.is_empty());
        assert!(input.max_tokens.is_none());
    }
}
