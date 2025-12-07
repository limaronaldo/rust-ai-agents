//! MCP Server implementation
//!
//! Exposes agents and tools as an MCP server that can be consumed by
//! MCP clients like Claude Desktop, VS Code extensions, etc.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                     MCP Server                          │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
//! │  │   Tools     │  │  Resources  │  │    Prompts      │  │
//! │  │  Handler    │  │   Handler   │  │    Handler      │  │
//! │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
//! │         │                │                  │           │
//! │  ┌──────┴────────────────┴──────────────────┴────────┐  │
//! │  │              Request Router                       │  │
//! │  └───────────────────────┬───────────────────────────┘  │
//! │                          │                              │
//! │  ┌───────────────────────┴───────────────────────────┐  │
//! │  │           Server Transport (STDIO/SSE)            │  │
//! │  └───────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_mcp::server::{McpServer, ServerConfig};
//!
//! let server = McpServer::builder()
//!     .name("my-agent-server")
//!     .version("1.0.0")
//!     .add_tool(my_tool)
//!     .add_agent_as_tool(my_agent)
//!     .build();
//!
//! // Run with STDIO transport
//! server.run_stdio().await?;
//! ```

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, info, warn};

use crate::error::McpError;
use crate::protocol::*;

// =============================================================================
// Server Configuration
// =============================================================================

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
    /// Optional instructions for clients
    pub instructions: Option<String>,
    /// Enable tools capability
    pub enable_tools: bool,
    /// Enable resources capability
    pub enable_resources: bool,
    /// Enable prompts capability
    pub enable_prompts: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            name: "rust-ai-agents-mcp-server".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            instructions: None,
            enable_tools: true,
            enable_resources: false,
            enable_prompts: false,
        }
    }
}

// =============================================================================
// Tool Handler Trait
// =============================================================================

/// Handler for a single tool
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Get the tool definition
    fn definition(&self) -> McpTool;

    /// Execute the tool
    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError>;
}

/// Handler for resources
#[async_trait]
pub trait ResourceHandler: Send + Sync {
    /// List available resources
    fn list(&self) -> Vec<McpResource>;

    /// Read a resource by URI
    async fn read(&self, uri: &str) -> Result<ResourceContent, McpError>;
}

/// Handler for prompts
#[async_trait]
pub trait PromptHandler: Send + Sync {
    /// List available prompts
    fn list(&self) -> Vec<McpPrompt>;

    /// Get a prompt with arguments applied
    async fn get(
        &self,
        name: &str,
        arguments: HashMap<String, String>,
    ) -> Result<Vec<PromptMessage>, McpError>;
}

/// Prompt message (for get_prompt response)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: PromptContent,
}

/// Prompt content
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum PromptContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { resource: ResourceContent },
}

// =============================================================================
// Simple Tool Wrapper
// =============================================================================

/// A simple function-based tool
pub struct FnTool<F>
where
    F: Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync + 'static,
{
    definition: McpTool,
    handler: F,
}

impl<F> FnTool<F>
where
    F: Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync + 'static,
{
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: serde_json::Value,
        handler: F,
    ) -> Self {
        Self {
            definition: McpTool {
                name: name.into(),
                description: Some(description.into()),
                input_schema: schema,
            },
            handler,
        }
    }
}

#[async_trait]
impl<F> ToolHandler for FnTool<F>
where
    F: Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync + 'static,
{
    fn definition(&self) -> McpTool {
        self.definition.clone()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
        match (self.handler)(arguments) {
            Ok(result) => Ok(CallToolResult {
                content: vec![ToolContent::text(result.to_string())],
                is_error: false,
            }),
            Err(e) => Ok(CallToolResult {
                content: vec![ToolContent::text(e)],
                is_error: true,
            }),
        }
    }
}

/// Async function-based tool
pub struct AsyncFnTool<F, Fut>
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<serde_json::Value, String>> + Send + 'static,
{
    definition: McpTool,
    handler: F,
}

impl<F, Fut> AsyncFnTool<F, Fut>
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<serde_json::Value, String>> + Send + 'static,
{
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: serde_json::Value,
        handler: F,
    ) -> Self {
        Self {
            definition: McpTool {
                name: name.into(),
                description: Some(description.into()),
                input_schema: schema,
            },
            handler,
        }
    }
}

#[async_trait]
impl<F, Fut> ToolHandler for AsyncFnTool<F, Fut>
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<serde_json::Value, String>> + Send + 'static,
{
    fn definition(&self) -> McpTool {
        self.definition.clone()
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
        match (self.handler)(arguments).await {
            Ok(result) => Ok(CallToolResult {
                content: vec![ToolContent::text(result.to_string())],
                is_error: false,
            }),
            Err(e) => Ok(CallToolResult {
                content: vec![ToolContent::text(e)],
                is_error: true,
            }),
        }
    }
}

// =============================================================================
// MCP Server
// =============================================================================

/// MCP Server that exposes tools, resources, and prompts
pub struct McpServer {
    config: ServerConfig,
    tools: RwLock<HashMap<String, Arc<dyn ToolHandler>>>,
    resources: RwLock<Option<Arc<dyn ResourceHandler>>>,
    prompts: RwLock<Option<Arc<dyn PromptHandler>>>,
    initialized: RwLock<bool>,
}

impl McpServer {
    /// Create a new server with default configuration
    pub fn new() -> Self {
        Self::with_config(ServerConfig::default())
    }

    /// Create a server with custom configuration
    pub fn with_config(config: ServerConfig) -> Self {
        Self {
            config,
            tools: RwLock::new(HashMap::new()),
            resources: RwLock::new(None),
            prompts: RwLock::new(None),
            initialized: RwLock::new(false),
        }
    }

    /// Create a builder for the server
    pub fn builder() -> McpServerBuilder {
        McpServerBuilder::new()
    }

    /// Register a tool handler
    pub fn add_tool(&self, handler: impl ToolHandler + 'static) {
        let def = handler.definition();
        self.tools
            .write()
            .insert(def.name.clone(), Arc::new(handler));
    }

    /// Register a resource handler
    pub fn set_resource_handler(&self, handler: impl ResourceHandler + 'static) {
        *self.resources.write() = Some(Arc::new(handler));
    }

    /// Register a prompt handler
    pub fn set_prompt_handler(&self, handler: impl PromptHandler + 'static) {
        *self.prompts.write() = Some(Arc::new(handler));
    }

    /// Run the server with STDIO transport
    pub async fn run_stdio(self: Arc<Self>) -> Result<(), McpError> {
        info!(
            "Starting MCP server '{}' v{} on STDIO",
            self.config.name, self.config.version
        );

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let reader = BufReader::new(stdin);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            debug!("Received: {}", line);

            let response = self.handle_message(&line).await;

            if let Some(resp) = response {
                let json = serde_json::to_string(&resp).unwrap();
                debug!("Sending: {}", json);

                if let Err(e) = stdout.write_all(format!("{}\n", json).as_bytes()).await {
                    error!("Failed to write response: {}", e);
                    break;
                }
                if let Err(e) = stdout.flush().await {
                    error!("Failed to flush response: {}", e);
                    break;
                }
            }
        }

        info!("MCP server shutting down");
        Ok(())
    }

    /// Handle an incoming JSON-RPC message
    async fn handle_message(&self, message: &str) -> Option<JsonRpcResponse> {
        // Try to parse as request
        if let Ok(request) = serde_json::from_str::<JsonRpcRequest>(message) {
            let response = self.handle_request(request).await;
            return Some(response);
        }

        // Try to parse as notification
        if let Ok(notification) = serde_json::from_str::<JsonRpcNotification>(message) {
            self.handle_notification(notification).await;
            return None;
        }

        warn!("Failed to parse message: {}", message);
        None
    }

    /// Handle a JSON-RPC request
    async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling request: {} (id={:?})", request.method, request.id);

        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(request.params).await,
            "ping" => Ok(serde_json::json!({})),
            "tools/list" => self.handle_tools_list(request.params).await,
            "tools/call" => self.handle_tools_call(request.params).await,
            "resources/list" => self.handle_resources_list(request.params).await,
            "resources/read" => self.handle_resources_read(request.params).await,
            "prompts/list" => self.handle_prompts_list(request.params).await,
            "prompts/get" => self.handle_prompts_get(request.params).await,
            _ => Err(McpError::MethodNotFound(request.method.clone())),
        };

        match result {
            Ok(value) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(value),
                error: None,
            },
            Err(e) => {
                let (code, message) = match &e {
                    McpError::MethodNotFound(_) => (-32601, e.to_string()),
                    McpError::InvalidParams(msg) => (-32602, msg.clone()),
                    _ => (-32000, e.to_string()),
                };
                JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code,
                        message,
                        data: None,
                    }),
                }
            }
        }
    }

    /// Handle a JSON-RPC notification
    async fn handle_notification(&self, notification: JsonRpcNotification) {
        debug!("Handling notification: {}", notification.method);

        match notification.method.as_str() {
            "notifications/initialized" => {
                info!("Client initialized");
            }
            "notifications/cancelled" => {
                debug!("Request cancelled");
            }
            _ => {
                debug!("Unknown notification: {}", notification.method);
            }
        }
    }

    // =========================================================================
    // Request Handlers
    // =========================================================================

    async fn handle_initialize(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let _params: InitializeParams = params
            .map(|p| serde_json::from_value(p))
            .transpose()
            .map_err(|e| McpError::InvalidParams(e.to_string()))?
            .unwrap_or_else(|| InitializeParams {
                protocol_version: PROTOCOL_VERSION.to_string(),
                capabilities: ClientCapabilities::default(),
                client_info: Implementation {
                    name: "unknown".to_string(),
                    version: "0.0.0".to_string(),
                },
            });

        *self.initialized.write() = true;

        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: if self.config.enable_tools {
                    Some(ToolsCapability {
                        list_changed: Some(true),
                    })
                } else {
                    None
                },
                resources: if self.config.enable_resources {
                    Some(ResourcesCapability {
                        subscribe: Some(false),
                        list_changed: Some(true),
                    })
                } else {
                    None
                },
                prompts: if self.config.enable_prompts {
                    Some(PromptsCapability {
                        list_changed: Some(true),
                    })
                } else {
                    None
                },
                logging: None,
                experimental: None,
            },
            server_info: Implementation {
                name: self.config.name.clone(),
                version: self.config.version.clone(),
            },
            instructions: self.config.instructions.clone(),
        };

        serde_json::to_value(result).map_err(|e| McpError::Internal(e.to_string()))
    }

    async fn handle_tools_list(
        &self,
        _params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let tools = self.tools.read();
        let tool_list: Vec<McpTool> = tools.values().map(|h| h.definition()).collect();

        let result = ListToolsResult {
            tools: tool_list,
            next_cursor: None,
        };

        serde_json::to_value(result).map_err(|e| McpError::Internal(e.to_string()))
    }

    async fn handle_tools_call(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let params: CallToolParams = params
            .map(|p| serde_json::from_value(p))
            .transpose()
            .map_err(|e| McpError::InvalidParams(e.to_string()))?
            .ok_or_else(|| McpError::InvalidParams("Missing params".to_string()))?;

        let tools = self.tools.read();
        let handler = tools
            .get(&params.name)
            .ok_or_else(|| McpError::ToolNotFound(params.name.clone()))?
            .clone();
        drop(tools);

        let arguments = params.arguments.unwrap_or(serde_json::json!({}));
        let result = handler.execute(arguments).await?;

        serde_json::to_value(result).map_err(|e| McpError::Internal(e.to_string()))
    }

    async fn handle_resources_list(
        &self,
        _params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let handler = self.resources.read();
        let resources = handler.as_ref().map(|h| h.list()).unwrap_or_default();

        let result = ListResourcesResult {
            resources,
            next_cursor: None,
        };

        serde_json::to_value(result).map_err(|e| McpError::Internal(e.to_string()))
    }

    async fn handle_resources_read(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        #[derive(serde::Deserialize)]
        struct ReadParams {
            uri: String,
        }

        let params: ReadParams = params
            .map(|p| serde_json::from_value(p))
            .transpose()
            .map_err(|e| McpError::InvalidParams(e.to_string()))?
            .ok_or_else(|| McpError::InvalidParams("Missing uri".to_string()))?;

        let handler = self.resources.read();
        let handler = handler
            .as_ref()
            .ok_or_else(|| McpError::CapabilityNotSupported("resources".to_string()))?;

        let content = handler.read(&params.uri).await?;

        let result = serde_json::json!({
            "contents": [content]
        });

        Ok(result)
    }

    async fn handle_prompts_list(
        &self,
        _params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let handler = self.prompts.read();
        let prompts = handler.as_ref().map(|h| h.list()).unwrap_or_default();

        let result = ListPromptsResult {
            prompts,
            next_cursor: None,
        };

        serde_json::to_value(result).map_err(|e| McpError::Internal(e.to_string()))
    }

    async fn handle_prompts_get(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        #[derive(serde::Deserialize)]
        struct GetParams {
            name: String,
            #[serde(default)]
            arguments: HashMap<String, String>,
        }

        let params: GetParams = params
            .map(|p| serde_json::from_value(p))
            .transpose()
            .map_err(|e| McpError::InvalidParams(e.to_string()))?
            .ok_or_else(|| McpError::InvalidParams("Missing name".to_string()))?;

        let handler = self.prompts.read();
        let handler = handler
            .as_ref()
            .ok_or_else(|| McpError::CapabilityNotSupported("prompts".to_string()))?;

        let messages = handler.get(&params.name, params.arguments).await?;

        let result = serde_json::json!({
            "messages": messages
        });

        Ok(result)
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for McpServer
pub struct McpServerBuilder {
    config: ServerConfig,
    tools: Vec<Arc<dyn ToolHandler>>,
    resource_handler: Option<Arc<dyn ResourceHandler>>,
    prompt_handler: Option<Arc<dyn PromptHandler>>,
}

impl McpServerBuilder {
    pub fn new() -> Self {
        Self {
            config: ServerConfig::default(),
            tools: Vec::new(),
            resource_handler: None,
            prompt_handler: None,
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.config.version = version.into();
        self
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.config.instructions = Some(instructions.into());
        self
    }

    pub fn enable_resources(mut self, enable: bool) -> Self {
        self.config.enable_resources = enable;
        self
    }

    pub fn enable_prompts(mut self, enable: bool) -> Self {
        self.config.enable_prompts = enable;
        self
    }

    pub fn add_tool(mut self, handler: impl ToolHandler + 'static) -> Self {
        self.tools.push(Arc::new(handler));
        self
    }

    pub fn resource_handler(mut self, handler: impl ResourceHandler + 'static) -> Self {
        self.config.enable_resources = true;
        self.resource_handler = Some(Arc::new(handler));
        self
    }

    pub fn prompt_handler(mut self, handler: impl PromptHandler + 'static) -> Self {
        self.config.enable_prompts = true;
        self.prompt_handler = Some(Arc::new(handler));
        self
    }

    pub fn build(self) -> Arc<McpServer> {
        let server = McpServer::with_config(self.config);

        for tool in self.tools {
            let def = tool.definition();
            server.tools.write().insert(def.name.clone(), tool);
        }

        if let Some(handler) = self.resource_handler {
            *server.resources.write() = Some(handler);
        }

        if let Some(handler) = self.prompt_handler {
            *server.prompts.write() = Some(handler);
        }

        Arc::new(server)
    }
}

impl Default for McpServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct EchoTool;

    #[async_trait]
    impl ToolHandler for EchoTool {
        fn definition(&self) -> McpTool {
            McpTool {
                name: "echo".to_string(),
                description: Some("Echoes the input".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }),
            }
        }

        async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError> {
            let message = arguments
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("No message");

            Ok(CallToolResult {
                content: vec![ToolContent::text(message)],
                is_error: false,
            })
        }
    }

    #[tokio::test]
    async fn test_server_builder() {
        let server = McpServer::builder()
            .name("test-server")
            .version("1.0.0")
            .add_tool(EchoTool)
            .build();

        assert_eq!(server.config.name, "test-server");
        assert_eq!(server.config.version, "1.0.0");
        assert!(server.tools.read().contains_key("echo"));
    }

    #[tokio::test]
    async fn test_handle_initialize() {
        let server = McpServer::builder()
            .name("test-server")
            .version("1.0.0")
            .build();

        let request = JsonRpcRequest::new(1i64, "initialize").with_params(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }));

        let response = server.handle_request(request).await;

        assert!(response.error.is_none());
        assert!(response.result.is_some());

        let result = response.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "test-server");
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
    }

    #[tokio::test]
    async fn test_handle_tools_list() {
        let server = McpServer::builder()
            .name("test-server")
            .add_tool(EchoTool)
            .build();

        // Initialize first
        *server.initialized.write() = true;

        let request = JsonRpcRequest::new(1i64, "tools/list");
        let response = server.handle_request(request).await;

        assert!(response.error.is_none());

        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");
    }

    #[tokio::test]
    async fn test_handle_tools_call() {
        let server = McpServer::builder()
            .name("test-server")
            .add_tool(EchoTool)
            .build();

        *server.initialized.write() = true;

        let request = JsonRpcRequest::new(1i64, "tools/call").with_params(json!({
            "name": "echo",
            "arguments": {
                "message": "Hello, MCP!"
            }
        }));

        let response = server.handle_request(request).await;

        assert!(response.error.is_none());

        let result = response.result.unwrap();
        assert_eq!(result["isError"], false);

        let content = &result["content"][0];
        assert_eq!(content["type"], "text");
        assert_eq!(content["text"], "Hello, MCP!");
    }

    #[tokio::test]
    async fn test_handle_unknown_method() {
        let server = McpServer::new();

        let request = JsonRpcRequest::new(1i64, "unknown/method");
        let response = server.handle_request(request).await;

        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_fn_tool() {
        let tool = FnTool::new(
            "add",
            "Adds two numbers",
            json!({
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            }),
            |args| {
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);
                Ok(json!(a + b))
            },
        );

        let result = tool.execute(json!({"a": 2, "b": 3})).await.unwrap();
        assert!(!result.is_error);
        // JSON serializes 5.0 as "5.0", so we check for either
        let text = result.content[0].as_text().unwrap();
        assert!(text == "5" || text == "5.0");
    }
}
