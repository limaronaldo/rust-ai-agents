//! MCP Client implementation

use std::sync::Arc;
use tracing::{debug, info};

use crate::error::McpError;
use crate::protocol::*;
use crate::transport::McpTransport;

/// MCP Client for communicating with MCP servers
pub struct McpClient<T: McpTransport> {
    transport: Arc<T>,
    server_info: Option<Implementation>,
    server_capabilities: Option<ServerCapabilities>,
    tools_cache: Option<Vec<McpTool>>,
}

impl<T: McpTransport> McpClient<T> {
    /// Create a new MCP client and initialize the connection
    pub async fn new(transport: T) -> Result<Self, McpError> {
        let mut client = Self {
            transport: Arc::new(transport),
            server_info: None,
            server_capabilities: None,
            tools_cache: None,
        };

        client.initialize().await?;
        Ok(client)
    }

    /// Create without auto-initialization (for testing)
    pub fn new_uninit(transport: T) -> Self {
        Self {
            transport: Arc::new(transport),
            server_info: None,
            server_capabilities: None,
            tools_cache: None,
        }
    }

    /// Initialize the MCP connection
    async fn initialize(&mut self) -> Result<(), McpError> {
        let params = InitializeParams {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ClientCapabilities {
                roots: Some(RootsCapability::default()),
                sampling: None,
                experimental: None,
            },
            client_info: Implementation {
                name: "rust-ai-agents".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };

        let request =
            JsonRpcRequest::new(1i64, "initialize").with_params(serde_json::to_value(&params)?);

        let response = self.transport.request(request).await?;

        if let Some(error) = response.error {
            return Err(McpError::JsonRpc {
                code: error.code,
                message: error.message,
            });
        }

        let result: InitializeResult =
            serde_json::from_value(response.result.ok_or_else(|| {
                McpError::InvalidResponse("No result in initialize response".to_string())
            })?)?;

        // Check version compatibility
        if result.protocol_version != PROTOCOL_VERSION {
            // Log warning but continue - servers may support multiple versions
            debug!(
                "Protocol version mismatch: client={}, server={}",
                PROTOCOL_VERSION, result.protocol_version
            );
        }

        self.server_info = Some(result.server_info.clone());
        self.server_capabilities = Some(result.capabilities.clone());

        info!(
            "MCP initialized: server={} v{}, tools={}, resources={}, prompts={}",
            result.server_info.name,
            result.server_info.version,
            result.capabilities.tools.is_some(),
            result.capabilities.resources.is_some(),
            result.capabilities.prompts.is_some(),
        );

        // Send initialized notification
        self.transport
            .notify("notifications/initialized", None)
            .await?;

        Ok(())
    }

    /// Get server info
    pub fn server_info(&self) -> Option<&Implementation> {
        self.server_info.as_ref()
    }

    /// Get server capabilities
    pub fn capabilities(&self) -> Option<&ServerCapabilities> {
        self.server_capabilities.as_ref()
    }

    /// Check if server supports tools
    pub fn supports_tools(&self) -> bool {
        self.server_capabilities
            .as_ref()
            .map(|c| c.tools.is_some())
            .unwrap_or(false)
    }

    /// Check if server supports resources
    pub fn supports_resources(&self) -> bool {
        self.server_capabilities
            .as_ref()
            .map(|c| c.resources.is_some())
            .unwrap_or(false)
    }

    /// Check if server supports prompts
    pub fn supports_prompts(&self) -> bool {
        self.server_capabilities
            .as_ref()
            .map(|c| c.prompts.is_some())
            .unwrap_or(false)
    }

    // =========================================================================
    // Tool Operations
    // =========================================================================

    /// List available tools
    pub async fn list_tools(&mut self) -> Result<Vec<McpTool>, McpError> {
        if !self.supports_tools() {
            return Err(McpError::CapabilityNotSupported("tools".to_string()));
        }

        let mut all_tools = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let params = ListToolsParams {
                cursor: cursor.clone(),
            };
            let request =
                JsonRpcRequest::new(0i64, "tools/list").with_params(serde_json::to_value(&params)?);

            let response = self.transport.request(request).await?;

            if let Some(error) = response.error {
                return Err(McpError::JsonRpc {
                    code: error.code,
                    message: error.message,
                });
            }

            let result: ListToolsResult =
                serde_json::from_value(response.result.ok_or_else(|| {
                    McpError::InvalidResponse("No result in list_tools response".to_string())
                })?)?;

            all_tools.extend(result.tools);

            match result.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }

        self.tools_cache = Some(all_tools.clone());
        Ok(all_tools)
    }

    /// Get cached tools (or fetch if not cached)
    pub async fn get_tools(&mut self) -> Result<&[McpTool], McpError> {
        if self.tools_cache.is_none() {
            self.list_tools().await?;
        }
        Ok(self.tools_cache.as_ref().unwrap())
    }

    /// Call a tool
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        let params = CallToolParams {
            name: name.to_string(),
            arguments: Some(arguments),
        };

        let request =
            JsonRpcRequest::new(0i64, "tools/call").with_params(serde_json::to_value(&params)?);

        let response = self.transport.request(request).await?;

        if let Some(error) = response.error {
            return Err(McpError::JsonRpc {
                code: error.code,
                message: error.message,
            });
        }

        let result: CallToolResult = serde_json::from_value(response.result.ok_or_else(|| {
            McpError::InvalidResponse("No result in call_tool response".to_string())
        })?)?;

        Ok(result)
    }

    // =========================================================================
    // Resource Operations
    // =========================================================================

    /// List available resources
    pub async fn list_resources(&self) -> Result<Vec<McpResource>, McpError> {
        if !self.supports_resources() {
            return Err(McpError::CapabilityNotSupported("resources".to_string()));
        }

        let mut all_resources = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let params = serde_json::json!({ "cursor": cursor });
            let request = JsonRpcRequest::new(0i64, "resources/list").with_params(params);

            let response = self.transport.request(request).await?;

            if let Some(error) = response.error {
                return Err(McpError::JsonRpc {
                    code: error.code,
                    message: error.message,
                });
            }

            let result: ListResourcesResult =
                serde_json::from_value(response.result.ok_or_else(|| {
                    McpError::InvalidResponse("No result in list_resources response".to_string())
                })?)?;

            all_resources.extend(result.resources);

            match result.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }

        Ok(all_resources)
    }

    /// Read a resource
    pub async fn read_resource(&self, uri: &str) -> Result<ResourceContent, McpError> {
        let params = serde_json::json!({ "uri": uri });
        let request = JsonRpcRequest::new(0i64, "resources/read").with_params(params);

        let response = self.transport.request(request).await?;

        if let Some(error) = response.error {
            return Err(McpError::JsonRpc {
                code: error.code,
                message: error.message,
            });
        }

        #[derive(serde::Deserialize)]
        struct ReadResult {
            contents: Vec<ResourceContent>,
        }

        let result: ReadResult = serde_json::from_value(response.result.ok_or_else(|| {
            McpError::InvalidResponse("No result in read_resource response".to_string())
        })?)?;

        result.contents.into_iter().next().ok_or_else(|| {
            McpError::InvalidResponse("Empty contents in read_resource response".to_string())
        })
    }

    // =========================================================================
    // Prompt Operations
    // =========================================================================

    /// List available prompts
    pub async fn list_prompts(&self) -> Result<Vec<McpPrompt>, McpError> {
        if !self.supports_prompts() {
            return Err(McpError::CapabilityNotSupported("prompts".to_string()));
        }

        let mut all_prompts = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let params = serde_json::json!({ "cursor": cursor });
            let request = JsonRpcRequest::new(0i64, "prompts/list").with_params(params);

            let response = self.transport.request(request).await?;

            if let Some(error) = response.error {
                return Err(McpError::JsonRpc {
                    code: error.code,
                    message: error.message,
                });
            }

            let result: ListPromptsResult =
                serde_json::from_value(response.result.ok_or_else(|| {
                    McpError::InvalidResponse("No result in list_prompts response".to_string())
                })?)?;

            all_prompts.extend(result.prompts);

            match result.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }

        Ok(all_prompts)
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// Close the connection
    pub async fn close(self) -> Result<(), McpError> {
        self.transport.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_capabilities() {
        let caps = ClientCapabilities::default();
        assert!(caps.roots.is_none());
        assert!(caps.sampling.is_none());
    }
}
