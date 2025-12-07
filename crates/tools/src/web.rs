//! Web-related tools

use async_trait::async_trait;
use rust_ai_agents_core::{errors::ToolError, ExecutionContext, Tool, ToolSchema};
use serde_json::json;

/// Web search tool (stub - requires external API)
pub struct WebSearchTool;

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("web_search", "Search the web for information").with_parameters(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5
                }
            },
            "required": ["query"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let query = arguments["query"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'query' field".to_string()))?;

        // TODO: Implement actual web search (e.g., using SerpAPI, Google Custom Search)
        Ok(json!({
            "query": query,
            "results": [
                {
                    "title": "Example Result 1",
                    "url": "https://example.com/1",
                    "snippet": "This is a sample search result..."
                }
            ],
            "note": "Web search not fully implemented - this is mock data"
        }))
    }
}

/// HTTP request tool
pub struct HttpRequestTool;

impl Default for HttpRequestTool {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpRequestTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("http_request", "Make an HTTP request to a URL")
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to request"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                        "default": "GET"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Request headers"
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body (for POST/PUT)"
                    }
                },
                "required": ["url"]
            }))
            .with_dangerous(true)
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let url = arguments["url"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'url' field".to_string()))?;

        let method = arguments["method"].as_str().unwrap_or("GET").to_uppercase();
        let body_content = arguments["body"].as_str().unwrap_or("").to_string();

        // Build the request
        let client = reqwest::Client::new();
        let mut request_builder = match method.as_str() {
            "GET" => client.get(url),
            "POST" => client.post(url).body(body_content),
            "PUT" => client.put(url).body(body_content),
            "DELETE" => client.delete(url),
            "PATCH" => client.patch(url).body(body_content),
            "HEAD" => client.head(url),
            _ => {
                return Err(ToolError::InvalidArguments(format!(
                    "Unsupported method: {}",
                    method
                )))
            }
        };

        // Add custom headers if provided
        if let Some(headers) = arguments["headers"].as_object() {
            for (key, value) in headers {
                if let Some(header_value) = value.as_str() {
                    request_builder = request_builder.header(key.as_str(), header_value);
                }
            }
        }

        // Execute the request
        let response = request_builder.send().await;

        match response {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let response_headers: serde_json::Map<String, serde_json::Value> = resp
                    .headers()
                    .iter()
                    .filter_map(|(k, v)| v.to_str().ok().map(|val| (k.to_string(), json!(val))))
                    .collect();
                let body = resp.text().await.unwrap_or_default();

                Ok(json!({
                    "status": status,
                    "headers": response_headers,
                    "body": body,
                    "success": (200..300).contains(&status)
                }))
            }
            Err(e) => Err(ToolError::ExecutionFailed(e.to_string())),
        }
    }
}
