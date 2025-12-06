//! Web-related tools

use async_trait::async_trait;
use rust_ai_agents_core::{Tool, ToolSchema, ExecutionContext, errors::ToolError};
use serde_json::json;

/// Web search tool (stub - requires external API)
pub struct WebSearchTool;

impl WebSearchTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "web_search",
            "Search the web for information"
        )
        .with_parameters(json!({
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
        let query = arguments["query"].as_str()
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

impl HttpRequestTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "http_request",
            "Make an HTTP request to a URL"
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to request"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
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
        let url = arguments["url"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'url' field".to_string()))?;

        let method = arguments["method"].as_str().unwrap_or("GET");

        // Make the request
        let client = reqwest::Client::new();
        let response = match method {
            "GET" => client.get(url).send().await,
            "POST" => {
                let body = arguments["body"].as_str().unwrap_or("");
                client.post(url).body(body.to_string()).send().await
            }
            _ => return Err(ToolError::InvalidArguments(format!("Unsupported method: {}", method))),
        };

        match response {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();

                Ok(json!({
                    "status": status,
                    "body": body,
                    "success": status >= 200 && status < 300
                }))
            }
            Err(e) => Err(ToolError::ExecutionFailed(e.to_string())),
        }
    }
}
