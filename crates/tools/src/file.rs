//! File system tools

use async_trait::async_trait;
use rust_ai_agents_core::{Tool, ToolSchema, ExecutionContext, errors::ToolError};
use serde_json::json;

/// Read file tool
pub struct ReadFileTool;

impl ReadFileTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "read_file",
            "Read contents of a file"
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let path = arguments["path"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'path' field".to_string()))?;

        match tokio::fs::read_to_string(path).await {
            Ok(content) => Ok(json!({
                "path": path,
                "content": content,
                "size": content.len()
            })),
            Err(e) => Err(ToolError::ExecutionFailed(e.to_string())),
        }
    }
}

/// Write file tool
pub struct WriteFileTool;

impl WriteFileTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "write_file",
            "Write content to a file"
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }))
        .with_dangerous(true)
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let path = arguments["path"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'path' field".to_string()))?;
        let content = arguments["content"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'content' field".to_string()))?;

        match tokio::fs::write(path, content).await {
            Ok(_) => Ok(json!({
                "path": path,
                "bytes_written": content.len(),
                "success": true
            })),
            Err(e) => Err(ToolError::ExecutionFailed(e.to_string())),
        }
    }
}

/// List directory tool
pub struct ListDirectoryTool;

impl ListDirectoryTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ListDirectoryTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "list_directory",
            "List contents of a directory"
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                }
            },
            "required": ["path"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let path = arguments["path"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'path' field".to_string()))?;

        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(path).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        while let Ok(Some(entry)) = dir.next_entry().await {
            if let Ok(file_type) = entry.file_type().await {
                entries.push(json!({
                    "name": entry.file_name().to_string_lossy().to_string(),
                    "is_file": file_type.is_file(),
                    "is_dir": file_type.is_dir(),
                }));
            }
        }

        Ok(json!({
            "path": path,
            "entries": entries,
            "count": entries.len()
        }))
    }
}
