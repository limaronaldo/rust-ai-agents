//! Message types for agent communication

use crate::types::AgentId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Message content types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Content {
    /// Plain text message
    Text(String),

    /// Tool/function call request
    ToolCall(Vec<ToolCall>),

    /// Tool/function execution result
    ToolResult(Vec<ToolResult>),

    /// Structured data
    StructuredData(serde_json::Value),

    /// Image data (base64 encoded)
    Image { data: String, mime_type: String },

    /// Audio data (base64 encoded)
    Audio { data: String, mime_type: String },

    /// File reference
    File { path: String, mime_type: String },

    /// Error message
    Error { code: String, message: String },
}

/// Tool/function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique call ID
    pub id: String,

    /// Tool/function name
    pub name: String,

    /// Arguments as JSON
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            arguments,
        }
    }
}

/// Tool/function execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Call ID this result corresponds to
    pub call_id: String,

    /// Success status
    pub success: bool,

    /// Result data
    pub data: serde_json::Value,

    /// Error message if failed
    pub error: Option<String>,
}

impl ToolResult {
    pub fn success(call_id: String, data: serde_json::Value) -> Self {
        Self {
            call_id,
            success: true,
            data,
            error: None,
        }
    }

    pub fn failure(call_id: String, error: impl Into<String>) -> Self {
        Self {
            call_id,
            success: false,
            data: serde_json::Value::Null,
            error: Some(error.into()),
        }
    }
}

/// Message between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message ID
    pub id: String,

    /// Sender agent ID
    pub from: AgentId,

    /// Recipient agent ID
    pub to: AgentId,

    /// Message content
    pub content: Content,

    /// Message timestamp
    pub timestamp: DateTime<Utc>,

    /// Message metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,

    /// Message priority (0-255, higher = more urgent)
    pub priority: u8,
}

impl Message {
    /// Create a new message
    pub fn new(from: AgentId, to: AgentId, content: Content) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            from,
            to,
            content,
            timestamp: Utc::now(),
            metadata: std::collections::HashMap::new(),
            priority: 128, // Default medium priority
        }
    }

    /// Create a system message (from system to agent)
    pub fn system(to: AgentId, content: Content) -> Self {
        Self::new(AgentId::new("system"), to, content)
    }

    /// Create a system welcome message
    pub fn system_welcome(to: &AgentId) -> Self {
        Self::system(
            to.clone(),
            Content::Text("Welcome! I'm ready to assist you.".to_string()),
        )
    }

    /// Create a user message
    pub fn user(to: AgentId, text: impl Into<String>) -> Self {
        Self::new(AgentId::new("user"), to, Content::Text(text.into()))
    }

    /// Set message priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Check if message is a tool call
    pub fn is_tool_call(&self) -> bool {
        matches!(self.content, Content::ToolCall(_))
    }

    /// Check if message is a tool result
    pub fn is_tool_result(&self) -> bool {
        matches!(self.content, Content::ToolResult(_))
    }

    /// Check if message is plain text
    pub fn is_text(&self) -> bool {
        matches!(self.content, Content::Text(_))
    }

    /// Get text content if available
    pub fn as_text(&self) -> Option<&str> {
        match &self.content {
            Content::Text(text) => Some(text),
            _ => None,
        }
    }
}

/// Message role for LLM conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    /// System message (instructions)
    System,
    /// User message
    User,
    /// Assistant message
    Assistant,
    /// Tool/function result
    Tool,
}

/// LLM conversation message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMMessage {
    /// Message role
    pub role: MessageRole,

    /// Message content
    pub content: String,

    /// Tool calls (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Tool call ID (for tool role)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Function name (for tool role)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl LLMMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant_with_tools(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: String::new(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn tool(tool_call_id: String, name: String, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
            name: Some(name),
        }
    }
}
