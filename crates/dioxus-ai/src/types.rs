//! Types for dioxus-ai

use serde::{Deserialize, Serialize};

/// A message in the chat conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    /// Unique identifier for the message
    pub id: String,
    /// Role: "user", "assistant", or "system"
    pub role: String,
    /// Message content
    pub content: String,
    /// Timestamp when created
    pub created_at: f64,
}

impl ChatMessage {
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            id: generate_id(),
            role: "user".to_string(),
            content: content.into(),
            created_at: now(),
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            id: generate_id(),
            role: "assistant".to_string(),
            content: content.into(),
            created_at: now(),
        }
    }

    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            id: generate_id(),
            role: "system".to_string(),
            content: content.into(),
            created_at: now(),
        }
    }
}

/// Options for the chat hook
#[derive(Debug, Clone)]
pub struct ChatOptions {
    /// LLM provider: "openai", "anthropic", "openrouter"
    pub provider: String,
    /// API key
    pub api_key: String,
    /// Model identifier (e.g., "gpt-4o-mini", "claude-3-sonnet")
    pub model: String,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Temperature (0.0 - 2.0)
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Enable streaming
    pub stream: bool,
    /// Initial messages
    pub initial_messages: Vec<ChatMessage>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            api_key: String::new(),
            model: "gpt-4o-mini".to_string(),
            system_prompt: None,
            temperature: 0.7,
            max_tokens: 4096,
            stream: true,
            initial_messages: Vec::new(),
        }
    }
}

/// Options for the completion hook
#[derive(Debug, Clone)]
pub struct CompletionOptions {
    /// LLM provider: "openai", "anthropic", "openrouter"
    pub provider: String,
    /// API key
    pub api_key: String,
    /// Model identifier
    pub model: String,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Temperature (0.0 - 2.0)
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
}

impl Default for CompletionOptions {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            api_key: String::new(),
            model: "gpt-4o-mini".to_string(),
            system_prompt: None,
            temperature: 0.7,
            max_tokens: 4096,
        }
    }
}

/// Generate a unique ID
fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("msg_{}", timestamp)
}

/// Get current timestamp
fn now() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
