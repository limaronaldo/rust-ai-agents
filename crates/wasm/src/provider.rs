//! Provider-specific helpers for WASM

use serde::{Deserialize, Serialize};

/// Supported LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    OpenAI,
    Anthropic,
    OpenRouter,
}

impl Provider {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(Provider::OpenAI),
            "anthropic" | "claude" => Some(Provider::Anthropic),
            "openrouter" => Some(Provider::OpenRouter),
            _ => None,
        }
    }

    pub fn api_url(&self) -> &'static str {
        match self {
            Provider::OpenAI => "https://api.openai.com/v1/chat/completions",
            Provider::Anthropic => "https://api.anthropic.com/v1/messages",
            Provider::OpenRouter => "https://openrouter.ai/api/v1/chat/completions",
        }
    }

    pub fn default_model(&self) -> &'static str {
        match self {
            Provider::OpenAI => "gpt-4-turbo",
            Provider::Anthropic => "claude-3-5-sonnet-20241022",
            Provider::OpenRouter => "openai/gpt-4-turbo",
        }
    }
}
