//! LLM Provider definitions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// OpenAI (GPT-4, GPT-3.5, etc.)
    OpenAI,
    /// Anthropic (Claude 3, etc.)
    Anthropic,
    /// OpenRouter (100+ models)
    OpenRouter,
}

impl Provider {
    /// Get the API endpoint URL for this provider.
    pub fn endpoint(&self) -> &'static str {
        match self {
            Self::OpenAI => "https://api.openai.com/v1/chat/completions",
            Self::Anthropic => "https://api.anthropic.com/v1/messages",
            Self::OpenRouter => "https://openrouter.ai/api/v1/chat/completions",
        }
    }

    /// Get the header name for the API key.
    pub fn auth_header(&self) -> &'static str {
        match self {
            Self::OpenAI | Self::OpenRouter => "Authorization",
            Self::Anthropic => "x-api-key",
        }
    }

    /// Format the API key for the authorization header.
    pub fn format_auth(&self, api_key: &str) -> String {
        match self {
            Self::OpenAI | Self::OpenRouter => format!("Bearer {}", api_key),
            Self::Anthropic => api_key.to_string(),
        }
    }

    /// Get additional required headers for this provider.
    pub fn extra_headers(&self) -> Vec<(&'static str, &'static str)> {
        match self {
            Self::OpenAI => vec![],
            Self::Anthropic => vec![
                ("anthropic-version", "2023-06-01"),
                ("anthropic-dangerous-direct-browser-access", "true"),
            ],
            Self::OpenRouter => vec![("HTTP-Referer", "https://rust-ai-agents.dev")],
        }
    }

    /// Check if this provider uses OpenAI-compatible format.
    pub fn is_openai_compatible(&self) -> bool {
        matches!(self, Self::OpenAI | Self::OpenRouter)
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenAI => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenRouter => write!(f, "openrouter"),
        }
    }
}

impl std::str::FromStr for Provider {
    type Err = crate::LlmClientError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Self::OpenAI),
            "anthropic" => Ok(Self::Anthropic),
            "openrouter" => Ok(Self::OpenRouter),
            _ => Err(crate::LlmClientError::UnknownProvider(s.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_endpoints() {
        assert!(Provider::OpenAI.endpoint().contains("openai.com"));
        assert!(Provider::Anthropic.endpoint().contains("anthropic.com"));
        assert!(Provider::OpenRouter.endpoint().contains("openrouter.ai"));
    }

    #[test]
    fn test_provider_from_str() {
        assert_eq!("openai".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert_eq!(
            "ANTHROPIC".parse::<Provider>().unwrap(),
            Provider::Anthropic
        );
        assert!("invalid".parse::<Provider>().is_err());
    }

    #[test]
    fn test_auth_format() {
        assert!(Provider::OpenAI.format_auth("key").starts_with("Bearer "));
        assert_eq!(Provider::Anthropic.format_auth("key"), "key");
    }
}
