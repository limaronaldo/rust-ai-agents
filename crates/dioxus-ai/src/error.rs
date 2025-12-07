//! Error types for dioxus-ai

use thiserror::Error;

/// Errors that can occur in dioxus-ai
#[derive(Debug, Error, Clone)]
pub enum DioxusAiError {
    /// HTTP request failed
    #[error("Request failed: {0}")]
    RequestFailed(String),

    /// Invalid provider
    #[error("Invalid provider: {0}")]
    InvalidProvider(String),

    /// API error response
    #[error("API error: {0}")]
    ApiError(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Stream error
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Missing configuration
    #[error("Missing configuration: {0}")]
    MissingConfig(String),
}

impl From<serde_json::Error> for DioxusAiError {
    fn from(e: serde_json::Error) -> Self {
        DioxusAiError::ParseError(e.to_string())
    }
}

impl From<rust_ai_agents_llm_client::LlmClientError> for DioxusAiError {
    fn from(e: rust_ai_agents_llm_client::LlmClientError) -> Self {
        DioxusAiError::ApiError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, DioxusAiError>;
