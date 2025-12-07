//! Error types for leptos-ai

use thiserror::Error;

/// Errors that can occur in leptos-ai
#[derive(Debug, Error, Clone)]
pub enum LeptosAiError {
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

impl From<gloo_net::Error> for LeptosAiError {
    fn from(e: gloo_net::Error) -> Self {
        LeptosAiError::RequestFailed(e.to_string())
    }
}

impl From<serde_json::Error> for LeptosAiError {
    fn from(e: serde_json::Error) -> Self {
        LeptosAiError::ParseError(e.to_string())
    }
}

impl From<rust_ai_agents_llm_client::LlmClientError> for LeptosAiError {
    fn from(e: rust_ai_agents_llm_client::LlmClientError) -> Self {
        LeptosAiError::ApiError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, LeptosAiError>;
