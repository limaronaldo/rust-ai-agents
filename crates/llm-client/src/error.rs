//! Error types for the LLM client.

use thiserror::Error;

/// Result type for LLM client operations.
pub type Result<T> = std::result::Result<T, LlmClientError>;

/// Errors that can occur in the LLM client.
#[derive(Debug, Error)]
pub enum LlmClientError {
    /// JSON serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid provider
    #[error("Unknown provider: {0}")]
    UnknownProvider(String),

    /// API error response
    #[error("API error: {message} (type: {error_type})")]
    ApiError {
        error_type: String,
        message: String,
        code: Option<String>,
    },

    /// Unexpected response format
    #[error("Unexpected response format: {0}")]
    UnexpectedFormat(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimited,

    /// Authentication error
    #[error("Authentication failed: {0}")]
    AuthError(String),
}

impl LlmClientError {
    /// Create a missing field error.
    pub fn missing(field: &str) -> Self {
        Self::MissingField(field.to_string())
    }

    /// Create an API error from response JSON.
    pub fn from_api_response(json: &serde_json::Value) -> Self {
        let error = &json["error"];
        Self::ApiError {
            error_type: error["type"].as_str().unwrap_or("unknown").to_string(),
            message: error["message"]
                .as_str()
                .unwrap_or("Unknown error")
                .to_string(),
            code: error["code"].as_str().map(String::from),
        }
    }
}
