//! Error types for Cloudflare Workers agent

use thiserror::Error;

/// Errors that can occur when using the Cloudflare agent
#[derive(Error, Debug)]
pub enum CloudflareError {
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    /// Failed to parse response
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// Provider API returned an error
    #[error("Provider error: {0}")]
    ProviderError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// KV store error
    #[error("KV store error: {0}")]
    KvError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Stream error
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Worker error
    #[error("Worker error: {0}")]
    WorkerError(String),
}

impl From<worker::Error> for CloudflareError {
    fn from(err: worker::Error) -> Self {
        CloudflareError::WorkerError(err.to_string())
    }
}

impl From<serde_json::Error> for CloudflareError {
    fn from(err: serde_json::Error) -> Self {
        CloudflareError::SerializationError(err.to_string())
    }
}

impl From<rust_ai_agents_llm_client::LlmClientError> for CloudflareError {
    fn from(err: rust_ai_agents_llm_client::LlmClientError) -> Self {
        CloudflareError::ProviderError(err.to_string())
    }
}

/// Result type for Cloudflare agent operations
pub type Result<T> = std::result::Result<T, CloudflareError>;
