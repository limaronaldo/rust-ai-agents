//! Cache error types

use thiserror::Error;

/// Errors that can occur during cache operations
#[derive(Debug, Error)]
pub enum CacheError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Key not found: {0}")]
    NotFound(String),

    #[error("Cache is full")]
    CacheFull,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[cfg(feature = "redis")]
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),
}

impl CacheError {
    /// Check if error is transient and operation can be retried
    pub fn is_transient(&self) -> bool {
        matches!(self, CacheError::Connection(_) | CacheError::Backend(_))
    }
}
