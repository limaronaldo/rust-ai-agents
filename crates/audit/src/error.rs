//! Error types for the audit module.

use thiserror::Error;

/// Errors that can occur during audit operations.
#[derive(Debug, Error)]
pub enum AuditError {
    /// I/O error during file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Buffer full error
    #[error("Audit buffer full, events may be dropped")]
    BufferFull,

    /// Logger not initialized
    #[error("Audit logger not initialized")]
    NotInitialized,

    /// Multiple errors from composite logger
    #[error("Multiple errors: {0:?}")]
    Multiple(Vec<String>),

    /// Rotation error
    #[error("Log rotation error: {0}")]
    Rotation(String),

    /// Channel send error
    #[error("Channel send error")]
    ChannelSend,

    /// Shutdown error
    #[error("Logger shutdown error: {0}")]
    Shutdown(String),

    /// Generic internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl AuditError {
    /// Create a configuration error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a rotation error.
    pub fn rotation(msg: impl Into<String>) -> Self {
        Self::Rotation(msg.into())
    }

    /// Create an internal error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}
