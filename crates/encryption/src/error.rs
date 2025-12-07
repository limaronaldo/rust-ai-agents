//! Error types for encryption operations.

use thiserror::Error;

/// Errors that can occur during encryption operations.
#[derive(Debug, Error)]
pub enum CryptoError {
    /// Encryption failed
    #[error("encryption failed: {0}")]
    EncryptionFailed(String),

    /// Decryption failed
    #[error("decryption failed: {0}")]
    DecryptionFailed(String),

    /// Invalid key length
    #[error("invalid key length: expected {expected}, got {got}")]
    InvalidKeyLength { expected: usize, got: usize },

    /// Invalid nonce length
    #[error("invalid nonce length: expected {expected}, got {got}")]
    InvalidNonceLength { expected: usize, got: usize },

    /// Key derivation failed
    #[error("key derivation failed: {0}")]
    KeyDerivationFailed(String),

    /// Invalid ciphertext format
    #[error("invalid ciphertext format: {0}")]
    InvalidCiphertext(String),

    /// Key not found for version
    #[error("key not found for version {0}")]
    KeyNotFound(u32),

    /// Serialization error
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// Base64 decoding error
    #[error("base64 decode error: {0}")]
    Base64Error(String),
}

impl From<serde_json::Error> for CryptoError {
    fn from(err: serde_json::Error) -> Self {
        CryptoError::SerializationError(err.to_string())
    }
}

impl From<base64::DecodeError> for CryptoError {
    fn from(err: base64::DecodeError) -> Self {
        CryptoError::Base64Error(err.to_string())
    }
}

/// Result type for encryption operations.
pub type CryptoResult<T> = Result<T, CryptoError>;
