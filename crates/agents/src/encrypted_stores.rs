//! Encrypted storage wrappers for sessions and checkpoints.
//!
//! This module provides encrypted implementations of `SessionStore` and `CheckpointStore`
//! that wrap existing store implementations with transparent encryption.
//!
//! # Feature Flag
//!
//! This module requires the `encryption` feature:
//!
//! ```toml
//! rust-ai-agents-agents = { version = "0.1", features = ["encryption"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_agents::{
//!     session::{MemorySessionStore, ConversationManager},
//!     encrypted_stores::EncryptedSessionStoreWrapper,
//! };
//! use rust_ai_agents_encryption::{EncryptionKey, EnvelopeEncryptor};
//! use std::sync::Arc;
//!
//! // Create encryption key and encryptor
//! let key = EncryptionKey::generate(32);
//! let encryptor = Arc::new(EnvelopeEncryptor::new(key));
//!
//! // Wrap memory store with encryption
//! let inner_store = MemorySessionStore::new();
//! let encrypted_store = EncryptedSessionStoreWrapper::new(inner_store, encryptor);
//!
//! // Use with ConversationManager
//! let manager = ConversationManager::with_store(Arc::new(encrypted_store));
//! ```

use rust_ai_agents_encryption::{CryptoError, EnvelopeEncryptor};

use crate::checkpoint::{CheckpointError, CheckpointId, CheckpointMetadata, CheckpointStore};
use crate::session::{ConversationId, ConversationSession, SessionError, SessionStore};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use std::sync::Arc;

/// Encrypted session store wrapper.
///
/// Wraps any `SessionStore` implementation to provide transparent encryption.
/// Sessions are serialized to JSON, encrypted, then stored.
pub struct EncryptedSessionStoreWrapper<S> {
    inner: S,
    encryptor: Arc<EnvelopeEncryptor>,
}

impl<S> EncryptedSessionStoreWrapper<S> {
    /// Create a new encrypted session store wrapper.
    pub fn new(inner: S, encryptor: Arc<EnvelopeEncryptor>) -> Self {
        Self { inner, encryptor }
    }

    /// Get a reference to the inner store.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get a reference to the encryptor.
    pub fn encryptor(&self) -> &EnvelopeEncryptor {
        &self.encryptor
    }
}

fn crypto_to_session_error(e: CryptoError) -> SessionError {
    SessionError::StorageError(format!("encryption error: {}", e))
}

#[async_trait]
impl<S: SessionStore> SessionStore for EncryptedSessionStoreWrapper<S> {
    async fn save(&self, session: &ConversationSession) -> Result<(), SessionError> {
        // Serialize and encrypt session
        let json = serde_json::to_vec(session)
            .map_err(|e| SessionError::SerializationError(e.to_string()))?;

        let ciphertext = self
            .encryptor
            .encrypt(&json, Some(session.id.as_bytes()))
            .map_err(crypto_to_session_error)?;

        // Create a wrapper session with encrypted data stored in state
        let mut encrypted_session = session.clone();
        encrypted_session
            .state
            .set("__encrypted_data", STANDARD.encode(&ciphertext));
        encrypted_session.messages.clear(); // Clear sensitive data from wrapper

        self.inner.save(&encrypted_session).await
    }

    async fn load(&self, session_id: &str) -> Result<Option<ConversationSession>, SessionError> {
        let encrypted_session = match self.inner.load(session_id).await? {
            Some(s) => s,
            None => return Ok(None),
        };

        // Get encrypted data from state
        let encrypted_data = match encrypted_session.state.get("__encrypted_data") {
            Some(s) => s.clone(),
            None => {
                // No encrypted data, return as-is (legacy unencrypted session)
                return Ok(Some(encrypted_session));
            }
        };

        // Decode and decrypt
        let ciphertext = STANDARD
            .decode(&encrypted_data)
            .map_err(|e| SessionError::StorageError(format!("base64 decode error: {}", e)))?;

        let plaintext = self
            .encryptor
            .decrypt(&ciphertext, Some(session_id.as_bytes()))
            .map_err(crypto_to_session_error)?;

        let session: ConversationSession = serde_json::from_slice(&plaintext)
            .map_err(|e| SessionError::SerializationError(e.to_string()))?;

        Ok(Some(session))
    }

    async fn delete(&self, session_id: &str) -> Result<(), SessionError> {
        self.inner.delete(session_id).await
    }

    async fn list(&self) -> Result<Vec<ConversationId>, SessionError> {
        self.inner.list().await
    }

    async fn list_for_user(&self, user_id: &str) -> Result<Vec<ConversationId>, SessionError> {
        self.inner.list_for_user(user_id).await
    }
}

/// Encrypted checkpoint store wrapper.
///
/// Wraps any `CheckpointStore` implementation to provide transparent encryption.
/// Checkpoint state is encrypted before storage.
pub struct EncryptedCheckpointStoreWrapper<S> {
    inner: S,
    encryptor: Arc<EnvelopeEncryptor>,
}

impl<S> EncryptedCheckpointStoreWrapper<S> {
    /// Create a new encrypted checkpoint store wrapper.
    pub fn new(inner: S, encryptor: Arc<EnvelopeEncryptor>) -> Self {
        Self { inner, encryptor }
    }

    /// Get a reference to the inner store.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get a reference to the encryptor.
    pub fn encryptor(&self) -> &EnvelopeEncryptor {
        &self.encryptor
    }
}

fn crypto_to_checkpoint_error(e: CryptoError) -> CheckpointError {
    CheckpointError::Storage(format!("encryption error: {}", e))
}

#[async_trait]
impl<S: CheckpointStore> CheckpointStore for EncryptedCheckpointStoreWrapper<S> {
    async fn save(
        &self,
        thread_id: &str,
        metadata: CheckpointMetadata,
        state: Vec<u8>,
    ) -> Result<CheckpointId, CheckpointError> {
        // Encrypt state with thread_id as associated data
        let ciphertext = self
            .encryptor
            .encrypt(&state, Some(thread_id.as_bytes()))
            .map_err(crypto_to_checkpoint_error)?;

        self.inner.save(thread_id, metadata, ciphertext).await
    }

    async fn load(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<(CheckpointMetadata, Vec<u8>), CheckpointError> {
        let (metadata, ciphertext) = self.inner.load(thread_id, checkpoint_id).await?;

        // Decrypt state
        let plaintext = self
            .encryptor
            .decrypt(&ciphertext, Some(thread_id.as_bytes()))
            .map_err(crypto_to_checkpoint_error)?;

        Ok((metadata, plaintext))
    }

    async fn load_latest(
        &self,
        thread_id: &str,
    ) -> Result<(CheckpointMetadata, Vec<u8>), CheckpointError> {
        let (metadata, ciphertext) = self.inner.load_latest(thread_id).await?;

        // Decrypt state
        let plaintext = self
            .encryptor
            .decrypt(&ciphertext, Some(thread_id.as_bytes()))
            .map_err(crypto_to_checkpoint_error)?;

        Ok((metadata, plaintext))
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>, CheckpointError> {
        self.inner.list(thread_id).await
    }

    async fn delete(&self, thread_id: &str, checkpoint_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete(thread_id, checkpoint_id).await
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_thread(thread_id).await
    }

    async fn count(&self, thread_id: &str) -> Result<usize, CheckpointError> {
        self.inner.count(thread_id).await
    }

    async fn list_threads(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_threads().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{ChatRole, MemorySessionStore};
    use rust_ai_agents_encryption::EncryptionKey;

    #[tokio::test]
    async fn test_encrypted_session_roundtrip() {
        let key = EncryptionKey::generate(32);
        let encryptor = Arc::new(EnvelopeEncryptor::new(key));
        let inner_store = MemorySessionStore::new();
        let store = EncryptedSessionStoreWrapper::new(inner_store, encryptor);

        // Create and save session
        let mut session = ConversationSession::new("test-session");
        session.add_message(ChatRole::User, "Hello!");
        session.add_message(ChatRole::Assistant, "Hi there!");

        store.save(&session).await.unwrap();

        // Load and verify
        let loaded = store.load("test-session").await.unwrap().unwrap();
        assert_eq!(loaded.id, session.id);
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].content, "Hello!");
        assert_eq!(loaded.messages[1].content, "Hi there!");
    }

    #[tokio::test]
    async fn test_encrypted_session_list() {
        let key = EncryptionKey::generate(32);
        let encryptor = Arc::new(EnvelopeEncryptor::new(key));
        let inner_store = MemorySessionStore::new();
        let store = EncryptedSessionStoreWrapper::new(inner_store, encryptor);

        // Create multiple sessions
        for i in 0..3 {
            let session = ConversationSession::new(format!("session-{}", i));
            store.save(&session).await.unwrap();
        }

        let sessions = store.list().await.unwrap();
        assert_eq!(sessions.len(), 3);
    }

    #[tokio::test]
    async fn test_encrypted_session_delete() {
        let key = EncryptionKey::generate(32);
        let encryptor = Arc::new(EnvelopeEncryptor::new(key));
        let inner_store = MemorySessionStore::new();
        let store = EncryptedSessionStoreWrapper::new(inner_store, encryptor);

        let session = ConversationSession::new("to-delete");
        store.save(&session).await.unwrap();

        assert!(store.load("to-delete").await.unwrap().is_some());

        store.delete("to-delete").await.unwrap();

        assert!(store.load("to-delete").await.unwrap().is_none());
    }
}
