//! Encrypted store wrappers for sessions and checkpoints.
//!
//! These wrappers provide transparent encryption for storage backends.
//! Data is encrypted before saving and decrypted after loading.

use std::sync::Arc;

use crate::envelope::EnvelopeEncryptor;
use crate::error::{CryptoError, CryptoResult};

/// Encrypted session store wrapper.
///
/// Wraps any session store implementation to provide transparent encryption.
/// Sessions are serialized to JSON, encrypted, then stored as bytes.
///
/// # Example
///
/// ```rust,ignore
/// use rust_ai_agents_encryption::{EncryptionKey, EnvelopeEncryptor, EncryptedSessionStore};
/// use rust_ai_agents_agents::session::MemorySessionStore;
///
/// let key = EncryptionKey::generate(32);
/// let encryptor = Arc::new(EnvelopeEncryptor::new(key));
/// let inner_store = MemorySessionStore::new();
///
/// let encrypted_store = EncryptedSessionStore::new(inner_store, encryptor);
/// ```
pub struct EncryptedSessionStore<S> {
    inner: S,
    encryptor: Arc<EnvelopeEncryptor>,
}

impl<S> EncryptedSessionStore<S> {
    /// Create a new encrypted session store.
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

    /// Encrypt session data.
    #[cfg(feature = "aes")]
    pub fn encrypt_session<T: serde::Serialize>(&self, session: &T) -> CryptoResult<Vec<u8>> {
        let json = serde_json::to_vec(session)?;
        self.encryptor.encrypt(&json, None)
    }

    /// Decrypt session data.
    #[cfg(feature = "aes")]
    pub fn decrypt_session<T: serde::de::DeserializeOwned>(
        &self,
        ciphertext: &[u8],
    ) -> CryptoResult<T> {
        let plaintext = self.encryptor.decrypt(ciphertext, None)?;
        let session = serde_json::from_slice(&plaintext)?;
        Ok(session)
    }
}

impl<S: Clone> Clone for EncryptedSessionStore<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            encryptor: self.encryptor.clone(),
        }
    }
}

impl<S: std::fmt::Debug> std::fmt::Debug for EncryptedSessionStore<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedSessionStore")
            .field("inner", &self.inner)
            .field("encryptor", &"[EnvelopeEncryptor]")
            .finish()
    }
}

/// Encrypted checkpoint store wrapper.
///
/// Wraps any checkpoint store implementation to provide transparent encryption.
/// Checkpoint state bytes are encrypted before storage.
///
/// # Example
///
/// ```rust,ignore
/// use rust_ai_agents_encryption::{EncryptionKey, EnvelopeEncryptor, EncryptedCheckpointStore};
/// use rust_ai_agents_agents::checkpoint::MemoryCheckpointStore;
///
/// let key = EncryptionKey::generate(32);
/// let encryptor = Arc::new(EnvelopeEncryptor::new(key));
/// let inner_store = MemoryCheckpointStore::new();
///
/// let encrypted_store = EncryptedCheckpointStore::new(inner_store, encryptor);
/// ```
pub struct EncryptedCheckpointStore<S> {
    inner: S,
    encryptor: Arc<EnvelopeEncryptor>,
}

impl<S> EncryptedCheckpointStore<S> {
    /// Create a new encrypted checkpoint store.
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

    /// Encrypt checkpoint state.
    ///
    /// Uses the thread_id as associated data for additional authentication.
    #[cfg(feature = "aes")]
    pub fn encrypt_state(&self, state: &[u8], thread_id: &str) -> CryptoResult<Vec<u8>> {
        self.encryptor.encrypt(state, Some(thread_id.as_bytes()))
    }

    /// Decrypt checkpoint state.
    ///
    /// Uses the thread_id as associated data for verification.
    #[cfg(feature = "aes")]
    pub fn decrypt_state(&self, ciphertext: &[u8], thread_id: &str) -> CryptoResult<Vec<u8>> {
        self.encryptor
            .decrypt(ciphertext, Some(thread_id.as_bytes()))
    }
}

impl<S: Clone> Clone for EncryptedCheckpointStore<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            encryptor: self.encryptor.clone(),
        }
    }
}

impl<S: std::fmt::Debug> std::fmt::Debug for EncryptedCheckpointStore<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedCheckpointStore")
            .field("inner", &self.inner)
            .field("encryptor", &"[EnvelopeEncryptor]")
            .finish()
    }
}

/// Configuration for encrypted storage.
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Whether encryption is enabled
    pub enabled: bool,
    /// Master key (base64 encoded)
    pub master_key: Option<String>,
    /// Key derivation salt (base64 encoded)
    pub salt: Option<String>,
    /// Algorithm to use (aes-256-gcm or chacha20-poly1305)
    pub algorithm: String,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            master_key: None,
            salt: None,
            algorithm: "aes-256-gcm".to_string(),
        }
    }
}

impl EncryptionConfig {
    /// Create a new encryption config with a generated key.
    #[cfg(feature = "aes")]
    pub fn with_generated_key() -> Self {
        use crate::key::EncryptionKey;
        let key = EncryptionKey::generate(32);
        Self {
            enabled: true,
            master_key: Some(key.to_base64()),
            salt: None,
            algorithm: "aes-256-gcm".to_string(),
        }
    }

    /// Create an encryptor from this config.
    #[cfg(feature = "aes")]
    pub fn create_encryptor(&self) -> CryptoResult<Option<EnvelopeEncryptor>> {
        use crate::key::EncryptionKey;

        if !self.enabled {
            return Ok(None);
        }

        let key = match &self.master_key {
            Some(k) => EncryptionKey::from_base64(k)?,
            None => {
                return Err(CryptoError::KeyDerivationFailed(
                    "master_key required when encryption is enabled".to_string(),
                ))
            }
        };

        Ok(Some(EnvelopeEncryptor::new(key)))
    }
}

#[cfg(all(test, feature = "aes"))]
mod tests {
    use super::*;
    use crate::key::EncryptionKey;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct MockSession {
        id: String,
        user_id: String,
        messages: Vec<String>,
    }

    #[test]
    fn test_encrypted_session_store_roundtrip() {
        let key = EncryptionKey::generate(32);
        let encryptor = Arc::new(EnvelopeEncryptor::new(key));

        // Using unit as a mock inner store for testing encryption only
        let store: EncryptedSessionStore<()> = EncryptedSessionStore::new((), encryptor);

        let session = MockSession {
            id: "sess-123".to_string(),
            user_id: "user-456".to_string(),
            messages: vec!["Hello".to_string(), "World".to_string()],
        };

        let encrypted = store.encrypt_session(&session).unwrap();
        let decrypted: MockSession = store.decrypt_session(&encrypted).unwrap();

        assert_eq!(session, decrypted);
    }

    #[test]
    fn test_encrypted_checkpoint_store_with_aad() {
        let key = EncryptionKey::generate(32);
        let encryptor = Arc::new(EnvelopeEncryptor::new(key));

        let store: EncryptedCheckpointStore<()> = EncryptedCheckpointStore::new((), encryptor);

        let state = b"checkpoint state data";
        let thread_id = "thread-123";

        let encrypted = store.encrypt_state(state, thread_id).unwrap();
        let decrypted = store.decrypt_state(&encrypted, thread_id).unwrap();

        assert_eq!(state.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_checkpoint_wrong_thread_id_fails() {
        let key = EncryptionKey::generate(32);
        let encryptor = Arc::new(EnvelopeEncryptor::new(key));

        let store: EncryptedCheckpointStore<()> = EncryptedCheckpointStore::new((), encryptor);

        let state = b"checkpoint state data";
        let encrypted = store.encrypt_state(state, "thread-123").unwrap();

        // Try to decrypt with wrong thread_id
        let result = store.decrypt_state(&encrypted, "thread-456");
        assert!(result.is_err());
    }

    #[test]
    fn test_encryption_config_create_encryptor() {
        let config = EncryptionConfig::with_generated_key();
        assert!(config.enabled);
        assert!(config.master_key.is_some());

        let encryptor = config.create_encryptor().unwrap();
        assert!(encryptor.is_some());
    }

    #[test]
    fn test_disabled_encryption_config() {
        let config = EncryptionConfig::default();
        assert!(!config.enabled);

        let encryptor = config.create_encryptor().unwrap();
        assert!(encryptor.is_none());
    }
}
