//! At-rest encryption for sensitive data in rust-ai-agents.
//!
//! This crate provides encryption primitives for protecting sessions, checkpoints,
//! and other sensitive data at rest. It supports:
//!
//! - **AES-256-GCM** authenticated encryption (default)
//! - **ChaCha20-Poly1305** authenticated encryption (optional)
//! - **Argon2id** password-based key derivation
//! - **Key rotation** via versioned keys and envelope encryption
//! - **Secure memory** handling with automatic zeroing
//!
//! # Quick Start
//!
//! ```rust
//! use rust_ai_agents_encryption::{EncryptionKey, EnvelopeEncryptor, DataEncryptor};
//!
//! // Generate a random 256-bit key
//! let key = EncryptionKey::generate(32);
//!
//! // Create an encryptor
//! let encryptor = EnvelopeEncryptor::new(key);
//!
//! // Encrypt structured data
//! let secret = serde_json::json!({"user": "alice", "token": "secret123"});
//! let ciphertext = encryptor.encrypt_data(&secret).unwrap();
//!
//! // Decrypt
//! let decrypted: serde_json::Value = encryptor.decrypt_data(&ciphertext).unwrap();
//! assert_eq!(secret, decrypted);
//! ```
//!
//! # Key Derivation
//!
//! For password-based encryption:
//!
//! ```rust
//! use rust_ai_agents_encryption::{Argon2KeyDerivation, KeyDerivation, EnvelopeEncryptor};
//!
//! let kdf = Argon2KeyDerivation::new();
//! let salt = kdf.generate_salt(16);
//! let key = kdf.derive_encryption_key(b"user-password", &salt, 32).unwrap();
//!
//! let encryptor = EnvelopeEncryptor::new(key);
//! ```
//!
//! # Key Rotation
//!
//! ```rust
//! use rust_ai_agents_encryption::{EncryptionKey, EnvelopeEncryptor};
//!
//! let key1 = EncryptionKey::generate(32);
//! let mut encryptor = EnvelopeEncryptor::new(key1);
//!
//! // Encrypt with v1
//! let ciphertext = encryptor.encrypt(b"secret", None).unwrap();
//!
//! // Rotate to v2
//! let key2 = EncryptionKey::generate(32);
//! encryptor.rotate_key(key2);
//!
//! // Old ciphertext still decrypts (key v1 retained)
//! let plaintext = encryptor.decrypt(&ciphertext, None).unwrap();
//!
//! // Re-encrypt with new key
//! let new_ciphertext = encryptor.re_encrypt(&ciphertext, None).unwrap();
//! ```
//!
//! # Store Wrappers
//!
//! For encrypting session and checkpoint stores:
//!
//! ```rust,ignore
//! use rust_ai_agents_encryption::{EncryptedSessionStore, EncryptedCheckpointStore};
//!
//! // Wrap existing stores with encryption
//! let encrypted_sessions = EncryptedSessionStore::new(session_store, encryptor.clone());
//! let encrypted_checkpoints = EncryptedCheckpointStore::new(checkpoint_store, encryptor);
//! ```

pub mod error;
pub mod key;
pub mod traits;

#[cfg(feature = "aes")]
pub mod aes_cipher;

#[cfg(feature = "chacha")]
pub mod chacha_cipher;

pub mod envelope;
pub mod stores;

// Re-exports
pub use error::{CryptoError, CryptoResult};
pub use key::{Argon2KeyDerivation, EncryptionKey, KeyRing, VersionedKey};
pub use traits::{Cipher, DataEncryptor, KeyDerivation};

#[cfg(feature = "aes")]
pub use aes_cipher::Aes256GcmCipher;

#[cfg(feature = "chacha")]
pub use chacha_cipher::ChaCha20Poly1305Cipher;

pub use envelope::EnvelopeEncryptor;
pub use stores::{EncryptedCheckpointStore, EncryptedSessionStore};
