//! Core traits for encryption operations.

use crate::error::CryptoResult;

/// Trait for symmetric encryption ciphers.
///
/// Implementors provide authenticated encryption with associated data (AEAD).
pub trait Cipher: Send + Sync {
    /// Encrypt plaintext with optional associated data.
    ///
    /// Returns the ciphertext with embedded nonce and authentication tag.
    fn encrypt(&self, plaintext: &[u8], associated_data: Option<&[u8]>) -> CryptoResult<Vec<u8>>;

    /// Decrypt ciphertext with optional associated data.
    ///
    /// The ciphertext must include the nonce and authentication tag.
    fn decrypt(&self, ciphertext: &[u8], associated_data: Option<&[u8]>) -> CryptoResult<Vec<u8>>;

    /// Get the cipher algorithm name.
    fn algorithm(&self) -> &'static str;

    /// Get the key size in bytes.
    fn key_size(&self) -> usize;

    /// Get the nonce size in bytes.
    fn nonce_size(&self) -> usize;

    /// Get the authentication tag size in bytes.
    fn tag_size(&self) -> usize;
}

/// Trait for key derivation functions.
pub trait KeyDerivation: Send + Sync {
    /// Derive a key from a password and salt.
    fn derive_key(&self, password: &[u8], salt: &[u8], key_length: usize) -> CryptoResult<Vec<u8>>;

    /// Generate a random salt of the specified length.
    fn generate_salt(&self, length: usize) -> Vec<u8>;

    /// Get the algorithm name.
    fn algorithm(&self) -> &'static str;
}

/// Trait for encrypting typed data with serialization.
pub trait DataEncryptor: Send + Sync {
    /// Encrypt a serializable value.
    fn encrypt_data<T: serde::Serialize>(&self, data: &T) -> CryptoResult<Vec<u8>>;

    /// Decrypt to a deserializable value.
    fn decrypt_data<T: serde::de::DeserializeOwned>(&self, ciphertext: &[u8]) -> CryptoResult<T>;
}
