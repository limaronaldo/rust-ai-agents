//! Encryption key management and derivation.

use crate::error::{CryptoError, CryptoResult};
use crate::traits::KeyDerivation;
use argon2::{Argon2, Params};
use rand::RngCore;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// A secure encryption key that is zeroed on drop.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct EncryptionKey {
    bytes: Vec<u8>,
}

impl EncryptionKey {
    /// Create a new encryption key from bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Generate a random encryption key of the specified length.
    pub fn generate(length: usize) -> Self {
        let mut bytes = vec![0u8; length];
        rand::thread_rng().fill_bytes(&mut bytes);
        Self { bytes }
    }

    /// Create a key from a base64-encoded string.
    pub fn from_base64(encoded: &str) -> CryptoResult<Self> {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let bytes = STANDARD.decode(encoded)?;
        Ok(Self { bytes })
    }

    /// Encode the key as base64.
    pub fn to_base64(&self) -> String {
        use base64::{engine::general_purpose::STANDARD, Engine};
        STANDARD.encode(&self.bytes)
    }

    /// Get the key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get the key length.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Check if the key is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl std::fmt::Debug for EncryptionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptionKey")
            .field("len", &self.bytes.len())
            .field("bytes", &"[REDACTED]")
            .finish()
    }
}

/// Argon2id key derivation function.
///
/// Uses Argon2id which provides resistance against both side-channel
/// and GPU-based attacks.
pub struct Argon2KeyDerivation {
    params: Params,
}

impl Default for Argon2KeyDerivation {
    fn default() -> Self {
        Self::new()
    }
}

impl Argon2KeyDerivation {
    /// Create with default parameters (OWASP recommended).
    pub fn new() -> Self {
        // OWASP recommended: m=19456 (19 MiB), t=2, p=1
        let params = Params::new(19456, 2, 1, Some(32)).expect("valid params");
        Self { params }
    }

    /// Create with custom parameters.
    pub fn with_params(memory_kib: u32, iterations: u32, parallelism: u32) -> CryptoResult<Self> {
        let params = Params::new(memory_kib, iterations, parallelism, Some(32))
            .map_err(|e| CryptoError::KeyDerivationFailed(e.to_string()))?;
        Ok(Self { params })
    }

    /// Derive an encryption key from a password.
    pub fn derive_encryption_key(
        &self,
        password: &[u8],
        salt: &[u8],
        key_length: usize,
    ) -> CryptoResult<EncryptionKey> {
        let key_bytes = self.derive_key(password, salt, key_length)?;
        Ok(EncryptionKey::new(key_bytes))
    }
}

impl KeyDerivation for Argon2KeyDerivation {
    fn derive_key(&self, password: &[u8], salt: &[u8], key_length: usize) -> CryptoResult<Vec<u8>> {
        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            self.params.clone(),
        );

        let mut output = vec![0u8; key_length];
        argon2
            .hash_password_into(password, salt, &mut output)
            .map_err(|e| CryptoError::KeyDerivationFailed(e.to_string()))?;

        Ok(output)
    }

    fn generate_salt(&self, length: usize) -> Vec<u8> {
        let mut salt = vec![0u8; length];
        rand::thread_rng().fill_bytes(&mut salt);
        salt
    }

    fn algorithm(&self) -> &'static str {
        "argon2id"
    }
}

/// Versioned key for key rotation support.
#[derive(Clone)]
pub struct VersionedKey {
    /// Key version number
    pub version: u32,
    /// The encryption key
    pub key: EncryptionKey,
    /// When this key was created (Unix timestamp)
    pub created_at: u64,
    /// Whether this key is active for new encryptions
    pub active: bool,
}

impl VersionedKey {
    /// Create a new versioned key.
    pub fn new(version: u32, key: EncryptionKey) -> Self {
        Self {
            version,
            key,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            active: true,
        }
    }
}

impl std::fmt::Debug for VersionedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VersionedKey")
            .field("version", &self.version)
            .field("created_at", &self.created_at)
            .field("active", &self.active)
            .field("key", &"[REDACTED]")
            .finish()
    }
}

/// Key ring for managing multiple key versions.
///
/// Supports key rotation by maintaining multiple key versions.
/// New encryptions use the active key, decryptions use the version
/// embedded in the ciphertext.
#[derive(Default)]
pub struct KeyRing {
    keys: Vec<VersionedKey>,
}

impl KeyRing {
    /// Create an empty key ring.
    pub fn new() -> Self {
        Self { keys: Vec::new() }
    }

    /// Add a key to the ring.
    pub fn add_key(&mut self, key: VersionedKey) {
        // Deactivate any previously active keys
        if key.active {
            for k in &mut self.keys {
                k.active = false;
            }
        }
        self.keys.push(key);
    }

    /// Get the active key for encryption.
    pub fn active_key(&self) -> Option<&VersionedKey> {
        self.keys.iter().find(|k| k.active)
    }

    /// Get a key by version for decryption.
    pub fn get_key(&self, version: u32) -> Option<&VersionedKey> {
        self.keys.iter().find(|k| k.version == version)
    }

    /// Rotate to a new key.
    pub fn rotate(&mut self, new_key: EncryptionKey) -> u32 {
        let new_version = self.keys.iter().map(|k| k.version).max().unwrap_or(0) + 1;
        self.add_key(VersionedKey::new(new_version, new_key));
        new_version
    }

    /// Get all keys (for re-encryption during rotation).
    pub fn all_keys(&self) -> &[VersionedKey] {
        &self.keys
    }

    /// Number of keys in the ring.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl std::fmt::Debug for KeyRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyRing")
            .field("num_keys", &self.keys.len())
            .field("active_version", &self.active_key().map(|k| k.version))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_key_generate() {
        let key = EncryptionKey::generate(32);
        assert_eq!(key.len(), 32);
        assert!(!key.is_empty());
    }

    #[test]
    fn test_encryption_key_base64_roundtrip() {
        let key = EncryptionKey::generate(32);
        let encoded = key.to_base64();
        let decoded = EncryptionKey::from_base64(&encoded).unwrap();
        assert_eq!(key.as_bytes(), decoded.as_bytes());
    }

    #[test]
    fn test_argon2_key_derivation() {
        let kdf = Argon2KeyDerivation::new();
        let password = b"test-password";
        let salt = kdf.generate_salt(16);

        let key1 = kdf.derive_key(password, &salt, 32).unwrap();
        let key2 = kdf.derive_key(password, &salt, 32).unwrap();

        // Same password + salt should produce same key
        assert_eq!(key1, key2);
        assert_eq!(key1.len(), 32);
    }

    #[test]
    fn test_argon2_different_salts() {
        let kdf = Argon2KeyDerivation::new();
        let password = b"test-password";
        let salt1 = kdf.generate_salt(16);
        let salt2 = kdf.generate_salt(16);

        let key1 = kdf.derive_key(password, &salt1, 32).unwrap();
        let key2 = kdf.derive_key(password, &salt2, 32).unwrap();

        // Different salts should produce different keys
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_ring_rotation() {
        let mut ring = KeyRing::new();

        let key1 = EncryptionKey::generate(32);
        ring.add_key(VersionedKey::new(1, key1));

        assert_eq!(ring.active_key().unwrap().version, 1);

        let key2 = EncryptionKey::generate(32);
        let v2 = ring.rotate(key2);

        assert_eq!(v2, 2);
        assert_eq!(ring.active_key().unwrap().version, 2);
        assert_eq!(ring.len(), 2);

        // Old key should still be retrievable
        assert!(ring.get_key(1).is_some());
        assert!(!ring.get_key(1).unwrap().active);
    }
}
