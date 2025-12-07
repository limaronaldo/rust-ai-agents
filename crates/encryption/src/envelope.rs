//! Envelope encryption with key versioning support.
//!
//! Envelope encryption wraps data with a version header to support key rotation.
//! The ciphertext format includes the key version, allowing decryption with
//! the correct key from the key ring.

use crate::error::{CryptoError, CryptoResult};
use crate::key::{EncryptionKey, KeyRing, VersionedKey};
use crate::traits::{Cipher, DataEncryptor};

#[cfg(feature = "aes")]
use crate::aes_cipher::Aes256GcmCipher;

use serde::{de::DeserializeOwned, Serialize};

/// Envelope encryption header format version.
const ENVELOPE_VERSION: u8 = 1;

/// Envelope header size: version (1) + key_version (4) = 5 bytes
const ENVELOPE_HEADER_SIZE: usize = 5;

/// Envelope encryptor with key rotation support.
///
/// # Ciphertext Format
///
/// ```text
/// [envelope_version: 1 byte][key_version: 4 bytes][cipher_data: variable]
/// ```
///
/// The cipher_data contains the nonce, ciphertext, and authentication tag
/// as produced by the underlying cipher.
pub struct EnvelopeEncryptor {
    key_ring: KeyRing,
}

impl EnvelopeEncryptor {
    /// Create a new envelope encryptor with a single key.
    pub fn new(key: EncryptionKey) -> Self {
        let mut key_ring = KeyRing::new();
        key_ring.add_key(VersionedKey::new(1, key));
        Self { key_ring }
    }

    /// Create from an existing key ring.
    pub fn with_key_ring(key_ring: KeyRing) -> Self {
        Self { key_ring }
    }

    /// Get a reference to the key ring.
    pub fn key_ring(&self) -> &KeyRing {
        &self.key_ring
    }

    /// Get a mutable reference to the key ring.
    pub fn key_ring_mut(&mut self) -> &mut KeyRing {
        &mut self.key_ring
    }

    /// Rotate to a new key.
    pub fn rotate_key(&mut self, new_key: EncryptionKey) -> u32 {
        self.key_ring.rotate(new_key)
    }

    /// Encrypt data with the active key.
    #[cfg(feature = "aes")]
    pub fn encrypt(
        &self,
        plaintext: &[u8],
        associated_data: Option<&[u8]>,
    ) -> CryptoResult<Vec<u8>> {
        let active = self
            .key_ring
            .active_key()
            .ok_or(CryptoError::KeyNotFound(0))?;

        let cipher = Aes256GcmCipher::new(&active.key)?;
        let cipher_data = cipher.encrypt(plaintext, associated_data)?;

        // Build envelope: version + key_version + cipher_data
        let mut envelope = Vec::with_capacity(ENVELOPE_HEADER_SIZE + cipher_data.len());
        envelope.push(ENVELOPE_VERSION);
        envelope.extend_from_slice(&active.version.to_le_bytes());
        envelope.extend_from_slice(&cipher_data);

        Ok(envelope)
    }

    /// Decrypt data using the key version from the envelope.
    #[cfg(feature = "aes")]
    pub fn decrypt(
        &self,
        ciphertext: &[u8],
        associated_data: Option<&[u8]>,
    ) -> CryptoResult<Vec<u8>> {
        if ciphertext.len() < ENVELOPE_HEADER_SIZE {
            return Err(CryptoError::InvalidCiphertext(
                "envelope too short".to_string(),
            ));
        }

        let envelope_version = ciphertext[0];
        if envelope_version != ENVELOPE_VERSION {
            return Err(CryptoError::InvalidCiphertext(format!(
                "unsupported envelope version: {}",
                envelope_version
            )));
        }

        let key_version =
            u32::from_le_bytes([ciphertext[1], ciphertext[2], ciphertext[3], ciphertext[4]]);

        let versioned_key = self
            .key_ring
            .get_key(key_version)
            .ok_or(CryptoError::KeyNotFound(key_version))?;

        let cipher = Aes256GcmCipher::new(&versioned_key.key)?;
        let cipher_data = &ciphertext[ENVELOPE_HEADER_SIZE..];

        cipher.decrypt(cipher_data, associated_data)
    }

    /// Re-encrypt data with the current active key.
    ///
    /// Useful during key rotation to migrate old data to new keys.
    #[cfg(feature = "aes")]
    pub fn re_encrypt(
        &self,
        ciphertext: &[u8],
        associated_data: Option<&[u8]>,
    ) -> CryptoResult<Vec<u8>> {
        let plaintext = self.decrypt(ciphertext, associated_data)?;
        self.encrypt(&plaintext, associated_data)
    }

    /// Check if ciphertext uses the active key.
    pub fn uses_active_key(&self, ciphertext: &[u8]) -> CryptoResult<bool> {
        if ciphertext.len() < ENVELOPE_HEADER_SIZE {
            return Err(CryptoError::InvalidCiphertext(
                "envelope too short".to_string(),
            ));
        }

        let key_version =
            u32::from_le_bytes([ciphertext[1], ciphertext[2], ciphertext[3], ciphertext[4]]);

        Ok(self
            .key_ring
            .active_key()
            .map(|k| k.version == key_version)
            .unwrap_or(false))
    }

    /// Get the key version from ciphertext.
    pub fn get_key_version(&self, ciphertext: &[u8]) -> CryptoResult<u32> {
        if ciphertext.len() < ENVELOPE_HEADER_SIZE {
            return Err(CryptoError::InvalidCiphertext(
                "envelope too short".to_string(),
            ));
        }

        Ok(u32::from_le_bytes([
            ciphertext[1],
            ciphertext[2],
            ciphertext[3],
            ciphertext[4],
        ]))
    }
}

#[cfg(feature = "aes")]
impl DataEncryptor for EnvelopeEncryptor {
    fn encrypt_data<T: Serialize>(&self, data: &T) -> CryptoResult<Vec<u8>> {
        let json = serde_json::to_vec(data)?;
        self.encrypt(&json, None)
    }

    fn decrypt_data<T: DeserializeOwned>(&self, ciphertext: &[u8]) -> CryptoResult<T> {
        let plaintext = self.decrypt(ciphertext, None)?;
        let data = serde_json::from_slice(&plaintext)?;
        Ok(data)
    }
}

#[cfg(all(test, feature = "aes"))]
mod tests {
    use super::*;
    use crate::aes_cipher::Aes256GcmCipher;

    #[test]
    fn test_envelope_encrypt_decrypt() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let encryptor = EnvelopeEncryptor::new(key);

        let plaintext = b"Secret message";
        let ciphertext = encryptor.encrypt(plaintext, None).unwrap();
        let decrypted = encryptor.decrypt(&ciphertext, None).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_envelope_with_aad() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let encryptor = EnvelopeEncryptor::new(key);

        let plaintext = b"Secret message";
        let aad = b"context-data";

        let ciphertext = encryptor.encrypt(plaintext, Some(aad)).unwrap();
        let decrypted = encryptor.decrypt(&ciphertext, Some(aad)).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_key_rotation() {
        let key1 = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let mut encryptor = EnvelopeEncryptor::new(key1);

        // Encrypt with key v1
        let plaintext = b"Secret message";
        let ciphertext_v1 = encryptor.encrypt(plaintext, None).unwrap();

        assert_eq!(encryptor.get_key_version(&ciphertext_v1).unwrap(), 1);
        assert!(encryptor.uses_active_key(&ciphertext_v1).unwrap());

        // Rotate to key v2
        let key2 = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let v2 = encryptor.rotate_key(key2);
        assert_eq!(v2, 2);

        // Encrypt with key v2
        let ciphertext_v2 = encryptor.encrypt(plaintext, None).unwrap();

        assert_eq!(encryptor.get_key_version(&ciphertext_v2).unwrap(), 2);
        assert!(encryptor.uses_active_key(&ciphertext_v2).unwrap());
        assert!(!encryptor.uses_active_key(&ciphertext_v1).unwrap());

        // Both ciphertexts should still decrypt
        let decrypted_v1 = encryptor.decrypt(&ciphertext_v1, None).unwrap();
        let decrypted_v2 = encryptor.decrypt(&ciphertext_v2, None).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted_v1.as_slice());
        assert_eq!(plaintext.as_slice(), decrypted_v2.as_slice());
    }

    #[test]
    fn test_re_encrypt() {
        let key1 = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let mut encryptor = EnvelopeEncryptor::new(key1);

        let plaintext = b"Secret message";
        let ciphertext_v1 = encryptor.encrypt(plaintext, None).unwrap();

        // Rotate to v2
        let key2 = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        encryptor.rotate_key(key2);

        // Re-encrypt with new key
        let ciphertext_v2 = encryptor.re_encrypt(&ciphertext_v1, None).unwrap();

        assert_eq!(encryptor.get_key_version(&ciphertext_v2).unwrap(), 2);

        let decrypted = encryptor.decrypt(&ciphertext_v2, None).unwrap();
        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_data_encryptor_json() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestData {
            name: String,
            value: i32,
        }

        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let encryptor = EnvelopeEncryptor::new(key);

        let data = TestData {
            name: "test".to_string(),
            value: 42,
        };

        let ciphertext = encryptor.encrypt_data(&data).unwrap();
        let decrypted: TestData = encryptor.decrypt_data(&ciphertext).unwrap();

        assert_eq!(data, decrypted);
    }

    #[test]
    fn test_envelope_header_format() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let encryptor = EnvelopeEncryptor::new(key);

        let plaintext = b"Test";
        let ciphertext = encryptor.encrypt(plaintext, None).unwrap();

        // Check envelope version
        assert_eq!(ciphertext[0], ENVELOPE_VERSION);

        // Check key version (should be 1)
        let key_version =
            u32::from_le_bytes([ciphertext[1], ciphertext[2], ciphertext[3], ciphertext[4]]);
        assert_eq!(key_version, 1);
    }

    #[test]
    fn test_missing_key_version() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let encryptor = EnvelopeEncryptor::new(key);

        let plaintext = b"Test";
        let mut ciphertext = encryptor.encrypt(plaintext, None).unwrap();

        // Change key version to non-existent
        ciphertext[1] = 99;
        ciphertext[2] = 0;
        ciphertext[3] = 0;
        ciphertext[4] = 0;

        let result = encryptor.decrypt(&ciphertext, None);
        assert!(matches!(result, Err(CryptoError::KeyNotFound(99))));
    }
}
