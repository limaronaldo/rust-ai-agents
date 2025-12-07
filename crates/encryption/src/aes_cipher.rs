//! AES-256-GCM authenticated encryption implementation.

#[cfg(feature = "aes")]
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use rand::RngCore;

use crate::error::{CryptoError, CryptoResult};
use crate::key::EncryptionKey;
use crate::traits::Cipher;

/// AES-256-GCM cipher implementation.
///
/// Provides authenticated encryption with associated data (AEAD).
/// - Key size: 256 bits (32 bytes)
/// - Nonce size: 96 bits (12 bytes)
/// - Tag size: 128 bits (16 bytes)
///
/// # Ciphertext Format
///
/// ```text
/// [nonce: 12 bytes][ciphertext + tag: variable]
/// ```
#[cfg(feature = "aes")]
pub struct Aes256GcmCipher {
    cipher: Aes256Gcm,
}

#[cfg(feature = "aes")]
impl Aes256GcmCipher {
    /// Key size in bytes (256 bits).
    pub const KEY_SIZE: usize = 32;
    /// Nonce size in bytes (96 bits).
    pub const NONCE_SIZE: usize = 12;
    /// Authentication tag size in bytes (128 bits).
    pub const TAG_SIZE: usize = 16;

    /// Create a new AES-256-GCM cipher with the given key.
    pub fn new(key: &EncryptionKey) -> CryptoResult<Self> {
        if key.len() != Self::KEY_SIZE {
            return Err(CryptoError::InvalidKeyLength {
                expected: Self::KEY_SIZE,
                got: key.len(),
            });
        }

        let cipher = Aes256Gcm::new_from_slice(key.as_bytes())
            .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?;

        Ok(Self { cipher })
    }

    /// Generate a random nonce.
    fn generate_nonce() -> [u8; Self::NONCE_SIZE] {
        let mut nonce = [0u8; Self::NONCE_SIZE];
        rand::thread_rng().fill_bytes(&mut nonce);
        nonce
    }
}

#[cfg(feature = "aes")]
impl Cipher for Aes256GcmCipher {
    fn encrypt(&self, plaintext: &[u8], associated_data: Option<&[u8]>) -> CryptoResult<Vec<u8>> {
        let nonce_bytes = Self::generate_nonce();
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = if let Some(aad) = associated_data {
            use aes_gcm::aead::Payload;
            self.cipher
                .encrypt(
                    nonce,
                    Payload {
                        msg: plaintext,
                        aad,
                    },
                )
                .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?
        } else {
            self.cipher
                .encrypt(nonce, plaintext)
                .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?
        };

        // Prepend nonce to ciphertext
        let mut result = Vec::with_capacity(Self::NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    fn decrypt(&self, ciphertext: &[u8], associated_data: Option<&[u8]>) -> CryptoResult<Vec<u8>> {
        if ciphertext.len() < Self::NONCE_SIZE + Self::TAG_SIZE {
            return Err(CryptoError::InvalidCiphertext(
                "ciphertext too short".to_string(),
            ));
        }

        let (nonce_bytes, encrypted) = ciphertext.split_at(Self::NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);

        let plaintext = if let Some(aad) = associated_data {
            use aes_gcm::aead::Payload;
            self.cipher
                .decrypt(
                    nonce,
                    Payload {
                        msg: encrypted,
                        aad,
                    },
                )
                .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?
        } else {
            self.cipher
                .decrypt(nonce, encrypted)
                .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?
        };

        Ok(plaintext)
    }

    fn algorithm(&self) -> &'static str {
        "AES-256-GCM"
    }

    fn key_size(&self) -> usize {
        Self::KEY_SIZE
    }

    fn nonce_size(&self) -> usize {
        Self::NONCE_SIZE
    }

    fn tag_size(&self) -> usize {
        Self::TAG_SIZE
    }
}

#[cfg(all(test, feature = "aes"))]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let cipher = Aes256GcmCipher::new(&key).unwrap();

        let plaintext = b"Hello, World! This is a secret message.";
        let ciphertext = cipher.encrypt(plaintext, None).unwrap();
        let decrypted = cipher.decrypt(&ciphertext, None).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_encrypt_decrypt_with_aad() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let cipher = Aes256GcmCipher::new(&key).unwrap();

        let plaintext = b"Secret data";
        let aad = b"session-id-12345";

        let ciphertext = cipher.encrypt(plaintext, Some(aad)).unwrap();
        let decrypted = cipher.decrypt(&ciphertext, Some(aad)).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_aad_mismatch_fails() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let cipher = Aes256GcmCipher::new(&key).unwrap();

        let plaintext = b"Secret data";
        let aad1 = b"session-id-12345";
        let aad2 = b"session-id-67890";

        let ciphertext = cipher.encrypt(plaintext, Some(aad1)).unwrap();
        let result = cipher.decrypt(&ciphertext, Some(aad2));

        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_ciphertext_fails() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let cipher = Aes256GcmCipher::new(&key).unwrap();

        let plaintext = b"Secret data";
        let mut ciphertext = cipher.encrypt(plaintext, None).unwrap();

        // Tamper with the ciphertext
        if let Some(byte) = ciphertext.last_mut() {
            *byte ^= 0xFF;
        }

        let result = cipher.decrypt(&ciphertext, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_different_keys_fail() {
        let key1 = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let key2 = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);

        let cipher1 = Aes256GcmCipher::new(&key1).unwrap();
        let cipher2 = Aes256GcmCipher::new(&key2).unwrap();

        let plaintext = b"Secret data";
        let ciphertext = cipher1.encrypt(plaintext, None).unwrap();

        let result = cipher2.decrypt(&ciphertext, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_key_length() {
        let key = EncryptionKey::generate(16); // Too short
        let result = Aes256GcmCipher::new(&key);

        assert!(matches!(
            result,
            Err(CryptoError::InvalidKeyLength {
                expected: 32,
                got: 16
            })
        ));
    }

    #[test]
    fn test_ciphertext_format() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let cipher = Aes256GcmCipher::new(&key).unwrap();

        let plaintext = b"Test";
        let ciphertext = cipher.encrypt(plaintext, None).unwrap();

        // Ciphertext should be: nonce (12) + plaintext (4) + tag (16) = 32 bytes
        assert_eq!(
            ciphertext.len(),
            Aes256GcmCipher::NONCE_SIZE + plaintext.len() + Aes256GcmCipher::TAG_SIZE
        );
    }

    #[test]
    fn test_empty_plaintext() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let cipher = Aes256GcmCipher::new(&key).unwrap();

        let plaintext = b"";
        let ciphertext = cipher.encrypt(plaintext, None).unwrap();
        let decrypted = cipher.decrypt(&ciphertext, None).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_large_plaintext() {
        let key = EncryptionKey::generate(Aes256GcmCipher::KEY_SIZE);
        let cipher = Aes256GcmCipher::new(&key).unwrap();

        let plaintext = vec![0xABu8; 1024 * 1024]; // 1 MB
        let ciphertext = cipher.encrypt(&plaintext, None).unwrap();
        let decrypted = cipher.decrypt(&ciphertext, None).unwrap();

        assert_eq!(plaintext, decrypted);
    }
}
