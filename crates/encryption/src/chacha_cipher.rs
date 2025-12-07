//! ChaCha20-Poly1305 authenticated encryption implementation.
//!
//! This module is only available with the `chacha` feature.

#[cfg(feature = "chacha")]
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use rand::RngCore;

use crate::error::{CryptoError, CryptoResult};
use crate::key::EncryptionKey;
use crate::traits::Cipher;

/// ChaCha20-Poly1305 cipher implementation.
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
#[cfg(feature = "chacha")]
pub struct ChaCha20Poly1305Cipher {
    cipher: ChaCha20Poly1305,
}

#[cfg(feature = "chacha")]
impl ChaCha20Poly1305Cipher {
    /// Key size in bytes (256 bits).
    pub const KEY_SIZE: usize = 32;
    /// Nonce size in bytes (96 bits).
    pub const NONCE_SIZE: usize = 12;
    /// Authentication tag size in bytes (128 bits).
    pub const TAG_SIZE: usize = 16;

    /// Create a new ChaCha20-Poly1305 cipher with the given key.
    pub fn new(key: &EncryptionKey) -> CryptoResult<Self> {
        if key.len() != Self::KEY_SIZE {
            return Err(CryptoError::InvalidKeyLength {
                expected: Self::KEY_SIZE,
                got: key.len(),
            });
        }

        let cipher = ChaCha20Poly1305::new_from_slice(key.as_bytes())
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

#[cfg(feature = "chacha")]
impl Cipher for ChaCha20Poly1305Cipher {
    fn encrypt(&self, plaintext: &[u8], associated_data: Option<&[u8]>) -> CryptoResult<Vec<u8>> {
        let nonce_bytes = Self::generate_nonce();
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = if let Some(aad) = associated_data {
            use chacha20poly1305::aead::Payload;
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
            use chacha20poly1305::aead::Payload;
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
        "ChaCha20-Poly1305"
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

#[cfg(all(test, feature = "chacha"))]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = EncryptionKey::generate(ChaCha20Poly1305Cipher::KEY_SIZE);
        let cipher = ChaCha20Poly1305Cipher::new(&key).unwrap();

        let plaintext = b"Hello, World!";
        let ciphertext = cipher.encrypt(plaintext, None).unwrap();
        let decrypted = cipher.decrypt(&ciphertext, None).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_encrypt_decrypt_with_aad() {
        let key = EncryptionKey::generate(ChaCha20Poly1305Cipher::KEY_SIZE);
        let cipher = ChaCha20Poly1305Cipher::new(&key).unwrap();

        let plaintext = b"Secret data";
        let aad = b"context";

        let ciphertext = cipher.encrypt(plaintext, Some(aad)).unwrap();
        let decrypted = cipher.decrypt(&ciphertext, Some(aad)).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }
}
