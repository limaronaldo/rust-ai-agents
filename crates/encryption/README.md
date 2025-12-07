# rust-ai-agents-encryption

At-rest encryption for sensitive data in rust-ai-agents.

## Features

- **AES-256-GCM** - Authenticated encryption (default)
- **ChaCha20-Poly1305** - Alternative cipher (optional)
- **Argon2id** - Password-based key derivation
- **Key Rotation** - Versioned keys with envelope encryption
- **Secure Memory** - Automatic zeroing of sensitive data

## Quick Start

```rust
use rust_ai_agents_encryption::{EncryptionKey, EnvelopeEncryptor, DataEncryptor};

// Generate a random 256-bit key
let key = EncryptionKey::generate(32);

// Create an encryptor
let encryptor = EnvelopeEncryptor::new(key);

// Encrypt structured data
let secret = serde_json::json!({"user": "alice", "token": "secret123"});
let ciphertext = encryptor.encrypt_data(&secret)?;

// Decrypt
let decrypted: serde_json::Value = encryptor.decrypt_data(&ciphertext)?;
```

## Key Derivation

For password-based encryption:

```rust
use rust_ai_agents_encryption::{Argon2KeyDerivation, KeyDerivation, EnvelopeEncryptor};

let kdf = Argon2KeyDerivation::new();
let salt = kdf.generate_salt(16);
let key = kdf.derive_encryption_key(b"user-password", &salt, 32)?;

let encryptor = EnvelopeEncryptor::new(key);
```

## Key Rotation

```rust
use rust_ai_agents_encryption::{EncryptionKey, EnvelopeEncryptor};

let key1 = EncryptionKey::generate(32);
let mut encryptor = EnvelopeEncryptor::new(key1);

// Encrypt with v1
let ciphertext = encryptor.encrypt(b"secret", None)?;

// Rotate to v2
let key2 = EncryptionKey::generate(32);
encryptor.rotate_key(key2);

// Old ciphertext still decrypts (key v1 retained)
let plaintext = encryptor.decrypt(&ciphertext, None)?;

// Re-encrypt with new key for migration
let new_ciphertext = encryptor.re_encrypt(&ciphertext, None)?;
```

## Store Wrappers

Transparent encryption for session and checkpoint stores:

```rust
use rust_ai_agents_encryption::{
    EncryptionKey, EnvelopeEncryptor, 
    EncryptedSessionStore, EncryptedCheckpointStore
};
use std::sync::Arc;

let key = EncryptionKey::generate(32);
let encryptor = Arc::new(EnvelopeEncryptor::new(key));

// Wrap existing stores
let encrypted_sessions = EncryptedSessionStore::new(session_store, encryptor.clone());
let encrypted_checkpoints = EncryptedCheckpointStore::new(checkpoint_store, encryptor);
```

## Ciphertext Format

### Envelope Format
```
[envelope_version: 1 byte][key_version: 4 bytes][cipher_data]
```

### AES-GCM Cipher Data
```
[nonce: 12 bytes][ciphertext + tag]
```

## Security Considerations

- **Key Storage**: Store master keys securely (environment variables, HSM, or vault)
- **Key Rotation**: Rotate keys periodically and after suspected compromise
- **Salt Uniqueness**: Always use unique salts for key derivation
- **Memory Safety**: Sensitive data is automatically zeroed when dropped

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `aes` | AES-256-GCM cipher | ✓ |
| `chacha` | ChaCha20-Poly1305 cipher | |
| `full` | All ciphers | |

## License

Apache-2.0 OR MIT
