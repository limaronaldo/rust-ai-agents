//! KV Store integration for conversation persistence

use crate::error::{CloudflareError, Result};
use rust_ai_agents_llm_client::Message;
use serde::{Deserialize, Serialize};
use worker::kv::KvStore as WorkerKvStore;

/// Wrapper around Cloudflare KV for conversation storage
pub struct KvStore {
    kv: WorkerKvStore,
    /// TTL for conversation entries in seconds (default: 24 hours)
    ttl: u64,
}

impl KvStore {
    /// Create a new KV store wrapper
    pub fn new(kv: WorkerKvStore) -> Self {
        Self {
            kv,
            ttl: 86400, // 24 hours default
        }
    }

    /// Set custom TTL for conversation entries
    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.ttl = ttl_seconds;
        self
    }

    /// Get conversation history for a session
    pub async fn get_conversation(&self, session_id: &str) -> Result<Vec<Message>> {
        let key = format!("conversation:{}", session_id);

        match self.kv.get(&key).text().await {
            Ok(Some(data)) => {
                let conversation: Conversation = serde_json::from_str(&data)
                    .map_err(|e| CloudflareError::KvError(e.to_string()))?;
                Ok(conversation.messages)
            }
            Ok(None) => Ok(Vec::new()),
            Err(e) => Err(CloudflareError::KvError(e.to_string())),
        }
    }

    /// Save conversation history for a session
    pub async fn save_conversation(&self, session_id: &str, messages: &[Message]) -> Result<()> {
        let key = format!("conversation:{}", session_id);
        let conversation = Conversation {
            messages: messages.to_vec(),
            updated_at: chrono_now(),
        };

        let data = serde_json::to_string(&conversation)
            .map_err(|e| CloudflareError::KvError(e.to_string()))?;

        self.kv
            .put(&key, data)
            .map_err(|e| CloudflareError::KvError(e.to_string()))?
            .expiration_ttl(self.ttl)
            .execute()
            .await
            .map_err(|e| CloudflareError::KvError(e.to_string()))?;

        Ok(())
    }

    /// Append a message to conversation history
    pub async fn append_message(&self, session_id: &str, message: Message) -> Result<()> {
        let mut messages = self.get_conversation(session_id).await?;
        messages.push(message);
        self.save_conversation(session_id, &messages).await
    }

    /// Clear conversation history for a session
    pub async fn clear_conversation(&self, session_id: &str) -> Result<()> {
        let key = format!("conversation:{}", session_id);
        self.kv
            .delete(&key)
            .await
            .map_err(|e| CloudflareError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Get session metadata
    pub async fn get_metadata(&self, session_id: &str) -> Result<Option<SessionMetadata>> {
        let key = format!("metadata:{}", session_id);

        match self.kv.get(&key).text().await {
            Ok(Some(data)) => {
                let metadata: SessionMetadata = serde_json::from_str(&data)
                    .map_err(|e| CloudflareError::KvError(e.to_string()))?;
                Ok(Some(metadata))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(CloudflareError::KvError(e.to_string())),
        }
    }

    /// Save session metadata
    pub async fn save_metadata(&self, session_id: &str, metadata: &SessionMetadata) -> Result<()> {
        let key = format!("metadata:{}", session_id);
        let data =
            serde_json::to_string(metadata).map_err(|e| CloudflareError::KvError(e.to_string()))?;

        self.kv
            .put(&key, data)
            .map_err(|e| CloudflareError::KvError(e.to_string()))?
            .expiration_ttl(self.ttl)
            .execute()
            .await
            .map_err(|e| CloudflareError::KvError(e.to_string()))?;

        Ok(())
    }
}

/// Stored conversation structure
#[derive(Debug, Serialize, Deserialize)]
struct Conversation {
    messages: Vec<Message>,
    updated_at: String,
}

/// Session metadata for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session identifier
    pub session_id: String,
    /// Total tokens used in session
    pub total_tokens: u32,
    /// Number of messages exchanged
    pub message_count: u32,
    /// When the session was created
    pub created_at: String,
    /// When the session was last updated
    pub updated_at: String,
    /// Custom metadata
    pub custom: Option<serde_json::Value>,
}

impl SessionMetadata {
    /// Create new session metadata
    pub fn new(session_id: impl Into<String>) -> Self {
        let now = chrono_now();
        Self {
            session_id: session_id.into(),
            total_tokens: 0,
            message_count: 0,
            created_at: now.clone(),
            updated_at: now,
            custom: None,
        }
    }

    /// Update token count
    pub fn add_tokens(&mut self, tokens: u32) {
        self.total_tokens += tokens;
        self.updated_at = chrono_now();
    }

    /// Increment message count
    pub fn increment_messages(&mut self) {
        self.message_count += 1;
        self.updated_at = chrono_now();
    }
}

/// Get current timestamp as ISO string (WASM-compatible)
fn chrono_now() -> String {
    // In WASM, we use js_sys for time
    let now = js_sys::Date::new_0();
    now.to_iso_string().as_string().unwrap_or_default()
}
