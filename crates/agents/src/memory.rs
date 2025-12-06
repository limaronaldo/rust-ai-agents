//! Agent memory management

use rust_ai_agents_core::{errors::MemoryError, MemoryConfig, Message, RetentionPolicy};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Agent memory
pub struct AgentMemory {
    config: MemoryConfig,
    messages: Arc<RwLock<Vec<Message>>>,
    current_size: Arc<RwLock<usize>>,
}

impl AgentMemory {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            messages: Arc::new(RwLock::new(Vec::new())),
            current_size: Arc::new(RwLock::new(0)),
        }
    }

    /// Add a message to memory
    pub async fn add_message(&self, message: Message) -> Result<(), MemoryError> {
        let message_size = self.estimate_message_size(&message);

        // Check if we need to free space
        {
            let current_size = self.current_size.write().await;
            if *current_size + message_size > self.config.max_size {
                drop(current_size); // Release lock before calling cleanup
                self.cleanup().await?;
            }
        }

        // Add message
        {
            let mut messages = self.messages.write().await;
            messages.push(message);

            let mut current_size = self.current_size.write().await;
            *current_size += message_size;
        }

        Ok(())
    }

    /// Get message history
    pub async fn get_history(&self) -> Result<Vec<Message>, MemoryError> {
        let messages = self.messages.read().await;
        Ok(messages.clone())
    }

    /// Get last N messages
    pub async fn get_last_n(&self, n: usize) -> Result<Vec<Message>, MemoryError> {
        let messages = self.messages.read().await;
        let start = messages.len().saturating_sub(n);
        Ok(messages[start..].to_vec())
    }

    /// Clear memory
    pub async fn clear(&self) -> Result<(), MemoryError> {
        let mut messages = self.messages.write().await;
        messages.clear();

        let mut current_size = self.current_size.write().await;
        *current_size = 0;

        Ok(())
    }

    /// Get memory size
    pub async fn size(&self) -> usize {
        *self.current_size.read().await
    }

    /// Get message count
    pub async fn count(&self) -> usize {
        self.messages.read().await.len()
    }

    /// Cleanup old messages based on retention policy
    async fn cleanup(&self) -> Result<(), MemoryError> {
        let mut messages = self.messages.write().await;

        let original_len = messages.len();

        match &self.config.retention_policy {
            RetentionPolicy::KeepAll => {
                // Don't remove anything, but this means we're over limit
                return Err(MemoryError::LimitExceeded(
                    *self.current_size.read().await,
                    self.config.max_size,
                ));
            }
            RetentionPolicy::KeepRecent(n) => {
                if messages.len() > *n {
                    let remove_count = messages.len() - n;
                    messages.drain(0..remove_count);
                }
            }
            RetentionPolicy::KeepImportant(_threshold) => {
                // For now, just keep recent messages
                // TODO: Implement importance scoring
                if messages.len() > 100 {
                    let remove_count = messages.len() - 100;
                    messages.drain(0..remove_count);
                }
            }
            RetentionPolicy::Custom => {
                // Keep last 50% of messages
                if messages.len() > 1 {
                    let half = messages.len() / 2;
                    messages.drain(0..half);
                }
            }
        }

        // Recalculate size
        let new_size: usize = messages.iter().map(|m| self.estimate_message_size(m)).sum();

        let mut current_size = self.current_size.write().await;
        *current_size = new_size;

        tracing::debug!(
            "Memory cleanup: removed {} messages, size: {} -> {} bytes",
            original_len - messages.len(),
            original_len,
            new_size
        );

        Ok(())
    }

    /// Estimate message size in bytes
    fn estimate_message_size(&self, message: &Message) -> usize {
        // Rough estimation: JSON serialization size
        serde_json::to_string(message)
            .map(|s| s.len())
            .unwrap_or(1024) // Default to 1KB if serialization fails
    }
}
