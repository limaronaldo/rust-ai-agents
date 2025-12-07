//! # Checkpointing
//!
//! State persistence system for agent execution with recovery and time-travel capabilities.
//!
//! Inspired by LangGraph's checkpointing pattern.
//!
//! ## Features
//!
//! - **State Snapshots**: Capture agent state at any point
//! - **Recovery**: Resume execution from any checkpoint
//! - **Time-Travel**: Navigate between checkpoints
//! - **Branching**: Create alternate execution paths
//! - **Multiple Backends**: Memory, SQLite, or custom storage
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents::checkpoint::{CheckpointManager, MemoryCheckpointStore};
//!
//! let store = MemoryCheckpointStore::new();
//! let manager = CheckpointManager::new(store);
//!
//! // Save a checkpoint
//! let checkpoint_id = manager.save("thread_1", &state).await?;
//!
//! // Later, restore from checkpoint
//! let state = manager.load("thread_1", &checkpoint_id).await?;
//!
//! // Time-travel: list all checkpoints
//! let history = manager.list("thread_1").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tracing::{debug, info};

/// Unique identifier for a checkpoint
pub type CheckpointId = String;

/// Unique identifier for a thread (conversation/session)
pub type ThreadId = String;

/// Metadata associated with a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Unique checkpoint ID
    pub id: CheckpointId,
    /// Thread this checkpoint belongs to
    pub thread_id: ThreadId,
    /// When the checkpoint was created
    pub created_at: u64,
    /// Optional parent checkpoint (for branching)
    pub parent_id: Option<CheckpointId>,
    /// Step number in the execution
    pub step: u64,
    /// Optional human-readable label
    pub label: Option<String>,
    /// Custom tags for filtering
    pub tags: Vec<String>,
    /// Size of the state in bytes
    pub state_size: usize,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl CheckpointMetadata {
    pub fn new(thread_id: impl Into<String>, step: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            thread_id: thread_id.into(),
            created_at: now,
            parent_id: None,
            step,
            label: None,
            tags: Vec::new(),
            state_size: 0,
            custom: HashMap::new(),
        }
    }

    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }
}

/// A complete checkpoint including state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint<S> {
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,
    /// The actual state
    pub state: S,
}

impl<S> Checkpoint<S> {
    pub fn new(thread_id: impl Into<String>, step: u64, state: S) -> Self
    where
        S: Serialize,
    {
        let state_size = serde_json::to_vec(&state).map(|v| v.len()).unwrap_or(0);
        Self {
            metadata: CheckpointMetadata {
                state_size,
                ..CheckpointMetadata::new(thread_id, step)
            },
            state,
        }
    }

    pub fn id(&self) -> &str {
        &self.metadata.id
    }

    pub fn thread_id(&self) -> &str {
        &self.metadata.thread_id
    }

    pub fn step(&self) -> u64 {
        self.metadata.step
    }
}

/// Error type for checkpoint operations
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("Checkpoint not found: {0}")]
    NotFound(CheckpointId),

    #[error("Thread not found: {0}")]
    ThreadNotFound(ThreadId),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Trait for checkpoint storage backends
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Save a checkpoint, returns the checkpoint ID
    async fn save(
        &self,
        thread_id: &str,
        metadata: CheckpointMetadata,
        state: Vec<u8>,
    ) -> Result<CheckpointId, CheckpointError>;

    /// Load a specific checkpoint
    async fn load(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<(CheckpointMetadata, Vec<u8>), CheckpointError>;

    /// Load the latest checkpoint for a thread
    async fn load_latest(
        &self,
        thread_id: &str,
    ) -> Result<(CheckpointMetadata, Vec<u8>), CheckpointError>;

    /// List all checkpoints for a thread
    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>, CheckpointError>;

    /// Delete a specific checkpoint
    async fn delete(&self, thread_id: &str, checkpoint_id: &str) -> Result<(), CheckpointError>;

    /// Delete all checkpoints for a thread
    async fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError>;

    /// Get checkpoint count for a thread
    async fn count(&self, thread_id: &str) -> Result<usize, CheckpointError>;

    /// List all threads with checkpoints
    async fn list_threads(&self) -> Result<Vec<ThreadId>, CheckpointError>;
}

/// In-memory checkpoint store (for testing and development)
#[derive(Default)]
pub struct MemoryCheckpointStore {
    checkpoints: RwLock<HashMap<ThreadId, Vec<(CheckpointMetadata, Vec<u8>)>>>,
}

impl MemoryCheckpointStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl CheckpointStore for MemoryCheckpointStore {
    async fn save(
        &self,
        thread_id: &str,
        metadata: CheckpointMetadata,
        state: Vec<u8>,
    ) -> Result<CheckpointId, CheckpointError> {
        let id = metadata.id.clone();
        let mut checkpoints = self.checkpoints.write();
        checkpoints
            .entry(thread_id.to_string())
            .or_default()
            .push((metadata, state));
        Ok(id)
    }

    async fn load(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<(CheckpointMetadata, Vec<u8>), CheckpointError> {
        let checkpoints = self.checkpoints.read();
        let thread_checkpoints = checkpoints
            .get(thread_id)
            .ok_or_else(|| CheckpointError::ThreadNotFound(thread_id.to_string()))?;

        thread_checkpoints
            .iter()
            .find(|(m, _)| m.id == checkpoint_id)
            .cloned()
            .ok_or_else(|| CheckpointError::NotFound(checkpoint_id.to_string()))
    }

    async fn load_latest(
        &self,
        thread_id: &str,
    ) -> Result<(CheckpointMetadata, Vec<u8>), CheckpointError> {
        let checkpoints = self.checkpoints.read();
        let thread_checkpoints = checkpoints
            .get(thread_id)
            .ok_or_else(|| CheckpointError::ThreadNotFound(thread_id.to_string()))?;

        thread_checkpoints
            .last()
            .cloned()
            .ok_or_else(|| CheckpointError::ThreadNotFound(thread_id.to_string()))
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>, CheckpointError> {
        let checkpoints = self.checkpoints.read();
        Ok(checkpoints
            .get(thread_id)
            .map(|v| v.iter().map(|(m, _)| m.clone()).collect())
            .unwrap_or_default())
    }

    async fn delete(&self, thread_id: &str, checkpoint_id: &str) -> Result<(), CheckpointError> {
        let mut checkpoints = self.checkpoints.write();
        if let Some(thread_checkpoints) = checkpoints.get_mut(thread_id) {
            thread_checkpoints.retain(|(m, _)| m.id != checkpoint_id);
        }
        Ok(())
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError> {
        let mut checkpoints = self.checkpoints.write();
        checkpoints.remove(thread_id);
        Ok(())
    }

    async fn count(&self, thread_id: &str) -> Result<usize, CheckpointError> {
        let checkpoints = self.checkpoints.read();
        Ok(checkpoints.get(thread_id).map(|v| v.len()).unwrap_or(0))
    }

    async fn list_threads(&self) -> Result<Vec<ThreadId>, CheckpointError> {
        let checkpoints = self.checkpoints.read();
        Ok(checkpoints.keys().cloned().collect())
    }
}

/// Configuration for checkpoint manager
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Maximum number of checkpoints to keep per thread
    pub max_checkpoints_per_thread: Option<usize>,
    /// Auto-checkpoint every N steps
    pub auto_checkpoint_interval: Option<u64>,
    /// Whether to compress state before storing
    pub compress: bool,
    /// TTL for checkpoints (auto-delete after this duration)
    pub ttl: Option<Duration>,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            max_checkpoints_per_thread: Some(100),
            auto_checkpoint_interval: None,
            compress: false,
            ttl: None,
        }
    }
}

/// Main checkpoint manager
pub struct CheckpointManager<Store: CheckpointStore> {
    store: Arc<Store>,
    config: CheckpointConfig,
    /// Current step per thread
    steps: RwLock<HashMap<ThreadId, u64>>,
}

impl<Store: CheckpointStore> CheckpointManager<Store> {
    pub fn new(store: Store) -> Self {
        Self {
            store: Arc::new(store),
            config: CheckpointConfig::default(),
            steps: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_config(mut self, config: CheckpointConfig) -> Self {
        self.config = config;
        self
    }

    /// Save current state as a checkpoint
    pub async fn save<S: Serialize + Send>(
        &self,
        thread_id: &str,
        state: &S,
    ) -> Result<CheckpointId, CheckpointError> {
        self.save_with_label(thread_id, state, None).await
    }

    /// Save current state with an optional label
    pub async fn save_with_label<S: Serialize + Send>(
        &self,
        thread_id: &str,
        state: &S,
        label: Option<String>,
    ) -> Result<CheckpointId, CheckpointError> {
        // Serialize state
        let state_bytes =
            serde_json::to_vec(state).map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        // Get and increment step
        let step = {
            let mut steps = self.steps.write();
            let step = steps.entry(thread_id.to_string()).or_insert(0);
            *step += 1;
            *step
        };

        // Get parent ID (previous checkpoint)
        let parent_id = self
            .store
            .load_latest(thread_id)
            .await
            .ok()
            .map(|(m, _)| m.id);

        // Create metadata
        let mut metadata = CheckpointMetadata::new(thread_id, step);
        metadata.state_size = state_bytes.len();
        if let Some(parent) = parent_id {
            metadata = metadata.with_parent(parent);
        }
        if let Some(lbl) = label {
            metadata = metadata.with_label(lbl);
        }

        // Save checkpoint
        let id = self.store.save(thread_id, metadata, state_bytes).await?;

        // Cleanup old checkpoints if needed
        if let Some(max) = self.config.max_checkpoints_per_thread {
            self.cleanup_old_checkpoints(thread_id, max).await?;
        }

        debug!(thread_id, checkpoint_id = %id, step, "Checkpoint saved");
        Ok(id)
    }

    /// Load a specific checkpoint
    pub async fn load<S: DeserializeOwned>(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<Checkpoint<S>, CheckpointError> {
        let (metadata, state_bytes) = self.store.load(thread_id, checkpoint_id).await?;
        let state = serde_json::from_slice(&state_bytes)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        debug!(
            thread_id,
            checkpoint_id,
            step = metadata.step,
            "Checkpoint loaded"
        );
        Ok(Checkpoint { metadata, state })
    }

    /// Load the latest checkpoint for a thread
    pub async fn load_latest<S: DeserializeOwned>(
        &self,
        thread_id: &str,
    ) -> Result<Checkpoint<S>, CheckpointError> {
        let (metadata, state_bytes) = self.store.load_latest(thread_id).await?;
        let state = serde_json::from_slice(&state_bytes)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        debug!(thread_id, checkpoint_id = %metadata.id, step = metadata.step, "Latest checkpoint loaded");
        Ok(Checkpoint { metadata, state })
    }

    /// Get checkpoint history for a thread
    pub async fn history(
        &self,
        thread_id: &str,
    ) -> Result<Vec<CheckpointMetadata>, CheckpointError> {
        self.store.list(thread_id).await
    }

    /// Fork from a checkpoint (create a new thread starting from this point)
    pub async fn fork<S: Serialize + DeserializeOwned + Send>(
        &self,
        source_thread_id: &str,
        checkpoint_id: &str,
        new_thread_id: &str,
    ) -> Result<CheckpointId, CheckpointError> {
        // Load the source checkpoint
        let checkpoint: Checkpoint<S> = self.load(source_thread_id, checkpoint_id).await?;

        // Save as first checkpoint in new thread
        let id = self.save(new_thread_id, &checkpoint.state).await?;

        info!(
            source_thread = source_thread_id,
            source_checkpoint = checkpoint_id,
            new_thread = new_thread_id,
            new_checkpoint = %id,
            "Thread forked"
        );

        Ok(id)
    }

    /// Rewind to a previous checkpoint (delete all checkpoints after it)
    pub async fn rewind(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<(), CheckpointError> {
        let history = self.store.list(thread_id).await?;

        // Find the target checkpoint
        let target_idx = history
            .iter()
            .position(|m| m.id == checkpoint_id)
            .ok_or_else(|| CheckpointError::NotFound(checkpoint_id.to_string()))?;

        // Delete all checkpoints after the target
        for checkpoint in history.iter().skip(target_idx + 1) {
            self.store.delete(thread_id, &checkpoint.id).await?;
        }

        // Update step counter
        {
            let mut steps = self.steps.write();
            if let Some(target) = history.get(target_idx) {
                steps.insert(thread_id.to_string(), target.step);
            }
        }

        info!(thread_id, checkpoint_id, "Rewound to checkpoint");
        Ok(())
    }

    /// Get checkpoints by tag
    pub async fn find_by_tag(
        &self,
        thread_id: &str,
        tag: &str,
    ) -> Result<Vec<CheckpointMetadata>, CheckpointError> {
        let history = self.store.list(thread_id).await?;
        Ok(history
            .into_iter()
            .filter(|m| m.tags.contains(&tag.to_string()))
            .collect())
    }

    /// Get checkpoints by label
    pub async fn find_by_label(
        &self,
        thread_id: &str,
        label: &str,
    ) -> Result<Option<CheckpointMetadata>, CheckpointError> {
        let history = self.store.list(thread_id).await?;
        Ok(history
            .into_iter()
            .find(|m| m.label.as_deref() == Some(label)))
    }

    /// Delete a thread and all its checkpoints
    pub async fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError> {
        self.store.delete_thread(thread_id).await?;
        self.steps.write().remove(thread_id);
        info!(thread_id, "Thread deleted");
        Ok(())
    }

    /// Get current step for a thread
    pub fn current_step(&self, thread_id: &str) -> u64 {
        self.steps.read().get(thread_id).copied().unwrap_or(0)
    }

    /// List all threads
    pub async fn list_threads(&self) -> Result<Vec<ThreadId>, CheckpointError> {
        self.store.list_threads().await
    }

    async fn cleanup_old_checkpoints(
        &self,
        thread_id: &str,
        max: usize,
    ) -> Result<(), CheckpointError> {
        let count = self.store.count(thread_id).await?;
        if count > max {
            let history = self.store.list(thread_id).await?;
            let to_delete = count - max;

            for checkpoint in history.iter().take(to_delete) {
                self.store.delete(thread_id, &checkpoint.id).await?;
                debug!(thread_id, checkpoint_id = %checkpoint.id, "Old checkpoint deleted");
            }
        }
        Ok(())
    }
}

/// Convenience type for the default memory-based checkpoint manager
pub type MemoryCheckpointManager = CheckpointManager<MemoryCheckpointStore>;

impl MemoryCheckpointManager {
    pub fn in_memory() -> Self {
        Self::new(MemoryCheckpointStore::new())
    }
}

/// Builder for creating checkpoints with fluent API
pub struct CheckpointBuilder<'a, S: Serialize + Send, Store: CheckpointStore> {
    manager: &'a CheckpointManager<Store>,
    thread_id: String,
    state: &'a S,
    label: Option<String>,
    tags: Vec<String>,
    custom: HashMap<String, String>,
}

impl<'a, S: Serialize + Send, Store: CheckpointStore> CheckpointBuilder<'a, S, Store> {
    pub fn new(
        manager: &'a CheckpointManager<Store>,
        thread_id: impl Into<String>,
        state: &'a S,
    ) -> Self {
        Self {
            manager,
            thread_id: thread_id.into(),
            state,
            label: None,
            tags: Vec::new(),
            custom: HashMap::new(),
        }
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    pub async fn save(self) -> Result<CheckpointId, CheckpointError> {
        self.manager
            .save_with_label(&self.thread_id, self.state, self.label)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestState {
        messages: Vec<String>,
        counter: u32,
    }

    #[tokio::test]
    async fn test_save_and_load_checkpoint() {
        let manager = MemoryCheckpointManager::in_memory();

        let state = TestState {
            messages: vec!["hello".to_string()],
            counter: 1,
        };

        let id = manager.save("thread1", &state).await.unwrap();

        let loaded: Checkpoint<TestState> = manager.load("thread1", &id).await.unwrap();
        assert_eq!(loaded.state, state);
        assert_eq!(loaded.metadata.step, 1);
    }

    #[tokio::test]
    async fn test_load_latest() {
        let manager = MemoryCheckpointManager::in_memory();

        let state1 = TestState {
            messages: vec!["first".to_string()],
            counter: 1,
        };
        let state2 = TestState {
            messages: vec!["second".to_string()],
            counter: 2,
        };

        manager.save("thread1", &state1).await.unwrap();
        manager.save("thread1", &state2).await.unwrap();

        let loaded: Checkpoint<TestState> = manager.load_latest("thread1").await.unwrap();
        assert_eq!(loaded.state, state2);
        assert_eq!(loaded.metadata.step, 2);
    }

    #[tokio::test]
    async fn test_checkpoint_history() {
        let manager = MemoryCheckpointManager::in_memory();

        let state = TestState {
            messages: vec![],
            counter: 0,
        };

        manager.save("thread1", &state).await.unwrap();
        manager.save("thread1", &state).await.unwrap();
        manager.save("thread1", &state).await.unwrap();

        let history = manager.history("thread1").await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].step, 1);
        assert_eq!(history[1].step, 2);
        assert_eq!(history[2].step, 3);
    }

    #[tokio::test]
    async fn test_fork_thread() {
        let manager = MemoryCheckpointManager::in_memory();

        let state = TestState {
            messages: vec!["original".to_string()],
            counter: 5,
        };

        let checkpoint_id = manager.save("thread1", &state).await.unwrap();

        manager
            .fork::<TestState>("thread1", &checkpoint_id, "thread2")
            .await
            .unwrap();

        let forked: Checkpoint<TestState> = manager.load_latest("thread2").await.unwrap();
        assert_eq!(forked.state, state);
    }

    #[tokio::test]
    async fn test_rewind() {
        let manager = MemoryCheckpointManager::in_memory();

        let states: Vec<TestState> = (0..5)
            .map(|i| TestState {
                messages: vec![format!("msg{}", i)],
                counter: i,
            })
            .collect();

        let mut checkpoint_ids = Vec::new();
        for state in &states {
            let id = manager.save("thread1", state).await.unwrap();
            checkpoint_ids.push(id);
        }

        // Rewind to checkpoint 2 (3rd checkpoint, index 2)
        manager.rewind("thread1", &checkpoint_ids[2]).await.unwrap();

        let history = manager.history("thread1").await.unwrap();
        assert_eq!(history.len(), 3);

        let latest: Checkpoint<TestState> = manager.load_latest("thread1").await.unwrap();
        assert_eq!(latest.state.counter, 2);
    }

    #[tokio::test]
    async fn test_max_checkpoints_cleanup() {
        let config = CheckpointConfig {
            max_checkpoints_per_thread: Some(3),
            ..Default::default()
        };
        let manager = MemoryCheckpointManager::in_memory().with_config(config);

        let state = TestState {
            messages: vec![],
            counter: 0,
        };

        // Save 5 checkpoints
        for _ in 0..5 {
            manager.save("thread1", &state).await.unwrap();
        }

        // Should only have 3 checkpoints
        let history = manager.history("thread1").await.unwrap();
        assert_eq!(history.len(), 3);

        // Should have kept the latest 3 (steps 3, 4, 5)
        assert_eq!(history[0].step, 3);
        assert_eq!(history[1].step, 4);
        assert_eq!(history[2].step, 5);
    }

    #[tokio::test]
    async fn test_delete_thread() {
        let manager = MemoryCheckpointManager::in_memory();

        let state = TestState {
            messages: vec![],
            counter: 0,
        };

        manager.save("thread1", &state).await.unwrap();
        manager.save("thread1", &state).await.unwrap();

        manager.delete_thread("thread1").await.unwrap();

        let history = manager.history("thread1").await.unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_list_threads() {
        let manager = MemoryCheckpointManager::in_memory();

        let state = TestState {
            messages: vec![],
            counter: 0,
        };

        manager.save("thread1", &state).await.unwrap();
        manager.save("thread2", &state).await.unwrap();
        manager.save("thread3", &state).await.unwrap();

        let threads = manager.list_threads().await.unwrap();
        assert_eq!(threads.len(), 3);
    }

    #[tokio::test]
    async fn test_current_step() {
        let manager = MemoryCheckpointManager::in_memory();

        let state = TestState {
            messages: vec![],
            counter: 0,
        };

        assert_eq!(manager.current_step("thread1"), 0);

        manager.save("thread1", &state).await.unwrap();
        assert_eq!(manager.current_step("thread1"), 1);

        manager.save("thread1", &state).await.unwrap();
        assert_eq!(manager.current_step("thread1"), 2);
    }

    #[tokio::test]
    async fn test_checkpoint_with_label() {
        let manager = MemoryCheckpointManager::in_memory();

        let state = TestState {
            messages: vec![],
            counter: 0,
        };

        manager
            .save_with_label("thread1", &state, Some("important".to_string()))
            .await
            .unwrap();

        let found = manager.find_by_label("thread1", "important").await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().label.as_deref(), Some("important"));
    }

    #[tokio::test]
    async fn test_parent_chain() {
        let manager = MemoryCheckpointManager::in_memory();

        let state = TestState {
            messages: vec![],
            counter: 0,
        };

        let id1 = manager.save("thread1", &state).await.unwrap();
        let id2 = manager.save("thread1", &state).await.unwrap();
        let _id3 = manager.save("thread1", &state).await.unwrap();

        let history = manager.history("thread1").await.unwrap();

        // First checkpoint has no parent
        assert!(history[0].parent_id.is_none());

        // Second checkpoint's parent is first
        assert_eq!(history[1].parent_id.as_deref(), Some(id1.as_str()));

        // Third checkpoint's parent is second
        assert_eq!(history[2].parent_id.as_deref(), Some(id2.as_str()));
    }
}
