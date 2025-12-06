//! Persistent memory with pluggable storage backends
//!
//! This module provides a trait-based abstraction for agent memory persistence,
//! allowing different storage backends (in-memory, sled, sqlite, redis, etc.)

use async_trait::async_trait;
use rust_ai_agents_core::{errors::MemoryError, Message};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Session data that can be persisted and resumed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier
    pub id: String,
    /// Agent identifier this session belongs to
    pub agent_id: String,
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Session metadata (arbitrary key-value pairs)
    pub metadata: HashMap<String, serde_json::Value>,
    /// Creation timestamp (Unix milliseconds)
    pub created_at: i64,
    /// Last updated timestamp (Unix milliseconds)
    pub updated_at: i64,
    /// Resume token for continuing interrupted sessions
    pub resume_token: Option<String>,
}

impl Session {
    /// Create a new session
    pub fn new(agent_id: impl Into<String>) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            id: Uuid::new_v4().to_string(),
            agent_id: agent_id.into(),
            messages: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            resume_token: None,
        }
    }

    /// Create a new session with a specific ID
    pub fn with_id(id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            id: id.into(),
            agent_id: agent_id.into(),
            messages: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            resume_token: None,
        }
    }

    /// Add a message to the session
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.updated_at = chrono::Utc::now().timestamp_millis();
    }

    /// Set metadata value
    pub fn set_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
        self.updated_at = chrono::Utc::now().timestamp_millis();
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Generate a resume token for this session
    pub fn generate_resume_token(&mut self) -> String {
        let token = format!("{}:{}", self.id, Uuid::new_v4());
        self.resume_token = Some(token.clone());
        self.updated_at = chrono::Utc::now().timestamp_millis();
        token
    }
}

/// Trait for pluggable memory storage backends
#[async_trait]
pub trait MemoryBackend: Send + Sync {
    /// Save a session to storage
    async fn save_session(&self, session: &Session) -> Result<(), MemoryError>;

    /// Load a session by ID
    async fn load_session(&self, session_id: &str) -> Result<Option<Session>, MemoryError>;

    /// Load a session by resume token
    async fn load_by_resume_token(&self, token: &str) -> Result<Option<Session>, MemoryError>;

    /// Delete a session
    async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError>;

    /// List all sessions for an agent
    async fn list_sessions(&self, agent_id: &str) -> Result<Vec<Session>, MemoryError>;

    /// List all session IDs (lightweight)
    async fn list_session_ids(&self, agent_id: &str) -> Result<Vec<String>, MemoryError>;

    /// Check if a session exists
    async fn session_exists(&self, session_id: &str) -> Result<bool, MemoryError>;

    /// Clear all sessions for an agent
    async fn clear_agent_sessions(&self, agent_id: &str) -> Result<usize, MemoryError>;

    /// Get backend name for logging/debugging
    fn backend_name(&self) -> &'static str;
}

/// In-memory backend (default, non-persistent)
pub struct InMemoryBackend {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    resume_tokens: Arc<RwLock<HashMap<String, String>>>, // token -> session_id
}

impl InMemoryBackend {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            resume_tokens: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryBackend for InMemoryBackend {
    async fn save_session(&self, session: &Session) -> Result<(), MemoryError> {
        let mut sessions = self.sessions.write().await;

        // Update resume token index
        if let Some(token) = &session.resume_token {
            let mut tokens = self.resume_tokens.write().await;
            tokens.insert(token.clone(), session.id.clone());
        }

        sessions.insert(session.id.clone(), session.clone());
        Ok(())
    }

    async fn load_session(&self, session_id: &str) -> Result<Option<Session>, MemoryError> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(session_id).cloned())
    }

    async fn load_by_resume_token(&self, token: &str) -> Result<Option<Session>, MemoryError> {
        let tokens = self.resume_tokens.read().await;
        if let Some(session_id) = tokens.get(token) {
            let sessions = self.sessions.read().await;
            Ok(sessions.get(session_id).cloned())
        } else {
            Ok(None)
        }
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.remove(session_id) {
            // Clean up resume token
            if let Some(token) = session.resume_token {
                let mut tokens = self.resume_tokens.write().await;
                tokens.remove(&token);
            }
        }
        Ok(())
    }

    async fn list_sessions(&self, agent_id: &str) -> Result<Vec<Session>, MemoryError> {
        let sessions = self.sessions.read().await;
        Ok(sessions
            .values()
            .filter(|s| s.agent_id == agent_id)
            .cloned()
            .collect())
    }

    async fn list_session_ids(&self, agent_id: &str) -> Result<Vec<String>, MemoryError> {
        let sessions = self.sessions.read().await;
        Ok(sessions
            .values()
            .filter(|s| s.agent_id == agent_id)
            .map(|s| s.id.clone())
            .collect())
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool, MemoryError> {
        let sessions = self.sessions.read().await;
        Ok(sessions.contains_key(session_id))
    }

    async fn clear_agent_sessions(&self, agent_id: &str) -> Result<usize, MemoryError> {
        let mut sessions = self.sessions.write().await;
        let mut tokens = self.resume_tokens.write().await;

        let to_remove: Vec<_> = sessions
            .iter()
            .filter(|(_, s)| s.agent_id == agent_id)
            .map(|(id, s)| (id.clone(), s.resume_token.clone()))
            .collect();

        let count = to_remove.len();

        for (id, token) in to_remove {
            sessions.remove(&id);
            if let Some(t) = token {
                tokens.remove(&t);
            }
        }

        Ok(count)
    }

    fn backend_name(&self) -> &'static str {
        "in-memory"
    }
}

/// Sled-based persistent backend
pub struct SledBackend {
    db: sled::Db,
    sessions_tree: sled::Tree,
    tokens_tree: sled::Tree,
    agent_index_tree: sled::Tree,
}

impl SledBackend {
    /// Create a new Sled backend with the given path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, MemoryError> {
        let db = sled::open(path).map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let sessions_tree = db
            .open_tree("sessions")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let tokens_tree = db
            .open_tree("resume_tokens")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let agent_index_tree = db
            .open_tree("agent_sessions")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(Self {
            db,
            sessions_tree,
            tokens_tree,
            agent_index_tree,
        })
    }

    /// Create a temporary in-memory Sled database (useful for testing)
    pub fn temporary() -> Result<Self, MemoryError> {
        let db = sled::Config::new()
            .temporary(true)
            .open()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let sessions_tree = db
            .open_tree("sessions")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let tokens_tree = db
            .open_tree("resume_tokens")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let agent_index_tree = db
            .open_tree("agent_sessions")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(Self {
            db,
            sessions_tree,
            tokens_tree,
            agent_index_tree,
        })
    }

    /// Flush all pending writes to disk
    pub fn flush(&self) -> Result<(), MemoryError> {
        self.db
            .flush()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(())
    }

    // Helper to create agent index key
    fn agent_session_key(agent_id: &str, session_id: &str) -> Vec<u8> {
        format!("{}:{}", agent_id, session_id).into_bytes()
    }
}

#[async_trait]
impl MemoryBackend for SledBackend {
    async fn save_session(&self, session: &Session) -> Result<(), MemoryError> {
        let session_bytes = serde_json::to_vec(session)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

        // Save session
        self.sessions_tree
            .insert(session.id.as_bytes(), session_bytes)
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        // Update resume token index
        if let Some(token) = &session.resume_token {
            self.tokens_tree
                .insert(token.as_bytes(), session.id.as_bytes())
                .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        }

        // Update agent index
        let agent_key = Self::agent_session_key(&session.agent_id, &session.id);
        self.agent_index_tree
            .insert(agent_key, &[1u8])
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(())
    }

    async fn load_session(&self, session_id: &str) -> Result<Option<Session>, MemoryError> {
        match self.sessions_tree.get(session_id.as_bytes()) {
            Ok(Some(bytes)) => {
                let session: Session = serde_json::from_slice(&bytes)
                    .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
                Ok(Some(session))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(MemoryError::StorageError(e.to_string())),
        }
    }

    async fn load_by_resume_token(&self, token: &str) -> Result<Option<Session>, MemoryError> {
        match self.tokens_tree.get(token.as_bytes()) {
            Ok(Some(session_id_bytes)) => {
                let session_id = String::from_utf8(session_id_bytes.to_vec())
                    .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
                self.load_session(&session_id).await
            }
            Ok(None) => Ok(None),
            Err(e) => Err(MemoryError::StorageError(e.to_string())),
        }
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        // Load session first to get agent_id and resume_token
        if let Some(session) = self.load_session(session_id).await? {
            // Remove from sessions
            self.sessions_tree
                .remove(session_id.as_bytes())
                .map_err(|e| MemoryError::StorageError(e.to_string()))?;

            // Remove resume token
            if let Some(token) = session.resume_token {
                self.tokens_tree
                    .remove(token.as_bytes())
                    .map_err(|e| MemoryError::StorageError(e.to_string()))?;
            }

            // Remove from agent index
            let agent_key = Self::agent_session_key(&session.agent_id, session_id);
            self.agent_index_tree
                .remove(agent_key)
                .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        }

        Ok(())
    }

    async fn list_sessions(&self, agent_id: &str) -> Result<Vec<Session>, MemoryError> {
        let prefix = format!("{}:", agent_id);
        let mut sessions = Vec::new();

        for result in self.agent_index_tree.scan_prefix(prefix.as_bytes()) {
            let (key, _) = result.map_err(|e| MemoryError::StorageError(e.to_string()))?;
            let key_str = String::from_utf8(key.to_vec())
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

            // Extract session_id from "agent_id:session_id"
            if let Some(session_id) = key_str.strip_prefix(&prefix) {
                if let Some(session) = self.load_session(session_id).await? {
                    sessions.push(session);
                }
            }
        }

        Ok(sessions)
    }

    async fn list_session_ids(&self, agent_id: &str) -> Result<Vec<String>, MemoryError> {
        let prefix = format!("{}:", agent_id);
        let mut ids = Vec::new();

        for result in self.agent_index_tree.scan_prefix(prefix.as_bytes()) {
            let (key, _) = result.map_err(|e| MemoryError::StorageError(e.to_string()))?;
            let key_str = String::from_utf8(key.to_vec())
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

            if let Some(session_id) = key_str.strip_prefix(&prefix) {
                ids.push(session_id.to_string());
            }
        }

        Ok(ids)
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool, MemoryError> {
        self.sessions_tree
            .contains_key(session_id.as_bytes())
            .map_err(|e| MemoryError::StorageError(e.to_string()))
    }

    async fn clear_agent_sessions(&self, agent_id: &str) -> Result<usize, MemoryError> {
        let session_ids = self.list_session_ids(agent_id).await?;
        let count = session_ids.len();

        for session_id in session_ids {
            self.delete_session(&session_id).await?;
        }

        Ok(count)
    }

    fn backend_name(&self) -> &'static str {
        "sled"
    }
}

/// Session manager for managing agent sessions with a pluggable backend
pub struct SessionManager {
    backend: Arc<dyn MemoryBackend>,
}

impl SessionManager {
    /// Create a new session manager with the given backend
    pub fn new(backend: Arc<dyn MemoryBackend>) -> Self {
        Self { backend }
    }

    /// Create a session manager with the default in-memory backend
    pub fn in_memory() -> Self {
        Self {
            backend: Arc::new(InMemoryBackend::new()),
        }
    }

    /// Create a session manager with Sled backend
    pub fn sled<P: AsRef<Path>>(path: P) -> Result<Self, MemoryError> {
        Ok(Self {
            backend: Arc::new(SledBackend::new(path)?),
        })
    }

    /// Create a session manager with temporary Sled backend (for testing)
    pub fn sled_temporary() -> Result<Self, MemoryError> {
        Ok(Self {
            backend: Arc::new(SledBackend::temporary()?),
        })
    }

    /// Get the backend reference
    pub fn backend(&self) -> &dyn MemoryBackend {
        self.backend.as_ref()
    }

    /// Create a new session for an agent
    pub async fn create_session(&self, agent_id: &str) -> Result<Session, MemoryError> {
        let session = Session::new(agent_id);
        self.backend.save_session(&session).await?;
        tracing::debug!(
            backend = self.backend.backend_name(),
            session_id = %session.id,
            agent_id = %agent_id,
            "Created new session"
        );
        Ok(session)
    }

    /// Get or create a session for an agent
    pub async fn get_or_create_session(
        &self,
        agent_id: &str,
        session_id: Option<&str>,
    ) -> Result<Session, MemoryError> {
        if let Some(id) = session_id {
            if let Some(session) = self.backend.load_session(id).await? {
                return Ok(session);
            }
        }
        self.create_session(agent_id).await
    }

    /// Resume a session using a resume token
    pub async fn resume_session(&self, token: &str) -> Result<Option<Session>, MemoryError> {
        let session = self.backend.load_by_resume_token(token).await?;
        if session.is_some() {
            tracing::debug!(
                backend = self.backend.backend_name(),
                token = %token,
                "Resumed session from token"
            );
        }
        Ok(session)
    }

    /// Save a session
    pub async fn save_session(&self, session: &Session) -> Result<(), MemoryError> {
        self.backend.save_session(session).await
    }

    /// Add a message to a session and save
    pub async fn add_message(
        &self,
        session_id: &str,
        message: Message,
    ) -> Result<Session, MemoryError> {
        let mut session = self
            .backend
            .load_session(session_id)
            .await?
            .ok_or_else(|| MemoryError::SessionNotFound(session_id.to_string()))?;

        session.add_message(message);
        self.backend.save_session(&session).await?;
        Ok(session)
    }

    /// Generate a resume token for a session
    pub async fn create_resume_token(&self, session_id: &str) -> Result<String, MemoryError> {
        let mut session = self
            .backend
            .load_session(session_id)
            .await?
            .ok_or_else(|| MemoryError::SessionNotFound(session_id.to_string()))?;

        let token = session.generate_resume_token();
        self.backend.save_session(&session).await?;

        tracing::debug!(
            backend = self.backend.backend_name(),
            session_id = %session_id,
            "Generated resume token"
        );

        Ok(token)
    }

    /// Delete a session
    pub async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        self.backend.delete_session(session_id).await
    }

    /// List all sessions for an agent
    pub async fn list_sessions(&self, agent_id: &str) -> Result<Vec<Session>, MemoryError> {
        self.backend.list_sessions(agent_id).await
    }

    /// Clear all sessions for an agent
    pub async fn clear_agent_sessions(&self, agent_id: &str) -> Result<usize, MemoryError> {
        let count = self.backend.clear_agent_sessions(agent_id).await?;
        tracing::info!(
            backend = self.backend.backend_name(),
            agent_id = %agent_id,
            count = count,
            "Cleared agent sessions"
        );
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_ai_agents_core::{types::AgentId, Content};

    fn make_user_message(text: &str) -> Message {
        Message::user(AgentId::new("test-agent"), text)
    }

    fn make_assistant_message(text: &str) -> Message {
        Message::new(
            AgentId::new("test-agent"),
            AgentId::new("user"),
            Content::Text(text.to_string()),
        )
    }

    #[tokio::test]
    async fn test_in_memory_backend_basic() {
        let backend = InMemoryBackend::new();

        // Create and save session
        let mut session = Session::new("test-agent");
        session.add_message(make_user_message("Hello"));
        session.add_message(make_assistant_message("Hi there!"));

        backend.save_session(&session).await.unwrap();

        // Load session
        let loaded = backend.load_session(&session.id).await.unwrap().unwrap();
        assert_eq!(loaded.id, session.id);
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.agent_id, "test-agent");
    }

    #[tokio::test]
    async fn test_in_memory_resume_token() {
        let backend = InMemoryBackend::new();

        let mut session = Session::new("test-agent");
        let token = session.generate_resume_token();
        backend.save_session(&session).await.unwrap();

        // Resume by token
        let resumed = backend.load_by_resume_token(&token).await.unwrap().unwrap();
        assert_eq!(resumed.id, session.id);
    }

    #[tokio::test]
    async fn test_in_memory_list_sessions() {
        let backend = InMemoryBackend::new();

        // Create sessions for different agents
        let session1 = Session::new("agent-1");
        let session2 = Session::new("agent-1");
        let session3 = Session::new("agent-2");

        backend.save_session(&session1).await.unwrap();
        backend.save_session(&session2).await.unwrap();
        backend.save_session(&session3).await.unwrap();

        // List agent-1 sessions
        let sessions = backend.list_sessions("agent-1").await.unwrap();
        assert_eq!(sessions.len(), 2);

        // List agent-2 sessions
        let sessions = backend.list_sessions("agent-2").await.unwrap();
        assert_eq!(sessions.len(), 1);
    }

    #[tokio::test]
    async fn test_sled_backend_basic() {
        let backend = SledBackend::temporary().unwrap();

        // Create and save session
        let mut session = Session::new("test-agent");
        session.add_message(make_user_message("Test message"));
        session.set_metadata("key", serde_json::json!("value"));

        backend.save_session(&session).await.unwrap();

        // Load session
        let loaded = backend.load_session(&session.id).await.unwrap().unwrap();
        assert_eq!(loaded.id, session.id);
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(
            loaded.get_metadata("key"),
            Some(&serde_json::json!("value"))
        );
    }

    #[tokio::test]
    async fn test_sled_backend_resume_token() {
        let backend = SledBackend::temporary().unwrap();

        let mut session = Session::new("test-agent");
        let token = session.generate_resume_token();
        backend.save_session(&session).await.unwrap();

        // Resume by token
        let resumed = backend.load_by_resume_token(&token).await.unwrap().unwrap();
        assert_eq!(resumed.id, session.id);
    }

    #[tokio::test]
    async fn test_sled_backend_delete() {
        let backend = SledBackend::temporary().unwrap();

        let session = Session::new("test-agent");
        backend.save_session(&session).await.unwrap();

        assert!(backend.session_exists(&session.id).await.unwrap());

        backend.delete_session(&session.id).await.unwrap();

        assert!(!backend.session_exists(&session.id).await.unwrap());
    }

    #[tokio::test]
    async fn test_session_manager_create_and_resume() {
        let manager = SessionManager::in_memory();

        // Create session
        let session = manager.create_session("my-agent").await.unwrap();

        // Generate resume token
        let token = manager.create_resume_token(&session.id).await.unwrap();

        // Resume session
        let resumed = manager.resume_session(&token).await.unwrap().unwrap();
        assert_eq!(resumed.id, session.id);
    }

    #[tokio::test]
    async fn test_session_manager_add_message() {
        let manager = SessionManager::in_memory();

        let session = manager.create_session("my-agent").await.unwrap();

        // Add messages
        let updated = manager
            .add_message(&session.id, make_user_message("Hello"))
            .await
            .unwrap();

        assert_eq!(updated.messages.len(), 1);
        assert!(updated.messages[0].is_text());
    }

    #[tokio::test]
    async fn test_session_manager_sled() {
        let manager = SessionManager::sled_temporary().unwrap();

        // Create session
        let session = manager.create_session("persistent-agent").await.unwrap();

        // Add message
        let _updated = manager
            .add_message(&session.id, make_user_message("Persistent message"))
            .await
            .unwrap();

        // Verify
        let sessions = manager.list_sessions("persistent-agent").await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].messages.len(), 1);
    }

    #[tokio::test]
    async fn test_session_manager_clear() {
        let manager = SessionManager::in_memory();

        // Create multiple sessions
        manager.create_session("agent-1").await.unwrap();
        manager.create_session("agent-1").await.unwrap();
        manager.create_session("agent-2").await.unwrap();

        // Clear agent-1 sessions
        let count = manager.clear_agent_sessions("agent-1").await.unwrap();
        assert_eq!(count, 2);

        // Verify
        let sessions = manager.list_sessions("agent-1").await.unwrap();
        assert_eq!(sessions.len(), 0);

        let sessions = manager.list_sessions("agent-2").await.unwrap();
        assert_eq!(sessions.len(), 1);
    }
}
