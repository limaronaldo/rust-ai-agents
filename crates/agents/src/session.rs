//! # Structured Sessions
//!
//! Conversation memory and session management for agents.
//!
//! Inspired by OpenAI Agents SDK session patterns.
//!
//! ## Features
//!
//! - **Conversation History**: Store and retrieve message history
//! - **Session State**: Persist arbitrary session state
//! - **Turn Management**: Track conversation turns
//! - **Context Windows**: Manage context length limits
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents::session::{ConversationSession, ChatMessage, ChatRole};
//!
//! let session = ConversationSession::new("user_123")
//!     .with_system_prompt("You are a helpful assistant");
//!
//! session.add_message(ChatRole::User, "Hello!");
//! session.add_message(ChatRole::Assistant, "Hi there! How can I help?");
//!
//! let context = session.get_context(4096)?; // Get messages within token limit
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Unique identifier for a conversation session
pub type ConversationId = String;

/// Role of a message sender in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatRole {
    /// System instructions
    System,
    /// User message
    User,
    /// Assistant response
    Assistant,
    /// Tool/function call
    Tool,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        }
    }
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the sender
    pub role: ChatRole,
    /// Message content
    pub content: String,
    /// When the message was created
    pub timestamp: u64,
    /// Optional name (for multi-agent scenarios)
    pub name: Option<String>,
    /// Optional tool call ID
    pub tool_call_id: Option<String>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            name: None,
            tool_call_id: None,
            metadata: HashMap::new(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        let mut msg = Self::new(ChatRole::Tool, content);
        msg.tool_call_id = Some(tool_call_id.into());
        msg
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Estimate token count (rough approximation: 4 chars per token)
    pub fn estimated_tokens(&self) -> usize {
        self.content.len() / 4 + 1
    }
}

/// A conversation turn (user message + assistant response)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Turn number (1-indexed)
    pub number: u32,
    /// User message
    pub user_message: ChatMessage,
    /// Assistant response (if available)
    pub assistant_response: Option<ChatMessage>,
    /// Tool calls made during this turn
    pub tool_calls: Vec<ChatMessage>,
    /// When the turn started
    pub started_at: u64,
    /// When the turn completed
    pub completed_at: Option<u64>,
}

impl Turn {
    pub fn new(number: u32, user_message: ChatMessage) -> Self {
        Self {
            number,
            started_at: user_message.timestamp,
            user_message,
            assistant_response: None,
            tool_calls: Vec::new(),
            completed_at: None,
        }
    }

    pub fn complete(&mut self, response: ChatMessage) {
        self.assistant_response = Some(response);
        self.completed_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
    }

    pub fn add_tool_call(&mut self, tool_message: ChatMessage) {
        self.tool_calls.push(tool_message);
    }

    pub fn is_complete(&self) -> bool {
        self.assistant_response.is_some()
    }

    pub fn all_messages(&self) -> Vec<&ChatMessage> {
        let mut messages = vec![&self.user_message];
        messages.extend(self.tool_calls.iter());
        if let Some(ref response) = self.assistant_response {
            messages.push(response);
        }
        messages
    }
}

/// Session state for arbitrary data
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionState {
    /// Key-value store for session variables
    pub variables: HashMap<String, String>,
    /// Structured data (JSON serialized)
    pub data: HashMap<String, String>,
}

impl SessionState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.variables.insert(key.into(), value.into());
    }

    pub fn get(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }

    pub fn set_data<T: Serialize>(
        &mut self,
        key: impl Into<String>,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        let json = serde_json::to_string(value)?;
        self.data.insert(key.into(), json);
        Ok(())
    }

    pub fn get_data<T: for<'de> Deserialize<'de>>(
        &self,
        key: &str,
    ) -> Option<Result<T, serde_json::Error>> {
        self.data.get(key).map(|json| serde_json::from_str(json))
    }

    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.variables.remove(key)
    }

    pub fn clear(&mut self) {
        self.variables.clear();
        self.data.clear();
    }
}

/// Configuration for a session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Maximum messages to keep in history
    pub max_messages: usize,
    /// Maximum tokens in context
    pub max_tokens: usize,
    /// Session timeout (auto-expire)
    pub timeout: Option<Duration>,
    /// Whether to persist system prompt in history
    pub persist_system_prompt: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_messages: 100,
            max_tokens: 8192,
            timeout: Some(Duration::from_secs(3600)), // 1 hour
            persist_system_prompt: true,
        }
    }
}

/// A session representing a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSession {
    /// Session ID
    pub id: ConversationId,
    /// User ID (if applicable)
    pub user_id: Option<String>,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Message history
    pub messages: Vec<ChatMessage>,
    /// Conversation turns
    pub turns: Vec<Turn>,
    /// Session state
    pub state: SessionState,
    /// When session was created
    pub created_at: u64,
    /// When session was last updated
    pub updated_at: u64,
    /// Session metadata
    pub metadata: HashMap<String, String>,
    /// Configuration (not serialized)
    #[serde(skip)]
    config: SessionConfig,
}

impl ConversationSession {
    pub fn new(id: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: id.into(),
            user_id: None,
            system_prompt: None,
            messages: Vec::new(),
            turns: Vec::new(),
            state: SessionState::new(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
            config: SessionConfig::default(),
        }
    }

    pub fn with_config(mut self, config: SessionConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        self.system_prompt = Some(prompt.clone());
        if self.config.persist_system_prompt {
            self.messages.insert(0, ChatMessage::system(prompt));
        }
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add a message to the session
    pub fn add_message(&mut self, role: ChatRole, content: impl Into<String>) {
        let message = ChatMessage::new(role, content);
        self.add_message_obj(message);
    }

    /// Add a pre-constructed message
    pub fn add_message_obj(&mut self, message: ChatMessage) {
        // Handle turn management
        match message.role {
            ChatRole::User => {
                let turn_num = self.turns.len() as u32 + 1;
                self.turns.push(Turn::new(turn_num, message.clone()));
            }
            ChatRole::Assistant => {
                if let Some(turn) = self.turns.last_mut() {
                    turn.complete(message.clone());
                }
            }
            ChatRole::Tool => {
                if let Some(turn) = self.turns.last_mut() {
                    turn.add_tool_call(message.clone());
                }
            }
            ChatRole::System => {
                // System messages don't affect turns
            }
        }

        self.messages.push(message);
        self.touch();
        self.enforce_limits();
    }

    /// Get messages for context (within token limit)
    pub fn get_context(&self, max_tokens: Option<usize>) -> Vec<&ChatMessage> {
        let max_tokens = max_tokens.unwrap_or(self.config.max_tokens);
        let mut result = Vec::new();
        let mut token_count = 0;

        // Always include system prompt if present
        if let Some(system_msg) = self.messages.first() {
            if system_msg.role == ChatRole::System {
                token_count += system_msg.estimated_tokens();
                result.push(system_msg);
            }
        }

        // Add messages from most recent, respecting token limit
        for message in self.messages.iter().rev() {
            if message.role == ChatRole::System {
                continue; // Already handled
            }

            let msg_tokens = message.estimated_tokens();
            if token_count + msg_tokens > max_tokens {
                break;
            }

            token_count += msg_tokens;
            result.push(message);
        }

        // Reverse to get chronological order (except system which is first)
        let system_msg = if result.first().map(|m| m.role) == Some(ChatRole::System) {
            Some(result.remove(0))
        } else {
            None
        };

        result.reverse();

        if let Some(sys) = system_msg {
            result.insert(0, sys);
        }

        result
    }

    /// Get all messages
    pub fn get_all_messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Get recent N messages
    pub fn get_recent(&self, n: usize) -> Vec<&ChatMessage> {
        self.messages.iter().rev().take(n).rev().collect()
    }

    /// Get messages by role
    pub fn get_by_role(&self, role: ChatRole) -> Vec<&ChatMessage> {
        self.messages.iter().filter(|m| m.role == role).collect()
    }

    /// Get current turn number
    pub fn current_turn(&self) -> u32 {
        self.turns.len() as u32
    }

    /// Get the last turn
    pub fn last_turn(&self) -> Option<&Turn> {
        self.turns.last()
    }

    /// Get a specific turn
    pub fn get_turn(&self, number: u32) -> Option<&Turn> {
        if number == 0 || number as usize > self.turns.len() {
            return None;
        }
        self.turns.get(number as usize - 1)
    }

    /// Check if session is expired
    pub fn is_expired(&self) -> bool {
        if let Some(timeout) = self.config.timeout {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            return now - self.updated_at > timeout.as_secs();
        }
        false
    }

    /// Get session age
    pub fn age(&self) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Duration::from_secs(now - self.created_at)
    }

    /// Clear all messages (keep system prompt)
    pub fn clear_messages(&mut self) {
        let system = if self.config.persist_system_prompt {
            self.messages
                .first()
                .filter(|m| m.role == ChatRole::System)
                .cloned()
        } else {
            None
        };

        self.messages.clear();
        self.turns.clear();

        if let Some(sys) = system {
            self.messages.push(sys);
        }

        self.touch();
    }

    /// Get total estimated tokens
    pub fn total_tokens(&self) -> usize {
        self.messages.iter().map(|m| m.estimated_tokens()).sum()
    }

    fn touch(&mut self) {
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    fn enforce_limits(&mut self) {
        // Enforce max messages
        while self.messages.len() > self.config.max_messages {
            // Keep system prompt at index 0 if present
            let remove_idx = if self.messages.first().map(|m| m.role) == Some(ChatRole::System) {
                1
            } else {
                0
            };
            if remove_idx < self.messages.len() {
                self.messages.remove(remove_idx);
            }
        }
    }
}

/// Trait for session storage
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Save a session
    async fn save(&self, session: &ConversationSession) -> Result<(), SessionError>;

    /// Load a session
    async fn load(&self, session_id: &str) -> Result<Option<ConversationSession>, SessionError>;

    /// Delete a session
    async fn delete(&self, session_id: &str) -> Result<(), SessionError>;

    /// List all session IDs
    async fn list(&self) -> Result<Vec<ConversationId>, SessionError>;

    /// List sessions for a user
    async fn list_for_user(&self, user_id: &str) -> Result<Vec<ConversationId>, SessionError>;
}

/// Error type for session operations
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("Session not found: {0}")]
    NotFound(ConversationId),

    #[error("Session expired: {0}")]
    Expired(ConversationId),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// In-memory session store
#[derive(Default)]
pub struct MemorySessionStore {
    sessions: RwLock<HashMap<ConversationId, ConversationSession>>,
}

impl MemorySessionStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionStore for MemorySessionStore {
    async fn save(&self, session: &ConversationSession) -> Result<(), SessionError> {
        self.sessions
            .write()
            .insert(session.id.clone(), session.clone());
        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Option<ConversationSession>, SessionError> {
        Ok(self.sessions.read().get(session_id).cloned())
    }

    async fn delete(&self, session_id: &str) -> Result<(), SessionError> {
        self.sessions.write().remove(session_id);
        Ok(())
    }

    async fn list(&self) -> Result<Vec<ConversationId>, SessionError> {
        Ok(self.sessions.read().keys().cloned().collect())
    }

    async fn list_for_user(&self, user_id: &str) -> Result<Vec<ConversationId>, SessionError> {
        Ok(self
            .sessions
            .read()
            .values()
            .filter(|s| s.user_id.as_deref() == Some(user_id))
            .map(|s| s.id.clone())
            .collect())
    }
}

/// Session manager for handling multiple sessions
pub struct ConversationManager<S: SessionStore> {
    store: Arc<S>,
    config: SessionConfig,
}

impl ConversationManager<MemorySessionStore> {
    /// Create with in-memory store
    pub fn in_memory() -> Self {
        Self::new(Arc::new(MemorySessionStore::new()))
    }
}

impl<S: SessionStore> ConversationManager<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            config: SessionConfig::default(),
        }
    }

    pub fn with_config(mut self, config: SessionConfig) -> Self {
        self.config = config;
        self
    }

    /// Create a new session
    pub async fn create(
        &self,
        session_id: impl Into<String>,
    ) -> Result<ConversationSession, SessionError> {
        let session = ConversationSession::new(session_id).with_config(self.config.clone());
        self.store.save(&session).await?;
        info!(session_id = %session.id, "Session created");
        Ok(session)
    }

    /// Get or create a session
    pub async fn get_or_create(
        &self,
        session_id: impl Into<String>,
    ) -> Result<ConversationSession, SessionError> {
        let session_id = session_id.into();
        match self.store.load(&session_id).await? {
            Some(session) => {
                if session.is_expired() {
                    debug!(session_id = %session_id, "Session expired, creating new");
                    self.store.delete(&session_id).await?;
                    self.create(session_id).await
                } else {
                    Ok(session)
                }
            }
            None => self.create(session_id).await,
        }
    }

    /// Update a session
    pub async fn update(&self, session: &ConversationSession) -> Result<(), SessionError> {
        self.store.save(session).await
    }

    /// Delete a session
    pub async fn delete(&self, session_id: &str) -> Result<(), SessionError> {
        self.store.delete(session_id).await?;
        info!(session_id, "Session deleted");
        Ok(())
    }

    /// List all sessions
    pub async fn list(&self) -> Result<Vec<ConversationId>, SessionError> {
        self.store.list().await
    }

    /// List sessions for a user
    pub async fn list_for_user(&self, user_id: &str) -> Result<Vec<ConversationId>, SessionError> {
        self.store.list_for_user(user_id).await
    }

    /// Cleanup expired sessions
    pub async fn cleanup_expired(&self) -> Result<usize, SessionError> {
        let mut cleaned = 0;
        let session_ids = self.store.list().await?;

        for id in session_ids {
            if let Some(session) = self.store.load(&id).await? {
                if session.is_expired() {
                    self.store.delete(&id).await?;
                    cleaned += 1;
                }
            }
        }

        if cleaned > 0 {
            info!(count = cleaned, "Cleaned up expired sessions");
        }

        Ok(cleaned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = ChatMessage::user("Hello!");
        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.content, "Hello!");
        assert!(msg.timestamp > 0);
    }

    #[test]
    fn test_message_builders() {
        let msg = ChatMessage::assistant("Hi there!")
            .with_name("Assistant")
            .with_metadata("source", "test");

        assert_eq!(msg.role, ChatRole::Assistant);
        assert_eq!(msg.name, Some("Assistant".to_string()));
        assert_eq!(msg.metadata.get("source").unwrap(), "test");
    }

    #[test]
    fn test_session_basic() {
        let mut session =
            ConversationSession::new("test_session").with_system_prompt("You are helpful");

        session.add_message(ChatRole::User, "Hello!");
        session.add_message(ChatRole::Assistant, "Hi there!");

        assert_eq!(session.messages.len(), 3); // system + user + assistant
        assert_eq!(session.current_turn(), 1);
    }

    #[test]
    fn test_session_turns() {
        let mut session = ConversationSession::new("test");

        session.add_message(ChatRole::User, "First question");
        session.add_message(ChatRole::Assistant, "First answer");
        session.add_message(ChatRole::User, "Second question");
        session.add_message(ChatRole::Assistant, "Second answer");

        assert_eq!(session.current_turn(), 2);

        let turn1 = session.get_turn(1).unwrap();
        assert_eq!(turn1.user_message.content, "First question");
        assert!(turn1.is_complete());

        let turn2 = session.get_turn(2).unwrap();
        assert_eq!(turn2.user_message.content, "Second question");
    }

    #[test]
    fn test_get_context_with_limit() {
        let mut session = ConversationSession::new("test").with_system_prompt("System");

        // Add many messages
        for i in 0..20 {
            session.add_message(ChatRole::User, format!("Message {}", i));
            session.add_message(ChatRole::Assistant, format!("Response {}", i));
        }

        // Get with token limit
        let context = session.get_context(Some(100));

        // Should include system prompt and recent messages within limit
        assert!(!context.is_empty());
        assert_eq!(context[0].role, ChatRole::System);
    }

    #[test]
    fn test_session_state() {
        let mut session = ConversationSession::new("test");

        session.state.set("user_name", "Alice");
        session.state.set("preference", "dark_mode");

        assert_eq!(session.state.get("user_name").unwrap(), "Alice");
        assert_eq!(session.state.get("preference").unwrap(), "dark_mode");

        // Test structured data
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct UserPrefs {
            theme: String,
            language: String,
        }

        let prefs = UserPrefs {
            theme: "dark".to_string(),
            language: "en".to_string(),
        };

        session.state.set_data("prefs", &prefs).unwrap();
        let loaded: UserPrefs = session.state.get_data("prefs").unwrap().unwrap();
        assert_eq!(loaded, prefs);
    }

    #[test]
    fn test_message_limit() {
        let config = SessionConfig {
            max_messages: 5,
            ..Default::default()
        };

        let mut session = ConversationSession::new("test").with_config(config);

        for i in 0..10 {
            session.add_message(ChatRole::User, format!("Message {}", i));
        }

        assert_eq!(session.messages.len(), 5);
    }

    #[test]
    fn test_clear_messages() {
        let mut session = ConversationSession::new("test").with_system_prompt("System prompt");

        session.add_message(ChatRole::User, "Hello");
        session.add_message(ChatRole::Assistant, "Hi");

        assert_eq!(session.messages.len(), 3);

        session.clear_messages();

        // Should keep system prompt
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].role, ChatRole::System);
        assert_eq!(session.turns.len(), 0);
    }

    #[test]
    fn test_get_by_role() {
        let mut session = ConversationSession::new("test");

        session.add_message(ChatRole::User, "Q1");
        session.add_message(ChatRole::Assistant, "A1");
        session.add_message(ChatRole::User, "Q2");
        session.add_message(ChatRole::Assistant, "A2");

        let user_messages = session.get_by_role(ChatRole::User);
        assert_eq!(user_messages.len(), 2);

        let assistant_messages = session.get_by_role(ChatRole::Assistant);
        assert_eq!(assistant_messages.len(), 2);
    }

    #[tokio::test]
    async fn test_session_manager() {
        let manager = ConversationManager::in_memory();

        let session = manager.create("session1").await.unwrap();
        assert_eq!(session.id, "session1");

        let loaded = manager.get_or_create("session1").await.unwrap();
        assert_eq!(loaded.id, "session1");

        let sessions = manager.list().await.unwrap();
        assert_eq!(sessions.len(), 1);

        manager.delete("session1").await.unwrap();
        let sessions = manager.list().await.unwrap();
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn test_session_store() {
        let store = MemorySessionStore::new();

        let mut session = ConversationSession::new("test").with_user_id("user1");
        session.add_message(ChatRole::User, "Hello");

        store.save(&session).await.unwrap();

        let loaded = store.load("test").await.unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 1);

        let user_sessions = store.list_for_user("user1").await.unwrap();
        assert_eq!(user_sessions.len(), 1);
    }

    #[test]
    fn test_tool_message() {
        let mut session = ConversationSession::new("test");

        session.add_message(ChatRole::User, "Calculate 2+2");
        session.add_message_obj(ChatMessage::tool("4", "call_123"));
        session.add_message(ChatRole::Assistant, "The result is 4");

        let turn = session.get_turn(1).unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(
            turn.tool_calls[0].tool_call_id,
            Some("call_123".to_string())
        );
    }

    #[test]
    fn test_token_estimation() {
        let msg = ChatMessage::user("Hello world"); // 11 chars
        assert_eq!(msg.estimated_tokens(), 3); // 11/4 + 1 = 3
    }

    #[test]
    fn test_total_tokens() {
        let mut session = ConversationSession::new("test");
        session.add_message(ChatRole::User, "Hello world"); // ~3 tokens
        session.add_message(ChatRole::Assistant, "Hi there"); // ~3 tokens

        assert!(session.total_tokens() >= 4);
    }
}
