//! Session and Trajectory MCP Resources
//!
//! Exposes conversation sessions and execution traces as MCP resources,
//! allowing external tools to browse agent history.
//!
//! # Resource URIs
//!
//! - `session://list` - List all sessions
//! - `session://{session_id}` - Get a specific session
//! - `session://{session_id}/messages` - Get session messages
//! - `session://{session_id}/turns/{turn_number}` - Get a specific turn
//!
//! - `trace://list` - List all traces
//! - `trace://{task_id}` - Get a specific trace
//! - `trace://{task_id}/steps` - Get all steps in a trace
//! - `trace://{task_id}/summary` - Get trace summary
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_mcp::{McpServer, SessionResourceHandler, TraceResourceHandler};
//! use rust_ai_agents_agents::session::MemorySessionStore;
//! use rust_ai_agents_agents::trajectory::TrajectoryStore;
//!
//! let session_store = Arc::new(MemorySessionStore::new());
//! let trace_store = Arc::new(TrajectoryStore::new(1000));
//!
//! let server = McpServer::builder()
//!     .name("agent-server")
//!     .add_resource(SessionResourceHandler::new(session_store))
//!     .add_resource(TraceResourceHandler::new(trace_store))
//!     .build();
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::debug;

use crate::error::McpError;
use crate::protocol::{McpResource, ResourceContent};
use crate::server::ResourceHandler;

// =============================================================================
// Session Resource Handler
// =============================================================================

/// Resource handler for conversation sessions
///
/// Exposes sessions stored in a SessionStore as MCP resources.
pub struct SessionResourceHandler<S: SessionStoreRead> {
    store: Arc<S>,
    /// Optional filter by user ID
    user_filter: Option<String>,
    /// Maximum sessions to list
    max_list_size: usize,
}

/// Trait for reading sessions (subset of SessionStore)
#[async_trait]
pub trait SessionStoreRead: Send + Sync {
    /// List all session IDs
    async fn list_ids(&self) -> Result<Vec<String>, String>;

    /// Load a session by ID and return as JSON
    async fn load_json(&self, session_id: &str) -> Result<Option<String>, String>;

    /// Get session metadata (id, user_id, created_at, updated_at, message_count)
    async fn get_metadata(&self, session_id: &str) -> Result<Option<SessionMetadata>, String>;

    /// Get messages for a session as JSON
    async fn get_messages_json(&self, session_id: &str) -> Result<Option<String>, String>;

    /// Get a specific turn as JSON
    async fn get_turn_json(
        &self,
        session_id: &str,
        turn_number: u32,
    ) -> Result<Option<String>, String>;
}

/// Session metadata for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub id: String,
    pub user_id: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub message_count: usize,
    pub turn_count: usize,
}

impl<S: SessionStoreRead> SessionResourceHandler<S> {
    /// Create a new session resource handler
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            user_filter: None,
            max_list_size: 100,
        }
    }

    /// Filter sessions by user ID
    pub fn with_user_filter(mut self, user_id: impl Into<String>) -> Self {
        self.user_filter = Some(user_id.into());
        self
    }

    /// Set maximum sessions to list
    pub fn with_max_list_size(mut self, size: usize) -> Self {
        self.max_list_size = size;
        self
    }

    /// Parse a session URI and extract components
    fn parse_uri(uri: &str) -> Option<SessionUriParts> {
        let uri = uri.strip_prefix("session://")?;

        if uri == "list" {
            return Some(SessionUriParts::List);
        }

        let parts: Vec<&str> = uri.split('/').collect();

        match parts.as_slice() {
            [session_id] => Some(SessionUriParts::Session(session_id.to_string())),
            [session_id, "messages"] => Some(SessionUriParts::Messages(session_id.to_string())),
            [session_id, "turns", turn_num] => {
                let turn: u32 = turn_num.parse().ok()?;
                Some(SessionUriParts::Turn(session_id.to_string(), turn))
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
enum SessionUriParts {
    List,
    Session(String),
    Messages(String),
    Turn(String, u32),
}

#[async_trait]
impl<S: SessionStoreRead + 'static> ResourceHandler for SessionResourceHandler<S> {
    fn list(&self) -> Vec<McpResource> {
        vec![McpResource {
            uri: "session://list".to_string(),
            name: "Sessions".to_string(),
            description: Some("List all conversation sessions".to_string()),
            mime_type: Some("application/json".to_string()),
        }]
    }

    async fn read(&self, uri: &str) -> Result<ResourceContent, McpError> {
        debug!(uri = %uri, "Reading session resource");

        let parts = Self::parse_uri(uri)
            .ok_or_else(|| McpError::ResourceNotFound(format!("Invalid session URI: {}", uri)))?;

        match parts {
            SessionUriParts::List => {
                let ids = self
                    .store
                    .list_ids()
                    .await
                    .map_err(|e| McpError::Internal(e))?;

                let mut sessions = Vec::new();
                for id in ids.into_iter().take(self.max_list_size) {
                    if let Ok(Some(meta)) = self.store.get_metadata(&id).await {
                        // Apply user filter if set
                        if let Some(ref filter) = self.user_filter {
                            if meta.user_id.as_ref() != Some(filter) {
                                continue;
                            }
                        }
                        sessions.push(meta);
                    }
                }

                let json = serde_json::to_string_pretty(&sessions)
                    .map_err(|e| McpError::Internal(e.to_string()))?;

                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: Some(json),
                    blob: None,
                })
            }

            SessionUriParts::Session(session_id) => {
                let json = self
                    .store
                    .load_json(&session_id)
                    .await
                    .map_err(|e| McpError::Internal(e))?
                    .ok_or_else(|| {
                        McpError::ResourceNotFound(format!("Session not found: {}", session_id))
                    })?;

                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: Some(json),
                    blob: None,
                })
            }

            SessionUriParts::Messages(session_id) => {
                let json = self
                    .store
                    .get_messages_json(&session_id)
                    .await
                    .map_err(|e| McpError::Internal(e))?
                    .ok_or_else(|| {
                        McpError::ResourceNotFound(format!("Session not found: {}", session_id))
                    })?;

                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: Some(json),
                    blob: None,
                })
            }

            SessionUriParts::Turn(session_id, turn_number) => {
                let json = self
                    .store
                    .get_turn_json(&session_id, turn_number)
                    .await
                    .map_err(|e| McpError::Internal(e))?
                    .ok_or_else(|| {
                        McpError::ResourceNotFound(format!(
                            "Turn {} not found in session {}",
                            turn_number, session_id
                        ))
                    })?;

                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: Some(json),
                    blob: None,
                })
            }
        }
    }
}

// =============================================================================
// Trace Resource Handler
// =============================================================================

/// Resource handler for execution traces
///
/// Exposes trajectories stored in a TrajectoryStore as MCP resources.
pub struct TraceResourceHandler<T: TraceStoreRead> {
    store: Arc<T>,
    /// Optional filter by agent name
    agent_filter: Option<String>,
    /// Optional filter by success status
    success_filter: Option<bool>,
    /// Maximum traces to list
    max_list_size: usize,
}

/// Trait for reading traces (subset of TrajectoryStore)
#[async_trait]
pub trait TraceStoreRead: Send + Sync {
    /// List all task IDs
    fn list_ids(&self) -> Vec<String>;

    /// Get a trace by task ID as JSON
    fn get_json(&self, task_id: &str) -> Option<String>;

    /// Get trace metadata
    fn get_metadata(&self, task_id: &str) -> Option<TraceMetadata>;

    /// Get steps for a trace as JSON
    fn get_steps_json(&self, task_id: &str) -> Option<String>;

    /// Get trace summary as JSON
    fn get_summary_json(&self, task_id: &str) -> Option<String>;

    /// Filter by agent name
    fn filter_by_agent(&self, agent_name: &str) -> Vec<String>;

    /// Filter by success status
    fn filter_by_success(&self, success: bool) -> Vec<String>;
}

/// Trace metadata for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    pub task_id: String,
    pub agent_name: String,
    pub success: bool,
    pub total_duration_ms: u64,
    pub step_count: usize,
    pub llm_calls: usize,
    pub tool_calls: usize,
}

impl<T: TraceStoreRead> TraceResourceHandler<T> {
    /// Create a new trace resource handler
    pub fn new(store: Arc<T>) -> Self {
        Self {
            store,
            agent_filter: None,
            success_filter: None,
            max_list_size: 100,
        }
    }

    /// Filter traces by agent name
    pub fn with_agent_filter(mut self, agent_name: impl Into<String>) -> Self {
        self.agent_filter = Some(agent_name.into());
        self
    }

    /// Filter traces by success status
    pub fn with_success_filter(mut self, success: bool) -> Self {
        self.success_filter = Some(success);
        self
    }

    /// Set maximum traces to list
    pub fn with_max_list_size(mut self, size: usize) -> Self {
        self.max_list_size = size;
        self
    }

    /// Parse a trace URI and extract components
    fn parse_uri(uri: &str) -> Option<TraceUriParts> {
        let uri = uri.strip_prefix("trace://")?;

        if uri == "list" {
            return Some(TraceUriParts::List);
        }

        let parts: Vec<&str> = uri.split('/').collect();

        match parts.as_slice() {
            [task_id] => Some(TraceUriParts::Trace(task_id.to_string())),
            [task_id, "steps"] => Some(TraceUriParts::Steps(task_id.to_string())),
            [task_id, "summary"] => Some(TraceUriParts::Summary(task_id.to_string())),
            _ => None,
        }
    }
}

#[derive(Debug)]
enum TraceUriParts {
    List,
    Trace(String),
    Steps(String),
    Summary(String),
}

#[async_trait]
impl<T: TraceStoreRead + 'static> ResourceHandler for TraceResourceHandler<T> {
    fn list(&self) -> Vec<McpResource> {
        vec![McpResource {
            uri: "trace://list".to_string(),
            name: "Traces".to_string(),
            description: Some("List all execution traces".to_string()),
            mime_type: Some("application/json".to_string()),
        }]
    }

    async fn read(&self, uri: &str) -> Result<ResourceContent, McpError> {
        debug!(uri = %uri, "Reading trace resource");

        let parts = Self::parse_uri(uri)
            .ok_or_else(|| McpError::ResourceNotFound(format!("Invalid trace URI: {}", uri)))?;

        match parts {
            TraceUriParts::List => {
                // Get IDs with optional filters
                let ids = if let Some(ref agent) = self.agent_filter {
                    self.store.filter_by_agent(agent)
                } else if let Some(success) = self.success_filter {
                    self.store.filter_by_success(success)
                } else {
                    self.store.list_ids()
                };

                let mut traces = Vec::new();
                for id in ids.into_iter().take(self.max_list_size) {
                    if let Some(meta) = self.store.get_metadata(&id) {
                        traces.push(meta);
                    }
                }

                let json = serde_json::to_string_pretty(&traces)
                    .map_err(|e| McpError::Internal(e.to_string()))?;

                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: Some(json),
                    blob: None,
                })
            }

            TraceUriParts::Trace(task_id) => {
                let json = self.store.get_json(&task_id).ok_or_else(|| {
                    McpError::ResourceNotFound(format!("Trace not found: {}", task_id))
                })?;

                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: Some(json),
                    blob: None,
                })
            }

            TraceUriParts::Steps(task_id) => {
                let json = self.store.get_steps_json(&task_id).ok_or_else(|| {
                    McpError::ResourceNotFound(format!("Trace not found: {}", task_id))
                })?;

                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: Some(json),
                    blob: None,
                })
            }

            TraceUriParts::Summary(task_id) => {
                let json = self.store.get_summary_json(&task_id).ok_or_else(|| {
                    McpError::ResourceNotFound(format!("Trace not found: {}", task_id))
                })?;

                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: Some("application/json".to_string()),
                    text: Some(json),
                    blob: None,
                })
            }
        }
    }
}

// =============================================================================
// In-Memory Implementations
// =============================================================================

/// In-memory session store adapter for MCP resources
pub struct MemorySessionStoreAdapter {
    sessions: parking_lot::RwLock<std::collections::HashMap<String, String>>,
}

impl MemorySessionStoreAdapter {
    pub fn new() -> Self {
        Self {
            sessions: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Store a session (JSON serialized)
    pub fn store(&self, session_id: &str, session_json: &str) {
        self.sessions
            .write()
            .insert(session_id.to_string(), session_json.to_string());
    }

    /// Store a session object
    pub fn store_session<T: Serialize>(&self, session_id: &str, session: &T) {
        if let Ok(json) = serde_json::to_string(session) {
            self.store(session_id, &json);
        }
    }
}

impl Default for MemorySessionStoreAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SessionStoreRead for MemorySessionStoreAdapter {
    async fn list_ids(&self) -> Result<Vec<String>, String> {
        Ok(self.sessions.read().keys().cloned().collect())
    }

    async fn load_json(&self, session_id: &str) -> Result<Option<String>, String> {
        Ok(self.sessions.read().get(session_id).cloned())
    }

    async fn get_metadata(&self, session_id: &str) -> Result<Option<SessionMetadata>, String> {
        let sessions = self.sessions.read();
        let json = match sessions.get(session_id) {
            Some(j) => j,
            None => return Ok(None),
        };

        // Parse just enough to get metadata
        let value: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;

        Ok(Some(SessionMetadata {
            id: value["id"].as_str().unwrap_or(session_id).to_string(),
            user_id: value["user_id"].as_str().map(|s| s.to_string()),
            created_at: value["created_at"].as_u64().unwrap_or(0),
            updated_at: value["updated_at"].as_u64().unwrap_or(0),
            message_count: value["messages"].as_array().map(|a| a.len()).unwrap_or(0),
            turn_count: value["turns"].as_array().map(|a| a.len()).unwrap_or(0),
        }))
    }

    async fn get_messages_json(&self, session_id: &str) -> Result<Option<String>, String> {
        let sessions = self.sessions.read();
        let json = match sessions.get(session_id) {
            Some(j) => j,
            None => return Ok(None),
        };

        let value: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;

        let messages = &value["messages"];
        Ok(Some(
            serde_json::to_string_pretty(messages).map_err(|e| e.to_string())?,
        ))
    }

    async fn get_turn_json(
        &self,
        session_id: &str,
        turn_number: u32,
    ) -> Result<Option<String>, String> {
        let sessions = self.sessions.read();
        let json = match sessions.get(session_id) {
            Some(j) => j,
            None => return Ok(None),
        };

        let value: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;

        let turns = value["turns"].as_array();
        let turn = turns.and_then(|t| t.get(turn_number as usize - 1));

        match turn {
            Some(t) => Ok(Some(
                serde_json::to_string_pretty(t).map_err(|e| e.to_string())?,
            )),
            None => Ok(None),
        }
    }
}

/// In-memory trace store adapter for MCP resources
pub struct MemoryTraceStoreAdapter {
    traces: parking_lot::RwLock<std::collections::HashMap<String, String>>,
}

impl MemoryTraceStoreAdapter {
    pub fn new() -> Self {
        Self {
            traces: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Store a trace (JSON serialized)
    pub fn store(&self, task_id: &str, trace_json: &str) {
        self.traces
            .write()
            .insert(task_id.to_string(), trace_json.to_string());
    }

    /// Store a trace object
    pub fn store_trace<T: Serialize>(&self, task_id: &str, trace: &T) {
        if let Ok(json) = serde_json::to_string(trace) {
            self.store(task_id, &json);
        }
    }
}

impl Default for MemoryTraceStoreAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceStoreRead for MemoryTraceStoreAdapter {
    fn list_ids(&self) -> Vec<String> {
        self.traces.read().keys().cloned().collect()
    }

    fn get_json(&self, task_id: &str) -> Option<String> {
        self.traces.read().get(task_id).cloned()
    }

    fn get_metadata(&self, task_id: &str) -> Option<TraceMetadata> {
        let traces = self.traces.read();
        let json = traces.get(task_id)?;

        let value: serde_json::Value = serde_json::from_str(json).ok()?;

        Some(TraceMetadata {
            task_id: value["task_id"].as_str().unwrap_or(task_id).to_string(),
            agent_name: value["agent_name"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            success: value["success"].as_bool().unwrap_or(false),
            total_duration_ms: value["total_duration_ms"].as_u64().unwrap_or(0),
            step_count: value["steps"].as_array().map(|a| a.len()).unwrap_or(0),
            llm_calls: value["steps"]
                .as_array()
                .map(|steps| {
                    steps
                        .iter()
                        .filter(|s| s["step_type"] == "llm_call")
                        .count()
                })
                .unwrap_or(0),
            tool_calls: value["steps"]
                .as_array()
                .map(|steps| {
                    steps
                        .iter()
                        .filter(|s| s["step_type"] == "tool_call")
                        .count()
                })
                .unwrap_or(0),
        })
    }

    fn get_steps_json(&self, task_id: &str) -> Option<String> {
        let traces = self.traces.read();
        let json = traces.get(task_id)?;

        let value: serde_json::Value = serde_json::from_str(json).ok()?;
        let steps = &value["steps"];

        serde_json::to_string_pretty(steps).ok()
    }

    fn get_summary_json(&self, task_id: &str) -> Option<String> {
        let meta = self.get_metadata(task_id)?;
        serde_json::to_string_pretty(&meta).ok()
    }

    fn filter_by_agent(&self, agent_name: &str) -> Vec<String> {
        self.traces
            .read()
            .iter()
            .filter_map(|(id, json)| {
                let value: serde_json::Value = serde_json::from_str(json).ok()?;
                if value["agent_name"].as_str() == Some(agent_name) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    fn filter_by_success(&self, success: bool) -> Vec<String> {
        self.traces
            .read()
            .iter()
            .filter_map(|(id, json)| {
                let value: serde_json::Value = serde_json::from_str(json).ok()?;
                if value["success"].as_bool() == Some(success) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_session_json() -> String {
        serde_json::json!({
            "id": "session-1",
            "user_id": "user-123",
            "created_at": 1700000000,
            "updated_at": 1700001000,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "turns": [
                {
                    "number": 1,
                    "user_message": {"role": "user", "content": "Hello"},
                    "assistant_response": {"role": "assistant", "content": "Hi there!"}
                }
            ]
        })
        .to_string()
    }

    fn sample_trace_json() -> String {
        serde_json::json!({
            "task_id": "task-1",
            "agent_name": "research-agent",
            "success": true,
            "total_duration_ms": 1500,
            "steps": [
                {"step_type": "llm_call", "duration_ms": 500, "success": true},
                {"step_type": "tool_call", "duration_ms": 200, "success": true},
                {"step_type": "llm_call", "duration_ms": 600, "success": true}
            ],
            "metadata": {}
        })
        .to_string()
    }

    #[tokio::test]
    async fn test_session_resource_list() {
        let store = Arc::new(MemorySessionStoreAdapter::new());
        store.store("session-1", &sample_session_json());
        store.store("session-2", &sample_session_json());

        let handler = SessionResourceHandler::new(store);
        let resources = handler.list();

        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].uri, "session://list");
    }

    #[tokio::test]
    async fn test_session_resource_read_list() {
        let store = Arc::new(MemorySessionStoreAdapter::new());
        store.store("session-1", &sample_session_json());

        let handler = SessionResourceHandler::new(store);
        let content = handler.read("session://list").await.unwrap();

        assert!(content.text.is_some());
        let text = content.text.unwrap();
        assert!(text.contains("session-1"));
    }

    #[tokio::test]
    async fn test_session_resource_read_session() {
        let store = Arc::new(MemorySessionStoreAdapter::new());
        store.store("session-1", &sample_session_json());

        let handler = SessionResourceHandler::new(store);
        let content = handler.read("session://session-1").await.unwrap();

        assert!(content.text.is_some());
        let text = content.text.unwrap();
        assert!(text.contains("user-123"));
    }

    #[tokio::test]
    async fn test_session_resource_read_messages() {
        let store = Arc::new(MemorySessionStoreAdapter::new());
        store.store("session-1", &sample_session_json());

        let handler = SessionResourceHandler::new(store);
        let content = handler.read("session://session-1/messages").await.unwrap();

        assert!(content.text.is_some());
        let text = content.text.unwrap();
        assert!(text.contains("Hello"));
        assert!(text.contains("Hi there!"));
    }

    #[tokio::test]
    async fn test_session_resource_read_turn() {
        let store = Arc::new(MemorySessionStoreAdapter::new());
        store.store("session-1", &sample_session_json());

        let handler = SessionResourceHandler::new(store);
        let content = handler.read("session://session-1/turns/1").await.unwrap();

        assert!(content.text.is_some());
        let text = content.text.unwrap();
        assert!(text.contains("Hello"));
    }

    #[tokio::test]
    async fn test_session_resource_not_found() {
        let store = Arc::new(MemorySessionStoreAdapter::new());
        let handler = SessionResourceHandler::new(store);

        let result = handler.read("session://nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_trace_resource_list() {
        let store = Arc::new(MemoryTraceStoreAdapter::new());
        store.store("task-1", &sample_trace_json());

        let handler = TraceResourceHandler::new(store);
        let resources = handler.list();

        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].uri, "trace://list");
    }

    #[tokio::test]
    async fn test_trace_resource_read_list() {
        let store = Arc::new(MemoryTraceStoreAdapter::new());
        store.store("task-1", &sample_trace_json());

        let handler = TraceResourceHandler::new(store);
        let content = handler.read("trace://list").await.unwrap();

        assert!(content.text.is_some());
        let text = content.text.unwrap();
        assert!(text.contains("task-1"));
        assert!(text.contains("research-agent"));
    }

    #[tokio::test]
    async fn test_trace_resource_read_trace() {
        let store = Arc::new(MemoryTraceStoreAdapter::new());
        store.store("task-1", &sample_trace_json());

        let handler = TraceResourceHandler::new(store);
        let content = handler.read("trace://task-1").await.unwrap();

        assert!(content.text.is_some());
        let text = content.text.unwrap();
        assert!(text.contains("research-agent"));
        assert!(text.contains("1500"));
    }

    #[tokio::test]
    async fn test_trace_resource_read_steps() {
        let store = Arc::new(MemoryTraceStoreAdapter::new());
        store.store("task-1", &sample_trace_json());

        let handler = TraceResourceHandler::new(store);
        let content = handler.read("trace://task-1/steps").await.unwrap();

        assert!(content.text.is_some());
        let text = content.text.unwrap();
        assert!(text.contains("llm_call"));
        assert!(text.contains("tool_call"));
    }

    #[tokio::test]
    async fn test_trace_resource_read_summary() {
        let store = Arc::new(MemoryTraceStoreAdapter::new());
        store.store("task-1", &sample_trace_json());

        let handler = TraceResourceHandler::new(store);
        let content = handler.read("trace://task-1/summary").await.unwrap();

        assert!(content.text.is_some());
        let text = content.text.unwrap();
        assert!(text.contains("llm_calls"));
        assert!(text.contains("tool_calls"));
    }

    #[tokio::test]
    async fn test_trace_filter_by_agent() {
        let store = Arc::new(MemoryTraceStoreAdapter::new());
        store.store("task-1", &sample_trace_json());
        store.store(
            "task-2",
            &serde_json::json!({
                "task_id": "task-2",
                "agent_name": "other-agent",
                "success": true,
                "total_duration_ms": 500,
                "steps": []
            })
            .to_string(),
        );

        let handler = TraceResourceHandler::new(store).with_agent_filter("research-agent");
        let content = handler.read("trace://list").await.unwrap();

        let text = content.text.unwrap();
        assert!(text.contains("task-1"));
        assert!(!text.contains("task-2"));
    }

    #[tokio::test]
    async fn test_trace_filter_by_success() {
        let store = Arc::new(MemoryTraceStoreAdapter::new());
        store.store("task-1", &sample_trace_json()); // success: true
        store.store(
            "task-2",
            &serde_json::json!({
                "task_id": "task-2",
                "agent_name": "agent",
                "success": false,
                "total_duration_ms": 500,
                "steps": []
            })
            .to_string(),
        );

        let handler = TraceResourceHandler::new(store).with_success_filter(false);
        let content = handler.read("trace://list").await.unwrap();

        let text = content.text.unwrap();
        assert!(!text.contains("task-1"));
        assert!(text.contains("task-2"));
    }

    #[test]
    fn test_session_uri_parsing() {
        assert!(matches!(
            SessionResourceHandler::<MemorySessionStoreAdapter>::parse_uri("session://list"),
            Some(SessionUriParts::List)
        ));

        assert!(matches!(
            SessionResourceHandler::<MemorySessionStoreAdapter>::parse_uri("session://abc"),
            Some(SessionUriParts::Session(id)) if id == "abc"
        ));

        assert!(matches!(
            SessionResourceHandler::<MemorySessionStoreAdapter>::parse_uri("session://abc/messages"),
            Some(SessionUriParts::Messages(id)) if id == "abc"
        ));

        assert!(matches!(
            SessionResourceHandler::<MemorySessionStoreAdapter>::parse_uri("session://abc/turns/1"),
            Some(SessionUriParts::Turn(id, 1)) if id == "abc"
        ));

        assert!(
            SessionResourceHandler::<MemorySessionStoreAdapter>::parse_uri("invalid").is_none()
        );
    }

    #[test]
    fn test_trace_uri_parsing() {
        assert!(matches!(
            TraceResourceHandler::<MemoryTraceStoreAdapter>::parse_uri("trace://list"),
            Some(TraceUriParts::List)
        ));

        assert!(matches!(
            TraceResourceHandler::<MemoryTraceStoreAdapter>::parse_uri("trace://task-1"),
            Some(TraceUriParts::Trace(id)) if id == "task-1"
        ));

        assert!(matches!(
            TraceResourceHandler::<MemoryTraceStoreAdapter>::parse_uri("trace://task-1/steps"),
            Some(TraceUriParts::Steps(id)) if id == "task-1"
        ));

        assert!(matches!(
            TraceResourceHandler::<MemoryTraceStoreAdapter>::parse_uri("trace://task-1/summary"),
            Some(TraceUriParts::Summary(id)) if id == "task-1"
        ));
    }
}
