//! Dashboard state management

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rust_ai_agents_monitoring::{CostStats, CostTracker};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Agent status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub id: String,
    pub name: String,
    pub role: String,
    pub status: String,
    pub messages_processed: u64,
    pub last_activity: Option<DateTime<Utc>>,
    pub current_task: Option<String>,
}

/// Trace entry representing a single step in agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub id: String,
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    pub entry_type: TraceEntryType,
    pub duration_ms: Option<f64>,
    pub metadata: Option<serde_json::Value>,
}

/// Type of trace entry
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum TraceEntryType {
    LlmRequest {
        model: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        cost: f64,
    },
    LlmResponse {
        content: String,
        finish_reason: Option<String>,
    },
    ToolCall {
        tool_name: String,
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_name: String,
        result: serde_json::Value,
        success: bool,
    },
    AgentThought {
        thought: String,
    },
    Error {
        message: String,
        error_type: String,
    },
}

/// Session representing a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: u32,
    pub status: SessionStatus,
    pub agent_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Completed,
    Failed,
    Archived,
}

/// Message in a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Dashboard metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub timestamp: DateTime<Utc>,
    pub cost_stats: CostStats,
    pub agents: Vec<AgentStatus>,
    pub active_agents: usize,
    pub total_messages: u64,
    pub uptime_seconds: u64,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessage {
    /// Full metrics update
    Metrics(DashboardMetrics),
    /// Agent status changed
    AgentUpdate(AgentStatus),
    /// New request recorded
    RequestRecorded {
        model: String,
        cost: f64,
        tokens: u64,
        latency_ms: f64,
    },
    /// Trace update
    TraceUpdate(TraceEntry),
    /// Session update
    SessionUpdate(Session),
    /// Error occurred
    Error {
        message: String,
    },
    /// Ping/pong for keepalive
    Ping,
    Pong,
}

/// Shared dashboard state
pub struct DashboardState {
    /// Cost tracker reference
    pub cost_tracker: Arc<CostTracker>,
    /// Active agents
    agents: RwLock<HashMap<String, AgentStatus>>,
    /// Sessions
    sessions: RwLock<HashMap<String, Session>>,
    /// Session messages
    session_messages: RwLock<HashMap<String, Vec<SessionMessage>>>,
    /// Traces
    traces: RwLock<Vec<TraceEntry>>,
    /// Total messages processed
    total_messages: RwLock<u64>,
    /// Server start time
    started_at: DateTime<Utc>,
    /// Broadcast channel for WebSocket updates
    pub broadcast_tx: broadcast::Sender<WsMessage>,
}

impl DashboardState {
    /// Create new dashboard state
    pub fn new(cost_tracker: Arc<CostTracker>) -> Self {
        let (broadcast_tx, _) = broadcast::channel(1024);

        Self {
            cost_tracker,
            agents: RwLock::new(HashMap::new()),
            sessions: RwLock::new(HashMap::new()),
            session_messages: RwLock::new(HashMap::new()),
            traces: RwLock::new(Vec::new()),
            total_messages: RwLock::new(0),
            started_at: Utc::now(),
            broadcast_tx,
        }
    }

    // ==================== Agents ====================

    /// Register or update an agent
    pub fn update_agent(&self, status: AgentStatus) {
        self.agents
            .write()
            .insert(status.id.clone(), status.clone());
        let _ = self.broadcast_tx.send(WsMessage::AgentUpdate(status));
    }

    /// Remove an agent
    pub fn remove_agent(&self, agent_id: &str) {
        self.agents.write().remove(agent_id);
    }

    /// Get all agents
    pub fn get_agents(&self) -> Vec<AgentStatus> {
        self.agents.read().values().cloned().collect()
    }

    /// Get a specific agent
    pub fn get_agent(&self, id: &str) -> Option<AgentStatus> {
        self.agents.read().get(id).cloned()
    }

    /// Start an agent
    pub fn start_agent(&self, id: &str) -> Result<(), String> {
        let mut agents = self.agents.write();
        if let Some(agent) = agents.get_mut(id) {
            agent.status = "running".to_string();
            agent.last_activity = Some(Utc::now());
            let _ = self
                .broadcast_tx
                .send(WsMessage::AgentUpdate(agent.clone()));
            Ok(())
        } else {
            Err(format!("Agent {} not found", id))
        }
    }

    /// Stop an agent
    pub fn stop_agent(&self, id: &str) -> Result<(), String> {
        let mut agents = self.agents.write();
        if let Some(agent) = agents.get_mut(id) {
            agent.status = "stopped".to_string();
            agent.current_task = None;
            agent.last_activity = Some(Utc::now());
            let _ = self
                .broadcast_tx
                .send(WsMessage::AgentUpdate(agent.clone()));
            Ok(())
        } else {
            Err(format!("Agent {} not found", id))
        }
    }

    /// Restart an agent
    pub fn restart_agent(&self, id: &str) -> Result<(), String> {
        self.stop_agent(id)?;
        self.start_agent(id)
    }

    // ==================== Sessions ====================

    /// Add or update a session
    pub fn update_session(&self, session: Session) {
        self.sessions
            .write()
            .insert(session.id.clone(), session.clone());
        let _ = self.broadcast_tx.send(WsMessage::SessionUpdate(session));
    }

    /// Get all sessions
    pub fn get_sessions(&self) -> Vec<Session> {
        let mut sessions: Vec<Session> = self.sessions.read().values().cloned().collect();
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        sessions
    }

    /// Get a specific session
    pub fn get_session(&self, id: &str) -> Option<Session> {
        self.sessions.read().get(id).cloned()
    }

    /// Add a message to a session
    pub fn add_session_message(&self, message: SessionMessage) {
        let session_id = message.session_id.clone();
        self.session_messages
            .write()
            .entry(session_id.clone())
            .or_default()
            .push(message);

        // Update session message count
        if let Some(session) = self.sessions.write().get_mut(&session_id) {
            session.message_count += 1;
            session.updated_at = Utc::now();
        }
    }

    /// Get messages for a session
    pub fn get_session_messages(&self, session_id: &str) -> Vec<SessionMessage> {
        self.session_messages
            .read()
            .get(session_id)
            .cloned()
            .unwrap_or_default()
    }

    // ==================== Traces ====================

    /// Add a trace entry
    pub fn add_trace(&self, trace: TraceEntry) {
        self.traces.write().push(trace.clone());
        let _ = self.broadcast_tx.send(WsMessage::TraceUpdate(trace));
    }

    /// Get all traces
    pub fn get_traces(&self) -> Vec<TraceEntry> {
        let mut traces = self.traces.read().clone();
        traces.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        traces.truncate(1000); // Limit to last 1000 traces
        traces
    }

    /// Get traces for a specific session
    pub fn get_session_traces(&self, session_id: &str) -> Vec<TraceEntry> {
        self.traces
            .read()
            .iter()
            .filter(|t| t.session_id == session_id)
            .cloned()
            .collect()
    }

    // ==================== Messages & Metrics ====================

    /// Increment message counter
    pub fn record_message(&self) {
        *self.total_messages.write() += 1;
    }

    /// Record a new LLM request (broadcasts to clients)
    pub fn record_request(&self, model: &str, cost: f64, tokens: u64, latency_ms: f64) {
        let _ = self.broadcast_tx.send(WsMessage::RequestRecorded {
            model: model.to_string(),
            cost,
            tokens,
            latency_ms,
        });
    }

    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> DashboardMetrics {
        let agents: Vec<AgentStatus> = self.agents.read().values().cloned().collect();
        let active_agents = agents.iter().filter(|a| a.status == "running").count();
        let uptime = Utc::now()
            .signed_duration_since(self.started_at)
            .num_seconds() as u64;

        DashboardMetrics {
            timestamp: Utc::now(),
            cost_stats: self.cost_tracker.stats(),
            agents,
            active_agents,
            total_messages: *self.total_messages.read(),
            uptime_seconds: uptime,
        }
    }

    /// Subscribe to updates
    pub fn subscribe(&self) -> broadcast::Receiver<WsMessage> {
        self.broadcast_tx.subscribe()
    }

    /// Broadcast metrics to all connected clients
    pub fn broadcast_metrics(&self) {
        let metrics = self.get_metrics();
        let _ = self.broadcast_tx.send(WsMessage::Metrics(metrics));
    }
}
