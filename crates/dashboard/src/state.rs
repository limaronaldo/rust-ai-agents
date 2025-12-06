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
            total_messages: RwLock::new(0),
            started_at: Utc::now(),
            broadcast_tx,
        }
    }

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
