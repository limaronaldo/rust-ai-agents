//! Shared types for the API

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Agent status information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentStatus {
    pub id: String,
    pub name: String,
    pub role: String,
    pub status: String,
    pub messages_processed: u64,
    pub last_activity: Option<DateTime<Utc>>,
    pub current_task: Option<String>,
}

/// Cost statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CostStats {
    pub total_cost: f64,
    pub total_tokens: u64,
    pub total_requests: u64,
    pub avg_latency_ms: f64,
}

/// Dashboard metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DashboardMetrics {
    pub timestamp: DateTime<Utc>,
    pub cost_stats: CostStats,
    pub agents: Vec<AgentStatus>,
    pub active_agents: usize,
    pub total_messages: u64,
    pub uptime_seconds: u64,
}

/// Trace entry representing a single step in agent execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceEntry {
    pub id: String,
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    pub entry_type: TraceEntryType,
    pub duration_ms: Option<f64>,
    pub metadata: Option<serde_json::Value>,
}

/// Type of trace entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Completed,
    Failed,
    Archived,
}

/// Message in a session
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionMessage {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// WebSocket message types (matching dashboard)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessage {
    Metrics(DashboardMetrics),
    AgentUpdate(AgentStatus),
    RequestRecorded {
        model: String,
        cost: f64,
        tokens: u64,
        latency_ms: f64,
    },
    TraceUpdate(TraceEntry),
    SessionUpdate(Session),
    Error {
        message: String,
    },
    Ping,
    Pong,
}
