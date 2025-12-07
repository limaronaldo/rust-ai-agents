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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_status_serialization() {
        let agent = AgentStatus {
            id: "agent-1".to_string(),
            name: "Test Agent".to_string(),
            role: "assistant".to_string(),
            status: "running".to_string(),
            messages_processed: 42,
            last_activity: None,
            current_task: Some("Processing".to_string()),
        };

        let json = serde_json::to_string(&agent).unwrap();
        let deserialized: AgentStatus = serde_json::from_str(&json).unwrap();

        assert_eq!(agent, deserialized);
    }

    #[test]
    fn test_cost_stats_default() {
        let stats = CostStats::default();

        assert_eq!(stats.total_cost, 0.0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.avg_latency_ms, 0.0);
    }

    #[test]
    fn test_session_status_serialization() {
        let statuses = vec![
            SessionStatus::Active,
            SessionStatus::Completed,
            SessionStatus::Failed,
            SessionStatus::Archived,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let deserialized: SessionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_trace_entry_type_llm_request() {
        let entry = TraceEntryType::LlmRequest {
            model: "gpt-4".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            cost: 0.005,
        };

        let json = serde_json::to_string(&entry).unwrap();
        // Uses #[serde(tag = "type", content = "data")] so variant is PascalCase
        assert!(json.contains("LlmRequest"));
        assert!(json.contains("gpt-4"));
    }

    #[test]
    fn test_trace_entry_type_tool_call() {
        let entry = TraceEntryType::ToolCall {
            tool_name: "search".to_string(),
            arguments: serde_json::json!({"query": "rust"}),
        };

        let json = serde_json::to_string(&entry).unwrap();
        // Uses #[serde(tag = "type", content = "data")] so variant is PascalCase
        assert!(json.contains("ToolCall"));
        assert!(json.contains("search"));
    }

    #[test]
    fn test_ws_message_ping_pong() {
        let ping = WsMessage::Ping;
        let pong = WsMessage::Pong;

        let ping_json = serde_json::to_string(&ping).unwrap();
        let pong_json = serde_json::to_string(&pong).unwrap();

        assert!(ping_json.contains("Ping"));
        assert!(pong_json.contains("Pong"));
    }

    #[test]
    fn test_ws_message_request_recorded() {
        let msg = WsMessage::RequestRecorded {
            model: "claude-3".to_string(),
            cost: 0.01,
            tokens: 500,
            latency_ms: 150.0,
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("RequestRecorded"));
        assert!(json.contains("claude-3"));
    }

    #[test]
    fn test_session_message_serialization() {
        let msg = SessionMessage {
            id: "msg-1".to_string(),
            session_id: "session-1".to_string(),
            role: "user".to_string(),
            content: "Hello!".to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: SessionMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(msg.id, deserialized.id);
        assert_eq!(msg.content, deserialized.content);
    }
}
