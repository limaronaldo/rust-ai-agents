//! Audit log types and schemas for compliance and debugging.
//!
//! This module defines the core types used throughout the audit system:
//! - `AuditEvent`: The main event structure
//! - `AuditLevel`: Severity/importance levels
//! - `EventKind`: Types of auditable events
//! - `AuditContext`: Request context and metadata

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Severity level for audit events.
///
/// Used to filter and categorize events based on importance.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum AuditLevel {
    /// Detailed debugging information
    Debug,
    /// General informational events
    #[default]
    Info,
    /// Warning conditions
    Warn,
    /// Error conditions
    Error,
    /// Critical issues requiring immediate attention
    Critical,
}

impl std::fmt::Display for AuditLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Debug => write!(f, "DEBUG"),
            Self::Info => write!(f, "INFO"),
            Self::Warn => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// The kind of event being audited.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    /// Tool invocation event
    ToolCall {
        /// Name of the tool being called
        tool_name: String,
        /// Input parameters (may be redacted)
        parameters: serde_json::Value,
        /// Tool output (may be redacted or truncated)
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<serde_json::Value>,
        /// Whether the call was approved
        approved: bool,
        /// Duration in milliseconds
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    /// LLM request event
    LlmRequest {
        /// Provider name (openai, anthropic, etc.)
        provider: String,
        /// Model identifier
        model: String,
        /// Number of input tokens
        #[serde(skip_serializing_if = "Option::is_none")]
        input_tokens: Option<u32>,
        /// Number of output tokens
        #[serde(skip_serializing_if = "Option::is_none")]
        output_tokens: Option<u32>,
        /// Request duration in milliseconds
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        /// Whether streaming was used
        streaming: bool,
    },
    /// LLM response event
    LlmResponse {
        /// Provider name
        provider: String,
        /// Model identifier
        model: String,
        /// Finish reason (stop, length, tool_use, etc.)
        #[serde(skip_serializing_if = "Option::is_none")]
        finish_reason: Option<String>,
        /// Number of tool calls in response
        tool_calls_count: usize,
    },
    /// Agent lifecycle event
    AgentLifecycle {
        /// Agent identifier
        agent_id: String,
        /// Lifecycle action
        action: LifecycleAction,
    },
    /// Approval decision event
    ApprovalDecision {
        /// Tool that required approval
        tool_name: String,
        /// Whether it was approved
        approved: bool,
        /// Reason for decision
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        /// Who/what made the decision
        approver: String,
    },
    /// Error event
    Error {
        /// Error code or type
        error_type: String,
        /// Error message
        message: String,
        /// Stack trace if available
        #[serde(skip_serializing_if = "Option::is_none")]
        stack_trace: Option<String>,
    },
    /// Security-related event
    Security {
        /// Type of security event
        event_type: SecurityEventType,
        /// Description
        description: String,
    },
    /// Custom event for extensibility
    Custom {
        /// Event name
        name: String,
        /// Custom payload
        payload: serde_json::Value,
    },
}

/// Agent lifecycle actions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleAction {
    /// Agent started
    Started,
    /// Agent completed successfully
    Completed,
    /// Agent failed
    Failed,
    /// Agent was cancelled
    Cancelled,
    /// Agent paused
    Paused,
    /// Agent resumed
    Resumed,
}

/// Security event types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityEventType {
    /// Authentication attempt
    AuthAttempt,
    /// Authorization failure
    AuthzFailure,
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Suspicious activity detected
    SuspiciousActivity,
    /// Data access event
    DataAccess,
    /// Configuration change
    ConfigChange,
}

/// Context information for an audit event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditContext {
    /// Unique request/trace ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Session identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// User identifier (may be anonymized)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Agent identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Source IP address (may be anonymized)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ip_address: Option<String>,
    /// User agent string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    /// Additional metadata
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AuditContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the trace ID.
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Set the session ID.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the user ID.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set the agent ID.
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// The main audit event structure.
///
/// Each audit event captures a single auditable action with full context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event identifier
    pub id: Uuid,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Severity level
    pub level: AuditLevel,
    /// The kind of event
    pub kind: EventKind,
    /// Event context
    pub context: AuditContext,
    /// Schema version for forward compatibility
    pub schema_version: u32,
}

impl AuditEvent {
    /// Current schema version.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Create a new audit event.
    pub fn new(level: AuditLevel, kind: EventKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            level,
            kind,
            context: AuditContext::default(),
            schema_version: Self::SCHEMA_VERSION,
        }
    }

    /// Create a debug-level event.
    pub fn debug(kind: EventKind) -> Self {
        Self::new(AuditLevel::Debug, kind)
    }

    /// Create an info-level event.
    pub fn info(kind: EventKind) -> Self {
        Self::new(AuditLevel::Info, kind)
    }

    /// Create a warn-level event.
    pub fn warn(kind: EventKind) -> Self {
        Self::new(AuditLevel::Warn, kind)
    }

    /// Create an error-level event.
    pub fn error(kind: EventKind) -> Self {
        Self::new(AuditLevel::Error, kind)
    }

    /// Create a critical-level event.
    pub fn critical(kind: EventKind) -> Self {
        Self::new(AuditLevel::Critical, kind)
    }

    /// Set the context for this event.
    pub fn with_context(mut self, context: AuditContext) -> Self {
        self.context = context;
        self
    }

    /// Convenience: create a tool call event.
    pub fn tool_call(
        tool_name: impl Into<String>,
        parameters: serde_json::Value,
        approved: bool,
    ) -> Self {
        Self::info(EventKind::ToolCall {
            tool_name: tool_name.into(),
            parameters,
            result: None,
            approved,
            duration_ms: None,
        })
    }

    /// Convenience: create an LLM request event.
    pub fn llm_request(
        provider: impl Into<String>,
        model: impl Into<String>,
        streaming: bool,
    ) -> Self {
        Self::info(EventKind::LlmRequest {
            provider: provider.into(),
            model: model.into(),
            input_tokens: None,
            output_tokens: None,
            duration_ms: None,
            streaming,
        })
    }

    /// Convenience: create an LLM response event.
    pub fn llm_response(
        provider: impl Into<String>,
        model: impl Into<String>,
        tool_calls_count: usize,
        finish_reason: Option<impl Into<String>>,
    ) -> Self {
        Self::info(EventKind::LlmResponse {
            provider: provider.into(),
            model: model.into(),
            finish_reason: finish_reason.map(|r| r.into()),
            tool_calls_count,
        })
    }

    /// Convenience: create an error event.
    pub fn error_event(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::error(EventKind::Error {
            error_type: error_type.into(),
            message: message.into(),
            stack_trace: None,
        })
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to pretty JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Builder for creating audit events with fluent API.
#[derive(Debug, Default)]
pub struct AuditEventBuilder {
    level: Option<AuditLevel>,
    kind: Option<EventKind>,
    context: AuditContext,
}

impl AuditEventBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the level.
    pub fn level(mut self, level: AuditLevel) -> Self {
        self.level = Some(level);
        self
    }

    /// Set the event kind.
    pub fn kind(mut self, kind: EventKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Set the context.
    pub fn context(mut self, context: AuditContext) -> Self {
        self.context = context;
        self
    }

    /// Set trace ID in context.
    pub fn trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.context.trace_id = Some(trace_id.into());
        self
    }

    /// Set session ID in context.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.context.session_id = Some(session_id.into());
        self
    }

    /// Set agent ID in context.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.context.agent_id = Some(agent_id.into());
        self
    }

    /// Build the audit event.
    ///
    /// Panics if level or kind is not set.
    pub fn build(self) -> AuditEvent {
        AuditEvent {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            level: self.level.expect("level is required"),
            kind: self.kind.expect("kind is required"),
            context: self.context,
            schema_version: AuditEvent::SCHEMA_VERSION,
        }
    }

    /// Try to build the audit event.
    pub fn try_build(self) -> Option<AuditEvent> {
        Some(AuditEvent {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            level: self.level?,
            kind: self.kind?,
            context: self.context,
            schema_version: AuditEvent::SCHEMA_VERSION,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_level_ordering() {
        assert!(AuditLevel::Debug < AuditLevel::Info);
        assert!(AuditLevel::Info < AuditLevel::Warn);
        assert!(AuditLevel::Warn < AuditLevel::Error);
        assert!(AuditLevel::Error < AuditLevel::Critical);
    }

    #[test]
    fn test_audit_event_serialization() {
        let event =
            AuditEvent::tool_call("read_file", serde_json::json!({"path": "/tmp/test"}), true);
        let json = event.to_json().unwrap();

        assert!(json.contains("tool_call"));
        assert!(json.contains("read_file"));
        assert!(json.contains("approved"));
    }

    #[test]
    fn test_audit_context_builder() {
        let context = AuditContext::new()
            .with_trace_id("trace-123")
            .with_session_id("session-456")
            .with_user_id("user-789")
            .with_metadata("env", serde_json::json!("production"));

        assert_eq!(context.trace_id, Some("trace-123".to_string()));
        assert_eq!(context.session_id, Some("session-456".to_string()));
        assert_eq!(context.user_id, Some("user-789".to_string()));
        assert!(context.metadata.contains_key("env"));
    }

    #[test]
    fn test_event_builder() {
        let event = AuditEventBuilder::new()
            .level(AuditLevel::Warn)
            .kind(EventKind::Security {
                event_type: SecurityEventType::RateLimitExceeded,
                description: "Too many requests".to_string(),
            })
            .trace_id("trace-abc")
            .build();

        assert_eq!(event.level, AuditLevel::Warn);
        assert_eq!(event.context.trace_id, Some("trace-abc".to_string()));
    }

    #[test]
    fn test_llm_request_event() {
        let event = AuditEvent::llm_request("anthropic", "claude-3-opus", true);

        if let EventKind::LlmRequest {
            provider,
            model,
            streaming,
            ..
        } = &event.kind
        {
            assert_eq!(provider, "anthropic");
            assert_eq!(model, "claude-3-opus");
            assert!(*streaming);
        } else {
            panic!("Expected LlmRequest event kind");
        }
    }
}
