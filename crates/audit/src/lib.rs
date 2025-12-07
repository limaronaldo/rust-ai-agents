//! Audit logging for AI agents.
//!
//! This crate provides comprehensive audit logging for all tool calls,
//! LLM requests, and agent lifecycle events. It supports multiple backends
//! and is designed for compliance and debugging in production environments.
//!
//! # Features
//!
//! - **Structured Events**: Rich event types for tool calls, LLM requests, errors, and more
//! - **Multiple Backends**: File, JSON, and async logging with rotation support
//! - **Configurable Levels**: Filter events by severity (Debug, Info, Warn, Error, Critical)
//! - **Log Rotation**: Size-based, daily, or hourly rotation with cleanup
//! - **Non-blocking**: Async wrapper for high-throughput scenarios
//! - **Compliance Ready**: Designed for GDPR, SOC2, and enterprise requirements
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use rust_ai_agents_audit::{
//!     AuditEvent, AuditContext, JsonFileLogger, RotationConfig, RotationPolicy, AuditLogger
//! };
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a JSON file logger with daily rotation
//! let rotation = RotationConfig::new(RotationPolicy::Daily)
//!     .with_max_files(30);
//!
//! let logger = JsonFileLogger::new(
//!     "/var/log/agents/audit.jsonl",
//!     Default::default(),
//!     rotation,
//! ).await?;
//!
//! // Log a tool call
//! let event = AuditEvent::tool_call("read_file", serde_json::json!({"path": "/tmp/data.txt"}), true)
//!     .with_context(
//!         AuditContext::new()
//!             .with_trace_id("req-123")
//!             .with_agent_id("research-agent")
//!     );
//!
//! logger.log(event).await?;
//! logger.flush().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Event Types
//!
//! The crate supports various event types through `EventKind`:
//!
//! - `ToolCall`: Tool invocations with parameters and results
//! - `LlmRequest`: Outgoing LLM API requests
//! - `LlmResponse`: Incoming LLM responses
//! - `AgentLifecycle`: Agent start, stop, pause, resume events
//! - `ApprovalDecision`: Human-in-the-loop approval decisions
//! - `Error`: Error events with stack traces
//! - `Security`: Security-related events (auth, rate limits)
//! - `Custom`: Extensible custom events
//!
//! # Backends
//!
//! ## FileLogger
//! Simple human-readable text format, good for development.
//!
//! ## JsonFileLogger
//! Structured JSON Lines format with rotation, ideal for production.
//!
//! ## AsyncLogger
//! Non-blocking wrapper around any logger for high-throughput scenarios.
//!
//! ## MemoryLogger
//! In-memory storage for testing.
//!
//! ## CompositeLogger
//! Writes to multiple backends simultaneously.
//!
//! # Configuration
//!
//! ```rust
//! use rust_ai_agents_audit::{AuditConfig, AuditLevel};
//!
//! // Development config - verbose logging
//! let dev_config = AuditConfig::development();
//!
//! // Production config - info level with redaction
//! let prod_config = AuditConfig::production();
//!
//! // Custom config
//! let custom_config = AuditConfig::new()
//!     .with_min_level(AuditLevel::Warn)
//!     .with_redaction(true)
//!     .with_max_payload_size(5 * 1024);
//! ```

pub mod backends;
pub mod error;
pub mod traits;
pub mod types;

// Re-export main types
pub use error::AuditError;
pub use traits::{AuditConfig, AuditLogger, AuditStats, CompositeLogger, MemoryLogger, NoOpLogger};
pub use types::{
    AuditContext, AuditEvent, AuditEventBuilder, AuditLevel, EventKind, LifecycleAction,
    SecurityEventType,
};

// Re-export backends
pub use backends::{
    AsyncLogger, AsyncLoggerBuilder, FileLogger, JsonFileLogger, RotationConfig, RotationPolicy,
};

/// Convenience function to create a production-ready logger.
///
/// Creates a JSON file logger with:
/// - Daily rotation
/// - 30 days retention
/// - Production-level configuration
pub async fn create_production_logger(
    path: impl Into<std::path::PathBuf>,
) -> Result<JsonFileLogger, AuditError> {
    let rotation = RotationConfig::new(RotationPolicy::Daily).with_max_files(30);
    JsonFileLogger::new(path, AuditConfig::production(), rotation).await
}

/// Convenience function to create a development logger.
///
/// Creates a file logger with debug-level logging.
pub async fn create_dev_logger(
    path: impl Into<std::path::PathBuf>,
) -> Result<FileLogger, AuditError> {
    FileLogger::new(path, AuditConfig::development()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_logger_integration() {
        let logger = MemoryLogger::new();

        // Log various events
        logger
            .log(AuditEvent::tool_call("tool1", serde_json::json!({}), true))
            .await
            .unwrap();
        logger
            .log(AuditEvent::llm_request("anthropic", "claude-3", false))
            .await
            .unwrap();
        logger
            .log(AuditEvent::error_event(
                "RuntimeError",
                "Something went wrong",
            ))
            .await
            .unwrap();

        assert_eq!(logger.count().await, 3);

        let events = logger.events().await;
        assert!(matches!(events[0].kind, EventKind::ToolCall { .. }));
        assert!(matches!(events[1].kind, EventKind::LlmRequest { .. }));
        assert!(matches!(events[2].kind, EventKind::Error { .. }));
    }

    #[test]
    fn test_event_builder() {
        let event = AuditEventBuilder::new()
            .level(AuditLevel::Info)
            .kind(EventKind::Custom {
                name: "test".to_string(),
                payload: serde_json::json!({"key": "value"}),
            })
            .trace_id("trace-1")
            .session_id("session-1")
            .agent_id("agent-1")
            .build();

        assert_eq!(event.level, AuditLevel::Info);
        assert_eq!(event.context.trace_id, Some("trace-1".to_string()));
        assert_eq!(event.context.session_id, Some("session-1".to_string()));
        assert_eq!(event.context.agent_id, Some("agent-1".to_string()));
    }

    #[test]
    fn test_config_presets() {
        let dev = AuditConfig::development();
        assert_eq!(dev.min_level, AuditLevel::Debug);
        assert!(dev.include_debug);
        assert!(!dev.redact_sensitive);

        let prod = AuditConfig::production();
        assert_eq!(prod.min_level, AuditLevel::Info);
        assert!(!prod.include_debug);
        assert!(prod.redact_sensitive);
    }
}
