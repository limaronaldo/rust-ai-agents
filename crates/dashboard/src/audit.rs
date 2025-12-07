//! Audit logging middleware and API endpoints for the dashboard.
//!
//! Provides:
//! - Request/response audit logging middleware
//! - API endpoints for querying audit logs
//! - Real-time audit event streaming via WebSocket

use axum::{
    body::Body,
    extract::{Query, State},
    http::Request,
    middleware::Next,
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use rust_ai_agents_audit::{AuditContext, AuditEvent, AuditLogger, EventKind, MemoryLogger};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// State for audit endpoints
pub struct AuditState {
    /// The audit logger (using MemoryLogger for queryable in-memory storage)
    pub logger: Arc<MemoryLogger>,
}

impl AuditState {
    /// Create new audit state with a memory logger for dashboard queries.
    pub fn new() -> Self {
        Self {
            logger: Arc::new(MemoryLogger::new()),
        }
    }

    /// Create with an existing memory logger.
    pub fn with_logger(logger: Arc<MemoryLogger>) -> Self {
        Self { logger }
    }
}

impl Default for AuditState {
    fn default() -> Self {
        Self::new()
    }
}

/// Query parameters for listing audit events
#[derive(Debug, Deserialize)]
pub struct AuditQuery {
    /// Filter by event type
    pub event_type: Option<String>,
    /// Filter by trace ID
    pub trace_id: Option<String>,
    /// Filter by session ID
    pub session_id: Option<String>,
    /// Filter by agent ID
    pub agent_id: Option<String>,
    /// Maximum number of events to return
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

/// Audit event response for API
#[derive(Debug, Serialize)]
pub struct AuditEventResponse {
    pub id: String,
    pub timestamp: String,
    pub level: String,
    pub event_type: String,
    pub details: serde_json::Value,
    pub context: AuditContextResponse,
}

/// Audit context response
#[derive(Debug, Serialize)]
pub struct AuditContextResponse {
    pub trace_id: Option<String>,
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub user_id: Option<String>,
}

impl From<&AuditEvent> for AuditEventResponse {
    fn from(event: &AuditEvent) -> Self {
        let event_type = match &event.kind {
            EventKind::ToolCall { .. } => "tool_call",
            EventKind::LlmRequest { .. } => "llm_request",
            EventKind::LlmResponse { .. } => "llm_response",
            EventKind::AgentLifecycle { .. } => "agent_lifecycle",
            EventKind::ApprovalDecision { .. } => "approval_decision",
            EventKind::Error { .. } => "error",
            EventKind::Security { .. } => "security",
            EventKind::Custom { .. } => "custom",
        };

        let details = serde_json::to_value(&event.kind).unwrap_or_default();

        Self {
            id: event.id.to_string(),
            timestamp: event.timestamp.to_rfc3339(),
            level: format!("{:?}", event.level),
            event_type: event_type.to_string(),
            details,
            context: AuditContextResponse {
                trace_id: event.context.trace_id.clone(),
                session_id: event.context.session_id.clone(),
                agent_id: event.context.agent_id.clone(),
                user_id: event.context.user_id.clone(),
            },
        }
    }
}

/// List audit events with optional filtering
pub async fn list_audit_events(
    State(state): State<Arc<AuditState>>,
    Query(query): Query<AuditQuery>,
) -> impl IntoResponse {
    let events = state.logger.events().await;

    let filtered: Vec<_> = events
        .iter()
        .filter(|e| {
            // Filter by event type
            if let Some(ref event_type) = query.event_type {
                let matches = match (&e.kind, event_type.as_str()) {
                    (EventKind::ToolCall { .. }, "tool_call") => true,
                    (EventKind::LlmRequest { .. }, "llm_request") => true,
                    (EventKind::LlmResponse { .. }, "llm_response") => true,
                    (EventKind::AgentLifecycle { .. }, "agent_lifecycle") => true,
                    (EventKind::ApprovalDecision { .. }, "approval_decision") => true,
                    (EventKind::Error { .. }, "error") => true,
                    (EventKind::Security { .. }, "security") => true,
                    (EventKind::Custom { .. }, "custom") => true,
                    _ => false,
                };
                if !matches {
                    return false;
                }
            }

            // Filter by trace ID
            if let Some(ref trace_id) = query.trace_id {
                if e.context.trace_id.as_ref() != Some(trace_id) {
                    return false;
                }
            }

            // Filter by session ID
            if let Some(ref session_id) = query.session_id {
                if e.context.session_id.as_ref() != Some(session_id) {
                    return false;
                }
            }

            // Filter by agent ID
            if let Some(ref agent_id) = query.agent_id {
                if e.context.agent_id.as_ref() != Some(agent_id) {
                    return false;
                }
            }

            true
        })
        .collect();

    // Apply pagination
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(100).min(1000);

    let paginated: Vec<AuditEventResponse> = filtered
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|e| e.into())
        .collect();

    Json(paginated)
}

/// Get audit statistics
pub async fn get_audit_stats(State(state): State<Arc<AuditState>>) -> impl IntoResponse {
    let events = state.logger.events().await;

    let mut stats = AuditStats::default();

    for event in events.iter() {
        stats.total += 1;
        match &event.kind {
            EventKind::ToolCall { .. } => stats.tool_calls += 1,
            EventKind::LlmRequest { .. } => stats.llm_requests += 1,
            EventKind::LlmResponse { .. } => stats.llm_responses += 1,
            EventKind::AgentLifecycle { .. } => stats.lifecycle_events += 1,
            EventKind::ApprovalDecision { approved, .. } => {
                if *approved {
                    stats.approvals += 1;
                } else {
                    stats.denials += 1;
                }
            }
            EventKind::Error { .. } => stats.errors += 1,
            EventKind::Security { .. } => stats.security_events += 1,
            EventKind::Custom { .. } => stats.custom_events += 1,
        }
    }

    Json(stats)
}

/// Audit statistics response
#[derive(Debug, Default, Serialize)]
pub struct AuditStats {
    pub total: usize,
    pub tool_calls: usize,
    pub llm_requests: usize,
    pub llm_responses: usize,
    pub lifecycle_events: usize,
    pub approvals: usize,
    pub denials: usize,
    pub errors: usize,
    pub security_events: usize,
    pub custom_events: usize,
}

/// Middleware that logs HTTP requests to the audit log.
pub async fn audit_middleware(
    State(state): State<Arc<AuditState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();
    let path = uri.path().to_string();

    // Skip audit logging for audit endpoints themselves to avoid recursion
    if path.starts_with("/api/audit") {
        return next.run(request).await;
    }

    // Extract trace ID from headers if present
    let trace_id = request
        .headers()
        .get("x-trace-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let response = next.run(request).await;

    let duration_ms = start.elapsed().as_millis() as u64;
    let status = response.status().as_u16();

    // Log the request
    let mut ctx = AuditContext::default();
    if let Some(tid) = trace_id {
        ctx = ctx.with_trace_id(tid);
    }
    ctx = ctx
        .with_metadata("method", serde_json::json!(method.as_str()))
        .with_metadata("path", serde_json::json!(path))
        .with_metadata("status", serde_json::json!(status))
        .with_metadata("duration_ms", serde_json::json!(duration_ms));

    let event = AuditEvent::new(
        rust_ai_agents_audit::AuditLevel::Info,
        EventKind::Custom {
            name: "http_request".to_string(),
            payload: serde_json::json!({
                "method": method.as_str(),
                "path": path,
                "status": status,
                "duration_ms": duration_ms,
            }),
        },
    )
    .with_context(ctx);

    // Log asynchronously to not block response
    let logger = state.logger.clone();
    tokio::spawn(async move {
        if let Err(e) = logger.log(event).await {
            tracing::warn!("Failed to log HTTP request audit event: {}", e);
        }
    });

    response
}

/// Create router with audit endpoints
pub fn audit_routes() -> Router<Arc<AuditState>> {
    Router::new()
        .route("/api/audit/events", get(list_audit_events))
        .route("/api/audit/stats", get(get_audit_stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_state_creation() {
        let state = AuditState::new();
        let events = state.logger.events().await;
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn test_audit_event_logging() {
        let state = AuditState::new();

        let event = AuditEvent::tool_call("test_tool", serde_json::json!({}), true);
        state.logger.log(event).await.unwrap();

        let events = state.logger.events().await;
        assert_eq!(events.len(), 1);
    }

    #[tokio::test]
    async fn test_audit_stats() {
        let state = Arc::new(AuditState::new());

        // Log various events
        state
            .logger
            .log(AuditEvent::tool_call("tool1", serde_json::json!({}), true))
            .await
            .unwrap();
        state
            .logger
            .log(AuditEvent::llm_request("anthropic", "claude-3", false))
            .await
            .unwrap();
        state
            .logger
            .log(AuditEvent::error_event("TestError", "test message"))
            .await
            .unwrap();

        let events = state.logger.events().await;
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_audit_event_response_conversion() {
        let event = AuditEvent::tool_call("my_tool", serde_json::json!({"key": "value"}), true)
            .with_context(
                AuditContext::default()
                    .with_trace_id("trace-123")
                    .with_agent_id("agent-1"),
            );

        let response: AuditEventResponse = (&event).into();

        assert_eq!(response.event_type, "tool_call");
        assert_eq!(response.context.trace_id, Some("trace-123".to_string()));
        assert_eq!(response.context.agent_id, Some("agent-1".to_string()));
    }
}
