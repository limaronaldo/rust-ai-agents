//! Audited tool executor wrapper
//!
//! Wraps `ToolExecutor` to add audit logging for tool executions and approval decisions.

#[cfg(feature = "audit")]
use rust_ai_agents_audit::{AuditContext, AuditEvent, AuditLogger, EventKind};

use std::sync::Arc;
use std::time::Instant;

use rust_ai_agents_core::{AgentId, ToolCall, ToolResult};
use rust_ai_agents_tools::ToolRegistry;

use crate::approvals::ApprovalHandler;
use crate::executor::{ExecutionOutcome, ToolExecutor};

/// Audited tool executor that logs all tool executions and approval decisions.
///
/// # Example
///
/// ```rust,ignore
/// use rust_ai_agents_agents::{AuditedExecutor, ToolExecutor, ExecutorConfig};
/// use rust_ai_agents_audit::{create_production_logger, MemoryLogger};
///
/// let executor = ToolExecutor::new(4);
/// let logger = Arc::new(MemoryLogger::new());
/// let audited = AuditedExecutor::new(executor, logger);
///
/// // All tool executions are now logged
/// let results = audited.execute_tools(&calls, &registry, &agent_id).await;
/// ```
pub struct AuditedExecutor<H: ApprovalHandler> {
    inner: ToolExecutor<H>,
    #[cfg(feature = "audit")]
    logger: Arc<dyn AuditLogger>,
    #[cfg(feature = "audit")]
    context: Option<AuditContext>,
}

impl<H: ApprovalHandler + 'static> AuditedExecutor<H> {
    /// Create a new audited executor wrapper.
    #[cfg(feature = "audit")]
    pub fn new(executor: ToolExecutor<H>, logger: Arc<dyn AuditLogger>) -> Self {
        Self {
            inner: executor,
            logger,
            context: None,
        }
    }

    /// Create without audit (no-op when feature disabled).
    #[cfg(not(feature = "audit"))]
    pub fn new(executor: ToolExecutor<H>) -> Self {
        Self { inner: executor }
    }

    /// Set the audit context for subsequent executions.
    #[cfg(feature = "audit")]
    pub fn with_context(mut self, context: AuditContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Set trace ID for request correlation.
    #[cfg(feature = "audit")]
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        let ctx = self.context.take().unwrap_or_default();
        self.context = Some(ctx.with_trace_id(trace_id));
        self
    }

    /// Set session ID.
    #[cfg(feature = "audit")]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        let ctx = self.context.take().unwrap_or_default();
        self.context = Some(ctx.with_session_id(session_id));
        self
    }

    /// Set agent ID in context.
    #[cfg(feature = "audit")]
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        let ctx = self.context.take().unwrap_or_default();
        self.context = Some(ctx.with_agent_id(agent_id));
        self
    }

    /// Execute multiple tools in parallel with approval checks and audit logging.
    pub async fn execute_tools(
        &self,
        tool_calls: &[ToolCall],
        registry: &Arc<ToolRegistry>,
        agent_id: &AgentId,
    ) -> Vec<ToolResult> {
        #[cfg(feature = "audit")]
        let start = Instant::now();

        let outcomes = self
            .inner
            .execute_tools_with_outcomes(tool_calls, registry, agent_id)
            .await;

        #[cfg(feature = "audit")]
        {
            let duration_ms = start.elapsed().as_millis() as u64;
            for (call, outcome) in tool_calls.iter().zip(outcomes.iter()) {
                self.log_tool_execution(call, outcome, duration_ms).await;
            }
        }

        outcomes.into_iter().map(|o| o.into_tool_result()).collect()
    }

    /// Execute tools and return detailed outcomes with audit logging.
    pub async fn execute_tools_with_outcomes(
        &self,
        tool_calls: &[ToolCall],
        registry: &Arc<ToolRegistry>,
        agent_id: &AgentId,
    ) -> Vec<ExecutionOutcome> {
        #[cfg(feature = "audit")]
        let start = Instant::now();

        let outcomes = self
            .inner
            .execute_tools_with_outcomes(tool_calls, registry, agent_id)
            .await;

        #[cfg(feature = "audit")]
        {
            let duration_ms = start.elapsed().as_millis() as u64;
            for (call, outcome) in tool_calls.iter().zip(outcomes.iter()) {
                self.log_tool_execution(call, outcome, duration_ms).await;
            }
        }

        outcomes
    }

    #[cfg(feature = "audit")]
    async fn log_tool_execution(
        &self,
        call: &ToolCall,
        outcome: &ExecutionOutcome,
        duration_ms: u64,
    ) {
        let (approved, result_json) = match outcome {
            ExecutionOutcome::Success(result) => {
                let result_value = serde_json::to_value(result).ok();
                (true, result_value)
            }
            ExecutionOutcome::Denied { reason, .. } => {
                // Log approval denial
                self.log_approval_decision(call, false, Some(reason)).await;
                (false, None)
            }
            ExecutionOutcome::Skipped { .. } => {
                self.log_approval_decision(call, false, Some("skipped"))
                    .await;
                (false, None)
            }
            ExecutionOutcome::ApprovalError { error, .. } => {
                self.log_error(&format!("Approval error for {}: {}", call.name, error))
                    .await;
                (false, None)
            }
        };

        // Log tool call event
        let mut event = AuditEvent::new(
            rust_ai_agents_audit::AuditLevel::Info,
            EventKind::ToolCall {
                tool_name: call.name.clone(),
                parameters: call.arguments.clone(),
                result: result_json,
                approved,
                duration_ms: Some(duration_ms),
            },
        );

        if let Some(ctx) = &self.context {
            event = event.with_context(ctx.clone());
        }

        if let Err(e) = self.logger.log(event).await {
            tracing::warn!("Failed to log tool execution audit event: {}", e);
        }
    }

    #[cfg(feature = "audit")]
    async fn log_approval_decision(&self, call: &ToolCall, approved: bool, reason: Option<&str>) {
        let mut event = AuditEvent::new(
            rust_ai_agents_audit::AuditLevel::Info,
            EventKind::ApprovalDecision {
                tool_name: call.name.clone(),
                approved,
                reason: reason.map(|r| r.to_string()),
                approver: "system".to_string(),
            },
        );

        if let Some(ctx) = &self.context {
            event = event.with_context(ctx.clone());
        }

        if let Err(e) = self.logger.log(event).await {
            tracing::warn!("Failed to log approval decision audit event: {}", e);
        }
    }

    #[cfg(feature = "audit")]
    async fn log_error(&self, message: &str) {
        let mut event = AuditEvent::error_event("ToolExecutionError", message);

        if let Some(ctx) = &self.context {
            event = event.with_context(ctx.clone());
        }

        if let Err(e) = self.logger.log(event).await {
            tracing::warn!("Failed to log error audit event: {}", e);
        }
    }

    /// Get reference to inner executor.
    pub fn inner(&self) -> &ToolExecutor<H> {
        &self.inner
    }

    /// Get mutable reference to inner executor.
    pub fn inner_mut(&mut self) -> &mut ToolExecutor<H> {
        &mut self.inner
    }

    /// Consume and return inner executor.
    pub fn into_inner(self) -> ToolExecutor<H> {
        self.inner
    }
}

#[cfg(all(test, feature = "audit"))]
mod tests {
    use super::*;
    use crate::approvals::{ApprovalConfig, ApprovalDecision, TestApprovalHandler};
    use crate::executor::ExecutorConfig;
    use rust_ai_agents_audit::MemoryLogger;
    use rust_ai_agents_core::{Tool, ToolSchema};
    use rust_ai_agents_tools::ExecutionContext;
    use serde_json::json;
    use std::collections::HashMap;

    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "echo".to_string(),
                description: "Echoes input".to_string(),
                parameters: json!({"type": "object", "properties": {"message": {"type": "string"}}}),
                dangerous: false,
                metadata: HashMap::new(),
            }
        }

        async fn execute(
            &self,
            _ctx: &ExecutionContext,
            args: serde_json::Value,
        ) -> Result<serde_json::Value, rust_ai_agents_core::errors::ToolError> {
            Ok(args)
        }
    }

    struct DangerousTool;

    #[async_trait::async_trait]
    impl Tool for DangerousTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "dangerous".to_string(),
                description: "A dangerous tool".to_string(),
                parameters: json!({"type": "object"}),
                dangerous: true,
                metadata: HashMap::new(),
            }
        }

        async fn execute(
            &self,
            _ctx: &ExecutionContext,
            _args: serde_json::Value,
        ) -> Result<serde_json::Value, rust_ai_agents_core::errors::ToolError> {
            Ok(json!({"result": "executed"}))
        }
    }

    #[tokio::test]
    async fn test_audited_executor_logs_tool_calls() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let registry = Arc::new(registry);

        let executor = ToolExecutor::new(4);
        let logger = Arc::new(MemoryLogger::new());
        let audited = AuditedExecutor::new(executor, logger.clone())
            .with_trace_id("test-trace-123")
            .with_agent_id("test-agent");

        let agent_id = AgentId::new("test");
        let calls = vec![ToolCall {
            id: "1".to_string(),
            name: "echo".to_string(),
            arguments: json!({"message": "hello"}),
        }];

        let results = audited.execute_tools(&calls, &registry, &agent_id).await;

        assert_eq!(results.len(), 1);
        assert!(results[0].success);

        // Check audit logs
        let events = logger.events().await;
        assert_eq!(events.len(), 1);

        // Verify tool call event
        match &events[0].kind {
            EventKind::ToolCall {
                tool_name,
                approved,
                ..
            } => {
                assert_eq!(tool_name, "echo");
                assert!(approved);
            }
            _ => panic!("Expected ToolCall event"),
        }

        assert_eq!(
            events[0].context.trace_id,
            Some("test-trace-123".to_string())
        );
        assert_eq!(events[0].context.agent_id, Some("test-agent".to_string()));
    }

    #[tokio::test]
    async fn test_audited_executor_logs_denials() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DangerousTool));
        let registry = Arc::new(registry);

        let config = ApprovalConfig::builder()
            .require_approval_for_dangerous(true)
            .build();

        let handler = TestApprovalHandler::with_config(config);
        handler.set_decision("dangerous", ApprovalDecision::denied("Not allowed"));

        let executor = ToolExecutor::with_approval_handler(ExecutorConfig::default(), handler);
        let logger = Arc::new(MemoryLogger::new());
        let audited = AuditedExecutor::new(executor, logger.clone());

        let agent_id = AgentId::new("test");
        let calls = vec![ToolCall {
            id: "1".to_string(),
            name: "dangerous".to_string(),
            arguments: json!({}),
        }];

        let results = audited.execute_tools(&calls, &registry, &agent_id).await;

        assert_eq!(results.len(), 1);
        assert!(!results[0].success);

        // Check audit logs - should have approval denial + tool call
        let events = logger.events().await;
        assert!(events.len() >= 1);

        // Find the approval decision event
        let approval_event = events.iter().find(|e| {
            matches!(
                &e.kind,
                EventKind::ApprovalDecision {
                    approved: false,
                    ..
                }
            )
        });
        assert!(approval_event.is_some());
    }

    #[tokio::test]
    async fn test_audited_executor_multiple_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let registry = Arc::new(registry);

        let executor = ToolExecutor::new(4);
        let logger = Arc::new(MemoryLogger::new());
        let audited = AuditedExecutor::new(executor, logger.clone());

        let agent_id = AgentId::new("test");
        let calls = vec![
            ToolCall {
                id: "1".to_string(),
                name: "echo".to_string(),
                arguments: json!({"message": "hello"}),
            },
            ToolCall {
                id: "2".to_string(),
                name: "echo".to_string(),
                arguments: json!({"message": "world"}),
            },
        ];

        let results = audited.execute_tools(&calls, &registry, &agent_id).await;

        assert_eq!(results.len(), 2);
        assert!(results[0].success);
        assert!(results[1].success);

        // Should have 2 tool call events
        let events = logger.events().await;
        assert_eq!(events.len(), 2);
    }
}
