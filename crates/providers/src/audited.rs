//! Audited LLM backend wrapper
//!
//! Wraps any `LLMBackend` to add audit logging for all LLM requests and responses.

#[cfg(feature = "audit")]
use rust_ai_agents_audit::{AuditContext, AuditEvent, AuditLogger};

use crate::{InferenceOutput, LLMBackend, ModelInfo, StreamResponse, TokenUsage};
use async_trait::async_trait;
use rust_ai_agents_core::{errors::LLMError, LLMMessage, ToolSchema};
use std::sync::Arc;

#[cfg(feature = "audit")]
use std::time::Instant;

/// Wrapper that adds audit logging to any LLM backend.
///
/// # Example
///
/// ```rust,ignore
/// use rust_ai_agents_providers::{AnthropicProvider, AuditedBackend};
/// use rust_ai_agents_audit::{JsonFileLogger, create_production_logger};
///
/// let claude = AnthropicProvider::claude_35_sonnet(api_key);
/// let logger = create_production_logger("/var/log/audit.jsonl").await?;
/// let audited = AuditedBackend::new(claude, Arc::new(logger));
///
/// // All LLM calls are now logged
/// let output = audited.infer(&messages, &tools, 0.7).await?;
/// ```
#[derive(Clone)]
pub struct AuditedBackend<B: LLMBackend> {
    inner: B,
    #[cfg(feature = "audit")]
    logger: Arc<dyn AuditLogger>,
    #[cfg(feature = "audit")]
    context: Option<AuditContext>,
}

impl<B: LLMBackend> AuditedBackend<B> {
    /// Create a new audited backend wrapper.
    #[cfg(feature = "audit")]
    pub fn new(backend: B, logger: Arc<dyn AuditLogger>) -> Self {
        Self {
            inner: backend,
            logger,
            context: None,
        }
    }

    /// Create without audit (no-op when feature disabled).
    #[cfg(not(feature = "audit"))]
    pub fn new(backend: B) -> Self {
        Self { inner: backend }
    }

    /// Set the audit context for subsequent calls.
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

    /// Set agent ID.
    #[cfg(feature = "audit")]
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        let ctx = self.context.take().unwrap_or_default();
        self.context = Some(ctx.with_agent_id(agent_id));
        self
    }

    /// Get reference to inner backend.
    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// Get mutable reference to inner backend.
    pub fn inner_mut(&mut self) -> &mut B {
        &mut self.inner
    }

    /// Consume and return inner backend.
    pub fn into_inner(self) -> B {
        self.inner
    }

    #[cfg(feature = "audit")]
    async fn log_request(&self, model: &str, provider: &str, streaming: bool) {
        let mut event = AuditEvent::llm_request(provider, model, streaming);
        if let Some(ctx) = &self.context {
            event = event.with_context(ctx.clone());
        }
        if let Err(e) = self.logger.log(event).await {
            tracing::warn!("Failed to log LLM request audit event: {}", e);
        }
    }

    #[cfg(feature = "audit")]
    async fn log_response(
        &self,
        model: &str,
        provider: &str,
        usage: &TokenUsage,
        tool_call_count: usize,
        duration_ms: u64,
        finish_reason: Option<&str>,
    ) {
        let mut event = AuditEvent::llm_response(provider, model, tool_call_count, finish_reason);

        // Add context with token usage and duration metadata
        let mut ctx = self.context.clone().unwrap_or_default();
        ctx = ctx
            .with_metadata("duration_ms", serde_json::json!(duration_ms))
            .with_metadata("input_tokens", serde_json::json!(usage.prompt_tokens))
            .with_metadata("output_tokens", serde_json::json!(usage.completion_tokens))
            .with_metadata("total_tokens", serde_json::json!(usage.total_tokens));

        event = event.with_context(ctx);

        if let Err(e) = self.logger.log(event).await {
            tracing::warn!("Failed to log LLM response audit event: {}", e);
        }
    }

    #[cfg(feature = "audit")]
    async fn log_error(&self, error: &LLMError) {
        let mut event = AuditEvent::error_event("LLMError", &error.to_string());
        if let Some(ctx) = &self.context {
            event = event.with_context(ctx.clone());
        }
        if let Err(e) = self.logger.log(event).await {
            tracing::warn!("Failed to log LLM error audit event: {}", e);
        }
    }
}

#[async_trait]
impl<B: LLMBackend + 'static> LLMBackend for AuditedBackend<B> {
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        let info = self.inner.model_info();

        #[cfg(feature = "audit")]
        let start = Instant::now();

        #[cfg(feature = "audit")]
        self.log_request(&info.model, &info.provider, false).await;

        let result = self.inner.infer(messages, tools, temperature).await;

        #[cfg(feature = "audit")]
        {
            let duration_ms = start.elapsed().as_millis() as u64;
            match &result {
                Ok(output) => {
                    let tool_count = output.tool_calls.as_ref().map(|t| t.len()).unwrap_or(0);
                    self.log_response(
                        &info.model,
                        &info.provider,
                        &output.token_usage,
                        tool_count,
                        duration_ms,
                        Some("stop"),
                    )
                    .await;
                }
                Err(e) => {
                    self.log_error(e).await;
                }
            }
        }

        result
    }

    async fn infer_stream(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<StreamResponse, LLMError> {
        let info = self.inner.model_info();

        #[cfg(feature = "audit")]
        self.log_request(&info.model, &info.provider, true).await;

        // For streaming, we log the request but can't easily log the response
        // until the stream is consumed. The caller should handle final logging.
        self.inner.infer_stream(messages, tools, temperature).await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError> {
        self.inner.embed(text).await
    }

    fn model_info(&self) -> ModelInfo {
        self.inner.model_info()
    }

    fn supports_function_calling(&self) -> bool {
        self.inner.supports_function_calling()
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }
}

#[cfg(all(test, feature = "audit"))]
mod tests {
    use super::*;
    use crate::mock::{MockBackend, MockResponse};
    use rust_ai_agents_audit::MemoryLogger;
    use rust_ai_agents_core::MessageRole;

    #[tokio::test]
    async fn test_audited_backend_logs_request_and_response() {
        let mock = MockBackend::new().with_response(MockResponse::text("Hello!"));
        let logger = Arc::new(MemoryLogger::new());
        let audited = AuditedBackend::new(mock, logger.clone())
            .with_trace_id("test-trace-123")
            .with_agent_id("test-agent");

        let messages = vec![LLMMessage {
            role: MessageRole::User,
            content: "Hi".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }];

        let result = audited.infer(&messages, &[], 0.7).await;
        assert!(result.is_ok());

        // Check audit logs
        let events = logger.events().await;
        assert_eq!(events.len(), 2); // request + response

        // Verify request event
        assert!(matches!(
            &events[0].kind,
            rust_ai_agents_audit::EventKind::LlmRequest { .. }
        ));
        assert_eq!(
            events[0].context.trace_id,
            Some("test-trace-123".to_string())
        );
        assert_eq!(events[0].context.agent_id, Some("test-agent".to_string()));

        // Verify response event
        assert!(matches!(
            &events[1].kind,
            rust_ai_agents_audit::EventKind::LlmResponse { .. }
        ));
    }

    #[tokio::test]
    async fn test_audited_backend_logs_errors() {
        let mock = MockBackend::new().with_response(MockResponse::error("API Error"));
        let logger = Arc::new(MemoryLogger::new());
        let audited = AuditedBackend::new(mock, logger.clone());

        let messages = vec![LLMMessage {
            role: MessageRole::User,
            content: "Hi".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }];

        let result = audited.infer(&messages, &[], 0.7).await;
        assert!(result.is_err());

        // Check audit logs
        let events = logger.events().await;
        assert_eq!(events.len(), 2); // request + error

        // Verify error event
        assert!(matches!(
            &events[1].kind,
            rust_ai_agents_audit::EventKind::Error { .. }
        ));
    }
}
