//! Instrumented LLM backend wrapper with Prometheus metrics
//!
//! This module provides a wrapper around any `LLMBackend` implementation that
//! automatically records Prometheus metrics for all LLM operations.
//!
//! # Example
//!
//! ```rust,no_run
//! use rust_ai_agents_providers::{OpenAIProvider, InstrumentedBackend};
//! use rust_ai_agents_monitoring::init_prometheus;
//!
//! // Initialize Prometheus first
//! let _handle = init_prometheus();
//!
//! // Wrap any backend with instrumentation
//! let openai = OpenAIProvider::new("api-key".to_string(), "gpt-4-turbo".to_string());
//! let instrumented = InstrumentedBackend::new(openai, "openai", "chat");
//!
//! // Now all calls will record metrics automatically
//! ```

use async_trait::async_trait;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Instant;

use crate::backend::{InferenceOutput, LLMBackend, ModelInfo, StreamEvent, StreamResponse};
use rust_ai_agents_core::{errors::LLMError, LLMMessage, ToolSchema};

/// Wrapper that adds Prometheus metrics to any LLM backend
pub struct InstrumentedBackend<B: LLMBackend> {
    /// The underlying backend
    backend: Arc<B>,
    /// Provider name for metrics labels (e.g., "openai", "anthropic")
    provider: String,
    /// Route name for metrics labels (e.g., "chat", "completion", "embeddings")
    route: String,
}

impl<B: LLMBackend> InstrumentedBackend<B> {
    /// Create a new instrumented backend
    ///
    /// # Arguments
    ///
    /// * `backend` - The underlying LLM backend to wrap
    /// * `provider` - Provider name for metrics (e.g., "openai")
    /// * `route` - Route name for metrics (e.g., "chat")
    pub fn new(backend: B, provider: impl Into<String>, route: impl Into<String>) -> Self {
        Self {
            backend: Arc::new(backend),
            provider: provider.into(),
            route: route.into(),
        }
    }

    /// Get a reference to the underlying backend
    pub fn inner(&self) -> &B {
        &self.backend
    }

    /// Calculate cost based on model and token usage
    fn calculate_cost(&self, model: &str, prompt_tokens: usize, completion_tokens: usize) -> f64 {
        // Pricing as of Dec 2025 (USD per 1K tokens)
        let (input_cost, output_cost) = match model {
            // OpenAI GPT-4 Turbo
            "gpt-4-turbo" | "gpt-4-turbo-preview" => (0.01, 0.03),
            "gpt-4-turbo-2024-04-09" => (0.01, 0.03),

            // OpenAI GPT-4
            "gpt-4" | "gpt-4-0613" => (0.03, 0.06),
            "gpt-4-32k" | "gpt-4-32k-0613" => (0.06, 0.12),

            // OpenAI GPT-3.5 Turbo
            "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" => (0.0005, 0.0015),
            "gpt-3.5-turbo-16k" => (0.003, 0.004),

            // Anthropic Claude 3
            "claude-3-opus-20240229" => (0.015, 0.075),
            "claude-3-sonnet-20240229" => (0.003, 0.015),
            "claude-3-haiku-20240307" => (0.00025, 0.00125),

            // Anthropic Claude 3.5
            "claude-3-5-sonnet-20241022" => (0.003, 0.015),

            // Default fallback
            _ => (0.001, 0.002),
        };

        let input_cost_total = (prompt_tokens as f64 / 1000.0) * input_cost;
        let output_cost_total = (completion_tokens as f64 / 1000.0) * output_cost;

        input_cost_total + output_cost_total
    }
}

#[async_trait]
impl<B: LLMBackend + 'static> LLMBackend for InstrumentedBackend<B> {
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        use rust_ai_agents_monitoring::prometheus::{
            dec_active_requests, inc_active_requests, record_llm_cost, record_llm_error,
            record_llm_latency, record_llm_tokens,
        };

        let start = Instant::now();
        let model = self.backend.model_info().model;

        // Track active requests
        inc_active_requests(&self.route, &model);

        // Execute the actual inference
        let result = self.backend.infer(messages, tools, temperature).await;

        // Always decrement active requests
        dec_active_requests(&self.route, &model);

        // Record latency
        let latency_ms = start.elapsed().as_millis() as u64;
        record_llm_latency(&self.route, &model, latency_ms);

        match &result {
            Ok(output) => {
                // Record token usage
                record_llm_tokens(
                    &self.route,
                    &model,
                    output.token_usage.prompt_tokens as u64,
                    output.token_usage.completion_tokens as u64,
                );

                // Calculate and record cost
                let cost = self.calculate_cost(
                    &model,
                    output.token_usage.prompt_tokens,
                    output.token_usage.completion_tokens,
                );
                record_llm_cost(&self.route, &model, cost);
            }
            Err(e) => {
                // Record error type
                let error_type = match e {
                    LLMError::RateLimitExceeded => "rate_limit",
                    LLMError::AuthenticationFailed(_) => "auth_failed",
                    LLMError::NetworkError(_) => "network",
                    LLMError::InvalidResponse(_) => "invalid_response",
                    LLMError::ApiError(_) => "api_error",
                    LLMError::SerializationError(_) => "serialization",
                    LLMError::ValidationError(_) => "validation",
                    LLMError::Timeout(_) => "timeout",
                    LLMError::TokenLimitExceeded(_, _) => "token_limit",
                    LLMError::ModelNotAvailable(_) => "model_unavailable",
                };
                record_llm_error(&self.route, &model, error_type);

                // Record rate limit events specifically
                if matches!(e, LLMError::RateLimitExceeded) {
                    rust_ai_agents_monitoring::prometheus::record_rate_limit(
                        &self.provider,
                        &model,
                    );
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
        use rust_ai_agents_monitoring::prometheus::{
            dec_active_requests, inc_active_requests, record_llm_cost, record_llm_latency,
            record_llm_tokens,
        };

        let start = Instant::now();
        let model = self.backend.model_info().model.clone();
        let route = self.route.clone();

        // Track active requests
        inc_active_requests(&route, &model);

        // Execute the actual streaming inference
        let stream = self
            .backend
            .infer_stream(messages, tools, temperature)
            .await?;

        // Wrap the stream to record metrics when done
        let instrumented_stream = stream.map(move |event| {
            if let Ok(StreamEvent::Done { token_usage }) = &event {
                // Record latency
                let latency_ms = start.elapsed().as_millis() as u64;
                record_llm_latency(&route, &model, latency_ms);

                if let Some(usage) = token_usage {
                    // Record token usage
                    record_llm_tokens(
                        &route,
                        &model,
                        usage.prompt_tokens as u64,
                        usage.completion_tokens as u64,
                    );

                    // Calculate and record cost
                    let cost = {
                        let (input_cost, output_cost) = match model.as_str() {
                            "gpt-4-turbo" | "gpt-4-turbo-preview" => (0.01, 0.03),
                            "gpt-4" | "gpt-4-0613" => (0.03, 0.06),
                            "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" => (0.0005, 0.0015),
                            "claude-3-opus-20240229" => (0.015, 0.075),
                            "claude-3-sonnet-20240229" => (0.003, 0.015),
                            "claude-3-haiku-20240307" => (0.00025, 0.00125),
                            "claude-3-5-sonnet-20241022" => (0.003, 0.015),
                            _ => (0.001, 0.002),
                        };
                        (usage.prompt_tokens as f64 / 1000.0) * input_cost
                            + (usage.completion_tokens as f64 / 1000.0) * output_cost
                    };
                    record_llm_cost(&route, &model, cost);
                }

                // Decrement active requests when stream is done
                dec_active_requests(&route, &model);
            }
            event
        });

        Ok(Box::pin(instrumented_stream))
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError> {
        use rust_ai_agents_monitoring::prometheus::{
            dec_active_requests, inc_active_requests, record_llm_latency,
        };

        let start = Instant::now();
        let model = self.backend.model_info().model;
        let route = "embeddings";

        inc_active_requests(route, &model);

        let result = self.backend.embed(text).await;

        dec_active_requests(route, &model);

        let latency_ms = start.elapsed().as_millis() as u64;
        record_llm_latency(route, &model, latency_ms);

        result
    }

    fn model_info(&self) -> ModelInfo {
        self.backend.model_info()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockBackend;
    use rust_ai_agents_core::MessageRole;

    #[tokio::test]
    async fn test_instrumented_backend_records_metrics() {
        // Initialize Prometheus
        let handle = rust_ai_agents_monitoring::init_prometheus();

        // Create a mock backend
        let mock = MockBackend::new().with_response(crate::MockResponse::text("Test response"));

        // Wrap with instrumentation
        let instrumented = InstrumentedBackend::new(mock, "mock", "test");

        // Make a request
        let messages = vec![LLMMessage {
            role: MessageRole::User,
            content: "Hello".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let result = instrumented.infer(&messages, &[], 0.7).await;
        assert!(result.is_ok());

        // Check that metrics were recorded
        let metrics = handle.render();
        assert!(metrics.contains("llm_tokens_total"));
        assert!(metrics.contains("llm_latency_ms"));
        assert!(metrics.contains("llm_active_requests"));
    }

    #[tokio::test]
    async fn test_calculate_cost() {
        let mock = MockBackend::new();
        let instrumented = InstrumentedBackend::new(mock, "mock", "test");

        // GPT-4 Turbo: $0.01/1K input, $0.03/1K output
        let cost = instrumented.calculate_cost("gpt-4-turbo", 1000, 500);
        assert!((cost - 0.025).abs() < 0.001); // 0.01 + 0.015 = 0.025

        // GPT-3.5 Turbo: $0.0005/1K input, $0.0015/1K output
        let cost = instrumented.calculate_cost("gpt-3.5-turbo", 2000, 1000);
        assert!((cost - 0.0025).abs() < 0.0001); // 0.001 + 0.0015 = 0.0025

        // Claude 3 Opus: $0.015/1K input, $0.075/1K output
        let cost = instrumented.calculate_cost("claude-3-opus-20240229", 1000, 1000);
        assert!((cost - 0.090).abs() < 0.001); // 0.015 + 0.075 = 0.090
    }
}
