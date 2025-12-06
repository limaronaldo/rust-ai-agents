//! Backend trait for LLM providers

use async_trait::async_trait;
use futures::Stream;
use rust_ai_agents_core::{errors::LLMError, LLMMessage, ToolSchema};
use serde::{Deserialize, Serialize};
use std::pin::Pin;

/// LLM inference output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    /// Generated content
    pub content: String,

    /// Tool calls (if any)
    pub tool_calls: Option<Vec<rust_ai_agents_core::ToolCall>>,

    /// Reasoning/thoughts (if available)
    pub reasoning: Option<String>,

    /// Model confidence (0.0 - 1.0)
    pub confidence: f64,

    /// Token usage
    pub token_usage: TokenUsage,

    /// Response metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input tokens
    pub prompt_tokens: usize,

    /// Output tokens
    pub completion_tokens: usize,

    /// Total tokens
    pub total_tokens: usize,

    /// Cached tokens (if applicable)
    pub cached_tokens: Option<usize>,
}

impl TokenUsage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cached_tokens: None,
        }
    }

    pub fn with_cache(mut self, cached: usize) -> Self {
        self.cached_tokens = Some(cached);
        self
    }
}

/// Streaming event from LLM
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text content delta
    TextDelta(String),

    /// Tool call started
    ToolCallStart { id: String, name: String },

    /// Tool call arguments delta (JSON fragment)
    ToolCallDelta { id: String, arguments_delta: String },

    /// Tool call completed
    ToolCallEnd { id: String },

    /// Stream completed with final usage stats
    Done { token_usage: Option<TokenUsage> },

    /// Error occurred during streaming
    Error(String),
}

/// Type alias for streaming response
pub type StreamResponse = Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>>;

/// LLM backend trait
#[async_trait]
pub trait LLMBackend: Send + Sync {
    /// Perform inference with the LLM
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError>;

    /// Perform streaming inference with the LLM
    ///
    /// Returns a stream of events as the model generates output.
    /// Use this for real-time display of responses.
    ///
    /// Default implementation falls back to non-streaming `infer`.
    async fn infer_stream(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<StreamResponse, LLMError> {
        // Default: fall back to non-streaming
        let output = self.infer(messages, tools, temperature).await?;

        let stream = async_stream::stream! {
            // Emit text content
            if !output.content.is_empty() {
                yield Ok(StreamEvent::TextDelta(output.content));
            }

            // Emit tool calls
            if let Some(tool_calls) = output.tool_calls {
                for call in tool_calls {
                    yield Ok(StreamEvent::ToolCallStart {
                        id: call.id.clone(),
                        name: call.name.clone(),
                    });
                    yield Ok(StreamEvent::ToolCallDelta {
                        id: call.id.clone(),
                        arguments_delta: serde_json::to_string(&call.arguments).unwrap_or_default(),
                    });
                    yield Ok(StreamEvent::ToolCallEnd { id: call.id });
                }
            }

            // Done
            yield Ok(StreamEvent::Done {
                token_usage: Some(output.token_usage),
            });
        };

        Ok(Box::pin(stream))
    }

    /// Generate embeddings for text
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError>;

    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Check if model supports function calling
    fn supports_function_calling(&self) -> bool {
        true
    }

    /// Check if model supports streaming
    fn supports_streaming(&self) -> bool {
        false
    }
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub model: String,

    /// Provider name
    pub provider: String,

    /// Maximum context window size
    pub max_tokens: usize,

    /// Input cost per 1M tokens
    pub input_cost_per_1m: f64,

    /// Output cost per 1M tokens
    pub output_cost_per_1m: f64,

    /// Supports function calling
    pub supports_functions: bool,

    /// Supports vision
    pub supports_vision: bool,
}

impl ModelInfo {
    pub fn calculate_cost(&self, usage: &TokenUsage) -> f64 {
        let input_cost = (usage.prompt_tokens as f64 / 1_000_000.0) * self.input_cost_per_1m;
        let output_cost = (usage.completion_tokens as f64 / 1_000_000.0) * self.output_cost_per_1m;
        input_cost + output_cost
    }
}

/// Rate limiter for API calls
pub struct RateLimiter {
    /// Requests per minute
    rpm: usize,
    /// Tokens per minute
    tpm: usize,
    /// Request timestamps
    requests: parking_lot::Mutex<Vec<std::time::Instant>>,
    /// Token counts
    tokens: parking_lot::Mutex<Vec<(std::time::Instant, usize)>>,
}

impl RateLimiter {
    pub fn new(rpm: usize, tpm: usize) -> Self {
        Self {
            rpm,
            tpm,
            requests: parking_lot::Mutex::new(Vec::new()),
            tokens: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Wait until request can be made
    pub async fn wait(&self, estimated_tokens: usize) {
        loop {
            let now = std::time::Instant::now();
            let one_minute_ago = now - std::time::Duration::from_secs(60);

            // Clean old entries
            {
                let mut requests = self.requests.lock();
                requests.retain(|&t| t > one_minute_ago);

                let mut tokens = self.tokens.lock();
                tokens.retain(|(t, _)| *t > one_minute_ago);

                let current_rpm = requests.len();
                let current_tpm: usize = tokens.iter().map(|(_, count)| count).sum();

                // Check if we can proceed
                if current_rpm < self.rpm && (current_tpm + estimated_tokens) < self.tpm {
                    requests.push(now);
                    tokens.push((now, estimated_tokens));
                    return;
                }
            }

            // Wait a bit before retrying
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
}
