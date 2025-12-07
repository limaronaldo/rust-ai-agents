//! Mock LLM backend for deterministic testing
//!
//! Provides a configurable mock backend that returns predefined responses,
//! enabling reliable unit and integration tests without calling real LLMs.
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_providers::{MockBackend, MockResponse};
//!
//! let backend = MockBackend::new()
//!     .with_response(MockResponse::text("Hello, world!"))
//!     .with_response(MockResponse::tool_call("search", json!({"query": "test"})));
//!
//! // First call returns "Hello, world!"
//! // Second call returns a tool call
//! ```

use async_trait::async_trait;
use parking_lot::Mutex;
use rand::SeedableRng;
use rust_ai_agents_core::{errors::LLMError, LLMMessage, ToolCall, ToolSchema};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::backend::{InferenceOutput, LLMBackend, ModelInfo, TokenUsage};

/// A mock response configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockResponse {
    /// Text content to return
    pub content: String,

    /// Tool calls to return
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Optional reasoning
    pub reasoning: Option<String>,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,

    /// Simulated latency in milliseconds
    pub latency_ms: u64,

    /// Whether this response should fail
    pub should_fail: Option<String>,
}

impl Default for MockResponse {
    fn default() -> Self {
        Self {
            content: String::new(),
            tool_calls: None,
            reasoning: None,
            confidence: 1.0,
            latency_ms: 0,
            should_fail: None,
        }
    }
}

impl MockResponse {
    /// Create a simple text response
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Default::default()
        }
    }

    /// Create a response with a tool call
    pub fn tool_call(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            tool_calls: Some(vec![ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4()),
                name: name.into(),
                arguments,
            }]),
            ..Default::default()
        }
    }

    /// Create a response with multiple tool calls
    pub fn tool_calls(calls: Vec<(impl Into<String>, serde_json::Value)>) -> Self {
        let tool_calls = calls
            .into_iter()
            .map(|(name, args)| ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4()),
                name: name.into(),
                arguments: args,
            })
            .collect();

        Self {
            tool_calls: Some(tool_calls),
            ..Default::default()
        }
    }

    /// Create a failing response
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            should_fail: Some(message.into()),
            ..Default::default()
        }
    }

    /// Add reasoning to the response
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add simulated latency
    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Add text content to a tool call response
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }
}

/// Matching strategy for pattern-based responses
pub enum MessageMatcher {
    /// Match any message
    Any,

    /// Match if last message contains this substring
    Contains(String),

    /// Match if last message matches this regex
    Regex(regex::Regex),

    /// Match based on a custom function
    Custom(Arc<dyn Fn(&[LLMMessage]) -> bool + Send + Sync>),
}

impl std::fmt::Debug for MessageMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "Any"),
            Self::Contains(s) => write!(f, "Contains({:?})", s),
            Self::Regex(r) => write!(f, "Regex({:?})", r.as_str()),
            Self::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

impl Clone for MessageMatcher {
    fn clone(&self) -> Self {
        match self {
            Self::Any => Self::Any,
            Self::Contains(s) => Self::Contains(s.clone()),
            Self::Regex(r) => Self::Regex(r.clone()),
            Self::Custom(f) => Self::Custom(Arc::clone(f)),
        }
    }
}

impl MessageMatcher {
    pub fn matches(&self, messages: &[LLMMessage]) -> bool {
        let last_content = messages.last().map(|m| m.content.as_str()).unwrap_or("");

        match self {
            Self::Any => true,
            Self::Contains(s) => last_content.contains(s),
            Self::Regex(r) => r.is_match(last_content),
            Self::Custom(f) => f(messages),
        }
    }
}

/// A pattern-response pair for conditional matching
#[derive(Clone)]
pub struct ConditionalResponse {
    pub matcher: MessageMatcher,
    pub response: MockResponse,
}

/// Configuration for MockBackend behavior
#[derive(Debug, Clone)]
pub struct MockConfig {
    /// Default token counts for responses
    pub default_prompt_tokens: usize,
    pub default_completion_tokens: usize,

    /// Whether to record all calls for later inspection
    pub record_calls: bool,

    /// Model info to return
    pub model_info: ModelInfo,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            default_prompt_tokens: 100,
            default_completion_tokens: 50,
            record_calls: true,
            model_info: ModelInfo {
                model: "mock-model".to_string(),
                provider: "mock".to_string(),
                max_tokens: 4096,
                input_cost_per_1m: 0.0,
                output_cost_per_1m: 0.0,
                supports_functions: true,
                supports_vision: false,
            },
        }
    }
}

/// A recorded call for inspection
#[derive(Debug, Clone)]
pub struct RecordedCall {
    pub messages: Vec<LLMMessage>,
    pub tools: Vec<ToolSchema>,
    pub temperature: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Mock LLM backend for testing
///
/// Supports several modes:
/// 1. **Sequential**: Returns responses in order, cycling if needed
/// 2. **Pattern-based**: Returns responses based on message content
/// 3. **Echo**: Returns the last message content (useful for simple tests)
pub struct MockBackend {
    /// Sequential responses (returned in order)
    responses: Mutex<Vec<MockResponse>>,

    /// Current response index
    response_index: AtomicUsize,

    /// Pattern-based responses
    conditional_responses: Mutex<Vec<ConditionalResponse>>,

    /// Configuration
    config: MockConfig,

    /// Recorded calls (if enabled)
    recorded_calls: Mutex<Vec<RecordedCall>>,

    /// Echo mode: if true, returns last message content
    echo_mode: bool,

    /// Embeddings to return (query -> embedding)
    embeddings: Mutex<HashMap<String, Vec<f32>>>,

    /// Default embedding dimensions
    embedding_dims: usize,
}

impl MockBackend {
    /// Create a new MockBackend with default configuration
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(Vec::new()),
            response_index: AtomicUsize::new(0),
            conditional_responses: Mutex::new(Vec::new()),
            config: MockConfig::default(),
            recorded_calls: Mutex::new(Vec::new()),
            echo_mode: false,
            embeddings: Mutex::new(HashMap::new()),
            embedding_dims: 1536,
        }
    }

    /// Create a MockBackend with custom configuration
    pub fn with_config(config: MockConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Create a MockBackend in echo mode
    pub fn echo() -> Self {
        Self {
            echo_mode: true,
            ..Self::new()
        }
    }

    /// Add a sequential response
    pub fn with_response(self, response: MockResponse) -> Self {
        self.responses.lock().push(response);
        self
    }

    /// Add multiple sequential responses
    pub fn with_responses(self, responses: Vec<MockResponse>) -> Self {
        self.responses.lock().extend(responses);
        self
    }

    /// Add a conditional response
    pub fn when(self, matcher: MessageMatcher, response: MockResponse) -> Self {
        self.conditional_responses
            .lock()
            .push(ConditionalResponse { matcher, response });
        self
    }

    /// Add a response that matches when message contains a string
    pub fn when_contains(self, substring: impl Into<String>, response: MockResponse) -> Self {
        self.when(MessageMatcher::Contains(substring.into()), response)
    }

    /// Set a specific embedding for a query
    pub fn with_embedding(self, query: impl Into<String>, embedding: Vec<f32>) -> Self {
        self.embeddings.lock().insert(query.into(), embedding);
        self
    }

    /// Set embedding dimensions for generated embeddings
    pub fn with_embedding_dims(mut self, dims: usize) -> Self {
        self.embedding_dims = dims;
        self
    }

    /// Get all recorded calls
    pub fn get_recorded_calls(&self) -> Vec<RecordedCall> {
        self.recorded_calls.lock().clone()
    }

    /// Get the number of calls made
    pub fn call_count(&self) -> usize {
        self.recorded_calls.lock().len()
    }

    /// Clear recorded calls
    pub fn clear_recorded_calls(&self) {
        self.recorded_calls.lock().clear();
    }

    /// Reset the response index to start from the beginning
    pub fn reset(&self) {
        self.response_index.store(0, Ordering::SeqCst);
        self.recorded_calls.lock().clear();
    }

    /// Get the last call made
    pub fn last_call(&self) -> Option<RecordedCall> {
        self.recorded_calls.lock().last().cloned()
    }

    /// Check if a specific tool was called
    pub fn was_tool_called(&self, tool_name: &str) -> bool {
        self.recorded_calls
            .lock()
            .iter()
            .any(|call| call.tools.iter().any(|t| t.name == tool_name))
    }

    fn get_next_response(&self, messages: &[LLMMessage]) -> MockResponse {
        // Check conditional responses first
        let conditionals = self.conditional_responses.lock();
        for cond in conditionals.iter() {
            if cond.matcher.matches(messages) {
                return cond.response.clone();
            }
        }
        drop(conditionals);

        // Echo mode
        if self.echo_mode {
            let content = messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            return MockResponse::text(content);
        }

        // Sequential responses
        let responses = self.responses.lock();
        if responses.is_empty() {
            return MockResponse::text("Mock response");
        }

        let index = self.response_index.fetch_add(1, Ordering::SeqCst);
        let actual_index = index % responses.len();
        responses[actual_index].clone()
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMBackend for MockBackend {
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        // Record the call
        if self.config.record_calls {
            self.recorded_calls.lock().push(RecordedCall {
                messages: messages.to_vec(),
                tools: tools.to_vec(),
                temperature,
                timestamp: chrono::Utc::now(),
            });
        }

        let response = self.get_next_response(messages);

        // Simulate latency
        if response.latency_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(response.latency_ms)).await;
        }

        // Check for intentional failure
        if let Some(error_msg) = &response.should_fail {
            return Err(LLMError::ApiError(error_msg.clone()));
        }

        Ok(InferenceOutput {
            content: response.content,
            tool_calls: response.tool_calls,
            reasoning: response.reasoning,
            confidence: response.confidence,
            token_usage: TokenUsage::new(
                self.config.default_prompt_tokens,
                self.config.default_completion_tokens,
            ),
            metadata: HashMap::new(),
        })
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError> {
        // Check for predefined embedding
        if let Some(embedding) = self.embeddings.lock().get(text) {
            return Ok(embedding.clone());
        }

        // Generate deterministic embedding based on text hash
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(text, &mut hasher);
        let hash = std::hash::Hasher::finish(&hasher);

        let mut rng = rand::rngs::StdRng::seed_from_u64(hash);
        use rand::Rng;

        let embedding: Vec<f32> = (0..self.embedding_dims)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        Ok(embedding)
    }

    fn model_info(&self) -> ModelInfo {
        self.config.model_info.clone()
    }

    fn supports_function_calling(&self) -> bool {
        self.config.model_info.supports_functions
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_mock_text_response() {
        let backend = MockBackend::new().with_response(MockResponse::text("Hello from mock!"));

        let messages = vec![LLMMessage::user("Hi")];

        let result = backend.infer(&messages, &[], 0.7).await.unwrap();
        assert_eq!(result.content, "Hello from mock!");
        assert!(result.tool_calls.is_none());
    }

    #[tokio::test]
    async fn test_mock_tool_call_response() {
        let backend = MockBackend::new()
            .with_response(MockResponse::tool_call("search", json!({"query": "rust"})));

        let messages = vec![LLMMessage::user("Search for rust")];

        let result = backend.infer(&messages, &[], 0.7).await.unwrap();
        assert!(result.tool_calls.is_some());

        let tool_calls = result.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "search");
        assert_eq!(tool_calls[0].arguments, json!({"query": "rust"}));
    }

    #[tokio::test]
    async fn test_mock_sequential_responses() {
        let backend = MockBackend::new()
            .with_response(MockResponse::text("First"))
            .with_response(MockResponse::text("Second"))
            .with_response(MockResponse::text("Third"));

        let messages = vec![LLMMessage::user("Hi")];

        let r1 = backend.infer(&messages, &[], 0.7).await.unwrap();
        assert_eq!(r1.content, "First");

        let r2 = backend.infer(&messages, &[], 0.7).await.unwrap();
        assert_eq!(r2.content, "Second");

        let r3 = backend.infer(&messages, &[], 0.7).await.unwrap();
        assert_eq!(r3.content, "Third");

        // Should cycle back
        let r4 = backend.infer(&messages, &[], 0.7).await.unwrap();
        assert_eq!(r4.content, "First");
    }

    #[tokio::test]
    async fn test_mock_echo_mode() {
        let backend = MockBackend::echo();

        let messages = vec![LLMMessage::user("Echo this back")];

        let result = backend.infer(&messages, &[], 0.7).await.unwrap();
        assert_eq!(result.content, "Echo this back");
    }

    #[tokio::test]
    async fn test_mock_conditional_response() {
        let backend = MockBackend::new()
            .when_contains("weather", MockResponse::text("It's sunny!"))
            .when_contains("time", MockResponse::text("It's 3:00 PM"))
            .with_response(MockResponse::text("Default response"));

        let weather_msg = vec![LLMMessage::user("What's the weather?")];
        let time_msg = vec![LLMMessage::user("What time is it?")];
        let other_msg = vec![LLMMessage::user("Hello")];

        let r1 = backend.infer(&weather_msg, &[], 0.7).await.unwrap();
        assert_eq!(r1.content, "It's sunny!");

        let r2 = backend.infer(&time_msg, &[], 0.7).await.unwrap();
        assert_eq!(r2.content, "It's 3:00 PM");

        let r3 = backend.infer(&other_msg, &[], 0.7).await.unwrap();
        assert_eq!(r3.content, "Default response");
    }

    #[tokio::test]
    async fn test_mock_error_response() {
        let backend = MockBackend::new().with_response(MockResponse::error("Simulated API error"));

        let messages = vec![LLMMessage::user("Hi")];

        let result = backend.infer(&messages, &[], 0.7).await;
        assert!(result.is_err());

        match result {
            Err(LLMError::ApiError(msg)) => {
                assert_eq!(msg, "Simulated API error");
            }
            _ => panic!("Expected ApiError"),
        }
    }

    #[tokio::test]
    async fn test_mock_recorded_calls() {
        let backend = MockBackend::new().with_response(MockResponse::text("Response"));

        let messages = vec![LLMMessage::user("Test message")];

        let tools = vec![ToolSchema {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: json!({}),
            dangerous: false,
            metadata: HashMap::new(),
        }];

        backend.infer(&messages, &tools, 0.5).await.unwrap();

        assert_eq!(backend.call_count(), 1);

        let last_call = backend.last_call().unwrap();
        assert_eq!(last_call.messages.len(), 1);
        assert_eq!(last_call.messages[0].content, "Test message");
        assert_eq!(last_call.tools.len(), 1);
        assert_eq!(last_call.tools[0].name, "test_tool");
        assert!((last_call.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_mock_embeddings() {
        let backend = MockBackend::new()
            .with_embedding("hello", vec![0.1, 0.2, 0.3])
            .with_embedding_dims(3);

        // Predefined embedding
        let embed1 = backend.embed("hello").await.unwrap();
        assert_eq!(embed1, vec![0.1, 0.2, 0.3]);

        // Generated embedding (deterministic based on hash)
        let embed2 = backend.embed("world").await.unwrap();
        assert_eq!(embed2.len(), 3);

        // Same input should give same embedding
        let embed3 = backend.embed("world").await.unwrap();
        assert_eq!(embed2, embed3);
    }

    #[tokio::test]
    async fn test_mock_with_latency() {
        let backend =
            MockBackend::new().with_response(MockResponse::text("Delayed").with_latency(100));

        let messages = vec![LLMMessage::user("Hi")];

        let start = std::time::Instant::now();
        backend.infer(&messages, &[], 0.7).await.unwrap();
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() >= 100);
    }

    #[tokio::test]
    async fn test_mock_multiple_tool_calls() {
        let backend = MockBackend::new().with_response(MockResponse::tool_calls(vec![
            ("search", json!({"query": "rust"})),
            ("calculate", json!({"expression": "2+2"})),
        ]));

        let messages = vec![LLMMessage::user("Do stuff")];

        let result = backend.infer(&messages, &[], 0.7).await.unwrap();
        let tool_calls = result.tool_calls.unwrap();

        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].name, "search");
        assert_eq!(tool_calls[1].name, "calculate");
    }

    #[tokio::test]
    async fn test_mock_reset() {
        let backend = MockBackend::new()
            .with_response(MockResponse::text("First"))
            .with_response(MockResponse::text("Second"));

        let messages = vec![LLMMessage::user("Hi")];

        backend.infer(&messages, &[], 0.7).await.unwrap();
        backend.infer(&messages, &[], 0.7).await.unwrap();

        assert_eq!(backend.call_count(), 2);

        backend.reset();

        assert_eq!(backend.call_count(), 0);

        let result = backend.infer(&messages, &[], 0.7).await.unwrap();
        assert_eq!(result.content, "First"); // Back to first response
    }

    #[tokio::test]
    async fn test_mock_tool_call_with_content() {
        let backend = MockBackend::new().with_response(
            MockResponse::tool_call("search", json!({"query": "test"}))
                .with_content("I'll search for that"),
        );

        let messages = vec![LLMMessage::user("Search")];

        let result = backend.infer(&messages, &[], 0.7).await.unwrap();

        assert_eq!(result.content, "I'll search for that");
        assert!(result.tool_calls.is_some());
    }
}
