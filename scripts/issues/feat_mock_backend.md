## Description

Create a `MockBackend` that implements the existing `LLMBackend` trait from `crates/providers` for deterministic testing without API calls.

## Context

The `LLMBackend` trait already exists in `crates/providers/src/backend.rs` with implementations for OpenAI, Anthropic, and OpenRouter. We need a mock implementation for tests.

## Acceptance Criteria

- [ ] `MockBackend` implements `LLMBackend` from `crates/providers/src/backend.rs`
- [ ] Configurable scripts (tool call → response sequences)
- [ ] Engine tests can use `MockBackend` instead of real providers

## Implementation

```rust
// crates/providers/src/mock.rs

use crate::{LLMBackend, InferenceOutput, TokenUsage, ModelInfo};
use rust_ai_agents_core::{LLMMessage, ToolSchema, ToolCall, errors::LLMError};
use std::sync::atomic::{AtomicUsize, Ordering};

pub enum MockStep {
    /// Return a text response
    Text(String),
    /// Return a tool call
    ToolCall { name: String, arguments: serde_json::Value },
    /// Return an error
    Error(LLMError),
}

pub struct MockBackend {
    steps: Vec<MockStep>,
    current: AtomicUsize,
    model_info: ModelInfo,
}

impl MockBackend {
    pub fn new(steps: Vec<MockStep>) -> Self;
    
    /// Single text response
    pub fn text(response: impl Into<String>) -> Self;
    
    /// Tool call followed by text response
    pub fn tool_then_text(tool: &str, args: Value, response: impl Into<String>) -> Self;
    
    /// Always return an error
    pub fn error(error: LLMError) -> Self;
}

#[async_trait]
impl LLMBackend for MockBackend {
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        let step_idx = self.current.fetch_add(1, Ordering::SeqCst);
        let step = self.steps.get(step_idx % self.steps.len())
            .expect("MockBackend has no steps configured");
        
        match step {
            MockStep::Text(content) => Ok(InferenceOutput {
                content: content.clone(),
                tool_calls: None,
                reasoning: None,
                confidence: 1.0,
                token_usage: TokenUsage::new(10, 10),
                metadata: Default::default(),
            }),
            MockStep::ToolCall { name, arguments } => Ok(InferenceOutput {
                content: String::new(),
                tool_calls: Some(vec![ToolCall::new(name, arguments.clone())]),
                reasoning: None,
                confidence: 1.0,
                token_usage: TokenUsage::new(10, 10),
                metadata: Default::default(),
            }),
            MockStep::Error(e) => Err(e.clone()),
        }
    }
    
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, LLMError> {
        Ok(vec![0.0; 1536]) // Fake embedding
    }
    
    fn model_info(&self) -> ModelInfo {
        self.model_info.clone()
    }
}
```

## Checklist

- [ ] Create `crates/providers/src/mock.rs`
- [ ] Implement `LLMBackend` trait for `MockBackend`
- [ ] Add convenience constructors (`text()`, `tool_then_text()`, `error()`)
- [ ] Add `pub mod mock;` to `crates/providers/src/lib.rs`
- [ ] Add unit tests in `mock.rs`
- [ ] Update some engine tests to demonstrate `MockBackend` usage
