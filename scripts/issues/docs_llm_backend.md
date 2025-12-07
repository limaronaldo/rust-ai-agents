## Description

Document the existing `LLMBackend` trait from `crates/providers` and the new `MockBackend`.

## Content Outline

### 1. Overview
- What is `LLMBackend`?
- Why pluggable backends matter
- Available implementations

### 2. Quick Start
```rust
use rust_ai_agents_providers::{OpenAIBackend, OpenAIConfig};

let backend = OpenAIBackend::new(OpenAIConfig::default());
let engine = AgentEngine::new(Arc::new(backend), config);
```

### 3. Available Backends

| Backend | Provider | Feature | Notes |
|---------|----------|---------|-------|
| `OpenAIBackend` | OpenAI | - | GPT-4, GPT-3.5 |
| `AnthropicBackend` | Anthropic | - | Claude models |
| `OpenRouterBackend` | OpenRouter | - | Multiple providers |
| `MockBackend` | Testing | - | Deterministic responses |

### 4. MockBackend for Testing
```rust
use rust_ai_agents_providers::{MockBackend, MockStep};

// Simple text response
let backend = MockBackend::text("Hello!");

// Tool call then response
let backend = MockBackend::tool_then_text(
    "search",
    json!({"q": "test"}),
    "Found results.",
);

// Custom sequence
let backend = MockBackend::new(vec![
    MockStep::ToolCall { name: "a".into(), arguments: json!({}) },
    MockStep::ToolCall { name: "b".into(), arguments: json!({}) },
    MockStep::Text("Done.".into()),
]);
```

### 5. Creating a Custom Backend
```rust
#[async_trait]
impl LLMBackend for MyCustomBackend {
    async fn infer(...) -> Result<InferenceOutput, LLMError> {
        // Your implementation
    }
    
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError> {
        // Your implementation
    }
    
    fn model_info(&self) -> ModelInfo {
        // Your implementation
    }
}
```

### 6. Configuration
- Environment variables
- Programmatic config
- Rate limiting

## Checklist

- [ ] Create `docs/llm_backends.md`
- [ ] Document all existing backends
- [ ] Document MockBackend usage patterns
- [ ] Add runnable code examples
- [ ] Link from README.md
- [ ] Link from ROADMAP.md
