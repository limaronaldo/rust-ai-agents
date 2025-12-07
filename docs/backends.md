# LLM Backends Guide

This guide covers all LLM backend implementations in HyperAgent.

## Overview

HyperAgent supports multiple LLM providers through a unified `LLMBackend` trait:

```
┌─────────────────────────────────────────────────────────────┐
│                      LLMBackend Trait                       │
├─────────────────────────────────────────────────────────────┤
│  infer()        - Single request inference                  │
│  infer_stream() - Streaming inference with SSE              │
│  embed()        - Text embeddings                           │
│  model_info()   - Model metadata and pricing                │
└─────────────────────────────────────────────────────────────┘
          │              │              │              │
          ▼              ▼              ▼              ▼
    ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────┐
    │  OpenAI  │  │ Anthropic │  │ OpenRouter │  │   Mock   │
    │  GPT-4   │  │  Claude   │  │  200+ LLMs │  │  Testing │
    └──────────┘  └───────────┘  └────────────┘  └──────────┘
```

## Quick Start

```rust
use rust_ai_agents_providers::{AnthropicProvider, OpenAIProvider, LLMBackend};

// Anthropic Claude (recommended)
let claude = AnthropicProvider::claude_35_sonnet(api_key);

// OpenAI GPT-4
let gpt4 = OpenAIProvider::new(api_key, "gpt-4-turbo".to_string());

// Use with agent
let agent = AgentBuilder::new()
    .name("my-agent")
    .backend(claude)
    .build();
```

## Providers

### Anthropic (Claude)

Best for: Complex reasoning, long context, tool use, coding tasks.

```rust
use rust_ai_agents_providers::{AnthropicProvider, ClaudeModel};

// Recommended: Claude 3.5 Sonnet - best balance of speed and intelligence
let claude = AnthropicProvider::claude_35_sonnet(api_key);

// High performance: Claude 3.5 Haiku - fast and efficient
let haiku = AnthropicProvider::claude_35_haiku(api_key);

// Maximum capability: Claude 3 Opus - most powerful
let opus = AnthropicProvider::claude_3_opus(api_key);

// Custom configuration
let custom = AnthropicProvider::new(api_key, ClaudeModel::Claude35Sonnet)
    .with_max_tokens(4096);
```

**Available Models:**

| Model | Context | Input $/1M | Output $/1M | Best For |
|-------|---------|------------|-------------|----------|
| Claude 3.5 Sonnet | 200K | $3.00 | $15.00 | General use, coding |
| Claude 3.5 Haiku | 200K | $1.00 | $5.00 | Fast tasks, high volume |
| Claude 3 Opus | 200K | $15.00 | $75.00 | Complex reasoning |
| Claude 3 Haiku | 200K | $0.25 | $1.25 | Simple tasks |

**Features:**
- Tool/function calling
- Vision (image analysis)
- Extended thinking
- 200K token context window
- Streaming support

### OpenAI (GPT-4)

Best for: General tasks, function calling, established ecosystem.

```rust
use rust_ai_agents_providers::OpenAIProvider;

// GPT-4 Turbo (recommended)
let gpt4 = OpenAIProvider::new(api_key, "gpt-4-turbo".to_string());

// GPT-4o (multimodal)
let gpt4o = OpenAIProvider::new(api_key, "gpt-4o".to_string());

// GPT-3.5 Turbo (budget option)
let gpt35 = OpenAIProvider::new(api_key, "gpt-3.5-turbo".to_string());

// With custom rate limits
let provider = OpenAIProvider::new(api_key, "gpt-4-turbo".to_string())
    .with_rate_limits(500, 150_000); // RPM, TPM
```

**Available Models:**

| Model | Context | Input $/1M | Output $/1M | Best For |
|-------|---------|------------|-------------|----------|
| gpt-4-turbo | 128K | $10.00 | $30.00 | Complex tasks |
| gpt-4o | 128K | $5.00 | $15.00 | Multimodal |
| gpt-4o-mini | 128K | $0.15 | $0.60 | Budget tasks |
| gpt-3.5-turbo | 16K | $0.50 | $1.50 | Simple tasks |

**Features:**
- Tool/function calling
- Vision (GPT-4o)
- JSON mode
- Streaming support

### OpenRouter

Best for: Access to 200+ models with a single API, model comparison.

```rust
use rust_ai_agents_providers::OpenRouterProvider;

// Access any model
let provider = OpenRouterProvider::new(api_key, "anthropic/claude-3.5-sonnet".to_string());

// Llama models
let llama = OpenRouterProvider::new(api_key, "meta-llama/llama-3.1-70b-instruct".to_string());

// Mistral
let mistral = OpenRouterProvider::new(api_key, "mistralai/mistral-large".to_string());
```

**Popular Models via OpenRouter:**

| Model | Provider | Best For |
|-------|----------|----------|
| anthropic/claude-3.5-sonnet | Anthropic | General use |
| openai/gpt-4-turbo | OpenAI | Complex tasks |
| meta-llama/llama-3.1-70b | Meta | Open source |
| mistralai/mistral-large | Mistral | European hosting |
| google/gemini-pro-1.5 | Google | Long context |

### MockBackend (Testing)

Best for: Unit tests, integration tests, deterministic behavior.

```rust
use rust_ai_agents_providers::{MockBackend, MockResponse};
use serde_json::json;

// Simple text responses
let backend = MockBackend::new()
    .with_response(MockResponse::text("Hello, world!"));

// Tool call responses
let backend = MockBackend::new()
    .with_response(MockResponse::tool_call("search", json!({"query": "rust"})));

// Sequence of responses
let backend = MockBackend::new()
    .with_response(MockResponse::text("Let me search for that"))
    .with_response(MockResponse::tool_call("search", json!({"q": "test"})))
    .with_response(MockResponse::text("Here are the results"));

// Pattern-based responses
let backend = MockBackend::new()
    .with_pattern_response(
        MessageMatcher::Contains("weather".to_string()),
        MockResponse::text("It's sunny today!")
    );

// Error simulation
let backend = MockBackend::new()
    .with_response(MockResponse::error("API rate limit exceeded"));

// With simulated latency
let backend = MockBackend::new()
    .with_response(MockResponse::text("Slow response").with_latency(1000));

// Echo mode (returns last user message)
let backend = MockBackend::echo();
```

**Testing Example:**

```rust
#[tokio::test]
async fn test_agent_tool_use() {
    let backend = MockBackend::new()
        .with_response(MockResponse::tool_call("calculator", json!({"expr": "2+2"})))
        .with_response(MockResponse::text("The answer is 4"));

    let agent = AgentBuilder::new()
        .backend(backend.clone())
        .tool(calculator_tool())
        .build();

    let result = agent.run("What is 2+2?").await.unwrap();
    
    // Verify calls were made
    assert_eq!(backend.call_count(), 2);
    assert!(backend.was_tool_called("calculator"));
}
```

## LLMBackend Trait

All providers implement the `LLMBackend` trait:

```rust
#[async_trait]
pub trait LLMBackend: Send + Sync {
    /// Perform inference
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError>;

    /// Streaming inference
    async fn infer_stream(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<StreamResponse, LLMError>;

    /// Generate embeddings
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError>;

    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Check capabilities
    fn supports_function_calling(&self) -> bool;
    fn supports_streaming(&self) -> bool;
}
```

## Inference Output

```rust
pub struct InferenceOutput {
    /// Generated text content
    pub content: String,
    
    /// Tool calls (if any)
    pub tool_calls: Option<Vec<ToolCall>>,
    
    /// Reasoning/chain-of-thought (if available)
    pub reasoning: Option<String>,
    
    /// Model confidence (0.0 - 1.0)
    pub confidence: f64,
    
    /// Token usage statistics
    pub token_usage: TokenUsage,
    
    /// Additional metadata
    pub metadata: HashMap<String, Value>,
}

pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub cached_tokens: Option<usize>,
}
```

## Streaming

Stream responses for real-time display:

```rust
use futures::StreamExt;

let mut stream = backend.infer_stream(&messages, &tools, 0.7).await?;

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta(text) => print!("{}", text),
        StreamEvent::ToolCallStart { id, name } => println!("Calling {}...", name),
        StreamEvent::ToolCallDelta { id, arguments_delta } => { /* accumulate */ },
        StreamEvent::ToolCallEnd { id } => println!("Tool call complete"),
        StreamEvent::Done { token_usage } => println!("Done! Tokens: {:?}", token_usage),
        StreamEvent::Error(e) => eprintln!("Error: {}", e),
    }
}
```

## Rate Limiting

Built-in rate limiting prevents API throttling:

```rust
use rust_ai_agents_providers::{GovernorRateLimiter, RateLimitConfig};

// Configure rate limits
let config = RateLimitConfig::new()
    .requests_per_minute(100)
    .tokens_per_minute(100_000);

let limiter = GovernorRateLimiter::new(config);

// Provider with rate limiting
let provider = AnthropicProvider::claude_35_sonnet(api_key)
    .with_rate_limiter(limiter);
```

**Default Rate Limits by Provider:**

| Provider | RPM | TPM |
|----------|-----|-----|
| Anthropic | 50 | 40,000 |
| OpenAI (GPT-4) | 500 | 150,000 |
| OpenRouter | Varies | Varies |

## Retry Configuration

Automatic retries for transient errors:

```rust
use rust_ai_agents_providers::{with_retry, RetryConfig};

let config = RetryConfig::new()
    .max_retries(3)
    .initial_delay_ms(1000)
    .max_delay_ms(30000)
    .exponential_backoff(true);

let result = with_retry(config, || async {
    backend.infer(&messages, &tools, 0.7).await
}).await?;
```

## Cost Tracking

Track API costs:

```rust
let output = backend.infer(&messages, &tools, 0.7).await?;
let model_info = backend.model_info();

let cost = model_info.calculate_cost(&output.token_usage);
println!("Request cost: ${:.4}", cost);
```

## Environment Variables

Configure providers via environment:

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...
```

```rust
let api_key = std::env::var("ANTHROPIC_API_KEY")?;
let claude = AnthropicProvider::claude_35_sonnet(api_key);
```

## Choosing a Provider

| Use Case | Recommended Provider |
|----------|---------------------|
| General agents | Anthropic Claude 3.5 Sonnet |
| High volume | Anthropic Claude 3.5 Haiku |
| Complex reasoning | Anthropic Claude 3 Opus |
| Budget constraints | OpenAI GPT-4o-mini |
| Model comparison | OpenRouter |
| Testing | MockBackend |

## Next Steps

- [Approvals Guide](./approvals.md) - Human-in-the-loop approval system
- [MCP Integration](../crates/mcp/README.md) - Model Context Protocol
