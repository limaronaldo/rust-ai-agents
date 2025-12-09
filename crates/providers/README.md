# rust-ai-agents-providers

LLM provider implementations with rate limiting, retry logic, and multi-provider routing.

## Supported Providers

| Provider | Models | Features |
|----------|--------|----------|
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku | Tools, Vision, Streaming |
| **OpenAI** | GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo | Tools, Vision, Streaming, JSON Mode |
| **OpenRouter** | 200+ models (unified API) | All provider features via proxy |
| **Mock** | Configurable responses | Testing, CI/CD |

## Installation

```toml
[dependencies]
rust-ai-agents-providers = "0.1"

# Optional features
rust-ai-agents-providers = { version = "0.1", features = ["audit"] }
```

## Quick Start

### Anthropic (Claude)

```rust
use rust_ai_agents_providers::{AnthropicProvider, LLMBackend};

// Claude 3.5 Sonnet (recommended)
let claude = AnthropicProvider::claude_35_sonnet("your-api-key");

// Other models
let opus = AnthropicProvider::claude_3_opus("your-api-key");
let haiku = AnthropicProvider::claude_3_haiku("your-api-key");

// Custom model
let provider = AnthropicProvider::new("your-api-key", "claude-3-5-sonnet-20241022");
```

### OpenAI (GPT)

```rust
use rust_ai_agents_providers::OpenAIProvider;

// GPT-4o (recommended)
let gpt4o = OpenAIProvider::new("your-api-key", "gpt-4o");

// GPT-4 Turbo
let gpt4_turbo = OpenAIProvider::new("your-api-key", "gpt-4-turbo");

// GPT-3.5 Turbo (faster, cheaper)
let gpt35 = OpenAIProvider::new("your-api-key", "gpt-3.5-turbo");
```

### OpenRouter (200+ Models)

```rust
use rust_ai_agents_providers::OpenRouterProvider;

// Access any model via OpenRouter
let provider = OpenRouterProvider::new(
    "your-openrouter-key",
    "anthropic/claude-3.5-sonnet",
);

// Llama, Mistral, etc.
let llama = OpenRouterProvider::new("key", "meta-llama/llama-3.1-405b-instruct");
let mistral = OpenRouterProvider::new("key", "mistralai/mistral-large");
```

### Mock Backend (Testing)

```rust
use rust_ai_agents_providers::{MockBackend, MockResponse, MockConfig};

// Simple mock
let mock = MockBackend::new()
    .with_response(MockResponse::text("Hello, I'm a mock!"));

// Sequence of responses
let mock = MockBackend::new()
    .with_responses(vec![
        MockResponse::text("First response"),
        MockResponse::text("Second response"),
        MockResponse::tool_call("search", json!({"query": "test"})),
    ]);

// Pattern matching
let mock = MockBackend::new()
    .with_matcher(|messages| {
        if messages.last()?.content.contains("weather") {
            Some(MockResponse::text("It's sunny!"))
        } else {
            None
        }
    })
    .with_default(MockResponse::text("I don't understand"));

// Record calls for assertions
let mock = MockBackend::new().with_recording(true);
// ... run agent ...
let calls = mock.recorded_calls();
assert_eq!(calls.len(), 3);
```

## Rate Limiting

Governor-based rate limiting with token bucket algorithm:

```rust
use rust_ai_agents_providers::{GovernorRateLimiter, RateLimitConfig};

// Configure rate limits
let config = RateLimitConfig {
    requests_per_minute: 60,
    tokens_per_minute: 100_000,
    burst_size: 10,
};

let limiter = GovernorRateLimiter::new(config);

// Apply to provider
let provider = AnthropicProvider::claude_35_sonnet("key")
    .with_rate_limiter(limiter);
```

### Provider Presets

```rust
// Anthropic defaults: 60 RPM, 100k TPM
let claude = AnthropicProvider::claude_35_sonnet("key"); // includes limiter

// OpenAI defaults: 500 RPM, 150k TPM  
let gpt4 = OpenAIProvider::new("key", "gpt-4o"); // includes limiter
```

## Retry Logic

Exponential backoff retry for transient errors:

```rust
use rust_ai_agents_providers::{with_retry, RetryConfig};

let config = RetryConfig {
    max_retries: 3,
    initial_delay: Duration::from_millis(500),
    max_delay: Duration::from_secs(30),
    multiplier: 2.0,
    retryable_errors: vec![429, 500, 502, 503, 504],
};

// Wrap any async operation
let result = with_retry(config, || async {
    provider.generate(messages, options).await
}).await?;
```

## Multi-Provider Routing (ModelRouter)

Route requests across multiple providers with fallback:

```rust
use rust_ai_agents_providers::{ModelRouter, RouteConfig, FallbackConfig};

let router = ModelRouter::builder()
    // Primary provider
    .add_provider("claude", AnthropicProvider::claude_35_sonnet("key"))
    // Fallback provider
    .add_provider("gpt4", OpenAIProvider::new("key", "gpt-4o"))
    // Cheap provider for simple tasks
    .add_provider("haiku", AnthropicProvider::claude_3_haiku("key"))
    // Configuration
    .default_provider("claude")
    .fallback(FallbackConfig {
        enabled: true,
        providers: vec!["gpt4", "haiku"],
        on_errors: vec![429, 500, 503],
    })
    .build();

// Route to specific provider
let response = router.generate("claude", messages, options).await?;

// Route to default with automatic fallback
let response = router.generate_default(messages, options).await?;
```

### Health Monitoring

```rust
// Check provider health
let health = router.health_status("claude").await;
match health {
    ProviderHealthStatus::Healthy => println!("Claude is up"),
    ProviderHealthStatus::Degraded { latency_ms } => println!("Slow: {}ms", latency_ms),
    ProviderHealthStatus::Unhealthy { reason } => println!("Down: {}", reason),
}

// Get all unhealthy providers
let unhealthy = router.unhealthy_providers().await;
```

## Structured Output

Force JSON schema compliance in responses:

```rust
use rust_ai_agents_providers::{StructuredOutput, SchemaBuilder, StructuredConfig};

// Define schema
let schema = SchemaBuilder::object()
    .property("name", PropertyType::String, true)
    .property("age", PropertyType::Integer, false)
    .property("email", PropertyType::String, true)
    .build();

// Configure structured output
let config = StructuredConfig {
    schema: schema.clone(),
    strict: true,  // Fail if response doesn't match
};

// Generate with schema enforcement
let response = provider.generate_structured(messages, config).await?;
let user: User = serde_json::from_value(response.json)?;
```

## Audited Backend

Wrap any provider with audit logging (requires `audit` feature):

```rust
use rust_ai_agents_providers::AuditedBackend;
use rust_ai_agents_audit::AuditLogger;

let logger = AuditLogger::new("llm-calls.log");
let audited = AuditedBackend::new(provider, logger);

// All calls are now logged with:
// - Timestamp
// - Model used
// - Input messages
// - Output response
// - Token usage
// - Latency
```

## Instrumented Backend

Add tracing instrumentation:

```rust
use rust_ai_agents_providers::InstrumentedBackend;

let instrumented = InstrumentedBackend::new(provider);

// Creates spans for:
// - llm.generate (with model, token counts)
// - llm.stream (with chunk counts)
// - llm.tool_call (with tool names)
```

## Streaming

```rust
use rust_ai_agents_providers::StreamResponse;
use futures::StreamExt;

let stream = provider.stream(messages, options).await?;

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::Text(chunk) => print!("{}", chunk),
        StreamEvent::ToolCall(call) => println!("Tool: {}", call.name),
        StreamEvent::Done(usage) => println!("Tokens: {}", usage.total),
    }
}
```

## Error Handling

```rust
use rust_ai_agents_providers::ProviderError;

match provider.generate(messages, options).await {
    Ok(response) => println!("{}", response.content),
    Err(ProviderError::RateLimited { retry_after }) => {
        println!("Rate limited, retry in {}s", retry_after.as_secs());
    }
    Err(ProviderError::InvalidApiKey) => {
        println!("Check your API key");
    }
    Err(ProviderError::ModelNotFound(model)) => {
        println!("Model {} not available", model);
    }
    Err(ProviderError::ContextLengthExceeded { max, requested }) => {
        println!("Too many tokens: {} > {}", requested, max);
    }
    Err(e) => println!("Error: {}", e),
}
```

## Environment Variables

```bash
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
OPENAI_API_KEY=sk-...

# OpenRouter
OPENROUTER_API_KEY=sk-or-...
```

## Model Comparison

| Model | Speed | Cost | Quality | Context | Best For |
|-------|-------|------|---------|---------|----------|
| Claude 3.5 Sonnet | Fast | $$ | Excellent | 200k | Complex reasoning, coding |
| Claude 3 Opus | Slow | $$$$ | Best | 200k | Research, analysis |
| Claude 3 Haiku | Fastest | $ | Good | 200k | Simple tasks, high volume |
| GPT-4o | Fast | $$ | Excellent | 128k | General purpose, vision |
| GPT-4 Turbo | Medium | $$$ | Excellent | 128k | Complex tasks |
| GPT-3.5 Turbo | Fastest | $ | Good | 16k | Simple tasks, chat |

## Related Documentation

- [Backends Guide](../../docs/backends.md) - Detailed provider documentation
- [Model Router Guide](../../docs/model-router.md) - Multi-provider routing patterns
- [Approvals Guide](../../docs/approvals.md) - Safety patterns

## Related Crates

- [`rust-ai-agents-core`](../core) - Shared types and traits
- [`rust-ai-agents-agents`](../agents) - Agent implementation
- [`rust-ai-agents-monitoring`](../monitoring) - Cost tracking
- [`rust-ai-agents-audit`](../audit) - Audit logging

## License

Apache-2.0
