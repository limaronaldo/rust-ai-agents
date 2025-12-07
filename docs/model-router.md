# Model Router Guide

**Last Updated:** December 7, 2025  
**Status:** Production Ready  
**Version:** 0.1.0

---

## Overview

The `ModelRouter` enables intelligent routing of LLM requests across multiple providers with automatic fallback, health tracking, and route-based configuration.

## Features

- **Route-based configuration**: Different routes can use different models/providers
- **Automatic fallback**: Switch to backup provider on failure
- **Health tracking**: Track provider health and avoid unhealthy providers
- **Builder pattern**: Fluent API for configuration

## Quick Start

```rust
use rust_ai_agents_providers::{
    ModelRouter, RouteConfig, OpenAIProvider, AnthropicProvider,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create router
    let mut router = ModelRouter::new();

    // Add providers
    router.add_provider("openai", Arc::new(
        OpenAIProvider::new(
            std::env::var("OPENAI_API_KEY")?,
            "gpt-4-turbo".to_string(),
        )
    ));
    router.add_provider("anthropic", Arc::new(
        AnthropicProvider::claude_35_sonnet(
            std::env::var("ANTHROPIC_API_KEY")?,
        )
    ));

    // Configure routes
    router.configure_route("chat", RouteConfig::new("anthropic")
        .with_model("claude-3-5-sonnet-20241022")
        .with_fallback("openai", Some("gpt-4-turbo".to_string()))
        .with_retries(2)
        .with_timeout(30_000)
    );

    router.configure_route("embeddings", RouteConfig::new("openai")
        .with_model("text-embedding-3-small")
    );

    // Use router
    let messages = vec![/* ... */];
    let response = router.infer("chat", &messages, &[], 0.7).await?;

    Ok(())
}
```

## Builder Pattern

Use `ModelRouterBuilder` for cleaner configuration:

```rust
use rust_ai_agents_providers::{ModelRouterBuilder, RouteConfig};

let router = ModelRouterBuilder::new()
    .provider("openai", Arc::new(openai_provider))
    .provider("anthropic", Arc::new(anthropic_provider))
    .default_provider("anthropic")
    .route("chat", RouteConfig::new("anthropic")
        .with_fallback("openai", None))
    .route("code", RouteConfig::new("openai")
        .with_model("gpt-4-turbo"))
    .build();
```

## Route Configuration

### RouteConfig Options

| Option | Description | Default |
|--------|-------------|---------|
| `provider` | Primary provider name | Required |
| `model` | Specific model to use | Provider default |
| `fallback` | Fallback provider/model | None |
| `max_retries` | Retries before fallback | 2 |
| `timeout_ms` | Request timeout | 30,000ms |
| `enabled` | Whether route is active | true |

### Example Configurations

**Simple route:**
```rust
RouteConfig::new("openai")
```

**With fallback:**
```rust
RouteConfig::new("anthropic")
    .with_fallback("openai", Some("gpt-4-turbo".to_string()))
```

**Full configuration:**
```rust
RouteConfig::new("anthropic")
    .with_model("claude-3-5-sonnet-20241022")
    .with_fallback("openai", Some("gpt-4-turbo".to_string()))
    .with_retries(3)
    .with_timeout(60_000)
```

## Automatic Fallback

The router automatically falls back to the configured backup provider when:

1. Primary provider returns an error
2. Primary provider is marked unhealthy (3+ consecutive failures in 60 seconds)

```rust
// Configure with fallback
router.configure_route("chat", RouteConfig::new("anthropic")
    .with_fallback("openai", None)
);

// If Anthropic fails, automatically tries OpenAI
let response = router.infer("chat", &messages, &[], 0.7).await?;
```

## Health Tracking

The router tracks provider health automatically:

```rust
// Get health status for all providers
let status = router.health_status();

for (name, health) in status {
    println!("{}: healthy={}, requests={}, failure_rate={:.2}%, avg_latency={}ms",
        name,
        health.healthy,
        health.total_requests,
        health.failure_rate * 100.0,
        health.avg_latency_ms,
    );
}
```

### Health Status Fields

| Field | Type | Description |
|-------|------|-------------|
| `healthy` | bool | Whether provider is considered healthy |
| `total_requests` | u64 | Total requests made |
| `failure_rate` | f64 | Failure rate (0.0 - 1.0) |
| `avg_latency_ms` | f64 | Average latency in milliseconds |
| `consecutive_failures` | u32 | Current consecutive failures |

### Health Algorithm

A provider is considered **unhealthy** when:
- 3 or more consecutive failures
- Most recent failure within last 60 seconds

Health is **restored** when:
- A successful request is made
- 60 seconds pass since last failure

## Streaming Support

```rust
// Streaming with automatic health-based fallback
let stream = router.infer_stream("chat", &messages, &[], 0.7).await?;

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta(text) => print!("{}", text),
        StreamEvent::Done { token_usage } => break,
        _ => {}
    }
}
```

## Integration with Instrumented Backend

Combine with `InstrumentedBackend` for metrics:

```rust
use rust_ai_agents_providers::{InstrumentedBackend, ModelRouter};

// Wrap providers with instrumentation before adding to router
let instrumented_openai = InstrumentedBackend::new(
    openai_provider,
    "openai",
    "chat",
);

let mut router = ModelRouter::new();
router.add_provider("openai", Arc::new(instrumented_openai));
```

## API Reference

### ModelRouter Methods

| Method | Description |
|--------|-------------|
| `new()` | Create new router |
| `add_provider(name, backend)` | Add a provider |
| `set_default(provider, model)` | Set default provider |
| `configure_route(route, config)` | Configure a route |
| `infer(route, messages, tools, temp)` | Perform inference |
| `infer_stream(route, messages, tools, temp)` | Streaming inference |
| `health_status()` | Get all provider health |
| `list_routes()` | List configured routes |
| `has_route(route)` | Check if route exists |
| `provider_names()` | Get all provider names |
| `model_info(provider)` | Get model info |

### ModelRouterBuilder Methods

| Method | Description |
|--------|-------------|
| `new()` | Create new builder |
| `provider(name, backend)` | Add provider |
| `default_provider(name)` | Set default |
| `default_model(model)` | Set default model |
| `route(name, config)` | Add route |
| `build()` | Build router |

## Best Practices

### 1. Always Configure Fallbacks for Critical Routes

```rust
// Good: Critical routes have fallbacks
router.configure_route("chat", RouteConfig::new("anthropic")
    .with_fallback("openai", None)
);

// Bad: No fallback for critical functionality
router.configure_route("chat", RouteConfig::new("anthropic"));
```

### 2. Use Appropriate Timeouts

```rust
// Fast routes (embeddings, simple completions)
RouteConfig::new("openai")
    .with_timeout(10_000)  // 10 seconds

// Complex routes (code generation, long responses)
RouteConfig::new("anthropic")
    .with_timeout(60_000)  // 60 seconds
```

### 3. Monitor Health Status

```rust
// Periodically check health
tokio::spawn(async move {
    loop {
        let status = router.health_status();
        for (name, health) in &status {
            if !health.healthy {
                warn!("Provider {} is unhealthy!", name);
            }
        }
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
});
```

### 4. Use Route Names Consistently

```rust
// Good: Descriptive, consistent route names
router.configure_route("chat/general", config);
router.configure_route("chat/code", config);
router.configure_route("embeddings", config);

// Bad: Inconsistent naming
router.configure_route("Chat", config);
router.configure_route("code-chat", config);
```

## Error Handling

```rust
match router.infer("chat", &messages, &[], 0.7).await {
    Ok(response) => {
        println!("Response: {}", response.content);
    }
    Err(LLMError::ApiError(msg)) => {
        // Both primary and fallback failed
        eprintln!("All providers failed: {}", msg);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Testing

Use `MockBackend` for testing:

```rust
use rust_ai_agents_providers::{MockBackend, MockResponse, ModelRouter};

#[tokio::test]
async fn test_routing() {
    let mock = MockBackend::new()
        .with_response(MockResponse::text("Test response"));

    let mut router = ModelRouter::new();
    router.add_provider("mock", Arc::new(mock));
    router.configure_route("test", RouteConfig::new("mock"));

    let result = router.infer("test", &messages, &[], 0.7).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_fallback() {
    let failing = MockBackend::new()
        .with_response(MockResponse::error("Failed"));
    let fallback = MockBackend::new()
        .with_response(MockResponse::text("Fallback response"));

    let mut router = ModelRouter::new();
    router.add_provider("primary", Arc::new(failing));
    router.add_provider("fallback", Arc::new(fallback));
    router.configure_route("test", RouteConfig::new("primary")
        .with_fallback("fallback", None));

    let result = router.infer("test", &messages, &[], 0.7).await;
    assert_eq!(result.unwrap().content, "Fallback response");
}
```

---

## References

- [LLMBackend Trait](../crates/providers/src/backend.rs)
- [Prometheus Metrics](./prometheus-metrics.md)
- [HyperAgent Roadmap](./ROADMAP.md)

---

**Maintained by:** Ronaldo Lima  
**License:** Apache-2.0
