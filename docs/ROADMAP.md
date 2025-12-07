# HyperAgent Roadmap - Advanced Observability & Multi-Provider

**Created:** December 7, 2025  
**Updated:** December 7, 2025  
**Status:** Active Development  
**Priority:** High

---

## Overview

This document outlines the next major features to implement in rust-ai-agents:

1. **Prometheus Metrics** - Real-time observability with Grafana integration ✅ COMPLETE
2. **Model Router** - Multi-provider routing with automatic fallback ✅ COMPLETE

---

## Phase 1: Prometheus Metrics Integration

### Goal
Add production-grade observability with Prometheus metrics, enabling real-time monitoring via Grafana dashboards.

### Dependencies to Add

```toml
# In crates/monitoring/Cargo.toml
[dependencies]
metrics = "0.23"
metrics-exporter-prometheus = "0.15"
metrics-util = "0.16"
```

### Files to Create/Modify

#### 1. Create `crates/monitoring/src/prometheus.rs`

```rust
//! Prometheus metrics integration

use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::time::Duration;

pub const LABEL_ROUTE: &str = "route";
pub const LABEL_MODEL: &str = "model";
pub const LABEL_PROVIDER: &str = "provider";

/// Initialize Prometheus metrics recorder
pub fn init_prometheus() -> PrometheusHandle {
    let builder = PrometheusBuilder::new()
        .idle_timeout(Duration::from_secs(15 * 60));
    
    builder.install_recorder().expect("failed to install Prometheus recorder")
}

/// Record LLM token usage
pub fn record_llm_tokens(route: &str, model: &str, input: u64, output: u64) {
    counter!("llm_tokens_input_total", 
        LABEL_ROUTE => route.to_string(), 
        LABEL_MODEL => model.to_string()
    ).increment(input);
    
    counter!("llm_tokens_output_total",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    ).increment(output);
    
    counter!("llm_tokens_total",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    ).increment(input + output);
}

/// Record LLM cost in USD
pub fn record_llm_cost(route: &str, model: &str, cost_usd: f64) {
    counter!("llm_cost_usd_total",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    ).increment(cost_usd as u64);
}

/// Record LLM latency
pub fn record_llm_latency(route: &str, model: &str, latency_ms: u64) {
    histogram!("llm_latency_ms",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    ).record(latency_ms as f64);
}

/// Increment cache hit counter
pub fn inc_cache_hit(route: &str) {
    counter!("llm_cache_hits_total",
        LABEL_ROUTE => route.to_string()
    ).increment(1);
}

/// Increment cache miss counter
pub fn inc_cache_miss(route: &str) {
    counter!("llm_cache_misses_total",
        LABEL_ROUTE => route.to_string()
    ).increment(1);
}

/// Increment rate limited counter
pub fn inc_rate_limited(route: &str) {
    counter!("llm_rate_limited_total",
        LABEL_ROUTE => route.to_string()
    ).increment(1);
}

/// Set active agents gauge
pub fn set_active_agents(count: u64) {
    gauge!("agents_active_total").set(count as f64);
}

/// Set active sessions gauge
pub fn set_active_sessions(count: u64) {
    gauge!("sessions_active_total").set(count as f64);
}
```

#### 2. Update `crates/monitoring/src/lib.rs`

```rust
pub mod prometheus;
pub use prometheus::{
    init_prometheus, 
    record_llm_tokens, 
    record_llm_cost, 
    record_llm_latency,
    inc_cache_hit, 
    inc_cache_miss, 
    inc_rate_limited,
    set_active_agents,
    set_active_sessions,
};
```

#### 3. Update `crates/dashboard/src/state.rs`

Add `PrometheusHandle` to `DashboardState`:

```rust
use metrics_exporter_prometheus::PrometheusHandle;

pub struct DashboardState {
    // ... existing fields ...
    pub metrics_handle: PrometheusHandle,
}
```

#### 4. Add `/metrics` endpoint in `crates/dashboard/src/handlers.rs`

```rust
use axum::{extract::State, response::IntoResponse};

pub async fn prometheus_metrics_handler(
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let body = state.metrics_handle.render();
    (
        [("Content-Type", "text/plain; version=0.0.4")],
        body,
    )
}
```

#### 5. Add route in `crates/dashboard/src/server.rs`

```rust
.route("/metrics", get(prometheus_metrics_handler))
```

### Metrics Exposed

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_tokens_input_total` | Counter | route, model | Input tokens consumed |
| `llm_tokens_output_total` | Counter | route, model | Output tokens generated |
| `llm_tokens_total` | Counter | route, model | Total tokens |
| `llm_cost_usd_total` | Counter | route, model | Cost in USD |
| `llm_latency_ms` | Histogram | route, model | Request latency |
| `llm_cache_hits_total` | Counter | route | Cache hits |
| `llm_cache_misses_total` | Counter | route | Cache misses |
| `llm_rate_limited_total` | Counter | route | Rate limited requests |
| `agents_active_total` | Gauge | - | Active agents |
| `sessions_active_total` | Gauge | - | Active sessions |

---

## Phase 2: LlmProvider Trait (Multi-Provider)

### Goal
Abstract LLM providers behind a trait to enable:
- Swapping providers without code changes
- Automatic fallback on failure
- A/B testing models by route/user tier

### Files to Create

#### 1. Create `crates/providers/src/types.rs`

```rust
//! Generic LLM types

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct LlmMessage {
    pub role: LlmRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct LlmChatRequest {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    pub temperature: f32,
    pub max_tokens: u16,
    pub stream: bool,
}

#[derive(Debug, Clone)]
pub struct LlmChunk {
    pub content_delta: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Provider error: {0}")]
    Provider(String),
    
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("Rate limited")]
    RateLimited,
    
    #[error("Timeout")]
    Timeout,
    
    #[error("Internal error: {0}")]
    Internal(String),
}
```

#### 2. Create `crates/providers/src/traits.rs`

```rust
//! LLM Provider trait

use std::pin::Pin;
use async_trait::async_trait;
use futures::Stream;

use crate::types::{LlmChatRequest, LlmChunk, LlmError};

/// Type-erased stream for LLM responses
pub type LlmStream = Pin<Box<dyn Stream<Item = Result<LlmChunk, LlmError>> + Send + 'static>>;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Provider name (e.g., "openai", "anthropic")
    fn name(&self) -> &str;
    
    /// Supported models
    fn models(&self) -> Vec<String>;
    
    /// Stream chat completion
    async fn stream_chat(&self, req: LlmChatRequest) -> Result<LlmStream, LlmError>;
    
    /// Non-streaming chat completion
    async fn chat(&self, req: LlmChatRequest) -> Result<String, LlmError> {
        use futures::StreamExt;
        
        let mut stream = self.stream_chat(req).await?;
        let mut result = String::new();
        
        while let Some(chunk) = stream.next().await {
            result.push_str(&chunk?.content_delta);
        }
        
        Ok(result)
    }
}
```

#### 3. Create `crates/providers/src/openai_provider.rs`

```rust
//! OpenAI provider implementation

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use futures::StreamExt;

use crate::traits::{LlmProvider, LlmStream};
use crate::types::{LlmChatRequest, LlmChunk, LlmError, LlmMessage, LlmRole};

pub struct OpenAiProvider {
    client: Client<OpenAIConfig>,
    models: Vec<String>,
}

impl OpenAiProvider {
    pub fn new(api_key: String) -> Self {
        let config = OpenAIConfig::new().with_api_key(api_key);
        let client = Client::with_config(config);
        
        Self {
            client,
            models: vec![
                "gpt-4o".to_string(),
                "gpt-4o-mini".to_string(),
                "gpt-4-turbo".to_string(),
                "gpt-3.5-turbo".to_string(),
            ],
        }
    }
    
    fn map_role(role: &LlmRole) -> &'static str {
        match role {
            LlmRole::System => "system",
            LlmRole::User => "user",
            LlmRole::Assistant => "assistant",
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }
    
    fn models(&self) -> Vec<String> {
        self.models.clone()
    }
    
    async fn stream_chat(&self, req: LlmChatRequest) -> Result<LlmStream, LlmError> {
        // Implementation using async-openai
        // Convert LlmChatRequest -> CreateChatCompletionRequest
        // Map stream to LlmChunk
        todo!()
    }
}
```

#### 4. Create `crates/providers/src/anthropic_provider.rs`

```rust
//! Anthropic Claude provider implementation

use async_trait::async_trait;

use crate::traits::{LlmProvider, LlmStream};
use crate::types::{LlmChatRequest, LlmError};

pub struct AnthropicProvider {
    api_key: String,
    models: Vec<String>,
}

impl AnthropicProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            models: vec![
                "claude-3-5-sonnet-20241022".to_string(),
                "claude-3-5-haiku-20241022".to_string(),
                "claude-3-opus-20240229".to_string(),
            ],
        }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }
    
    fn models(&self) -> Vec<String> {
        self.models.clone()
    }
    
    async fn stream_chat(&self, req: LlmChatRequest) -> Result<LlmStream, LlmError> {
        // Implementation using reqwest for Anthropic API
        todo!()
    }
}
```

#### 5. Create `crates/providers/src/router.rs`

```rust
//! Model router for selecting provider/model by route

use std::collections::HashMap;
use std::sync::Arc;

use crate::traits::LlmProvider;
use crate::types::{LlmChatRequest, LlmError};

pub struct ModelRouter {
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    route_config: HashMap<String, RouteConfig>,
    default_provider: String,
    default_model: String,
}

#[derive(Clone)]
pub struct RouteConfig {
    pub provider: String,
    pub model: String,
    pub fallback_provider: Option<String>,
    pub fallback_model: Option<String>,
}

impl ModelRouter {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            route_config: HashMap::new(),
            default_provider: "openai".to_string(),
            default_model: "gpt-4o-mini".to_string(),
        }
    }
    
    pub fn add_provider(&mut self, name: &str, provider: Arc<dyn LlmProvider>) {
        self.providers.insert(name.to_string(), provider);
    }
    
    pub fn configure_route(&mut self, route: &str, config: RouteConfig) {
        self.route_config.insert(route.to_string(), config);
    }
    
    pub fn get_provider_for_route(&self, route: &str) -> (Arc<dyn LlmProvider>, String) {
        if let Some(config) = self.route_config.get(route) {
            if let Some(provider) = self.providers.get(&config.provider) {
                return (provider.clone(), config.model.clone());
            }
        }
        
        // Default
        let provider = self.providers.get(&self.default_provider)
            .expect("default provider not configured");
        (provider.clone(), self.default_model.clone())
    }
}
```

---

## Implementation Order

### Step 1: Prometheus Metrics ✅ COMPLETE (PR #37)
- Added `crates/monitoring/src/prometheus.rs`
- Added `crates/providers/src/instrumented.rs`
- Added `/metrics` endpoint to dashboard
- Full documentation at `docs/prometheus-metrics.md`

### Step 2: Model Router ✅ COMPLETE (PR #38)
- Added `crates/providers/src/router.rs`
- Route-based provider/model selection
- Automatic fallback on provider failure
- Health tracking with auto-recovery
- Full documentation at `docs/model-router.md`

---

## Docker Compose for Full Stack

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - DATABASE_URL=postgres://postgres:password@postgres:5432/hyperagent
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - postgres
      - redis
      - prometheus

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: hyperagent
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  grafana_data:
```

---

## Success Metrics

After implementing all phases:

| Metric | Before | After |
|--------|--------|-------|
| Observability | Basic logs | Full Prometheus + Grafana ✅ |
| Provider lock-in | OpenAI only | Multi-provider with fallback ✅ |
| Cost visibility | Manual calculation | Real-time dashboards ✅ |
| Latency tracking | None | P50/P90/P99 per route ✅ |

---

## References

- [metrics crate docs](https://docs.rs/metrics)
- [metrics-exporter-prometheus](https://docs.rs/metrics-exporter-prometheus)
- [async-openai](https://docs.rs/async-openai)
- [Anthropic API docs](https://docs.anthropic.com/en/api)
