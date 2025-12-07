# Prometheus Metrics Guide

**Last Updated:** December 7, 2025  
**Status:** Production Ready  
**Version:** 0.1.0

---

## Overview

HyperAgent provides production-grade observability through Prometheus metrics integration. This enables real-time monitoring, alerting, and cost tracking for all LLM operations.

## Quick Start

### 1. Initialize Prometheus

```rust
use rust_ai_agents_monitoring::init_prometheus;
use rust_ai_agents_dashboard::DashboardServer;

// Initialize Prometheus metrics recorder (call once at startup)
let prometheus_handle = init_prometheus();

// Start dashboard with /metrics endpoint
let server = DashboardServer::new(cost_tracker);
server.run("0.0.0.0:8080").await?;
```

### 2. Instrument LLM Providers

Wrap any LLM backend with `InstrumentedBackend` to automatically record metrics:

```rust
use rust_ai_agents_providers::{OpenAIProvider, InstrumentedBackend};

// Create base provider
let openai = OpenAIProvider::new(api_key, "gpt-4-turbo".to_string());

// Wrap with instrumentation
let instrumented = InstrumentedBackend::new(
    openai,
    "openai",  // provider label
    "chat"     // route label
);

// All calls now record metrics automatically
let response = instrumented.infer(&messages, &tools, 0.7).await?;
```

### 3. Access Metrics

**Prometheus Endpoint:**
```
GET http://localhost:8080/metrics
```

**Grafana Dashboard:** Import the provided dashboard (see [Grafana Setup](#grafana-setup))

---

## Available Metrics

### Token Usage

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_tokens_input_total` | Counter | `route`, `model` | Total input tokens sent to LLM |
| `llm_tokens_output_total` | Counter | `route`, `model` | Total output tokens received |
| `llm_tokens_total` | Counter | `route`, `model` | Total tokens (input + output) |

**Example Query:**
```promql
# Total tokens by model (last 24h)
sum by (model) (increase(llm_tokens_total[24h]))
```

### Cost Tracking

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_cost_usd_total` | Counter | `route`, `model` | Total cost in USD (stored as cents) |

**Example Query:**
```promql
# Total cost by route (last 7 days)
sum by (route) (increase(llm_cost_usd_total[7d])) / 100
```

**Supported Models & Pricing:**
- GPT-4 Turbo: $0.01/1K input, $0.03/1K output
- GPT-4: $0.03/1K input, $0.06/1K output
- GPT-3.5 Turbo: $0.0005/1K input, $0.0015/1K output
- Claude 3.5 Sonnet: $0.003/1K input, $0.015/1K output
- Claude 3 Opus: $0.015/1K input, $0.075/1K output
- Claude 3 Haiku: $0.00025/1K input, $0.00125/1K output

### Latency

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_latency_ms` | Histogram | `route`, `model` | Request latency in milliseconds |

**Example Queries:**
```promql
# P50 latency by model
histogram_quantile(0.5, sum by (model, le) (rate(llm_latency_ms_bucket[5m])))

# P90 latency by model
histogram_quantile(0.9, sum by (model, le) (rate(llm_latency_ms_bucket[5m])))

# P99 latency by model
histogram_quantile(0.99, sum by (model, le) (rate(llm_latency_ms_bucket[5m])))
```

### Cache Performance

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_cache_hits_total` | Counter | `route` | Cache hit count |
| `llm_cache_misses_total` | Counter | `route` | Cache miss count |

**Example Query:**
```promql
# Cache hit rate by route
sum by (route) (rate(llm_cache_hits_total[5m])) 
  / 
(sum by (route) (rate(llm_cache_hits_total[5m])) + sum by (route) (rate(llm_cache_misses_total[5m])))
```

### Rate Limiting

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_rate_limited_total` | Counter | `provider`, `model` | Rate limit events |

**Example Query:**
```promql
# Rate limit events per hour
sum by (provider, model) (increase(llm_rate_limited_total[1h]))
```

### Errors

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_errors_total` | Counter | `route`, `model`, `error_type` | Error count by type |

**Error Types:**
- `rate_limit` - Rate limit exceeded
- `auth_failed` - Authentication failed
- `network` - Network error
- `invalid_response` - Invalid API response
- `api_error` - API error
- `serialization` - Serialization error
- `validation` - Validation error
- `timeout` - Request timeout
- `token_limit` - Token limit exceeded
- `model_unavailable` - Model not available

**Example Query:**
```promql
# Error rate by type (per minute)
sum by (error_type) (rate(llm_errors_total[1m]))
```

### Active Requests

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_active_requests` | Gauge | `route`, `model` | Current active requests |

**Example Query:**
```promql
# Active requests by model
sum by (model) (llm_active_requests)
```

---

## Manual Metrics Recording

For custom use cases, you can record metrics manually:

```rust
use rust_ai_agents_monitoring::prometheus::{
    record_llm_tokens,
    record_llm_cost,
    record_llm_latency,
    inc_cache_hit,
    inc_cache_miss,
    record_rate_limit,
    record_llm_error,
    inc_active_requests,
    dec_active_requests,
};

// Record token usage
record_llm_tokens("custom_route", "gpt-4-turbo", 150, 50);

// Record cost
record_llm_cost("custom_route", "gpt-4-turbo", 0.0025);

// Record latency
record_llm_latency("custom_route", "gpt-4-turbo", 1250);

// Cache metrics
inc_cache_hit("custom_route");
inc_cache_miss("custom_route");

// Rate limit event
record_rate_limit("openai", "gpt-4-turbo");

// Error
record_llm_error("custom_route", "gpt-4-turbo", "timeout");

// Active requests (always pair inc/dec)
inc_active_requests("custom_route", "gpt-4-turbo");
// ... do work ...
dec_active_requests("custom_route", "gpt-4-turbo");
```

---

## Grafana Setup

### 1. Add Prometheus Data Source

In Grafana:
1. Go to **Configuration > Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. URL: `http://localhost:9090` (or your Prometheus server)
5. Click **Save & Test**

### 2. Import Dashboard

Create a new dashboard with these panels:

#### Panel 1: Total Cost (Last 24h)

```promql
sum(increase(llm_cost_usd_total[24h])) / 100
```

**Visualization:** Stat  
**Unit:** USD ($)

#### Panel 2: Token Usage by Model

```promql
sum by (model) (increase(llm_tokens_total[24h]))
```

**Visualization:** Time series  
**Unit:** Tokens

#### Panel 3: Latency Percentiles

```promql
# P50
histogram_quantile(0.5, sum by (le) (rate(llm_latency_ms_bucket[5m])))

# P90
histogram_quantile(0.9, sum by (le) (rate(llm_latency_ms_bucket[5m])))

# P99
histogram_quantile(0.99, sum by (le) (rate(llm_latency_ms_bucket[5m])))
```

**Visualization:** Time series  
**Unit:** Milliseconds (ms)

#### Panel 4: Cache Hit Rate

```promql
sum(rate(llm_cache_hits_total[5m])) 
  / 
(sum(rate(llm_cache_hits_total[5m])) + sum(rate(llm_cache_misses_total[5m])))
```

**Visualization:** Gauge  
**Unit:** Percent (0-1)

#### Panel 5: Error Rate

```promql
sum by (error_type) (rate(llm_errors_total[1m]))
```

**Visualization:** Time series  
**Unit:** Errors/sec

#### Panel 6: Active Requests

```promql
sum by (model) (llm_active_requests)
```

**Visualization:** Time series  
**Unit:** Requests

### 3. Alerting Rules

Create alerts in Prometheus (`/etc/prometheus/rules.yml`):

```yaml
groups:
  - name: llm_alerts
    interval: 30s
    rules:
      # High cost alert
      - alert: HighLLMCost
        expr: increase(llm_cost_usd_total[1h]) / 100 > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM cost detected"
          description: "Cost exceeded $50 in the last hour"

      # High error rate
      - alert: HighLLMErrorRate
        expr: sum(rate(llm_errors_total[5m])) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High LLM error rate"
          description: "Error rate is {{ $value }} errors/sec"

      # High latency
      - alert: HighLLMLatency
        expr: histogram_quantile(0.99, sum by (le) (rate(llm_latency_ms_bucket[5m]))) > 5000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High LLM latency (P99 > 5s)"
          description: "P99 latency is {{ $value }}ms"

      # Rate limiting
      - alert: LLMRateLimited
        expr: increase(llm_rate_limited_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Frequent rate limiting"
          description: "{{ $value }} rate limit events in 5 minutes"
```

---

## Architecture

### Metrics Flow

```
┌─────────────────────────────────────────────────────────┐
│                  LLM Request Flow                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. inc_active_requests()                               │
│         ↓                                               │
│  2. Execute LLM call (infer/infer_stream)               │
│         ↓                                               │
│  3. record_llm_latency()                                │
│  4. record_llm_tokens()                                 │
│  5. record_llm_cost()                                   │
│  6. record_llm_error() (if error)                       │
│  7. record_rate_limit() (if rate limited)               │
│         ↓                                               │
│  8. dec_active_requests()                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Global Recorder

Prometheus metrics use a **global recorder** that must be initialized once at application startup:

```rust
use rust_ai_agents_monitoring::init_prometheus;

// Call once in main()
let handle = init_prometheus();

// The handle can be used to render metrics
let metrics_text = handle.render();
```

**Important:** `init_prometheus()` can only be called once per process. Subsequent calls will panic. This is by design to ensure a single global metrics registry.

---

## Best Practices

### 1. Label Cardinality

Keep label values bounded to avoid high cardinality:

✅ **Good:**
```rust
// Fixed set of routes
record_llm_tokens("chat", model, input, output);
record_llm_tokens("completion", model, input, output);
record_llm_tokens("embeddings", model, input, output);
```

❌ **Bad:**
```rust
// Unbounded user IDs as labels (creates millions of time series)
record_llm_tokens(&format!("user_{}", user_id), model, input, output);
```

### 2. Cost Tracking

Costs are stored as **cents** (multiplied by 100) to avoid floating-point precision issues:

```rust
// Internally: 0.0025 USD → 0.25 cents → stored as 0 (rounded)
// Better: handle stores full precision, query divides by 100
record_llm_cost("chat", "gpt-4-turbo", 0.0025);

// Query in Grafana
sum(llm_cost_usd_total) / 100  // Convert back to USD
```

### 3. Active Requests

Always pair `inc_active_requests()` and `dec_active_requests()`:

```rust
inc_active_requests(route, model);

// Use defer pattern or RAII
let _guard = scopeguard::guard((), |_| {
    dec_active_requests(route, model);
});

// Or manually
match result {
    Ok(_) | Err(_) => {
        dec_active_requests(route, model);
    }
}
```

### 4. Streaming Metrics

For streaming responses, record metrics when the stream completes:

```rust
let stream = backend.infer_stream(&messages, &tools, 0.7).await?;

let instrumented_stream = stream.map(|event| {
    if let Ok(StreamEvent::Done { token_usage }) = &event {
        if let Some(usage) = token_usage {
            record_llm_tokens(route, model, usage.prompt_tokens, usage.completion_tokens);
            // ... record cost, etc.
        }
        dec_active_requests(route, model);
    }
    event
});
```

---

## Performance Impact

### Overhead

Prometheus metrics add **minimal overhead**:

- Counter increment: ~10ns
- Histogram record: ~50ns
- Gauge update: ~10ns

For a typical LLM request (1-5 seconds), metrics add **<1μs total overhead** (~0.0001%).

### Memory Usage

Each unique label combination creates a new time series:

- **Typical:** 10 routes × 5 models × 8 metrics = **400 time series**
- **Memory:** ~100KB total for time series metadata

### Scrape Interval

Recommended Prometheus scrape interval: **15-30 seconds**

---

## Troubleshooting

### Metrics Not Appearing

**Problem:** `/metrics` endpoint returns empty or missing metrics.

**Solution:**
1. Verify `init_prometheus()` was called at startup
2. Check that instrumented backend is being used
3. Make at least one LLM request to populate metrics
4. Verify Prometheus is scraping the correct endpoint

### Duplicate Initialization Error

**Problem:** `failed to install Prometheus recorder: FailedToSetGlobalRecorder`

**Solution:** `init_prometheus()` can only be called once. Store the handle globally or pass it to components that need it:

```rust
// Store in application state
struct AppState {
    prometheus_handle: PrometheusHandle,
}

// Or use lazy_static
lazy_static! {
    static ref PROMETHEUS: PrometheusHandle = init_prometheus();
}
```

### High Cardinality Warning

**Problem:** Prometheus shows "high cardinality" warnings.

**Solution:** Reduce label values. Never use unbounded values (user IDs, session IDs) as labels. Use fixed routes/models only.

---

## Examples

### Complete Example

See `examples/prometheus_monitoring.rs`:

```rust
use rust_ai_agents_monitoring::init_prometheus;
use rust_ai_agents_providers::{OpenAIProvider, InstrumentedBackend};
use rust_ai_agents_dashboard::DashboardServer;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize Prometheus
    let prometheus_handle = init_prometheus();

    // 2. Create instrumented LLM backend
    let openai = OpenAIProvider::new(api_key, "gpt-4-turbo".to_string());
    let llm = InstrumentedBackend::new(openai, "openai", "chat");

    // 3. Start dashboard with /metrics endpoint
    let server = DashboardServer::new(cost_tracker);
    tokio::spawn(async move {
        server.run("0.0.0.0:8080").await.unwrap();
    });

    // 4. Use the LLM - metrics recorded automatically
    let response = llm.infer(&messages, &tools, 0.7).await?;

    // 5. Access metrics
    println!("Metrics:\n{}", prometheus_handle.render());

    Ok(())
}
```

---

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [HyperAgent Roadmap](./ROADMAP.md)

---

**Maintained by:** Ronaldo Lima  
**License:** Apache-2.0
