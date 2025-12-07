//! Prometheus metrics integration for LLM operations
//!
//! This module provides production-grade observability with Prometheus metrics,
//! enabling real-time monitoring via Grafana dashboards.
//!
//! # Metrics Exposed
//!
//! - `llm_tokens_input_total` - Counter of input tokens by route/model
//! - `llm_tokens_output_total` - Counter of output tokens by route/model
//! - `llm_tokens_total` - Counter of total tokens by route/model
//! - `llm_cost_usd_total` - Counter of cost in USD by route/model
//! - `llm_latency_ms` - Histogram of latency in milliseconds
//! - `llm_cache_hits_total` - Counter of cache hits by route
//! - `llm_cache_misses_total` - Counter of cache misses by route
//! - `llm_rate_limited_total` - Counter of rate limit events by provider/model
//! - `llm_errors_total` - Counter of errors by route/model/error_type
//!
//! # Example
//!
//! ```rust,no_run
//! use rust_ai_agents_monitoring::prometheus::{init_prometheus, record_llm_tokens};
//!
//! // Initialize once at startup
//! let handle = init_prometheus();
//!
//! // Record metrics during operation
//! record_llm_tokens("chat", "gpt-4-turbo", 150, 50);
//!
//! // Get metrics in Prometheus format
//! let metrics = handle.render();
//! ```

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::time::Duration;

/// Label key for route (e.g., "chat", "completion", "embeddings")
pub const LABEL_ROUTE: &str = "route";

/// Label key for model (e.g., "gpt-4-turbo", "claude-3-opus")
pub const LABEL_MODEL: &str = "model";

/// Label key for provider (e.g., "openai", "anthropic", "openrouter")
pub const LABEL_PROVIDER: &str = "provider";

/// Label key for error type (e.g., "timeout", "rate_limit", "invalid_request")
pub const LABEL_ERROR_TYPE: &str = "error_type";

/// Initialize Prometheus metrics recorder
///
/// This should be called once at application startup. Returns a handle
/// that can be used to render metrics in Prometheus format.
///
/// # Example
///
/// ```rust,no_run
/// use rust_ai_agents_monitoring::prometheus::init_prometheus;
///
/// let handle = init_prometheus();
/// println!("{}", handle.render());
/// ```
pub fn init_prometheus() -> PrometheusHandle {
    let builder = PrometheusBuilder::new().idle_timeout(
        metrics_util::MetricKindMask::ALL,
        Some(Duration::from_secs(15 * 60)),
    );

    let handle = builder
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    // Describe all metrics for better Prometheus integration
    describe_metrics();

    handle
}

/// Describe all metrics with help text
fn describe_metrics() {
    describe_counter!(
        "llm_tokens_input_total",
        "Total number of input tokens sent to LLM providers"
    );
    describe_counter!(
        "llm_tokens_output_total",
        "Total number of output tokens received from LLM providers"
    );
    describe_counter!(
        "llm_tokens_total",
        "Total number of tokens (input + output) processed"
    );
    describe_counter!("llm_cost_usd_total", "Total cost in USD for LLM operations");
    describe_histogram!("llm_latency_ms", "Latency of LLM requests in milliseconds");
    describe_counter!(
        "llm_cache_hits_total",
        "Total number of cache hits for LLM requests"
    );
    describe_counter!(
        "llm_cache_misses_total",
        "Total number of cache misses for LLM requests"
    );
    describe_counter!(
        "llm_rate_limited_total",
        "Total number of rate limit events from LLM providers"
    );
    describe_counter!(
        "llm_errors_total",
        "Total number of errors during LLM operations"
    );
    describe_gauge!(
        "llm_active_requests",
        "Current number of active LLM requests"
    );
}

/// Record LLM token usage
///
/// # Arguments
///
/// * `route` - The route/endpoint (e.g., "chat", "completion")
/// * `model` - The model name (e.g., "gpt-4-turbo")
/// * `input` - Number of input tokens
/// * `output` - Number of output tokens
///
/// # Example
///
/// ```rust,no_run
/// use rust_ai_agents_monitoring::prometheus::record_llm_tokens;
///
/// record_llm_tokens("chat", "gpt-4-turbo", 150, 50);
/// ```
pub fn record_llm_tokens(route: &str, model: &str, input: u64, output: u64) {
    counter!(
        "llm_tokens_input_total",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    )
    .increment(input);

    counter!(
        "llm_tokens_output_total",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    )
    .increment(output);

    counter!(
        "llm_tokens_total",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    )
    .increment(input + output);
}

/// Record LLM cost in USD
///
/// # Arguments
///
/// * `route` - The route/endpoint
/// * `model` - The model name
/// * `cost_usd` - Cost in USD (will be converted to cents for precision)
///
/// # Example
///
/// ```rust,no_run
/// use rust_ai_agents_monitoring::prometheus::record_llm_cost;
///
/// record_llm_cost("chat", "gpt-4-turbo", 0.0025);
/// ```
pub fn record_llm_cost(route: &str, model: &str, cost_usd: f64) {
    // Store as cents to avoid floating point precision issues
    let cost_cents = (cost_usd * 100.0) as u64;
    counter!(
        "llm_cost_usd_total",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    )
    .increment(cost_cents);
}

/// Record LLM request latency
///
/// # Arguments
///
/// * `route` - The route/endpoint
/// * `model` - The model name
/// * `latency_ms` - Latency in milliseconds
///
/// # Example
///
/// ```rust,no_run
/// use rust_ai_agents_monitoring::prometheus::record_llm_latency;
///
/// record_llm_latency("chat", "gpt-4-turbo", 1250);
/// ```
pub fn record_llm_latency(route: &str, model: &str, latency_ms: u64) {
    histogram!(
        "llm_latency_ms",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    )
    .record(latency_ms as f64);
}

/// Increment cache hit counter
///
/// # Arguments
///
/// * `route` - The route/endpoint that had a cache hit
///
/// # Example
///
/// ```rust,no_run
/// use rust_ai_agents_monitoring::prometheus::inc_cache_hit;
///
/// inc_cache_hit("chat");
/// ```
pub fn inc_cache_hit(route: &str) {
    counter!("llm_cache_hits_total", LABEL_ROUTE => route.to_string()).increment(1);
}

/// Increment cache miss counter
///
/// # Arguments
///
/// * `route` - The route/endpoint that had a cache miss
///
/// # Example
///
/// ```rust,no_run
/// use rust_ai_agents_monitoring::prometheus::inc_cache_miss;
///
/// inc_cache_miss("chat");
/// ```
pub fn inc_cache_miss(route: &str) {
    counter!("llm_cache_misses_total", LABEL_ROUTE => route.to_string()).increment(1);
}

/// Record rate limit event
///
/// # Arguments
///
/// * `provider` - The provider that rate limited (e.g., "openai")
/// * `model` - The model that was rate limited
///
/// # Example
///
/// ```rust,no_run
/// use rust_ai_agents_monitoring::prometheus::record_rate_limit;
///
/// record_rate_limit("openai", "gpt-4-turbo");
/// ```
pub fn record_rate_limit(provider: &str, model: &str) {
    counter!(
        "llm_rate_limited_total",
        LABEL_PROVIDER => provider.to_string(),
        LABEL_MODEL => model.to_string()
    )
    .increment(1);
}

/// Record LLM error
///
/// # Arguments
///
/// * `route` - The route/endpoint where the error occurred
/// * `model` - The model being used
/// * `error_type` - Type of error (e.g., "timeout", "invalid_request")
///
/// # Example
///
/// ```rust,no_run
/// use rust_ai_agents_monitoring::prometheus::record_llm_error;
///
/// record_llm_error("chat", "gpt-4-turbo", "timeout");
/// ```
pub fn record_llm_error(route: &str, model: &str, error_type: &str) {
    counter!(
        "llm_errors_total",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string(),
        LABEL_ERROR_TYPE => error_type.to_string()
    )
    .increment(1);
}

/// Increment active requests gauge
///
/// Call this when starting an LLM request.
///
/// # Example
///
/// ```rust,no_run
/// use rust_ai_agents_monitoring::prometheus::{inc_active_requests, dec_active_requests};
///
/// inc_active_requests("chat", "gpt-4-turbo");
/// // ... do request ...
/// dec_active_requests("chat", "gpt-4-turbo");
/// ```
pub fn inc_active_requests(route: &str, model: &str) {
    gauge!(
        "llm_active_requests",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    )
    .increment(1.0);
}

/// Decrement active requests gauge
///
/// Call this when completing an LLM request.
pub fn dec_active_requests(route: &str, model: &str) {
    gauge!(
        "llm_active_requests",
        LABEL_ROUTE => route.to_string(),
        LABEL_MODEL => model.to_string()
    )
    .decrement(1.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();
    static mut HANDLE: Option<PrometheusHandle> = None;

    fn get_handle() -> &'static PrometheusHandle {
        unsafe {
            INIT.call_once(|| {
                HANDLE = Some(init_prometheus());
            });
            HANDLE.as_ref().unwrap()
        }
    }

    #[test]
    fn test_init_prometheus() {
        let handle = get_handle();
        let metrics = handle.render();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_record_llm_tokens() {
        let handle = get_handle();

        record_llm_tokens("test_route", "test_model", 100, 50);

        let metrics = handle.render();
        assert!(metrics.contains("llm_tokens_input_total"));
        assert!(metrics.contains("llm_tokens_output_total"));
        assert!(metrics.contains("llm_tokens_total"));
    }

    #[test]
    fn test_record_llm_cost() {
        let handle = get_handle();

        record_llm_cost("test_route", "test_model", 0.05);

        let metrics = handle.render();
        assert!(metrics.contains("llm_cost_usd_total"));
    }

    #[test]
    fn test_record_llm_latency() {
        let handle = get_handle();

        record_llm_latency("test_route", "test_model", 1500);

        let metrics = handle.render();
        assert!(metrics.contains("llm_latency_ms"));
    }

    #[test]
    fn test_cache_metrics() {
        let handle = get_handle();

        inc_cache_hit("test_route");
        inc_cache_miss("test_route");

        let metrics = handle.render();
        assert!(metrics.contains("llm_cache_hits_total"));
        assert!(metrics.contains("llm_cache_misses_total"));
    }

    #[test]
    fn test_record_rate_limit() {
        let handle = get_handle();

        record_rate_limit("openai", "gpt-4");

        let metrics = handle.render();
        assert!(metrics.contains("llm_rate_limited_total"));
    }

    #[test]
    fn test_record_llm_error() {
        let handle = get_handle();

        record_llm_error("test_route", "test_model", "timeout");

        let metrics = handle.render();
        assert!(metrics.contains("llm_errors_total"));
    }

    #[test]
    fn test_active_requests() {
        let handle = get_handle();

        inc_active_requests("test_route", "test_model");
        dec_active_requests("test_route", "test_model");

        let metrics = handle.render();
        assert!(metrics.contains("llm_active_requests"));
    }
}
