//! Resilience Patterns for rust-ai-agents
//!
//! This crate provides battle-tested resilience patterns for building
//! robust AI agent systems:
//!
//! - **Circuit Breaker**: Prevents cascading failures by blocking requests to failing services
//! - **Retry**: Automatic retries with exponential backoff and jitter
//! - **Rate Limiting**: Token bucket rate limiting with provider-specific presets
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_resilience::{
//!     CircuitBreaker, CircuitBreakerConfig,
//!     RetryExecutor, RetryConfig,
//!     GovernorRateLimiter, RateLimitConfig,
//! };
//!
//! // Circuit breaker for LLM provider
//! let circuit = CircuitBreaker::new("openai", CircuitBreakerConfig::for_llm());
//!
//! // Retry executor for transient failures
//! let retry = RetryExecutor::new(RetryConfig::for_llm());
//!
//! // Rate limiter for API calls
//! let rate_limiter = GovernorRateLimiter::new("openai", RateLimitConfig::openai_tier1());
//!
//! // Combined resilient call
//! let result = circuit.execute(|| async {
//!     rate_limiter.acquire().await;
//!     retry.execute(|| call_llm_api()).await
//! }).await;
//! ```

pub mod circuit_breaker;
pub mod rate_limit;
pub mod retry;

pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerMetrics,
    CircuitBreakerRegistry, CircuitState, CircuitStatus,
};

pub use rate_limit::{GovernorRateLimiter, RateLimitConfig, RateLimitError, RateLimiterRegistry};

pub use retry::{
    AlwaysRetryable, NeverRetryable, RetryConfig, RetryError, RetryExecutor, RetryableError,
};

/// Convenience function to retry and helper
pub use retry::{retry, retry_default};

/// Combined resilience wrapper that applies all patterns
pub struct ResilientExecutor {
    circuit_breaker: Option<std::sync::Arc<CircuitBreaker>>,
    rate_limiter: Option<std::sync::Arc<GovernorRateLimiter>>,
    retry_config: RetryConfig,
}

impl ResilientExecutor {
    /// Create a new resilient executor with all patterns
    pub fn new(
        circuit_breaker: Option<std::sync::Arc<CircuitBreaker>>,
        rate_limiter: Option<std::sync::Arc<GovernorRateLimiter>>,
        retry_config: RetryConfig,
    ) -> Self {
        Self {
            circuit_breaker,
            rate_limiter,
            retry_config,
        }
    }

    /// Create with just retry
    pub fn with_retry(retry_config: RetryConfig) -> Self {
        Self {
            circuit_breaker: None,
            rate_limiter: None,
            retry_config,
        }
    }

    /// Create for LLM calls with sensible defaults
    pub fn for_llm(name: &str) -> Self {
        Self {
            circuit_breaker: Some(std::sync::Arc::new(CircuitBreaker::new(
                name,
                CircuitBreakerConfig::for_llm(),
            ))),
            rate_limiter: Some(std::sync::Arc::new(GovernorRateLimiter::new(
                name,
                RateLimitConfig::openrouter(),
            ))),
            retry_config: RetryConfig::for_llm(),
        }
    }

    /// Execute an operation with all configured resilience patterns
    pub async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, ResilientError<E>>
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display + RetryableError,
    {
        // Rate limit first
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire().await;
        }

        // Then circuit breaker
        if let Some(ref cb) = self.circuit_breaker {
            let retry_executor = RetryExecutor::new(self.retry_config.clone());
            let op = operation.clone();

            cb.execute(|| async { retry_executor.execute(op.clone()).await })
                .await
                .map_err(|e| match e {
                    CircuitBreakerError::CircuitOpen {
                        name,
                        state,
                        retry_after,
                    } => ResilientError::CircuitOpen {
                        name,
                        state,
                        retry_after,
                    },
                    CircuitBreakerError::OperationFailed(retry_err) => match retry_err {
                        RetryError::MaxRetriesExceeded {
                            attempts,
                            last_error,
                        } => ResilientError::MaxRetriesExceeded {
                            attempts,
                            last_error,
                        },
                        RetryError::NonRetryable(e) => ResilientError::NonRetryable(e),
                    },
                })
        } else {
            // Just retry without circuit breaker
            let retry_executor = RetryExecutor::new(self.retry_config.clone());
            retry_executor
                .execute(operation)
                .await
                .map_err(|e| match e {
                    RetryError::MaxRetriesExceeded {
                        attempts,
                        last_error,
                    } => ResilientError::MaxRetriesExceeded {
                        attempts,
                        last_error,
                    },
                    RetryError::NonRetryable(e) => ResilientError::NonRetryable(e),
                })
        }
    }
}

/// Combined error type for resilient execution
#[derive(Debug, thiserror::Error)]
pub enum ResilientError<E> {
    #[error("Circuit '{name}' is {state} (retry in {retry_after:?})")]
    CircuitOpen {
        name: String,
        state: CircuitState,
        retry_after: Option<std::time::Duration>,
    },

    #[error("Max retries ({attempts}) exceeded: {last_error}")]
    MaxRetriesExceeded { attempts: u32, last_error: E },

    #[error("Non-retryable error: {0}")]
    NonRetryable(E),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[derive(Debug, thiserror::Error)]
    #[error("Test error")]
    struct TestError(bool);

    impl RetryableError for TestError {
        fn is_retryable(&self) -> bool {
            self.0
        }
    }

    #[tokio::test]
    async fn test_resilient_executor_success() {
        let executor = ResilientExecutor::with_retry(RetryConfig::fast());

        let result: Result<i32, ResilientError<TestError>> =
            executor.execute(|| async { Ok(42) }).await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_resilient_executor_with_retries() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let executor = ResilientExecutor::with_retry(RetryConfig {
            max_retries: 5,
            initial_backoff: Duration::from_millis(10),
            jitter: false,
            ..RetryConfig::fast()
        });

        let result: Result<i32, ResilientError<TestError>> = executor
            .execute(|| {
                let c = counter_clone.clone();
                async move {
                    let attempt = c.fetch_add(1, Ordering::SeqCst) + 1;
                    if attempt < 3 {
                        Err(TestError(true))
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_resilient_executor_for_llm() {
        let executor = ResilientExecutor::for_llm("test-provider");

        let result: Result<i32, ResilientError<TestError>> =
            executor.execute(|| async { Ok(42) }).await;

        assert_eq!(result.unwrap(), 42);
    }
}
