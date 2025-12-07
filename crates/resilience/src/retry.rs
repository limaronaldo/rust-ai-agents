//! Retry with Exponential Backoff
//!
//! Provides resilient execution of async operations with automatic retries,
//! exponential backoff, and jitter to prevent thundering herd.

use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};

/// Configuration for resilient execution
#[derive(Clone, Debug)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Whether to add jitter to backoff
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create a fast config for testing or low-latency operations
    pub fn fast() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(2),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }

    /// Create an aggressive config for critical operations
    pub fn aggressive() -> Self {
        Self {
            max_retries: 10,
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(60),
            backoff_multiplier: 1.5,
            jitter: true,
        }
    }

    /// Config optimized for LLM API calls
    pub fn for_llm() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }

    /// Config for tool execution
    pub fn for_tools() -> Self {
        Self {
            max_retries: 2,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }

    /// Builder: set max retries
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Builder: set initial backoff
    pub fn with_initial_backoff(mut self, initial_backoff: Duration) -> Self {
        self.initial_backoff = initial_backoff;
        self
    }

    /// Builder: set max backoff
    pub fn with_max_backoff(mut self, max_backoff: Duration) -> Self {
        self.max_backoff = max_backoff;
        self
    }

    /// Builder: set jitter
    pub fn with_jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }
}

/// Trait for errors that can be retried
pub trait RetryableError: std::fmt::Display {
    /// Whether this error should trigger a retry
    fn is_retryable(&self) -> bool;

    /// Suggested retry delay in milliseconds (if any)
    fn retry_delay_hint(&self) -> Option<u64> {
        None
    }
}

/// Error returned when all retries are exhausted
#[derive(Debug, thiserror::Error)]
pub enum RetryError<E> {
    #[error("Max retries ({attempts}) exceeded: {last_error}")]
    MaxRetriesExceeded { attempts: u32, last_error: E },

    #[error("Non-retryable error: {0}")]
    NonRetryable(E),
}

impl<E> RetryError<E> {
    /// Get the underlying error
    pub fn into_inner(self) -> E {
        match self {
            RetryError::MaxRetriesExceeded { last_error, .. } => last_error,
            RetryError::NonRetryable(e) => e,
        }
    }

    /// Check if max retries were exceeded
    pub fn is_max_retries(&self) -> bool {
        matches!(self, RetryError::MaxRetriesExceeded { .. })
    }

    /// Get number of attempts if max retries exceeded
    pub fn attempts(&self) -> Option<u32> {
        match self {
            RetryError::MaxRetriesExceeded { attempts, .. } => Some(*attempts),
            _ => None,
        }
    }
}

/// Executor that wraps operations with retry logic
pub struct RetryExecutor {
    config: RetryConfig,
}

impl RetryExecutor {
    /// Create a new retry executor
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RetryConfig::default())
    }

    /// Execute an operation with retry logic
    pub async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, RetryError<E>>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Display + RetryableError,
    {
        self.execute_with_context("operation", operation).await
    }

    /// Execute with a context name for logging
    pub async fn execute_with_context<F, Fut, T, E>(
        &self,
        context: &str,
        operation: F,
    ) -> Result<T, RetryError<E>>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Display + RetryableError,
    {
        let mut attempt = 0;
        let mut backoff = self.config.initial_backoff;

        loop {
            attempt += 1;

            match operation().await {
                Ok(result) => {
                    if attempt > 1 {
                        info!(
                            context = %context,
                            attempt = %attempt,
                            max_retries = %self.config.max_retries,
                            "Succeeded after retries"
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    // Check if error is retryable
                    if !e.is_retryable() {
                        return Err(RetryError::NonRetryable(e));
                    }

                    // Check if we've exhausted retries
                    if attempt >= self.config.max_retries {
                        return Err(RetryError::MaxRetriesExceeded {
                            attempts: attempt,
                            last_error: e,
                        });
                    }

                    // Calculate wait time
                    let wait = if let Some(hint) = e.retry_delay_hint() {
                        Duration::from_millis(hint)
                    } else if self.config.jitter {
                        self.with_jitter(backoff)
                    } else {
                        backoff
                    };

                    warn!(
                        context = %context,
                        attempt = %attempt,
                        max_retries = %self.config.max_retries,
                        error = %e,
                        wait_ms = %wait.as_millis(),
                        "Retry attempt failed, waiting before next try"
                    );

                    sleep(wait).await;

                    // Increase backoff for next attempt
                    backoff = self.next_backoff(backoff);
                }
            }
        }
    }

    /// Calculate next backoff duration
    fn next_backoff(&self, current: Duration) -> Duration {
        let next = Duration::from_secs_f64(
            (current.as_secs_f64() * self.config.backoff_multiplier)
                .min(self.config.max_backoff.as_secs_f64()),
        );
        next.min(self.config.max_backoff)
    }

    /// Add jitter to a duration (±15%)
    fn with_jitter(&self, duration: Duration) -> Duration {
        let jitter_factor = 0.85 + (rand::random::<f64>() * 0.3); // 0.85 to 1.15
        Duration::from_secs_f64(duration.as_secs_f64() * jitter_factor)
    }
}

/// Helper function for one-off retries
pub async fn retry<F, Fut, T, E>(config: RetryConfig, operation: F) -> Result<T, RetryError<E>>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display + RetryableError,
{
    RetryExecutor::new(config).execute(operation).await
}

/// Helper for quick retries with default config
pub async fn retry_default<F, Fut, T, E>(operation: F) -> Result<T, RetryError<E>>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display + RetryableError,
{
    retry(RetryConfig::default(), operation).await
}

// Implement RetryableError for common error types

impl RetryableError for std::io::Error {
    fn is_retryable(&self) -> bool {
        matches!(
            self.kind(),
            std::io::ErrorKind::ConnectionRefused
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::TimedOut
                | std::io::ErrorKind::Interrupted
        )
    }
}

#[cfg(feature = "reqwest")]
impl RetryableError for reqwest::Error {
    fn is_retryable(&self) -> bool {
        self.is_timeout() || self.is_connect() || self.is_request()
    }
}

/// Wrapper to make any error retryable
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct AlwaysRetryable<E: std::fmt::Display>(pub E);

impl<E: std::fmt::Display> RetryableError for AlwaysRetryable<E> {
    fn is_retryable(&self) -> bool {
        true
    }
}

/// Wrapper to make any error non-retryable
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct NeverRetryable<E: std::fmt::Display>(pub E);

impl<E: std::fmt::Display> RetryableError for NeverRetryable<E> {
    fn is_retryable(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[derive(Debug, thiserror::Error)]
    #[error("Test error: {0}")]
    struct TestError(String, bool);

    impl RetryableError for TestError {
        fn is_retryable(&self) -> bool {
            self.1
        }
    }

    #[tokio::test]
    async fn test_success_first_try() {
        let executor = RetryExecutor::new(RetryConfig::fast());

        let result: Result<i32, RetryError<TestError>> =
            executor.execute(|| async { Ok(42) }).await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_success_after_retries() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let executor = RetryExecutor::new(RetryConfig {
            max_retries: 5,
            initial_backoff: Duration::from_millis(10),
            jitter: false,
            ..RetryConfig::fast()
        });

        let result: Result<i32, RetryError<TestError>> = executor
            .execute(|| {
                let c = counter_clone.clone();
                async move {
                    let attempt = c.fetch_add(1, Ordering::SeqCst) + 1;
                    if attempt < 3 {
                        Err(TestError("transient".into(), true))
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
    async fn test_non_retryable_error() {
        let executor = RetryExecutor::new(RetryConfig::fast());

        let result: Result<i32, RetryError<TestError>> = executor
            .execute(|| async { Err(TestError("permanent".into(), false)) })
            .await;

        assert!(matches!(result, Err(RetryError::NonRetryable(_))));
    }

    #[tokio::test]
    async fn test_max_retries_exceeded() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let executor = RetryExecutor::new(RetryConfig {
            max_retries: 3,
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(50),
            backoff_multiplier: 2.0,
            jitter: false,
        });

        let result: Result<i32, RetryError<TestError>> = executor
            .execute(|| {
                let c = counter_clone.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err(TestError("always fails".into(), true))
                }
            })
            .await;

        assert!(result.unwrap_err().is_max_retries());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_always_retryable_wrapper() {
        #[derive(Debug, thiserror::Error)]
        #[error("custom error")]
        struct CustomError;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let executor = RetryExecutor::new(RetryConfig {
            max_retries: 3,
            initial_backoff: Duration::from_millis(10),
            jitter: false,
            ..RetryConfig::fast()
        });

        let result: Result<i32, RetryError<AlwaysRetryable<CustomError>>> = executor
            .execute(|| {
                let c = counter_clone.clone();
                async move {
                    let attempt = c.fetch_add(1, Ordering::SeqCst) + 1;
                    if attempt < 2 {
                        Err(AlwaysRetryable(CustomError))
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
    }
}
