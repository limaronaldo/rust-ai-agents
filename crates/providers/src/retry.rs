//! Retry logic with exponential backoff for LLM API calls

use backoff::{backoff::Backoff, ExponentialBackoff, ExponentialBackoffBuilder};
use rust_ai_agents_core::errors::LLMError;
use std::future::Future;
use std::time::Duration;
use tracing::{debug, warn};

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial retry interval
    pub initial_interval: Duration,
    /// Maximum retry interval
    pub max_interval: Duration,
    /// Multiplier for exponential backoff
    pub multiplier: f64,
    /// Maximum total elapsed time for retries
    pub max_elapsed_time: Option<Duration>,
    /// Whether to add randomization to intervals
    pub randomization_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_interval: Duration::from_millis(500),
            max_interval: Duration::from_secs(30),
            multiplier: 2.0,
            max_elapsed_time: Some(Duration::from_secs(120)),
            randomization_factor: 0.5,
        }
    }
}

impl RetryConfig {
    /// Create a new retry config with custom settings
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Set initial interval
    pub fn with_initial_interval(mut self, interval: Duration) -> Self {
        self.initial_interval = interval;
        self
    }

    /// Set max interval
    pub fn with_max_interval(mut self, interval: Duration) -> Self {
        self.max_interval = interval;
        self
    }

    /// Set multiplier
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.multiplier = multiplier;
        self
    }

    /// Build the backoff strategy
    fn build_backoff(&self) -> ExponentialBackoff {
        ExponentialBackoffBuilder::new()
            .with_initial_interval(self.initial_interval)
            .with_max_interval(self.max_interval)
            .with_multiplier(self.multiplier)
            .with_randomization_factor(self.randomization_factor)
            .with_max_elapsed_time(self.max_elapsed_time)
            .build()
    }
}

/// Determines if an error is retryable
pub fn is_retryable(error: &LLMError) -> bool {
    matches!(
        error,
        LLMError::RateLimitExceeded | LLMError::NetworkError(_) | LLMError::Timeout(_)
    )
}

/// Execute a future with retry logic
pub async fn with_retry<F, Fut, T>(
    config: &RetryConfig,
    operation_name: &str,
    mut operation: F,
) -> Result<T, LLMError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LLMError>>,
{
    let mut backoff = config.build_backoff();
    let mut attempt = 0;

    loop {
        attempt += 1;

        match operation().await {
            Ok(result) => {
                if attempt > 1 {
                    debug!(
                        operation = operation_name,
                        attempt, "Operation succeeded after retry"
                    );
                }
                return Ok(result);
            }
            Err(e) => {
                if !is_retryable(&e) {
                    debug!(
                        operation = operation_name,
                        error = %e,
                        "Non-retryable error"
                    );
                    return Err(e);
                }

                if attempt >= config.max_retries {
                    warn!(
                        operation = operation_name,
                        attempt,
                        max_retries = config.max_retries,
                        error = %e,
                        "Max retries exceeded"
                    );
                    return Err(e);
                }

                if let Some(duration) = backoff.next_backoff() {
                    warn!(
                        operation = operation_name,
                        attempt,
                        wait_ms = duration.as_millis(),
                        error = %e,
                        "Retrying after backoff"
                    );
                    tokio::time::sleep(duration).await;
                } else {
                    // Backoff exhausted (max elapsed time reached)
                    warn!(
                        operation = operation_name,
                        error = %e,
                        "Backoff exhausted"
                    );
                    return Err(e);
                }
            }
        }
    }
}

/// A wrapper that adds retry capability to any LLM backend
pub struct RetryWrapper<B> {
    inner: B,
    config: RetryConfig,
}

impl<B> RetryWrapper<B> {
    pub fn new(inner: B, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    pub fn inner(&self) -> &B {
        &self.inner
    }

    pub fn config(&self) -> &RetryConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_on_rate_limit() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = RetryConfig::new(3)
            .with_initial_interval(Duration::from_millis(10))
            .with_max_interval(Duration::from_millis(50));

        let result: Result<(), LLMError> = with_retry(&config, "test_op", || {
            let attempts = attempts_clone.clone();
            async move {
                let current = attempts.fetch_add(1, Ordering::SeqCst);
                if current < 2 {
                    Err(LLMError::RateLimitExceeded)
                } else {
                    Ok(())
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_no_retry_on_auth_error() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = RetryConfig::new(3);

        let result: Result<(), LLMError> = with_retry(&config, "test_op", || {
            let attempts = attempts_clone.clone();
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(LLMError::AuthenticationFailed("Invalid key".to_string()))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1); // No retry
    }
}
