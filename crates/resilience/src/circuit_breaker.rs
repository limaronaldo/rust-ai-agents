//! Circuit Breaker Pattern
//!
//! Prevents cascading failures by temporarily blocking requests
//! to failing services. Implements the standard three-state pattern:
//! - Closed: Normal operation, requests pass through
//! - Open: Requests blocked due to failures
//! - Half-Open: Testing recovery with limited requests

use parking_lot::RwLock;
use std::future::Future;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// State of the circuit breaker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation - requests allowed
    Closed,
    /// Blocking requests due to failures
    Open,
    /// Testing recovery - limited requests allowed
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "Closed"),
            CircuitState::Open => write!(f, "Open"),
            CircuitState::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

/// Configuration for circuit breaker
#[derive(Clone, Debug)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Number of successes in half-open to close circuit
    pub success_threshold: u32,
    /// Time to wait before trying half-open
    pub timeout: Duration,
    /// Maximum concurrent calls in half-open state
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            half_open_max_calls: 3,
        }
    }
}

impl CircuitBreakerConfig {
    /// Strict config for critical services (LLM providers)
    pub fn strict() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 5,
            timeout: Duration::from_secs(60),
            half_open_max_calls: 1,
        }
    }

    /// Lenient config for less critical services
    pub fn lenient() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 2,
            timeout: Duration::from_secs(15),
            half_open_max_calls: 5,
        }
    }

    /// Config optimized for LLM API calls
    pub fn for_llm() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(45),
            half_open_max_calls: 1,
        }
    }

    /// Config for tool execution
    pub fn for_tools() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(20),
            half_open_max_calls: 2,
        }
    }
}

/// Metrics for circuit breaker
#[derive(Default)]
pub struct CircuitBreakerMetrics {
    pub total_calls: AtomicU64,
    pub successful_calls: AtomicU64,
    pub failed_calls: AtomicU64,
    pub rejected_calls: AtomicU64,
    pub state_transitions: AtomicU64,
}

impl CircuitBreakerMetrics {
    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.total_calls.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        let successful = self.successful_calls.load(Ordering::Relaxed);
        successful as f64 / total as f64
    }

    /// Get failure rate (0.0 to 1.0)
    pub fn failure_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }

    /// Get rejection rate (0.0 to 1.0)
    pub fn rejection_rate(&self) -> f64 {
        let total = self.total_calls.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let rejected = self.rejected_calls.load(Ordering::Relaxed);
        rejected as f64 / total as f64
    }
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    name: String,
    state: RwLock<CircuitState>,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    half_open_calls: AtomicU32,
    last_failure_time: RwLock<Option<Instant>>,
    config: CircuitBreakerConfig,
    metrics: CircuitBreakerMetrics,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(name: impl Into<String>, config: CircuitBreakerConfig) -> Self {
        Self {
            name: name.into(),
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            half_open_calls: AtomicU32::new(0),
            last_failure_time: RwLock::new(None),
            config,
            metrics: CircuitBreakerMetrics::default(),
        }
    }

    /// Create with default config
    pub fn with_defaults(name: impl Into<String>) -> Self {
        Self::new(name, CircuitBreakerConfig::default())
    }

    /// Get circuit name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        *self.state.read()
    }

    /// Get metrics
    pub fn metrics(&self) -> &CircuitBreakerMetrics {
        &self.metrics
    }

    /// Get config
    pub fn config(&self) -> &CircuitBreakerConfig {
        &self.config
    }

    /// Check if the circuit allows execution
    pub fn can_execute(&self) -> bool {
        let state = *self.state.read();

        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = *self.last_failure_time.read() {
                    if last_failure.elapsed() >= self.config.timeout {
                        self.transition_to(CircuitState::HalfOpen);
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => {
                // Allow limited calls in half-open
                let current = self.half_open_calls.load(Ordering::Relaxed);
                current < self.config.half_open_max_calls
            }
        }
    }

    /// Execute an operation through the circuit breaker
    pub async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        self.metrics.total_calls.fetch_add(1, Ordering::Relaxed);

        if !self.can_execute() {
            self.metrics.rejected_calls.fetch_add(1, Ordering::Relaxed);
            return Err(CircuitBreakerError::CircuitOpen {
                name: self.name.clone(),
                state: self.state(),
                retry_after: self.time_until_retry(),
            });
        }

        // Track half-open calls
        if self.state() == CircuitState::HalfOpen {
            self.half_open_calls.fetch_add(1, Ordering::Relaxed);
        }

        match operation().await {
            Ok(result) => {
                self.record_success();
                self.metrics
                    .successful_calls
                    .fetch_add(1, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) => {
                self.record_failure();
                self.metrics.failed_calls.fetch_add(1, Ordering::Relaxed);
                Err(CircuitBreakerError::OperationFailed(e))
            }
        }
    }

    /// Execute with a fallback when circuit is open
    pub async fn execute_with_fallback<F, Fut, T, E, FB, FBFut>(
        &self,
        operation: F,
        fallback: FB,
    ) -> Result<T, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        FB: FnOnce() -> FBFut,
        FBFut: Future<Output = Result<T, E>>,
    {
        match self.execute(operation).await {
            Ok(result) => Ok(result),
            Err(CircuitBreakerError::CircuitOpen { .. }) => fallback().await,
            Err(CircuitBreakerError::OperationFailed(e)) => Err(e),
        }
    }

    /// Time until retry is allowed (when open)
    pub fn time_until_retry(&self) -> Option<Duration> {
        let state = *self.state.read();
        if state != CircuitState::Open {
            return None;
        }

        if let Some(last_failure) = *self.last_failure_time.read() {
            let elapsed = last_failure.elapsed();
            if elapsed < self.config.timeout {
                return Some(self.config.timeout - elapsed);
            }
        }
        None
    }

    /// Record a successful operation
    fn record_success(&self) {
        let state = *self.state.read();

        match state {
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= self.config.success_threshold {
                    self.transition_to(CircuitState::Closed);
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::Open => {
                // Should not happen
            }
        }
    }

    /// Record a failed operation
    fn record_failure(&self) {
        *self.last_failure_time.write() = Some(Instant::now());

        let state = *self.state.read();

        match state {
            CircuitState::Closed => {
                let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= self.config.failure_threshold {
                    self.transition_to(CircuitState::Open);
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open reopens the circuit
                self.transition_to(CircuitState::Open);
            }
            CircuitState::Open => {
                // Already open
            }
        }
    }

    /// Transition to a new state
    fn transition_to(&self, new_state: CircuitState) {
        let old_state = *self.state.read();
        if old_state != new_state {
            *self.state.write() = new_state;
            self.metrics
                .state_transitions
                .fetch_add(1, Ordering::Relaxed);

            // Reset counters on transition
            match new_state {
                CircuitState::Closed => {
                    self.failure_count.store(0, Ordering::Relaxed);
                    self.success_count.store(0, Ordering::Relaxed);
                    self.half_open_calls.store(0, Ordering::Relaxed);
                    info!(
                        circuit = %self.name,
                        "Circuit CLOSED (recovered)"
                    );
                }
                CircuitState::Open => {
                    self.success_count.store(0, Ordering::Relaxed);
                    self.half_open_calls.store(0, Ordering::Relaxed);
                    warn!(
                        circuit = %self.name,
                        failures = %self.failure_count.load(Ordering::Relaxed),
                        "Circuit OPEN after failures"
                    );
                }
                CircuitState::HalfOpen => {
                    self.success_count.store(0, Ordering::Relaxed);
                    self.half_open_calls.store(0, Ordering::Relaxed);
                    info!(
                        circuit = %self.name,
                        "Circuit HALF-OPEN (testing recovery)"
                    );
                }
            }
        }
    }

    /// Manually reset the circuit to closed state
    pub fn reset(&self) {
        self.transition_to(CircuitState::Closed);
        *self.last_failure_time.write() = None;
    }

    /// Force open the circuit (for maintenance)
    pub fn force_open(&self) {
        self.transition_to(CircuitState::Open);
        *self.last_failure_time.write() = Some(Instant::now());
    }

    /// Get status summary
    pub fn status(&self) -> CircuitStatus {
        CircuitStatus {
            name: self.name.clone(),
            state: self.state(),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            total_calls: self.metrics.total_calls.load(Ordering::Relaxed),
            failed_calls: self.metrics.failed_calls.load(Ordering::Relaxed),
            rejected_calls: self.metrics.rejected_calls.load(Ordering::Relaxed),
            success_rate: self.metrics.success_rate(),
            time_until_retry: self.time_until_retry(),
        }
    }
}

/// Error from circuit breaker
#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError<E> {
    #[error("Circuit '{name}' is {state} - requests blocked (retry in {retry_after:?})")]
    CircuitOpen {
        name: String,
        state: CircuitState,
        retry_after: Option<Duration>,
    },

    #[error("Operation failed: {0}")]
    OperationFailed(#[source] E),
}

impl<E> CircuitBreakerError<E> {
    /// Check if this is a circuit open error
    pub fn is_circuit_open(&self) -> bool {
        matches!(self, CircuitBreakerError::CircuitOpen { .. })
    }

    /// Get the underlying operation error if any
    pub fn operation_error(&self) -> Option<&E> {
        match self {
            CircuitBreakerError::OperationFailed(e) => Some(e),
            _ => None,
        }
    }

    /// Get time until retry if circuit is open
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            CircuitBreakerError::CircuitOpen { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

/// Status snapshot of a circuit breaker
#[derive(Debug, Clone)]
pub struct CircuitStatus {
    pub name: String,
    pub state: CircuitState,
    pub failure_count: u32,
    pub success_count: u32,
    pub total_calls: u64,
    pub failed_calls: u64,
    pub rejected_calls: u64,
    pub success_rate: f64,
    pub time_until_retry: Option<Duration>,
}

/// Registry of circuit breakers for multiple services
pub struct CircuitBreakerRegistry {
    breakers: dashmap::DashMap<String, std::sync::Arc<CircuitBreaker>>,
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreakerRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            breakers: dashmap::DashMap::new(),
        }
    }

    /// Register a circuit breaker
    pub fn register(&self, breaker: CircuitBreaker) -> std::sync::Arc<CircuitBreaker> {
        let name = breaker.name().to_string();
        let arc = std::sync::Arc::new(breaker);
        self.breakers.insert(name, arc.clone());
        arc
    }

    /// Get or create a circuit breaker with default config
    pub fn get_or_create(&self, name: &str) -> std::sync::Arc<CircuitBreaker> {
        if let Some(breaker) = self.breakers.get(name) {
            return breaker.clone();
        }

        let breaker = CircuitBreaker::with_defaults(name);
        self.register(breaker)
    }

    /// Get or create with specific config
    pub fn get_or_create_with_config(
        &self,
        name: &str,
        config: CircuitBreakerConfig,
    ) -> std::sync::Arc<CircuitBreaker> {
        if let Some(breaker) = self.breakers.get(name) {
            return breaker.clone();
        }

        let breaker = CircuitBreaker::new(name, config);
        self.register(breaker)
    }

    /// Get a circuit breaker by name
    pub fn get(&self, name: &str) -> Option<std::sync::Arc<CircuitBreaker>> {
        self.breakers.get(name).map(|r| r.clone())
    }

    /// List all registered circuit breakers
    pub fn list(&self) -> Vec<String> {
        self.breakers.iter().map(|r| r.key().clone()).collect()
    }

    /// Get status of all circuit breakers
    pub fn all_status(&self) -> Vec<CircuitStatus> {
        self.breakers.iter().map(|r| r.status()).collect()
    }

    /// Reset all circuit breakers
    pub fn reset_all(&self) {
        for breaker in self.breakers.iter() {
            breaker.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("Test error")]
    struct TestError;

    #[tokio::test]
    async fn test_circuit_starts_closed() {
        let cb = CircuitBreaker::with_defaults("test");
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.can_execute());
    }

    #[tokio::test]
    async fn test_success_keeps_closed() {
        let cb = CircuitBreaker::with_defaults("test");

        for _ in 0..10 {
            let result: Result<i32, CircuitBreakerError<TestError>> =
                cb.execute(|| async { Ok(42) }).await;
            assert!(result.is_ok());
        }

        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.metrics().success_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_failures_open_circuit() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::new("test", config);

        for _ in 0..3 {
            let _: Result<i32, _> = cb.execute(|| async { Err::<i32, _>(TestError) }).await;
        }

        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.can_execute());
    }

    #[tokio::test]
    async fn test_open_circuit_rejects() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let cb = CircuitBreaker::new("test", config);

        // Open the circuit
        let _: Result<i32, _> = cb.execute(|| async { Err::<i32, _>(TestError) }).await;
        assert_eq!(cb.state(), CircuitState::Open);

        // Try to execute - should be rejected
        let result: Result<i32, CircuitBreakerError<TestError>> =
            cb.execute(|| async { Ok(42) }).await;

        assert!(result.unwrap_err().is_circuit_open());
    }

    #[tokio::test]
    async fn test_half_open_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let cb = CircuitBreaker::new("test", config);

        // Open the circuit
        let _: Result<i32, _> = cb.execute(|| async { Err::<i32, _>(TestError) }).await;
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Should transition to half-open
        assert!(cb.can_execute());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_half_open_to_closed() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout: Duration::from_millis(10),
            half_open_max_calls: 5,
        };
        let cb = CircuitBreaker::new("test", config);

        // Open the circuit
        let _: Result<i32, _> = cb.execute(|| async { Err::<i32, _>(TestError) }).await;

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Succeed twice to close
        for _ in 0..2 {
            let _: Result<i32, _> = cb.execute(|| async { Ok::<i32, TestError>(42) }).await;
        }

        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_fallback() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let cb = CircuitBreaker::new("test", config);

        // Open the circuit
        let _: Result<i32, TestError> = cb
            .execute_with_fallback(
                || async { Err(TestError) },
                || async { Ok::<i32, TestError>(0) },
            )
            .await;

        // Now circuit is open, fallback should be used
        let result: Result<i32, TestError> = cb
            .execute_with_fallback(
                || async { Ok::<i32, TestError>(42) },
                || async { Ok::<i32, TestError>(99) },
            )
            .await;

        assert_eq!(result.unwrap(), 99);
    }

    #[tokio::test]
    async fn test_registry() {
        let registry = CircuitBreakerRegistry::new();

        let cb1 = registry.get_or_create("service1");
        let cb2 = registry.get_or_create("service2");

        assert_eq!(cb1.name(), "service1");
        assert_eq!(cb2.name(), "service2");

        // Should return same instance
        let cb1_again = registry.get_or_create("service1");
        assert_eq!(cb1.name(), cb1_again.name());

        assert_eq!(registry.list().len(), 2);
    }
}
