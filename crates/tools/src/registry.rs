//! Enhanced tool registry with circuit breaker, retry, and timeout

use parking_lot::RwLock;
use rust_ai_agents_core::{
    errors::ToolError,
    tool::{ExecutionContext, Tool, ToolSchema},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Duration to keep circuit open before testing
    pub reset_timeout: Duration,
    /// Number of successes in half-open state to close circuit
    pub success_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            reset_timeout: Duration::from_secs(30),
            success_threshold: 2,
        }
    }
}

/// Circuit breaker state tracker
#[derive(Debug)]
struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<Instant>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            config,
        }
    }

    fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if reset timeout has elapsed
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.config.reset_timeout {
                        self.state = CircuitState::HalfOpen;
                        self.success_count = 0;
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.config.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitState::Open => {}
        }
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        match self.state {
            CircuitState::Closed => {
                if self.failure_count >= self.config.failure_threshold {
                    self.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.success_count = 0;
            }
            CircuitState::Open => {}
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier
    pub multiplier: f64,
    /// Whether to add jitter to backoff
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Calculate backoff duration for given attempt
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        let base = self.initial_backoff.as_millis() as f64;
        let backoff = base * self.multiplier.powi(attempt as i32);
        let capped = backoff.min(self.max_backoff.as_millis() as f64);

        let final_backoff = if self.jitter {
            let jitter = rand_jitter() * 0.3 * capped;
            capped + jitter
        } else {
            capped
        };

        Duration::from_millis(final_backoff as u64)
    }
}

/// Simple jitter using system time
fn rand_jitter() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}

/// Tool execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolStats {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub total_retries: u64,
    pub circuit_breaks: u64,
    pub timeouts: u64,
    pub total_latency_ms: f64,
}

impl ToolStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_calls > 0 {
            self.successful_calls as f64 / self.total_calls as f64
        } else {
            1.0
        }
    }

    pub fn avg_latency_ms(&self) -> f64 {
        if self.successful_calls > 0 {
            self.total_latency_ms / self.successful_calls as f64
        } else {
            0.0
        }
    }
}

/// Tool wrapper with enhanced features
struct EnhancedTool {
    tool: Arc<dyn Tool>,
    circuit_breaker: RwLock<CircuitBreaker>,
    stats: RwLock<ToolStats>,
    retry_config: RetryConfig,
    timeout_duration: Duration,
}

/// Enhanced tool registry with circuit breaker, retry, and timeout
pub struct EnhancedToolRegistry {
    tools: Arc<RwLock<HashMap<String, Arc<EnhancedTool>>>>,
    default_circuit_config: CircuitBreakerConfig,
    default_retry_config: RetryConfig,
    default_timeout: Duration,
}

impl EnhancedToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            default_circuit_config: CircuitBreakerConfig::default(),
            default_retry_config: RetryConfig::default(),
            default_timeout: Duration::from_secs(30),
        }
    }

    /// Configure default circuit breaker settings
    pub fn with_circuit_breaker(mut self, config: CircuitBreakerConfig) -> Self {
        self.default_circuit_config = config;
        self
    }

    /// Configure default retry settings
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.default_retry_config = config;
        self
    }

    /// Configure default timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Register a tool with default settings
    pub fn register(&self, tool: Arc<dyn Tool>) {
        self.register_with_config(
            tool,
            self.default_circuit_config.clone(),
            self.default_retry_config.clone(),
            self.default_timeout,
        );
    }

    /// Register a tool with custom settings
    pub fn register_with_config(
        &self,
        tool: Arc<dyn Tool>,
        circuit_config: CircuitBreakerConfig,
        retry_config: RetryConfig,
        timeout_duration: Duration,
    ) {
        let schema = tool.schema();
        let enhanced = Arc::new(EnhancedTool {
            tool,
            circuit_breaker: RwLock::new(CircuitBreaker::new(circuit_config)),
            stats: RwLock::new(ToolStats::default()),
            retry_config,
            timeout_duration,
        });

        self.tools.write().insert(schema.name.clone(), enhanced);
    }

    /// Execute a tool with circuit breaker, retry, and timeout
    pub async fn execute(
        &self,
        name: &str,
        context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let enhanced_tool = {
            let tools = self.tools.read();
            tools.get(name).cloned()
        };

        let enhanced_tool = enhanced_tool.ok_or_else(|| ToolError::NotFound(name.to_string()))?;

        // Check circuit breaker
        {
            let mut cb = enhanced_tool.circuit_breaker.write();
            if !cb.can_execute() {
                enhanced_tool.stats.write().circuit_breaks += 1;
                return Err(ToolError::CircuitOpen(name.to_string()));
            }
        }

        // Update call count
        enhanced_tool.stats.write().total_calls += 1;

        let start = Instant::now();
        let mut last_error = None;
        let mut retries = 0;

        // Retry loop
        for attempt in 0..=enhanced_tool.retry_config.max_retries {
            if attempt > 0 {
                retries += 1;
                let backoff = enhanced_tool.retry_config.backoff_duration(attempt - 1);
                tokio::time::sleep(backoff).await;
            }

            // Execute with timeout
            let result = timeout(
                enhanced_tool.timeout_duration,
                enhanced_tool.tool.execute(context, arguments.clone()),
            )
            .await;

            match result {
                Ok(Ok(value)) => {
                    // Success
                    let latency = start.elapsed().as_millis() as f64;
                    {
                        let mut stats = enhanced_tool.stats.write();
                        stats.successful_calls += 1;
                        stats.total_retries += retries;
                        stats.total_latency_ms += latency;
                    }
                    enhanced_tool.circuit_breaker.write().record_success();
                    return Ok(value);
                }
                Ok(Err(e)) => {
                    // Tool error
                    last_error = Some(e);
                }
                Err(_) => {
                    // Timeout
                    enhanced_tool.stats.write().timeouts += 1;
                    last_error = Some(ToolError::Timeout(name.to_string()));
                }
            }
        }

        // All retries exhausted
        {
            let mut stats = enhanced_tool.stats.write();
            stats.failed_calls += 1;
            stats.total_retries += retries;
        }
        enhanced_tool.circuit_breaker.write().record_failure();

        Err(last_error.unwrap_or_else(|| ToolError::Execution("Unknown error".to_string())))
    }

    /// Get tool by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.read().get(name).map(|et| et.tool.clone())
    }

    /// List all tool schemas
    pub fn list_schemas(&self) -> Vec<ToolSchema> {
        self.tools
            .read()
            .values()
            .map(|et| et.tool.schema())
            .collect()
    }

    /// Get statistics for a tool
    pub fn get_stats(&self, name: &str) -> Option<ToolStats> {
        self.tools
            .read()
            .get(name)
            .map(|et| et.stats.read().clone())
    }

    /// Get circuit state for a tool
    pub fn get_circuit_state(&self, name: &str) -> Option<CircuitState> {
        self.tools
            .read()
            .get(name)
            .map(|et| et.circuit_breaker.read().state)
    }

    /// Get all tool statistics
    pub fn all_stats(&self) -> HashMap<String, ToolStats> {
        self.tools
            .read()
            .iter()
            .map(|(name, et)| (name.clone(), et.stats.read().clone()))
            .collect()
    }

    /// Reset circuit breaker for a tool
    pub fn reset_circuit(&self, name: &str) -> bool {
        if let Some(et) = self.tools.read().get(name) {
            let mut cb = et.circuit_breaker.write();
            cb.state = CircuitState::Closed;
            cb.failure_count = 0;
            cb.success_count = 0;
            cb.last_failure_time = None;
            true
        } else {
            false
        }
    }

    /// Reset all circuit breakers
    pub fn reset_all_circuits(&self) {
        for et in self.tools.read().values() {
            let mut cb = et.circuit_breaker.write();
            cb.state = CircuitState::Closed;
            cb.failure_count = 0;
            cb.success_count = 0;
            cb.last_failure_time = None;
        }
    }

    /// Check if tool exists
    pub fn has(&self, name: &str) -> bool {
        self.tools.read().contains_key(name)
    }

    /// Get tool count
    pub fn len(&self) -> usize {
        self.tools.read().len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.read().is_empty()
    }

    /// Print tool health report
    pub fn print_health_report(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║               TOOL REGISTRY HEALTH REPORT                    ║");
        println!("╠══════════════════════════════════════════════════════════════╣");

        let tools = self.tools.read();
        for (name, et) in tools.iter() {
            let stats = et.stats.read();
            let cb = et.circuit_breaker.read();

            let state_icon = match cb.state {
                CircuitState::Closed => "🟢",
                CircuitState::HalfOpen => "🟡",
                CircuitState::Open => "🔴",
            };

            println!(
                "║ {} {:<30} {:>6.1}% success              ║",
                state_icon,
                if name.len() > 30 { &name[..30] } else { name },
                stats.success_rate() * 100.0
            );
            println!(
                "║   Calls: {:>8} | Retries: {:>6} | Avg: {:>6.0}ms           ║",
                stats.total_calls,
                stats.total_retries,
                stats.avg_latency_ms()
            );
        }

        println!("╚══════════════════════════════════════════════════════════════╝\n");
    }
}

impl Default for EnhancedToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EnhancedToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
            default_circuit_config: self.default_circuit_config.clone(),
            default_retry_config: self.default_retry_config.clone(),
            default_timeout: self.default_timeout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rust_ai_agents_core::types::AgentId;

    fn test_ctx() -> ExecutionContext {
        ExecutionContext::new(AgentId::new("test-agent"))
    }

    struct TestTool {
        should_fail: std::sync::atomic::AtomicBool,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("test_tool", "A test tool")
        }

        async fn execute(
            &self,
            _context: &ExecutionContext,
            _arguments: serde_json::Value,
        ) -> Result<serde_json::Value, ToolError> {
            if self.should_fail.load(std::sync::atomic::Ordering::SeqCst) {
                Err(ToolError::Execution("Test failure".to_string()))
            } else {
                Ok(serde_json::json!({"result": "success"}))
            }
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let registry = EnhancedToolRegistry::new().with_circuit_breaker(CircuitBreakerConfig {
            failure_threshold: 2,
            reset_timeout: Duration::from_millis(100),
            success_threshold: 1,
        });

        let tool = Arc::new(TestTool {
            should_fail: std::sync::atomic::AtomicBool::new(true),
        });

        registry.register(tool.clone());

        let ctx = test_ctx();

        // Fail twice to open circuit
        let _ = registry
            .execute("test_tool", &ctx, serde_json::json!({}))
            .await;
        let _ = registry
            .execute("test_tool", &ctx, serde_json::json!({}))
            .await;

        // Circuit should be open
        assert_eq!(
            registry.get_circuit_state("test_tool"),
            Some(CircuitState::Open)
        );

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Circuit should transition to half-open
        tool.should_fail
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let result = registry
            .execute("test_tool", &ctx, serde_json::json!({}))
            .await;

        assert!(result.is_ok());
        assert_eq!(
            registry.get_circuit_state("test_tool"),
            Some(CircuitState::Closed)
        );
    }

    #[test]
    fn test_retry_backoff() {
        let config = RetryConfig {
            max_retries: 5,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            multiplier: 2.0,
            jitter: false,
        };

        assert_eq!(config.backoff_duration(0), Duration::from_millis(100));
        assert_eq!(config.backoff_duration(1), Duration::from_millis(200));
        assert_eq!(config.backoff_duration(2), Duration::from_millis(400));
    }
}
