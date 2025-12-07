//! Rate Limiting with Governor
//!
//! Token bucket rate limiting for API calls with provider-specific presets.
//! Supports both blocking and non-blocking acquisition with timeouts.

use governor::{
    clock::DefaultClock,
    middleware::NoOpMiddleware,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

/// Configuration for rate limiting
#[derive(Clone, Debug)]
pub struct RateLimitConfig {
    /// Requests per period
    pub requests_per_period: u32,
    /// Period duration
    pub period: Duration,
    /// Maximum burst size
    pub burst_size: u32,
}

impl RateLimitConfig {
    /// Create a new rate limit config
    pub fn new(requests_per_period: u32, period: Duration) -> Self {
        Self {
            requests_per_period,
            period,
            burst_size: requests_per_period,
        }
    }

    /// Set burst size
    pub fn with_burst(mut self, burst_size: u32) -> Self {
        self.burst_size = burst_size;
        self
    }

    // === LLM Provider Presets ===

    /// OpenAI GPT-4 tier limits (Tier 1: 500 RPM)
    pub fn openai_tier1() -> Self {
        Self::new(500, Duration::from_secs(60))
    }

    /// OpenAI GPT-4 tier limits (Tier 2: 5000 RPM)
    pub fn openai_tier2() -> Self {
        Self::new(5000, Duration::from_secs(60))
    }

    /// OpenAI GPT-4o-mini limits (higher throughput)
    pub fn openai_mini() -> Self {
        Self::new(10000, Duration::from_secs(60))
    }

    /// Anthropic Claude limits (Tier 1: 60 RPM)
    pub fn anthropic_tier1() -> Self {
        Self::new(60, Duration::from_secs(60))
    }

    /// Anthropic Claude limits (Tier 2: 1000 RPM)
    pub fn anthropic_tier2() -> Self {
        Self::new(1000, Duration::from_secs(60))
    }

    /// Anthropic Claude limits (Tier 3: 2000 RPM)
    pub fn anthropic_tier3() -> Self {
        Self::new(2000, Duration::from_secs(60))
    }

    /// Google Gemini limits
    pub fn gemini() -> Self {
        Self::new(60, Duration::from_secs(60))
    }

    /// OpenRouter limits (varies by model, conservative default)
    pub fn openrouter() -> Self {
        Self::new(200, Duration::from_secs(60))
    }

    /// Groq limits (high throughput)
    pub fn groq() -> Self {
        Self::new(30, Duration::from_secs(60))
    }

    /// Together AI limits
    pub fn together() -> Self {
        Self::new(600, Duration::from_secs(60))
    }

    // === Other Service Presets ===

    /// Search service (Meilisearch, Elasticsearch)
    pub fn search_service() -> Self {
        Self::new(1000, Duration::from_secs(1)).with_burst(100)
    }

    /// Database operations
    pub fn database() -> Self {
        Self::new(500, Duration::from_secs(1)).with_burst(50)
    }

    /// External API (conservative)
    pub fn external_api() -> Self {
        Self::new(30, Duration::from_secs(60))
    }

    /// Conservative default for unknown providers
    pub fn conservative() -> Self {
        Self::new(30, Duration::from_secs(60))
    }

    /// Unlimited (for testing or local services)
    pub fn unlimited() -> Self {
        Self::new(100000, Duration::from_secs(1))
    }
}

type GovernorRateLimiterType = RateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>;

/// Rate limiter using Governor's token bucket algorithm
pub struct GovernorRateLimiter {
    limiter: Arc<GovernorRateLimiterType>,
    config: RateLimitConfig,
    name: String,
}

impl GovernorRateLimiter {
    /// Create a new rate limiter
    pub fn new(name: impl Into<String>, config: RateLimitConfig) -> Self {
        let quota = Self::config_to_quota(&config);
        let limiter = Arc::new(RateLimiter::direct(quota));

        Self {
            limiter,
            config,
            name: name.into(),
        }
    }

    /// Convert config to Governor quota
    fn config_to_quota(config: &RateLimitConfig) -> Quota {
        let period_nanos = config.period.as_nanos() as u64;
        let replenish_interval_nanos = period_nanos / config.requests_per_period as u64;
        let replenish_interval = Duration::from_nanos(replenish_interval_nanos);

        let burst = NonZeroU32::new(config.burst_size).unwrap_or(NonZeroU32::new(1).unwrap());

        Quota::with_period(replenish_interval)
            .expect("Invalid replenish interval")
            .allow_burst(burst)
    }

    /// Get rate limiter name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get current config
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }

    /// Check if a request can proceed (non-blocking)
    pub fn check(&self) -> bool {
        self.limiter.check().is_ok()
    }

    /// Wait until a request can proceed
    pub async fn acquire(&self) {
        loop {
            match self.limiter.check() {
                Ok(_) => {
                    debug!(
                        limiter = %self.name,
                        "Rate limit permit acquired"
                    );
                    return;
                }
                Err(not_until) => {
                    let wait_time = not_until.wait_time_from(governor::clock::Clock::now(
                        &governor::clock::DefaultClock::default(),
                    ));
                    debug!(
                        limiter = %self.name,
                        wait_ms = %wait_time.as_millis(),
                        "Rate limited, waiting"
                    );
                    tokio::time::sleep(wait_time).await;
                }
            }
        }
    }

    /// Try to acquire a permit with timeout
    pub async fn acquire_timeout(&self, timeout: Duration) -> Result<(), RateLimitError> {
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            match self.limiter.check() {
                Ok(_) => {
                    debug!(
                        limiter = %self.name,
                        "Rate limit permit acquired"
                    );
                    return Ok(());
                }
                Err(not_until) => {
                    let wait_time = not_until.wait_time_from(governor::clock::Clock::now(
                        &governor::clock::DefaultClock::default(),
                    ));

                    if tokio::time::Instant::now() + wait_time > deadline {
                        warn!(
                            limiter = %self.name,
                            wait_ms = %wait_time.as_millis(),
                            "Rate limit timeout exceeded"
                        );
                        return Err(RateLimitError::Timeout {
                            name: self.name.clone(),
                            wait_time,
                        });
                    }

                    tokio::time::sleep(wait_time).await;
                }
            }
        }
    }

    /// Execute a function with rate limiting
    pub async fn execute<F, Fut, T>(&self, f: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        self.acquire().await;
        f().await
    }

    /// Execute with timeout on rate limit wait
    pub async fn execute_timeout<F, Fut, T>(
        &self,
        timeout: Duration,
        f: F,
    ) -> Result<T, RateLimitError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        self.acquire_timeout(timeout).await?;
        Ok(f().await)
    }
}

/// Error from rate limiting
#[derive(Debug, thiserror::Error)]
pub enum RateLimitError {
    #[error("Rate limit timeout for '{name}': would need to wait {wait_time:?}")]
    Timeout { name: String, wait_time: Duration },
}

/// Multi-provider rate limiter registry
pub struct RateLimiterRegistry {
    limiters: dashmap::DashMap<String, Arc<GovernorRateLimiter>>,
}

impl Default for RateLimiterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiterRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            limiters: dashmap::DashMap::new(),
        }
    }

    /// Create with common LLM provider presets
    pub fn with_llm_defaults() -> Self {
        let registry = Self::new();

        // OpenAI
        registry.register("openai", RateLimitConfig::openai_tier1());
        registry.register("openai-mini", RateLimitConfig::openai_mini());
        registry.register("gpt-4", RateLimitConfig::openai_tier1());
        registry.register("gpt-4o", RateLimitConfig::openai_tier1());
        registry.register("gpt-4o-mini", RateLimitConfig::openai_mini());

        // Anthropic
        registry.register("anthropic", RateLimitConfig::anthropic_tier1());
        registry.register("claude", RateLimitConfig::anthropic_tier1());
        registry.register("claude-3", RateLimitConfig::anthropic_tier1());
        registry.register("claude-3.5", RateLimitConfig::anthropic_tier2());

        // Google
        registry.register("gemini", RateLimitConfig::gemini());
        registry.register("google", RateLimitConfig::gemini());

        // Others
        registry.register("openrouter", RateLimitConfig::openrouter());
        registry.register("groq", RateLimitConfig::groq());
        registry.register("together", RateLimitConfig::together());

        registry
    }

    /// Register a rate limiter
    pub fn register(&self, name: impl Into<String>, config: RateLimitConfig) {
        let name = name.into();
        let limiter = Arc::new(GovernorRateLimiter::new(name.clone(), config));
        self.limiters.insert(name, limiter);
    }

    /// Get a rate limiter by name
    pub fn get(&self, name: &str) -> Option<Arc<GovernorRateLimiter>> {
        self.limiters.get(name).map(|r| r.clone())
    }

    /// Get or create with default config
    pub fn get_or_default(&self, name: &str) -> Arc<GovernorRateLimiter> {
        if let Some(limiter) = self.get(name) {
            return limiter;
        }

        // Create with conservative default
        let config = RateLimitConfig::conservative();
        let limiter = Arc::new(GovernorRateLimiter::new(name, config));
        self.limiters.insert(name.to_string(), limiter.clone());
        limiter
    }

    /// Get or create with specific config
    pub fn get_or_create(&self, name: &str, config: RateLimitConfig) -> Arc<GovernorRateLimiter> {
        if let Some(limiter) = self.get(name) {
            return limiter;
        }

        let limiter = Arc::new(GovernorRateLimiter::new(name, config));
        self.limiters.insert(name.to_string(), limiter.clone());
        limiter
    }

    /// List all registered limiters
    pub fn list(&self) -> Vec<String> {
        self.limiters.iter().map(|r| r.key().clone()).collect()
    }

    /// Remove a limiter
    pub fn remove(&self, name: &str) -> Option<Arc<GovernorRateLimiter>> {
        self.limiters.remove(name).map(|(_, v)| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_allows_within_limit() {
        let limiter =
            GovernorRateLimiter::new("test", RateLimitConfig::new(10, Duration::from_secs(1)));

        // Should allow 10 requests quickly
        for _ in 0..10 {
            assert!(limiter.check());
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_over_limit() {
        let limiter = GovernorRateLimiter::new(
            "test",
            RateLimitConfig::new(2, Duration::from_secs(1)).with_burst(2),
        );

        // First 2 should pass
        assert!(limiter.check());
        assert!(limiter.check());

        // Third should be blocked
        assert!(!limiter.check());
    }

    #[tokio::test]
    async fn test_rate_limiter_acquire_waits() {
        let limiter = GovernorRateLimiter::new(
            "test",
            RateLimitConfig::new(10, Duration::from_millis(100)).with_burst(1),
        );

        // Use up the burst
        limiter.acquire().await;

        // Next acquire should wait
        let start = std::time::Instant::now();
        limiter.acquire().await;
        let elapsed = start.elapsed();

        // Should have waited approximately 10ms (100ms / 10 requests)
        assert!(elapsed >= Duration::from_millis(5));
    }

    #[tokio::test]
    async fn test_rate_limiter_timeout() {
        let limiter = GovernorRateLimiter::new(
            "test",
            RateLimitConfig::new(1, Duration::from_secs(1)).with_burst(1),
        );

        // Use up the quota
        limiter.acquire().await;

        // Try to acquire with short timeout
        let result = limiter.acquire_timeout(Duration::from_millis(10)).await;
        assert!(matches!(result, Err(RateLimitError::Timeout { .. })));
    }

    #[tokio::test]
    async fn test_registry_with_defaults() {
        let registry = RateLimiterRegistry::with_llm_defaults();

        assert!(registry.get("openai").is_some());
        assert!(registry.get("anthropic").is_some());
        assert!(registry.get("gemini").is_some());
        assert!(registry.get("openrouter").is_some());
    }

    #[tokio::test]
    async fn test_registry_get_or_default() {
        let registry = RateLimiterRegistry::new();

        // Should create a new limiter with conservative config
        let limiter = registry.get_or_default("unknown-provider");
        assert_eq!(limiter.name(), "unknown-provider");

        // Should return the same limiter
        let limiter2 = registry.get_or_default("unknown-provider");
        assert_eq!(limiter.name(), limiter2.name());
    }

    #[test]
    fn test_provider_presets() {
        let openai = RateLimitConfig::openai_tier1();
        assert_eq!(openai.requests_per_period, 500);

        let anthropic = RateLimitConfig::anthropic_tier1();
        assert_eq!(anthropic.requests_per_period, 60);

        let search = RateLimitConfig::search_service();
        assert_eq!(search.requests_per_period, 1000);
        assert_eq!(search.burst_size, 100);
    }
}
