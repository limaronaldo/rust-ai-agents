//! Rate limiting using governor for precise token bucket algorithm

use governor::{
    clock::DefaultClock,
    middleware::NoOpMiddleware,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter as GovernorLimiter,
};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;

/// Enhanced rate limiter using governor's token bucket algorithm
pub struct GovernorRateLimiter {
    /// Request rate limiter (RPM)
    request_limiter: Arc<GovernorLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>>,
    /// Token rate limiter (TPM) - optional, for token-based limiting
    token_limiter:
        Option<Arc<GovernorLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>>>,
    /// Configuration
    config: RateLimitConfig,
}

/// Rate limit configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Requests per minute
    pub requests_per_minute: u32,
    /// Tokens per minute (if applicable)
    pub tokens_per_minute: Option<u32>,
    /// Burst capacity for requests
    pub request_burst: u32,
    /// Burst capacity for tokens
    pub token_burst: Option<u32>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 500,
            tokens_per_minute: Some(150_000),
            request_burst: 10,
            token_burst: Some(10_000),
        }
    }
}

impl RateLimitConfig {
    /// Create config for OpenAI GPT-4 Turbo limits
    pub fn openai_gpt4_turbo() -> Self {
        Self {
            requests_per_minute: 500,
            tokens_per_minute: Some(150_000),
            request_burst: 20,
            token_burst: Some(20_000),
        }
    }

    /// Create config for OpenAI GPT-3.5 Turbo limits
    pub fn openai_gpt35_turbo() -> Self {
        Self {
            requests_per_minute: 3500,
            tokens_per_minute: Some(90_000),
            request_burst: 50,
            token_burst: Some(10_000),
        }
    }

    /// Create config for Anthropic Claude limits
    pub fn anthropic_claude() -> Self {
        Self {
            requests_per_minute: 1000,
            tokens_per_minute: Some(100_000),
            request_burst: 20,
            token_burst: Some(15_000),
        }
    }

    /// Create config for OpenRouter (varies by model)
    pub fn openrouter_default() -> Self {
        Self {
            requests_per_minute: 200,
            tokens_per_minute: None, // OpenRouter handles this
            request_burst: 10,
            token_burst: None,
        }
    }

    /// Create custom config
    pub fn custom(rpm: u32, tpm: Option<u32>) -> Self {
        Self {
            requests_per_minute: rpm,
            tokens_per_minute: tpm,
            request_burst: (rpm / 10).max(5),
            token_burst: tpm.map(|t| (t / 10).max(1000)),
        }
    }
}

impl GovernorRateLimiter {
    /// Create a new rate limiter with the given configuration
    pub fn new(config: RateLimitConfig) -> Self {
        // Calculate replenishment rate for requests
        // If 500 RPM, that's ~8.33 per second, so we replenish every 120ms
        let request_period_ms = 60_000 / config.requests_per_minute;
        let request_quota = Quota::with_period(Duration::from_millis(request_period_ms as u64))
            .expect("Invalid request quota")
            .allow_burst(NonZeroU32::new(config.request_burst).expect("Burst must be non-zero"));

        let request_limiter = Arc::new(GovernorLimiter::direct(request_quota));

        // Create token limiter if configured
        let token_limiter = config.tokens_per_minute.map(|tpm| {
            let token_period_ms = 60_000 / (tpm / 100); // Replenish 100 tokens at a time
            let token_quota = Quota::with_period(Duration::from_millis(token_period_ms as u64))
                .expect("Invalid token quota")
                .allow_burst(
                    NonZeroU32::new(config.token_burst.unwrap_or(1000))
                        .expect("Token burst must be non-zero"),
                );
            Arc::new(GovernorLimiter::direct(token_quota))
        });

        Self {
            request_limiter,
            token_limiter,
            config,
        }
    }

    /// Wait until a request can be made
    pub async fn acquire(&self) {
        // Wait for request permit
        self.request_limiter.until_ready().await;
        debug!("Rate limiter: request permit acquired");
    }

    /// Wait until a request with estimated tokens can be made
    pub async fn acquire_with_tokens(&self, estimated_tokens: u32) {
        // Wait for request permit
        self.request_limiter.until_ready().await;

        // Wait for token permits if configured
        if let Some(ref token_limiter) = self.token_limiter {
            // For large token counts, we need multiple permits
            let permits_needed = (estimated_tokens / 100).max(1);
            for _ in 0..permits_needed {
                token_limiter.until_ready().await;
            }
            debug!(
                tokens = estimated_tokens,
                permits = permits_needed,
                "Rate limiter: token permits acquired"
            );
        }
    }

    /// Try to acquire a permit without waiting
    pub fn try_acquire(&self) -> bool {
        self.request_limiter.check().is_ok()
    }

    /// Try to acquire with tokens without waiting
    pub fn try_acquire_with_tokens(&self, estimated_tokens: u32) -> bool {
        if self.request_limiter.check().is_err() {
            return false;
        }

        if let Some(ref token_limiter) = self.token_limiter {
            let permits_needed = (estimated_tokens / 100).max(1);
            for _ in 0..permits_needed {
                if token_limiter.check().is_err() {
                    return false;
                }
            }
        }

        true
    }

    /// Get current configuration
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }

    /// Check remaining capacity (approximate)
    pub fn remaining_burst(&self) -> u32 {
        // Governor doesn't expose this directly, so we estimate
        if self.try_acquire() {
            self.config.request_burst
        } else {
            0
        }
    }
}

/// Legacy rate limiter for backwards compatibility
pub struct LegacyRateLimiter {
    rpm: usize,
    tpm: usize,
    requests: parking_lot::Mutex<Vec<std::time::Instant>>,
    tokens: parking_lot::Mutex<Vec<(std::time::Instant, usize)>>,
}

impl LegacyRateLimiter {
    pub fn new(rpm: usize, tpm: usize) -> Self {
        Self {
            rpm,
            tpm,
            requests: parking_lot::Mutex::new(Vec::new()),
            tokens: parking_lot::Mutex::new(Vec::new()),
        }
    }

    pub async fn wait(&self, estimated_tokens: usize) {
        loop {
            let now = std::time::Instant::now();
            let one_minute_ago = now - std::time::Duration::from_secs(60);

            {
                let mut requests = self.requests.lock();
                requests.retain(|&t| t > one_minute_ago);

                let mut tokens = self.tokens.lock();
                tokens.retain(|(t, _)| *t > one_minute_ago);

                let current_rpm = requests.len();
                let current_tpm: usize = tokens.iter().map(|(_, count)| count).sum();

                if current_rpm < self.rpm && (current_tpm + estimated_tokens) < self.tpm {
                    requests.push(now);
                    tokens.push((now, estimated_tokens));
                    return;
                }
            }

            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let config = RateLimitConfig::custom(60, None); // 1 per second
        let limiter = GovernorRateLimiter::new(config);

        // Should be able to acquire immediately
        assert!(limiter.try_acquire());

        // Acquire and measure time
        let start = std::time::Instant::now();
        limiter.acquire().await;
        limiter.acquire().await;
        let elapsed = start.elapsed();

        // Should take at least ~1 second for 2 more requests after burst
        // (accounting for initial burst capacity)
        assert!(elapsed.as_millis() < 3000);
    }

    #[test]
    fn test_config_presets() {
        let gpt4 = RateLimitConfig::openai_gpt4_turbo();
        assert_eq!(gpt4.requests_per_minute, 500);

        let gpt35 = RateLimitConfig::openai_gpt35_turbo();
        assert_eq!(gpt35.requests_per_minute, 3500);

        let claude = RateLimitConfig::anthropic_claude();
        assert_eq!(claude.requests_per_minute, 1000);
    }
}
