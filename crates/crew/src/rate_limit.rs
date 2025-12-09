//! # Per-Model Rate Limiting
//!
//! MassGen-inspired rate limiting system for controlling agent startup
//! and API calls per model/provider.
//!
//! ## Features
//!
//! - **Per-Model Limits**: Different rate limits for different models
//! - **Token Bucket**: Smooth rate limiting with burst capacity
//! - **Sliding Window**: Track requests over time windows
//! - **Queuing**: Queue requests when rate limited
//! - **Priority**: Priority-based request ordering
//! - **Cost Tracking**: Track token/cost usage per model
//!
//! ## Example
//!
//! ```ignore
//! use rust_ai_agents_crew::rate_limit::*;
//!
//! let limiter = RateLimiter::builder()
//!     .add_model_limit("claude-3-opus", ModelLimit::new(10, Duration::from_secs(60)))
//!     .add_model_limit("claude-3-haiku", ModelLimit::new(100, Duration::from_secs(60)))
//!     .default_limit(ModelLimit::new(50, Duration::from_secs(60)))
//!     .build();
//!
//! // Check if we can make a request
//! if limiter.try_acquire("claude-3-opus").await {
//!     // Make the request
//! }
//!
//! // Or wait for permission
//! limiter.acquire("claude-3-opus").await;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, RwLock, Semaphore};

/// Rate limit configuration for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLimit {
    /// Maximum requests per window
    pub max_requests: usize,
    /// Time window for rate limiting
    pub window: Duration,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    /// Burst capacity (extra requests allowed in short bursts)
    pub burst_capacity: usize,
    /// Token limit per minute (if applicable)
    pub tokens_per_minute: Option<usize>,
    /// Cost limit per hour in cents (if applicable)
    pub cost_limit_cents_per_hour: Option<u64>,
}

impl ModelLimit {
    /// Create a new model limit
    pub fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            max_concurrent: max_requests / 2,
            burst_capacity: max_requests / 10,
            tokens_per_minute: None,
            cost_limit_cents_per_hour: None,
        }
    }

    /// Set maximum concurrent requests
    pub fn with_concurrent(mut self, max_concurrent: usize) -> Self {
        self.max_concurrent = max_concurrent;
        self
    }

    /// Set burst capacity
    pub fn with_burst(mut self, burst: usize) -> Self {
        self.burst_capacity = burst;
        self
    }

    /// Set token limit per minute
    pub fn with_tokens_per_minute(mut self, tokens: usize) -> Self {
        self.tokens_per_minute = Some(tokens);
        self
    }

    /// Set cost limit per hour
    pub fn with_cost_limit(mut self, cents_per_hour: u64) -> Self {
        self.cost_limit_cents_per_hour = Some(cents_per_hour);
        self
    }

    /// Preset for high-tier models (expensive, limited)
    pub fn high_tier() -> Self {
        Self {
            max_requests: 10,
            window: Duration::from_secs(60),
            max_concurrent: 2,
            burst_capacity: 2,
            tokens_per_minute: Some(100_000),
            cost_limit_cents_per_hour: Some(500), // $5/hour
        }
    }

    /// Preset for mid-tier models
    pub fn mid_tier() -> Self {
        Self {
            max_requests: 50,
            window: Duration::from_secs(60),
            max_concurrent: 10,
            burst_capacity: 5,
            tokens_per_minute: Some(500_000),
            cost_limit_cents_per_hour: Some(100), // $1/hour
        }
    }

    /// Preset for low-tier models (cheap, high throughput)
    pub fn low_tier() -> Self {
        Self {
            max_requests: 200,
            window: Duration::from_secs(60),
            max_concurrent: 50,
            burst_capacity: 20,
            tokens_per_minute: Some(2_000_000),
            cost_limit_cents_per_hour: Some(50), // $0.50/hour
        }
    }
}

impl Default for ModelLimit {
    fn default() -> Self {
        Self::mid_tier()
    }
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Lowest priority - can be delayed significantly
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority - processed before normal
    High = 2,
    /// Critical - bypass normal queuing
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

/// A request waiting in the queue
#[derive(Debug)]
struct QueuedRequest {
    #[allow(dead_code)]
    model: String,
    priority: Priority,
    #[allow(dead_code)]
    queued_at: Instant,
    notify: tokio::sync::oneshot::Sender<()>,
}

/// Sliding window tracker for a model
struct WindowTracker {
    requests: Vec<Instant>,
    limit: ModelLimit,
    tokens_used: usize,
    cost_cents: u64,
    last_reset: Instant,
}

impl WindowTracker {
    fn new(limit: ModelLimit) -> Self {
        Self {
            requests: Vec::new(),
            limit,
            tokens_used: 0,
            cost_cents: 0,
            last_reset: Instant::now(),
        }
    }

    fn cleanup_old_requests(&mut self) {
        let cutoff = Instant::now() - self.limit.window;
        self.requests.retain(|&t| t > cutoff);
    }

    fn can_acquire(&mut self) -> bool {
        self.cleanup_old_requests();

        // Check request limit
        if self.requests.len() >= self.limit.max_requests + self.limit.burst_capacity {
            return false;
        }

        // Check token limit
        if let Some(token_limit) = self.limit.tokens_per_minute {
            if self.tokens_used >= token_limit {
                // Reset if minute has passed
                if self.last_reset.elapsed() > Duration::from_secs(60) {
                    self.tokens_used = 0;
                    self.last_reset = Instant::now();
                } else {
                    return false;
                }
            }
        }

        // Check cost limit
        if let Some(cost_limit) = self.limit.cost_limit_cents_per_hour {
            if self.cost_cents >= cost_limit {
                // Reset if hour has passed
                if self.last_reset.elapsed() > Duration::from_secs(3600) {
                    self.cost_cents = 0;
                    self.last_reset = Instant::now();
                } else {
                    return false;
                }
            }
        }

        true
    }

    fn record_request(&mut self) {
        self.requests.push(Instant::now());
    }

    fn record_usage(&mut self, tokens: usize, cost_cents: u64) {
        self.tokens_used += tokens;
        self.cost_cents += cost_cents;
    }

    fn time_until_available(&mut self) -> Duration {
        self.cleanup_old_requests();

        if self.requests.len() < self.limit.max_requests {
            return Duration::ZERO;
        }

        // Find when oldest request will expire
        if let Some(&oldest) = self.requests.first() {
            let elapsed = oldest.elapsed();
            if elapsed < self.limit.window {
                return self.limit.window - elapsed;
            }
        }

        Duration::ZERO
    }

    fn current_usage(&self) -> UsageStats {
        UsageStats {
            requests_in_window: self.requests.len(),
            max_requests: self.limit.max_requests,
            tokens_used: self.tokens_used,
            tokens_limit: self.limit.tokens_per_minute,
            cost_cents: self.cost_cents,
            cost_limit_cents: self.limit.cost_limit_cents_per_hour,
        }
    }
}

/// Usage statistics for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    /// Requests made in current window
    pub requests_in_window: usize,
    /// Maximum requests allowed
    pub max_requests: usize,
    /// Tokens used in current period
    pub tokens_used: usize,
    /// Token limit (if any)
    pub tokens_limit: Option<usize>,
    /// Cost in cents for current period
    pub cost_cents: u64,
    /// Cost limit in cents (if any)
    pub cost_limit_cents: Option<u64>,
}

impl UsageStats {
    /// Get request utilization as percentage
    pub fn request_utilization(&self) -> f32 {
        if self.max_requests == 0 {
            return 0.0;
        }
        (self.requests_in_window as f32 / self.max_requests as f32) * 100.0
    }

    /// Get token utilization as percentage
    pub fn token_utilization(&self) -> Option<f32> {
        self.tokens_limit.map(|limit| {
            if limit == 0 {
                0.0
            } else {
                (self.tokens_used as f32 / limit as f32) * 100.0
            }
        })
    }

    /// Get cost utilization as percentage
    pub fn cost_utilization(&self) -> Option<f32> {
        self.cost_limit_cents.map(|limit| {
            if limit == 0 {
                0.0
            } else {
                (self.cost_cents as f32 / limit as f32) * 100.0
            }
        })
    }
}

/// Rate limiter events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitEvent {
    /// Request was rate limited
    RateLimited { model: String, wait_time_ms: u64 },
    /// Request was queued
    Queued {
        model: String,
        priority: Priority,
        queue_position: usize,
    },
    /// Request was dequeued
    Dequeued { model: String, wait_time_ms: u64 },
    /// Limit was reached
    LimitReached { model: String, limit_type: String },
    /// Usage warning (approaching limit)
    UsageWarning {
        model: String,
        utilization_percent: f32,
    },
}

/// Result of trying to acquire a rate limit permit
#[derive(Debug, Clone)]
pub struct AcquireResult {
    /// Whether acquisition was successful
    pub acquired: bool,
    /// Time waited (if any)
    pub wait_time: Duration,
    /// Current usage stats
    pub usage: UsageStats,
    /// Whether request was queued
    pub was_queued: bool,
}

/// Rate limiter configuration
#[derive(Debug, Clone, Default)]
pub struct RateLimiterConfig {
    /// Per-model limits
    pub model_limits: HashMap<String, ModelLimit>,
    /// Default limit for unknown models
    pub default_limit: ModelLimit,
    /// Maximum queue size per model
    pub max_queue_size: usize,
    /// Enable queuing when rate limited
    pub enable_queuing: bool,
    /// Warning threshold (percentage)
    pub warning_threshold: f32,
}

/// Builder for RateLimiter
pub struct RateLimiterBuilder {
    config: RateLimiterConfig,
}

impl Default for RateLimiterBuilder {
    fn default() -> Self {
        Self {
            config: RateLimiterConfig {
                model_limits: HashMap::new(),
                default_limit: ModelLimit::default(),
                max_queue_size: 100,
                enable_queuing: true,
                warning_threshold: 80.0,
            },
        }
    }
}

impl RateLimiterBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a limit for a specific model
    pub fn add_model_limit(mut self, model: impl Into<String>, limit: ModelLimit) -> Self {
        self.config.model_limits.insert(model.into(), limit);
        self
    }

    /// Set the default limit
    pub fn default_limit(mut self, limit: ModelLimit) -> Self {
        self.config.default_limit = limit;
        self
    }

    /// Set maximum queue size
    pub fn max_queue_size(mut self, size: usize) -> Self {
        self.config.max_queue_size = size;
        self
    }

    /// Enable or disable queuing
    pub fn enable_queuing(mut self, enable: bool) -> Self {
        self.config.enable_queuing = enable;
        self
    }

    /// Set warning threshold
    pub fn warning_threshold(mut self, percent: f32) -> Self {
        self.config.warning_threshold = percent;
        self
    }

    /// Build the rate limiter
    pub fn build(self) -> RateLimiter {
        RateLimiter::new(self.config)
    }
}

/// Per-model rate limiter
pub struct RateLimiter {
    config: RateLimiterConfig,
    trackers: Arc<RwLock<HashMap<String, WindowTracker>>>,
    #[allow(dead_code)]
    semaphores: Arc<RwLock<HashMap<String, Arc<Semaphore>>>>,
    queues: Arc<Mutex<HashMap<String, Vec<QueuedRequest>>>>,
    event_tx: broadcast::Sender<RateLimitEvent>,
}

impl RateLimiter {
    /// Create a new rate limiter with configuration
    pub fn new(config: RateLimiterConfig) -> Self {
        let (event_tx, _) = broadcast::channel(256);

        let mut trackers = HashMap::new();
        let mut semaphores = HashMap::new();

        for (model, limit) in &config.model_limits {
            trackers.insert(model.clone(), WindowTracker::new(limit.clone()));
            semaphores.insert(
                model.clone(),
                Arc::new(Semaphore::new(limit.max_concurrent)),
            );
        }

        Self {
            config,
            trackers: Arc::new(RwLock::new(trackers)),
            semaphores: Arc::new(RwLock::new(semaphores)),
            queues: Arc::new(Mutex::new(HashMap::new())),
            event_tx,
        }
    }

    /// Create a builder
    pub fn builder() -> RateLimiterBuilder {
        RateLimiterBuilder::new()
    }

    /// Subscribe to rate limit events
    pub fn subscribe(&self) -> broadcast::Receiver<RateLimitEvent> {
        self.event_tx.subscribe()
    }

    /// Try to acquire a permit without waiting
    pub async fn try_acquire(&self, model: &str) -> bool {
        self.try_acquire_with_priority(model, Priority::Normal)
            .await
    }

    /// Try to acquire with specific priority
    pub async fn try_acquire_with_priority(&self, model: &str, priority: Priority) -> bool {
        // Critical priority bypasses rate limiting
        if priority == Priority::Critical {
            self.record_request(model).await;
            return true;
        }

        let mut trackers = self.trackers.write().await;
        let tracker = self.get_or_create_tracker(&mut trackers, model);

        if tracker.can_acquire() {
            tracker.record_request();
            self.check_warning(model, tracker).await;
            true
        } else {
            let _ = self.event_tx.send(RateLimitEvent::RateLimited {
                model: model.to_string(),
                wait_time_ms: tracker.time_until_available().as_millis() as u64,
            });
            false
        }
    }

    /// Acquire a permit, waiting if necessary
    pub async fn acquire(&self, model: &str) -> AcquireResult {
        self.acquire_with_priority(model, Priority::Normal).await
    }

    /// Acquire with specific priority
    pub async fn acquire_with_priority(&self, model: &str, priority: Priority) -> AcquireResult {
        let start = Instant::now();

        // Critical priority bypasses rate limiting
        if priority == Priority::Critical {
            self.record_request(model).await;
            return AcquireResult {
                acquired: true,
                wait_time: Duration::ZERO,
                usage: self.get_usage(model).await,
                was_queued: false,
            };
        }

        // Try immediate acquisition
        if self.try_acquire_with_priority(model, priority).await {
            return AcquireResult {
                acquired: true,
                wait_time: Duration::ZERO,
                usage: self.get_usage(model).await,
                was_queued: false,
            };
        }

        // Queue if enabled
        if self.config.enable_queuing {
            let (tx, rx) = tokio::sync::oneshot::channel();

            {
                let mut queues = self.queues.lock().await;
                let queue = queues.entry(model.to_string()).or_default();

                if queue.len() >= self.config.max_queue_size {
                    return AcquireResult {
                        acquired: false,
                        wait_time: start.elapsed(),
                        usage: self.get_usage(model).await,
                        was_queued: false,
                    };
                }

                let position = queue.len();
                queue.push(QueuedRequest {
                    model: model.to_string(),
                    priority,
                    queued_at: Instant::now(),
                    notify: tx,
                });

                // Sort by priority (highest first)
                queue.sort_by(|a, b| b.priority.cmp(&a.priority));

                let _ = self.event_tx.send(RateLimitEvent::Queued {
                    model: model.to_string(),
                    priority,
                    queue_position: position,
                });
            }

            // Wait for our turn
            let _ = rx.await;

            let _ = self.event_tx.send(RateLimitEvent::Dequeued {
                model: model.to_string(),
                wait_time_ms: start.elapsed().as_millis() as u64,
            });

            AcquireResult {
                acquired: true,
                wait_time: start.elapsed(),
                usage: self.get_usage(model).await,
                was_queued: true,
            }
        } else {
            // Wait without queuing
            loop {
                let wait_time = {
                    let mut trackers = self.trackers.write().await;
                    let tracker = self.get_or_create_tracker(&mut trackers, model);
                    tracker.time_until_available()
                };

                if wait_time.is_zero() {
                    if self.try_acquire_with_priority(model, priority).await {
                        return AcquireResult {
                            acquired: true,
                            wait_time: start.elapsed(),
                            usage: self.get_usage(model).await,
                            was_queued: false,
                        };
                    }
                }

                tokio::time::sleep(wait_time.max(Duration::from_millis(10))).await;
            }
        }
    }

    /// Record token and cost usage
    pub async fn record_usage(&self, model: &str, tokens: usize, cost_cents: u64) {
        let mut trackers = self.trackers.write().await;
        let tracker = self.get_or_create_tracker(&mut trackers, model);
        tracker.record_usage(tokens, cost_cents);
    }

    /// Get current usage stats for a model
    pub async fn get_usage(&self, model: &str) -> UsageStats {
        let mut trackers = self.trackers.write().await;
        let tracker = self.get_or_create_tracker(&mut trackers, model);
        tracker.current_usage()
    }

    /// Get usage stats for all models
    pub async fn get_all_usage(&self) -> HashMap<String, UsageStats> {
        let mut trackers = self.trackers.write().await;
        trackers
            .iter_mut()
            .map(|(model, tracker)| (model.clone(), tracker.current_usage()))
            .collect()
    }

    /// Release a queued request (call after request completes)
    pub async fn release(&self, model: &str) {
        let mut queues = self.queues.lock().await;
        if let Some(queue) = queues.get_mut(model) {
            if let Some(request) = queue.pop() {
                let _ = request.notify.send(());
            }
        }
    }

    /// Get queue length for a model
    pub async fn queue_length(&self, model: &str) -> usize {
        let queues = self.queues.lock().await;
        queues.get(model).map(|q| q.len()).unwrap_or(0)
    }

    /// Check if a model is rate limited
    pub async fn is_rate_limited(&self, model: &str) -> bool {
        let mut trackers = self.trackers.write().await;
        let tracker = self.get_or_create_tracker(&mut trackers, model);
        !tracker.can_acquire()
    }

    /// Get time until a model is available
    pub async fn time_until_available(&self, model: &str) -> Duration {
        let mut trackers = self.trackers.write().await;
        let tracker = self.get_or_create_tracker(&mut trackers, model);
        tracker.time_until_available()
    }

    // Internal helpers

    fn get_or_create_tracker<'a>(
        &self,
        trackers: &'a mut HashMap<String, WindowTracker>,
        model: &str,
    ) -> &'a mut WindowTracker {
        if !trackers.contains_key(model) {
            let limit = self
                .config
                .model_limits
                .get(model)
                .cloned()
                .unwrap_or_else(|| self.config.default_limit.clone());
            trackers.insert(model.to_string(), WindowTracker::new(limit));
        }
        trackers.get_mut(model).unwrap()
    }

    async fn record_request(&self, model: &str) {
        let mut trackers = self.trackers.write().await;
        let tracker = self.get_or_create_tracker(&mut trackers, model);
        tracker.record_request();
    }

    async fn check_warning(&self, model: &str, tracker: &WindowTracker) {
        let utilization = (tracker.requests.len() as f32
            / (tracker.limit.max_requests + tracker.limit.burst_capacity) as f32)
            * 100.0;

        if utilization >= self.config.warning_threshold {
            let _ = self.event_tx.send(RateLimitEvent::UsageWarning {
                model: model.to_string(),
                utilization_percent: utilization,
            });
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::builder().build()
    }
}

/// Presets for common model rate limit configurations
pub mod rate_limit_presets {
    use super::*;

    /// Claude models rate limits
    pub fn claude_limits() -> HashMap<String, ModelLimit> {
        let mut limits = HashMap::new();
        limits.insert("claude-3-opus".to_string(), ModelLimit::high_tier());
        limits.insert("claude-3-sonnet".to_string(), ModelLimit::mid_tier());
        limits.insert("claude-3-haiku".to_string(), ModelLimit::low_tier());
        limits.insert("claude-3-5-sonnet".to_string(), ModelLimit::mid_tier());
        limits
    }

    /// OpenAI models rate limits
    pub fn openai_limits() -> HashMap<String, ModelLimit> {
        let mut limits = HashMap::new();
        limits.insert("gpt-4-turbo".to_string(), ModelLimit::high_tier());
        limits.insert("gpt-4".to_string(), ModelLimit::high_tier());
        limits.insert("gpt-3.5-turbo".to_string(), ModelLimit::low_tier());
        limits
    }

    /// Create a rate limiter with Claude presets
    pub fn claude_limiter() -> RateLimiter {
        let mut builder = RateLimiter::builder();
        for (model, limit) in claude_limits() {
            builder = builder.add_model_limit(model, limit);
        }
        builder.build()
    }

    /// Create a rate limiter with OpenAI presets
    pub fn openai_limiter() -> RateLimiter {
        let mut builder = RateLimiter::builder();
        for (model, limit) in openai_limits() {
            builder = builder.add_model_limit(model, limit);
        }
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_rate_limiting() {
        let limiter = RateLimiter::builder()
            .add_model_limit("test-model", ModelLimit::new(2, Duration::from_secs(60)))
            .build();

        // First two should succeed
        assert!(limiter.try_acquire("test-model").await);
        assert!(limiter.try_acquire("test-model").await);

        // Third should fail (no burst capacity by default in this limit)
        let limit = ModelLimit::new(2, Duration::from_secs(60)).with_burst(0);
        let limiter = RateLimiter::builder()
            .add_model_limit("test-model", limit)
            .build();

        assert!(limiter.try_acquire("test-model").await);
        assert!(limiter.try_acquire("test-model").await);
        assert!(!limiter.try_acquire("test-model").await);
    }

    #[tokio::test]
    async fn test_default_limit() {
        let limiter = RateLimiter::builder()
            .default_limit(ModelLimit::new(1, Duration::from_secs(60)).with_burst(0))
            .build();

        // Unknown model should use default limit
        assert!(limiter.try_acquire("unknown-model").await);
        assert!(!limiter.try_acquire("unknown-model").await);
    }

    #[tokio::test]
    async fn test_usage_stats() {
        let limiter = RateLimiter::builder()
            .add_model_limit("test-model", ModelLimit::new(10, Duration::from_secs(60)))
            .build();

        limiter.try_acquire("test-model").await;
        limiter.try_acquire("test-model").await;

        let usage = limiter.get_usage("test-model").await;
        assert_eq!(usage.requests_in_window, 2);
        assert_eq!(usage.max_requests, 10);
    }

    #[tokio::test]
    async fn test_priority_bypass() {
        let limiter = RateLimiter::builder()
            .add_model_limit(
                "test-model",
                ModelLimit::new(1, Duration::from_secs(60)).with_burst(0),
            )
            .build();

        // Use up the limit
        assert!(limiter.try_acquire("test-model").await);
        assert!(!limiter.try_acquire("test-model").await);

        // Critical priority should still work
        assert!(
            limiter
                .try_acquire_with_priority("test-model", Priority::Critical)
                .await
        );
    }

    #[tokio::test]
    async fn test_record_usage() {
        let limiter = RateLimiter::builder()
            .add_model_limit(
                "test-model",
                ModelLimit::new(10, Duration::from_secs(60)).with_tokens_per_minute(1000),
            )
            .build();

        limiter.record_usage("test-model", 500, 10).await;

        let usage = limiter.get_usage("test-model").await;
        assert_eq!(usage.tokens_used, 500);
        assert_eq!(usage.cost_cents, 10);
    }

    #[tokio::test]
    async fn test_is_rate_limited() {
        let limiter = RateLimiter::builder()
            .add_model_limit(
                "test-model",
                ModelLimit::new(1, Duration::from_secs(60)).with_burst(0),
            )
            .build();

        assert!(!limiter.is_rate_limited("test-model").await);
        limiter.try_acquire("test-model").await;
        assert!(limiter.is_rate_limited("test-model").await);
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let limiter = RateLimiter::builder()
            .add_model_limit(
                "test-model",
                ModelLimit::new(1, Duration::from_secs(60)).with_burst(0),
            )
            .build();

        let mut rx = limiter.subscribe();

        // Use up limit
        limiter.try_acquire("test-model").await;
        // This should fail and emit event
        let result = limiter.try_acquire("test-model").await;
        assert!(!result);

        // Allow time for broadcast
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Check we received a rate limited event
        let mut found_rate_limited = false;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, RateLimitEvent::RateLimited { .. }) {
                found_rate_limited = true;
                break;
            }
        }
        assert!(found_rate_limited);
    }

    #[tokio::test]
    async fn test_model_limit_presets() {
        let high = ModelLimit::high_tier();
        let mid = ModelLimit::mid_tier();
        let low = ModelLimit::low_tier();

        assert!(high.max_requests < mid.max_requests);
        assert!(mid.max_requests < low.max_requests);
    }

    #[tokio::test]
    async fn test_presets() {
        let claude = rate_limit_presets::claude_limits();
        assert!(claude.contains_key("claude-3-opus"));
        assert!(claude.contains_key("claude-3-haiku"));

        let openai = rate_limit_presets::openai_limits();
        assert!(openai.contains_key("gpt-4"));
        assert!(openai.contains_key("gpt-3.5-turbo"));
    }

    #[tokio::test]
    async fn test_utilization_stats() {
        let usage = UsageStats {
            requests_in_window: 50,
            max_requests: 100,
            tokens_used: 500,
            tokens_limit: Some(1000),
            cost_cents: 25,
            cost_limit_cents: Some(100),
        };

        assert!((usage.request_utilization() - 50.0).abs() < 0.1);
        assert!((usage.token_utilization().unwrap() - 50.0).abs() < 0.1);
        assert!((usage.cost_utilization().unwrap() - 25.0).abs() < 0.1);
    }
}
