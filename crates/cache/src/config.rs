//! Cache configuration

use std::time::Duration;

/// Configuration for cache behavior
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Maximum number of entries (for memory cache)
    pub max_entries: usize,
    /// Time-to-live for cache entries
    pub ttl: Duration,
    /// Prefix for cache keys (for Redis/Valkey)
    pub key_prefix: String,
    /// Whether to enable semantic matching (requires embeddings)
    pub semantic_matching: bool,
    /// Similarity threshold for semantic matching (0.0 to 1.0)
    pub similarity_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Duration::from_secs(3600), // 1 hour
            key_prefix: "ai-agents:cache:".to_string(),
            semantic_matching: false,
            similarity_threshold: 0.92,
        }
    }
}

impl CacheConfig {
    /// Create a new config with custom settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum entries
    pub fn max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Set TTL
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Set key prefix
    pub fn key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = prefix.into();
        self
    }

    /// Enable semantic matching
    pub fn with_semantic_matching(mut self, threshold: f32) -> Self {
        self.semantic_matching = true;
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Config for short-lived cache (5 minutes)
    pub fn short_lived() -> Self {
        Self {
            ttl: Duration::from_secs(300),
            ..Default::default()
        }
    }

    /// Config for long-lived cache (24 hours)
    pub fn long_lived() -> Self {
        Self {
            ttl: Duration::from_secs(86400),
            max_entries: 5000,
            ..Default::default()
        }
    }

    /// Config for session cache (2 hours)
    pub fn session() -> Self {
        Self {
            ttl: Duration::from_secs(7200),
            max_entries: 500,
            key_prefix: "ai-agents:session:".to_string(),
            ..Default::default()
        }
    }

    /// Config for agent responses
    pub fn for_agents() -> Self {
        Self {
            ttl: Duration::from_secs(3600),
            max_entries: 2000,
            key_prefix: "ai-agents:responses:".to_string(),
            ..Default::default()
        }
    }
}
