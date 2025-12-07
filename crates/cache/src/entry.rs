//! Cache entry types

use serde::{Deserialize, Serialize};

/// A cached entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Original query
    pub query: String,
    /// Context used for the query
    pub context: String,
    /// Cached response
    pub response: String,
    /// Function/tool calls made
    pub function_calls: Vec<String>,
    /// When the entry was created (Unix timestamp)
    pub created_at: i64,
    /// Number of times this entry was hit
    pub hit_count: u32,
    /// Optional embedding for semantic matching
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(
        query: impl Into<String>,
        context: impl Into<String>,
        response: impl Into<String>,
        function_calls: Vec<String>,
    ) -> Self {
        Self {
            query: query.into(),
            context: context.into(),
            response: response.into(),
            function_calls,
            created_at: chrono::Utc::now().timestamp(),
            hit_count: 0,
            embedding: None,
        }
    }

    /// Create with embedding for semantic matching
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Increment hit count
    pub fn record_hit(&mut self) {
        self.hit_count += 1;
    }

    /// Check if entry is expired
    pub fn is_expired(&self, ttl_secs: i64) -> bool {
        let now = chrono::Utc::now().timestamp();
        now - self.created_at > ttl_secs
    }

    /// Get age in seconds
    pub fn age_secs(&self) -> i64 {
        chrono::Utc::now().timestamp() - self.created_at
    }
}

/// Statistics for cache operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of entries
    pub entries: usize,
    /// Total hits
    pub hits: u64,
    /// Total misses
    pub misses: u64,
    /// Total stores
    pub stores: u64,
    /// Total evictions
    pub evictions: u64,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}
