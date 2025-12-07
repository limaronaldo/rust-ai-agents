//! Semantic Cache for rust-ai-agents
//!
//! Provides caching with optional semantic similarity matching for LLM responses.
//! Supports multiple backends:
//!
//! - **In-Memory**: Fast LRU cache with TTL support (default)
//! - **Redis/Valkey**: Distributed cache for multi-instance deployments
//!
//! # Features
//!
//! - `memory` (default): Enable in-memory LRU cache
//! - `redis`: Enable Redis/Valkey backend
//! - `full`: Enable all backends
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_cache::{Cache, MemoryCache, CacheConfig};
//!
//! // In-memory cache
//! let cache = MemoryCache::new(CacheConfig::default());
//!
//! // Store a response
//! cache.store("What is Rust?", "context", "Rust is a systems programming language", vec![]).await?;
//!
//! // Retrieve (exact match)
//! if let Some(entry) = cache.get("What is Rust?", "context").await? {
//!     println!("Cached response: {}", entry.response);
//! }
//! ```

mod config;
mod entry;
mod error;
mod traits;

#[cfg(feature = "memory")]
mod memory;

#[cfg(feature = "redis")]
mod redis_backend;

pub use config::CacheConfig;
pub use entry::{CacheEntry, CacheStats};
pub use error::CacheError;
pub use traits::{Cache, SemanticCache};

#[cfg(feature = "memory")]
pub use memory::{MemoryCache, SemanticMemoryCache};

#[cfg(feature = "redis")]
pub use redis_backend::RedisCache;

/// Optional cache wrapper for graceful degradation
pub enum OptionalCache {
    #[cfg(feature = "memory")]
    Memory(MemoryCache),
    #[cfg(feature = "redis")]
    Redis(RedisCache),
    Disabled,
}

impl OptionalCache {
    /// Create a memory cache
    #[cfg(feature = "memory")]
    pub fn memory(config: CacheConfig) -> Self {
        OptionalCache::Memory(MemoryCache::new(config))
    }

    /// Create a Redis/Valkey cache
    #[cfg(feature = "redis")]
    pub async fn redis(url: &str, config: CacheConfig) -> Self {
        match RedisCache::new(url, config).await {
            Ok(cache) => OptionalCache::Redis(cache),
            Err(e) => {
                tracing::warn!(
                    "Failed to connect to Redis/Valkey: {}. Using disabled cache.",
                    e
                );
                OptionalCache::Disabled
            }
        }
    }

    /// Create disabled cache
    pub fn disabled() -> Self {
        OptionalCache::Disabled
    }

    /// Check if cache is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, OptionalCache::Disabled)
    }

    /// Get from cache
    pub async fn get(&self, query: &str, context: &str) -> Option<CacheEntry> {
        match self {
            #[cfg(feature = "memory")]
            OptionalCache::Memory(cache) => cache.get(query, context).await.ok().flatten(),
            #[cfg(feature = "redis")]
            OptionalCache::Redis(cache) => cache.get(query, context).await.ok().flatten(),
            OptionalCache::Disabled => None,
        }
    }

    /// Store in cache
    pub async fn store(
        &self,
        query: &str,
        context: &str,
        response: &str,
        function_calls: Vec<String>,
    ) {
        match self {
            #[cfg(feature = "memory")]
            OptionalCache::Memory(cache) => {
                let _ = cache.store(query, context, response, function_calls).await;
            }
            #[cfg(feature = "redis")]
            OptionalCache::Redis(cache) => {
                let _ = cache.store(query, context, response, function_calls).await;
            }
            OptionalCache::Disabled => {}
        }
    }
}

/// Builder for creating caches with fallback
pub struct CacheBuilder {
    config: CacheConfig,
    #[cfg(feature = "redis")]
    redis_url: Option<String>,
    enable_fallback: bool,
}

impl CacheBuilder {
    /// Create a new cache builder
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
            #[cfg(feature = "redis")]
            redis_url: None,
            enable_fallback: true,
        }
    }

    /// Set cache configuration
    pub fn config(mut self, config: CacheConfig) -> Self {
        self.config = config;
        self
    }

    /// Set Redis/Valkey URL
    #[cfg(feature = "redis")]
    pub fn redis_url(mut self, url: impl Into<String>) -> Self {
        self.redis_url = Some(url.into());
        self
    }

    /// Enable/disable fallback to memory cache
    pub fn with_fallback(mut self, enable: bool) -> Self {
        self.enable_fallback = enable;
        self
    }

    /// Build the cache
    pub async fn build(self) -> OptionalCache {
        #[cfg(feature = "redis")]
        if let Some(url) = self.redis_url {
            match RedisCache::new(&url, self.config.clone()).await {
                Ok(cache) => {
                    tracing::info!("Connected to Redis/Valkey cache at {}", url);
                    return OptionalCache::Redis(cache);
                }
                Err(e) => {
                    tracing::warn!("Failed to connect to Redis/Valkey: {}", e);
                    if self.enable_fallback {
                        tracing::info!("Falling back to in-memory cache");
                        #[cfg(feature = "memory")]
                        return OptionalCache::Memory(MemoryCache::new(self.config));
                    }
                    return OptionalCache::Disabled;
                }
            }
        }

        #[cfg(feature = "memory")]
        return OptionalCache::Memory(MemoryCache::new(self.config));

        #[cfg(not(feature = "memory"))]
        OptionalCache::Disabled
    }
}

impl Default for CacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}
