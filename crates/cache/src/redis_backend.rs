//! Redis/Valkey cache backend
//!
//! Works with both Redis and Valkey (same protocol).
//! Connection URL format: `redis://localhost:6379` or `redis://user:pass@host:port/db`

use async_trait::async_trait;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Cache, CacheConfig, CacheEntry, CacheError, CacheStats};

/// Redis/Valkey cache backend
pub struct RedisCache {
    conn: ConnectionManager,
    config: CacheConfig,
    stats: RedisCacheStats,
}

struct RedisCacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    stores: AtomicU64,
}

impl Default for RedisCacheStats {
    fn default() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            stores: AtomicU64::new(0),
        }
    }
}

impl RedisCache {
    /// Create a new Redis/Valkey cache
    ///
    /// # Arguments
    /// * `url` - Redis/Valkey connection URL (e.g., `redis://localhost:6379`)
    /// * `config` - Cache configuration
    ///
    /// # Example
    /// ```rust,ignore
    /// // Redis
    /// let cache = RedisCache::new("redis://localhost:6379", CacheConfig::default()).await?;
    ///
    /// // Valkey (same URL format)
    /// let cache = RedisCache::new("redis://localhost:6379", CacheConfig::default()).await?;
    ///
    /// // With authentication
    /// let cache = RedisCache::new("redis://user:password@localhost:6379/0", config).await?;
    /// ```
    pub async fn new(url: &str, config: CacheConfig) -> Result<Self, CacheError> {
        let client = redis::Client::open(url)
            .map_err(|e| CacheError::Connection(format!("Failed to create client: {}", e)))?;

        let conn = ConnectionManager::new(client)
            .await
            .map_err(|e| CacheError::Connection(format!("Failed to connect: {}", e)))?;

        tracing::info!(
            url = %url,
            prefix = %config.key_prefix,
            "Connected to Redis/Valkey"
        );

        Ok(Self {
            conn,
            config,
            stats: RedisCacheStats::default(),
        })
    }

    /// Generate cache key
    fn make_key(&self, query: &str, context: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(query.as_bytes());
        if !context.is_empty() {
            hasher.update(b"|ctx:");
            let ctx_bytes = context.as_bytes();
            let limit = ctx_bytes.len().min(500);
            hasher.update(&ctx_bytes[..limit]);
        }
        format!("{}{:x}", self.config.key_prefix, hasher.finalize())
    }

    /// Check connection health
    pub async fn ping(&self) -> Result<bool, CacheError> {
        let mut conn = self.conn.clone();
        let result: Result<String, _> = redis::cmd("PING").query_async(&mut conn).await;
        Ok(result.map(|r| r == "PONG").unwrap_or(false))
    }
}

#[async_trait]
impl Cache for RedisCache {
    async fn get(&self, query: &str, context: &str) -> Result<Option<CacheEntry>, CacheError> {
        let key = self.make_key(query, context);
        let mut conn = self.conn.clone();

        let data: Option<String> = conn.get(&key).await?;

        match data {
            Some(json) => {
                let mut entry: CacheEntry = serde_json::from_str(&json)?;

                // Increment hit count
                entry.record_hit();

                // Update in Redis
                let updated = serde_json::to_string(&entry)?;
                let ttl_secs = self.config.ttl.as_secs();
                let _: () = conn.set_ex(&key, &updated, ttl_secs).await?;

                self.stats.hits.fetch_add(1, Ordering::Relaxed);

                tracing::debug!(
                    query = %query,
                    hits = %entry.hit_count,
                    "Redis cache HIT"
                );

                Ok(Some(entry))
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    async fn store(
        &self,
        query: &str,
        context: &str,
        response: &str,
        function_calls: Vec<String>,
    ) -> Result<(), CacheError> {
        let key = self.make_key(query, context);
        let entry = CacheEntry::new(query, context, response, function_calls);
        let json = serde_json::to_string(&entry)?;

        let mut conn = self.conn.clone();
        let ttl_secs = self.config.ttl.as_secs();

        let _: () = conn.set_ex(&key, &json, ttl_secs).await?;

        self.stats.stores.fetch_add(1, Ordering::Relaxed);

        tracing::debug!(
            query = %query,
            ttl_secs = %ttl_secs,
            "Redis cache STORE"
        );

        Ok(())
    }

    async fn delete(&self, query: &str, context: &str) -> Result<bool, CacheError> {
        let key = self.make_key(query, context);
        let mut conn = self.conn.clone();

        let deleted: i64 = conn.del(&key).await?;
        Ok(deleted > 0)
    }

    async fn clear(&self) -> Result<usize, CacheError> {
        let mut conn = self.conn.clone();
        let pattern = format!("{}*", self.config.key_prefix);

        // Use SCAN to avoid blocking
        let mut cursor: u64 = 0;
        let mut count = 0usize;

        loop {
            let (new_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(100)
                .query_async(&mut conn)
                .await?;

            for key in keys {
                let _: () = conn.del(&key).await?;
                count += 1;
            }

            cursor = new_cursor;
            if cursor == 0 {
                break;
            }
        }

        Ok(count)
    }

    async fn stats(&self) -> Result<CacheStats, CacheError> {
        let mut conn = self.conn.clone();
        let pattern = format!("{}*", self.config.key_prefix);

        // Count entries using SCAN
        let mut cursor: u64 = 0;
        let mut entries = 0usize;

        loop {
            let (new_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(100)
                .query_async(&mut conn)
                .await?;

            entries += keys.len();

            cursor = new_cursor;
            if cursor == 0 {
                break;
            }
        }

        Ok(CacheStats {
            entries,
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            stores: self.stats.stores.load(Ordering::Relaxed),
            evictions: 0, // Redis handles evictions internally
        })
    }
}

/// Semantic Redis cache with embedding support
pub struct SemanticRedisCache {
    inner: RedisCache,
}

impl SemanticRedisCache {
    /// Create a new semantic Redis cache
    pub async fn new(url: &str, config: CacheConfig) -> Result<Self, CacheError> {
        Ok(Self {
            inner: RedisCache::new(url, config).await?,
        })
    }

    /// Cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Store with embedding
    pub async fn store_with_embedding(
        &self,
        query: &str,
        context: &str,
        response: &str,
        function_calls: Vec<String>,
        embedding: Vec<f32>,
    ) -> Result<(), CacheError> {
        let key = self.inner.make_key(query, context);
        let mut entry = CacheEntry::new(query, context, response, function_calls);
        entry.embedding = Some(embedding);

        let json = serde_json::to_string(&entry)?;
        let mut conn = self.inner.conn.clone();
        let ttl_secs = self.inner.config.ttl.as_secs();

        let _: () = conn.set_ex(&key, &json, ttl_secs).await?;

        Ok(())
    }

    /// Find similar entries
    pub async fn find_similar(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        max_scan: usize,
    ) -> Result<Option<(CacheEntry, f32)>, CacheError> {
        let mut conn = self.inner.conn.clone();
        let pattern = format!("{}*", self.inner.config.key_prefix);

        let mut cursor: u64 = 0;
        let mut scanned = 0usize;
        let mut best_match: Option<(CacheEntry, f32)> = None;

        loop {
            let (new_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(100)
                .query_async(&mut conn)
                .await?;

            for key in &keys {
                if scanned >= max_scan {
                    break;
                }
                scanned += 1;

                if let Ok(Some(data)) = conn.get::<_, Option<String>>(key).await {
                    if let Ok(entry) = serde_json::from_str::<CacheEntry>(&data) {
                        if let Some(ref embedding) = entry.embedding {
                            let similarity = Self::cosine_similarity(query_embedding, embedding);

                            if similarity >= threshold {
                                if best_match.is_none()
                                    || similarity > best_match.as_ref().unwrap().1
                                {
                                    best_match = Some((entry, similarity));
                                }
                            }
                        }
                    }
                }
            }

            cursor = new_cursor;
            if cursor == 0 || scanned >= max_scan {
                break;
            }
        }

        if let Some((ref entry, similarity)) = best_match {
            tracing::info!(
                similarity = %format!("{:.1}%", similarity * 100.0),
                query = %entry.query,
                "Semantic cache HIT"
            );
        }

        Ok(best_match)
    }
}

#[async_trait]
impl Cache for SemanticRedisCache {
    async fn get(&self, query: &str, context: &str) -> Result<Option<CacheEntry>, CacheError> {
        self.inner.get(query, context).await
    }

    async fn store(
        &self,
        query: &str,
        context: &str,
        response: &str,
        function_calls: Vec<String>,
    ) -> Result<(), CacheError> {
        self.inner
            .store(query, context, response, function_calls)
            .await
    }

    async fn delete(&self, query: &str, context: &str) -> Result<bool, CacheError> {
        self.inner.delete(query, context).await
    }

    async fn clear(&self) -> Result<usize, CacheError> {
        self.inner.clear().await
    }

    async fn stats(&self) -> Result<CacheStats, CacheError> {
        self.inner.stats().await
    }
}
