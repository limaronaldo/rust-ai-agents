//! In-memory LRU cache implementation

use async_trait::async_trait;
use lru::LruCache;
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Cache, CacheConfig, CacheEntry, CacheError, CacheStats};

/// In-memory LRU cache with TTL support
pub struct MemoryCache {
    cache: RwLock<LruCache<String, CacheEntry>>,
    config: CacheConfig,
    stats: MemoryCacheStats,
}

struct MemoryCacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    stores: AtomicU64,
    evictions: AtomicU64,
}

impl Default for MemoryCacheStats {
    fn default() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            stores: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }
}

impl MemoryCache {
    /// Create a new memory cache
    pub fn new(config: CacheConfig) -> Self {
        let capacity =
            NonZeroUsize::new(config.max_entries).unwrap_or(NonZeroUsize::new(1000).unwrap());

        Self {
            cache: RwLock::new(LruCache::new(capacity)),
            config,
            stats: MemoryCacheStats::default(),
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Generate cache key from query and context
    fn make_key(query: &str, context: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(query.as_bytes());
        if !context.is_empty() {
            hasher.update(b"|ctx:");
            // Limit context to prevent huge keys
            let ctx_bytes = context.as_bytes();
            let limit = ctx_bytes.len().min(500);
            hasher.update(&ctx_bytes[..limit]);
        }
        format!("{:x}", hasher.finalize())
    }

    /// Get current entry count
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.config.max_entries
    }
}

#[async_trait]
impl Cache for MemoryCache {
    async fn get(&self, query: &str, context: &str) -> Result<Option<CacheEntry>, CacheError> {
        let key = Self::make_key(query, context);
        let ttl_secs = self.config.ttl.as_secs() as i64;

        let mut cache = self.cache.write();

        if let Some(entry) = cache.get_mut(&key) {
            // Check if expired
            if entry.is_expired(ttl_secs) {
                cache.pop(&key);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }

            entry.record_hit();
            self.stats.hits.fetch_add(1, Ordering::Relaxed);

            tracing::debug!(
                query = %query,
                hits = %entry.hit_count,
                "Cache HIT"
            );

            return Ok(Some(entry.clone()));
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }

    async fn store(
        &self,
        query: &str,
        context: &str,
        response: &str,
        function_calls: Vec<String>,
    ) -> Result<(), CacheError> {
        let key = Self::make_key(query, context);
        let entry = CacheEntry::new(query, context, response, function_calls);

        let mut cache = self.cache.write();

        // Check if we're at capacity and will evict
        if cache.len() >= self.config.max_entries && !cache.contains(&key) {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }

        cache.put(key, entry);
        self.stats.stores.fetch_add(1, Ordering::Relaxed);

        tracing::debug!(
            query = %query,
            "Cache STORE"
        );

        Ok(())
    }

    async fn delete(&self, query: &str, context: &str) -> Result<bool, CacheError> {
        let key = Self::make_key(query, context);
        let mut cache = self.cache.write();
        Ok(cache.pop(&key).is_some())
    }

    async fn clear(&self) -> Result<usize, CacheError> {
        let mut cache = self.cache.write();
        let count = cache.len();
        cache.clear();
        Ok(count)
    }

    async fn stats(&self) -> Result<CacheStats, CacheError> {
        Ok(CacheStats {
            entries: self.cache.read().len(),
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            stores: self.stats.stores.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
        })
    }
}

/// Memory cache with semantic similarity support
pub struct SemanticMemoryCache {
    inner: MemoryCache,
    embeddings: RwLock<LruCache<String, Vec<f32>>>,
}

impl SemanticMemoryCache {
    /// Create a new semantic memory cache
    pub fn new(config: CacheConfig) -> Self {
        let capacity =
            NonZeroUsize::new(config.max_entries).unwrap_or(NonZeroUsize::new(1000).unwrap());

        Self {
            inner: MemoryCache::new(config),
            embeddings: RwLock::new(LruCache::new(capacity)),
        }
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

    /// Find most similar entry
    pub async fn find_similar(
        &self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Option<(CacheEntry, f32)> {
        let embeddings = self.embeddings.read();
        let cache = self.inner.cache.read();
        let ttl_secs = self.inner.config.ttl.as_secs() as i64;

        let mut best_match: Option<(String, f32)> = None;

        for (key, embedding) in embeddings.iter() {
            let similarity = Self::cosine_similarity(query_embedding, embedding);

            if similarity >= threshold
                && (best_match.is_none() || similarity > best_match.as_ref().unwrap().1)
            {
                best_match = Some((key.clone(), similarity));
            }
        }

        if let Some((key, similarity)) = best_match {
            if let Some(entry) = cache.peek(&key) {
                if !entry.is_expired(ttl_secs) {
                    return Some((entry.clone(), similarity));
                }
            }
        }

        None
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
        let key = MemoryCache::make_key(query, context);

        // Store embedding
        self.embeddings.write().put(key.clone(), embedding);

        // Store entry
        self.inner
            .store(query, context, response, function_calls)
            .await
    }
}

#[async_trait]
impl Cache for SemanticMemoryCache {
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
        let key = MemoryCache::make_key(query, context);
        self.embeddings.write().pop(&key);
        self.inner.delete(query, context).await
    }

    async fn clear(&self) -> Result<usize, CacheError> {
        self.embeddings.write().clear();
        self.inner.clear().await
    }

    async fn stats(&self) -> Result<CacheStats, CacheError> {
        self.inner.stats().await
    }
}

#[async_trait]
impl crate::SemanticCache for SemanticMemoryCache {
    async fn find_similar_by_embedding(
        &self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Result<Option<(CacheEntry, f32)>, CacheError> {
        Ok(self.find_similar(query_embedding, threshold).await)
    }

    async fn store_with_embedding(
        &self,
        query: &str,
        context: &str,
        response: &str,
        function_calls: Vec<String>,
        embedding: Vec<f32>,
    ) -> Result<(), CacheError> {
        SemanticMemoryCache::store_with_embedding(
            self,
            query,
            context,
            response,
            function_calls,
            embedding,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_memory_cache_basic() {
        let cache = MemoryCache::with_defaults();

        // Store
        cache
            .store("query1", "ctx", "response1", vec![])
            .await
            .unwrap();

        // Get
        let entry = cache.get("query1", "ctx").await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().response, "response1");

        // Miss
        let miss = cache.get("nonexistent", "ctx").await.unwrap();
        assert!(miss.is_none());
    }

    #[tokio::test]
    async fn test_memory_cache_hit_count() {
        let cache = MemoryCache::with_defaults();

        cache
            .store("query", "ctx", "response", vec![])
            .await
            .unwrap();

        // Multiple gets should increment hit count
        for i in 1..=3 {
            let entry = cache.get("query", "ctx").await.unwrap().unwrap();
            assert_eq!(entry.hit_count, i);
        }
    }

    #[tokio::test]
    async fn test_memory_cache_expiry() {
        // Use 1 second TTL since is_expired() uses seconds granularity
        let config = CacheConfig {
            ttl: Duration::from_secs(1),
            ..Default::default()
        };
        let cache = MemoryCache::new(config);

        cache
            .store("query", "ctx", "response", vec![])
            .await
            .unwrap();

        // Should exist
        assert!(cache.get("query", "ctx").await.unwrap().is_some());

        // Wait for expiry (TTL is 1 second, wait 2.1 seconds to ensure > 1 second passes)
        // The is_expired check uses strictly greater than, so we need > ttl_secs
        tokio::time::sleep(Duration::from_millis(2100)).await;

        // Should be expired
        assert!(cache.get("query", "ctx").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_memory_cache_lru_eviction() {
        let config = CacheConfig {
            max_entries: 3,
            ..Default::default()
        };
        let cache = MemoryCache::new(config);

        // Fill cache
        cache.store("q1", "", "r1", vec![]).await.unwrap();
        cache.store("q2", "", "r2", vec![]).await.unwrap();
        cache.store("q3", "", "r3", vec![]).await.unwrap();

        assert_eq!(cache.len(), 3);

        // Access q1 to make it recently used
        cache.get("q1", "").await.unwrap();

        // Add new entry, q2 should be evicted (LRU)
        cache.store("q4", "", "r4", vec![]).await.unwrap();

        assert_eq!(cache.len(), 3);
        assert!(cache.get("q1", "").await.unwrap().is_some());
        assert!(cache.get("q2", "").await.unwrap().is_none()); // Evicted
        assert!(cache.get("q3", "").await.unwrap().is_some());
        assert!(cache.get("q4", "").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_memory_cache_stats() {
        let cache = MemoryCache::with_defaults();

        cache.store("q1", "", "r1", vec![]).await.unwrap();
        cache.store("q2", "", "r2", vec![]).await.unwrap();

        cache.get("q1", "").await.unwrap(); // Hit
        cache.get("q1", "").await.unwrap(); // Hit
        cache.get("q3", "").await.unwrap(); // Miss

        let stats = cache.stats().await.unwrap();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.stores, 2);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_memory_cache_delete() {
        let cache = MemoryCache::with_defaults();

        cache
            .store("query", "ctx", "response", vec![])
            .await
            .unwrap();
        assert!(cache.get("query", "ctx").await.unwrap().is_some());

        let deleted = cache.delete("query", "ctx").await.unwrap();
        assert!(deleted);

        assert!(cache.get("query", "ctx").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_memory_cache_clear() {
        let cache = MemoryCache::with_defaults();

        cache.store("q1", "", "r1", vec![]).await.unwrap();
        cache.store("q2", "", "r2", vec![]).await.unwrap();

        let cleared = cache.clear().await.unwrap();
        assert_eq!(cleared, 2);
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn test_semantic_cache_similarity() {
        let cache = SemanticMemoryCache::new(CacheConfig::default());

        // Store with embedding
        let embedding1 = vec![1.0, 0.0, 0.0];
        cache
            .store_with_embedding("q1", "", "r1", vec![], embedding1)
            .await
            .unwrap();

        // Find similar (exact match)
        let query_embedding = vec![1.0, 0.0, 0.0];
        let result = cache.find_similar(&query_embedding, 0.9).await;
        assert!(result.is_some());
        let (entry, similarity) = result.unwrap();
        assert_eq!(entry.response, "r1");
        assert!((similarity - 1.0).abs() < 0.001);

        // Find similar (partial match)
        let query_embedding2 = vec![0.9, 0.1, 0.0];
        let result2 = cache.find_similar(&query_embedding2, 0.9).await;
        assert!(result2.is_some());

        // No match (too different)
        let query_embedding3 = vec![0.0, 1.0, 0.0];
        let result3 = cache.find_similar(&query_embedding3, 0.9).await;
        assert!(result3.is_none());
    }
}
