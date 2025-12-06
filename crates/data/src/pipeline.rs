//! Data pipeline with LRU caching
//!
//! Async data processing with TTL-based cache eviction.

use crate::types::{DataError, DataSource};
use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Cached data entry with TTL
#[derive(Debug, Clone)]
pub struct CachedData {
    /// The cached data source
    pub data: DataSource,
    /// When this entry was cached
    pub cached_at: Instant,
    /// Time-to-live for this entry
    pub ttl: Duration,
}

impl CachedData {
    /// Create a new cached entry
    pub fn new(data: DataSource, ttl: Duration) -> Self {
        Self {
            data,
            cached_at: Instant::now(),
            ttl,
        }
    }

    /// Check if this entry has expired
    pub fn is_expired(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }

    /// Get remaining TTL
    pub fn remaining_ttl(&self) -> Duration {
        self.ttl.saturating_sub(self.cached_at.elapsed())
    }
}

/// LRU cache for data sources
pub struct DataCache {
    /// The LRU cache
    cache: Mutex<LruCache<String, CachedData>>,
    /// Default TTL for entries
    default_ttl: Duration,
    /// Cache statistics
    stats: Arc<CacheStats>,
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: std::sync::atomic::AtomicU64,
    pub misses: std::sync::atomic::AtomicU64,
    pub evictions: std::sync::atomic::AtomicU64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        use std::sync::atomic::Ordering;
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

impl DataCache {
    /// Create a new cache with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("capacity must be > 0"),
            )),
            default_ttl: Duration::from_secs(300), // 5 minutes default
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Set default TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = ttl;
        self
    }

    /// Get from cache
    pub fn get(&self, key: &str) -> Option<DataSource> {
        use std::sync::atomic::Ordering;

        let mut cache = self.cache.lock();

        if let Some(entry) = cache.get(key) {
            if entry.is_expired() {
                debug!(key = key, "Cache entry expired");
                cache.pop(key);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            debug!(key = key, remaining_ttl_ms = ?entry.remaining_ttl().as_millis(), "Cache hit");
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.data.clone());
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert into cache
    pub fn insert(&self, key: String, data: DataSource) {
        self.insert_with_ttl(key, data, self.default_ttl);
    }

    /// Insert with custom TTL
    pub fn insert_with_ttl(&self, key: String, data: DataSource, ttl: Duration) {
        use std::sync::atomic::Ordering;

        let mut cache = self.cache.lock();

        // Check if we're evicting an entry
        if cache.len() >= cache.cap().get() {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }

        cache.put(key, CachedData::new(data, ttl));
    }

    /// Remove from cache
    pub fn remove(&self, key: &str) -> Option<DataSource> {
        self.cache.lock().pop(key).map(|e| e.data)
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.cache.lock().clear();
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.lock().is_empty()
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Check if key exists and is not expired
    pub fn contains(&self, key: &str) -> bool {
        let cache = self.cache.lock();
        if let Some(entry) = cache.peek(key) {
            !entry.is_expired()
        } else {
            false
        }
    }
}

/// Data pipeline for async processing
pub struct DataPipeline {
    /// Cache for processed data
    cache: DataCache,
    /// Data loader function (customizable)
    loader: Option<Arc<dyn Fn(&str) -> DataSource + Send + Sync>>,
}

impl DataPipeline {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self {
            cache: DataCache::new(100),
            loader: None,
        }
    }

    /// Set cache capacity
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.cache = DataCache::new(capacity);
        self
    }

    /// Set cache TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.cache = self.cache.with_ttl(ttl);
        self
    }

    /// Set custom data loader
    pub fn with_loader<F>(mut self, loader: F) -> Self
    where
        F: Fn(&str) -> DataSource + Send + Sync + 'static,
    {
        self.loader = Some(Arc::new(loader));
        self
    }

    /// Process a data source (with caching)
    pub async fn process(&self, source_id: &str) -> Result<DataSource, DataError> {
        // Check cache first
        if let Some(cached) = self.cache.get(source_id) {
            info!(source_id = source_id, "Returning cached data");
            return Ok(cached);
        }

        // Load data
        let data = self.load_source(source_id).await?;

        // Cache it
        self.cache.insert(source_id.to_string(), data.clone());

        info!(source_id = source_id, "Loaded and cached data");
        Ok(data)
    }

    /// Process multiple sources
    pub async fn process_batch(
        &self,
        source_ids: &[String],
    ) -> Vec<Result<DataSource, DataError>> {
        let mut results = Vec::with_capacity(source_ids.len());

        for source_id in source_ids {
            results.push(self.process(source_id).await);
        }

        results
    }

    /// Load a data source
    async fn load_source(&self, source_id: &str) -> Result<DataSource, DataError> {
        // Simulate async load
        tokio::time::sleep(Duration::from_millis(10)).await;

        if let Some(ref loader) = self.loader {
            Ok(loader(source_id))
        } else {
            // Default: return empty source
            Ok(DataSource::new(source_id, format!("Source {}", source_id)))
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> &CacheStats {
        self.cache.stats()
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Invalidate specific cache entry
    pub fn invalidate(&self, source_id: &str) {
        self.cache.remove(source_id);
    }
}

impl Default for DataPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache = DataCache::new(2);

        let source = DataSource::new("test", "Test Source");
        cache.insert("a".to_string(), source.clone());

        assert!(cache.contains("a"));
        assert!(!cache.contains("b"));

        let retrieved = cache.get("a");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test");
    }

    #[test]
    fn test_cache_lru_eviction() {
        let cache = DataCache::new(2);

        cache.insert("a".to_string(), DataSource::new("a", "A"));
        cache.insert("b".to_string(), DataSource::new("b", "B"));

        // Access 'a' to make it recently used
        let _ = cache.get("a");

        // Insert 'c' - should evict 'b'
        cache.insert("c".to_string(), DataSource::new("c", "C"));

        assert!(cache.contains("a"), "a should still exist (recently used)");
        assert!(cache.contains("c"), "c should exist (just inserted)");
        assert!(!cache.contains("b"), "b should be evicted (LRU)");
    }

    #[test]
    fn test_cache_ttl_expiration() {
        let cache = DataCache::new(10).with_ttl(Duration::from_millis(50));

        cache.insert("short".to_string(), DataSource::new("short", "Short TTL"));

        // Should exist immediately
        assert!(cache.get("short").is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(60));

        // Should be expired
        assert!(cache.get("short").is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = DataCache::new(10);

        cache.insert("a".to_string(), DataSource::new("a", "A"));

        let _ = cache.get("a"); // Hit
        let _ = cache.get("a"); // Hit
        let _ = cache.get("b"); // Miss

        use std::sync::atomic::Ordering;
        assert_eq!(cache.stats().hits.load(Ordering::Relaxed), 2);
        assert_eq!(cache.stats().misses.load(Ordering::Relaxed), 1);
        assert!(cache.stats().hit_rate() > 0.6);
    }

    #[tokio::test]
    async fn test_pipeline_caching() {
        let pipeline = DataPipeline::new()
            .with_ttl(Duration::from_secs(60))
            .with_loader(|id| DataSource::new(id, format!("Loaded {}", id)));

        // First call - cache miss
        let start = Instant::now();
        let _ = pipeline.process("test").await.unwrap();
        let first_duration = start.elapsed();

        // Second call - cache hit
        let start2 = Instant::now();
        let _ = pipeline.process("test").await.unwrap();
        let second_duration = start2.elapsed();

        // Cache hit should be faster (no simulated load delay)
        assert!(
            second_duration < first_duration,
            "Cache hit should be faster: {:?} vs {:?}",
            second_duration,
            first_duration
        );
    }

    #[tokio::test]
    async fn test_pipeline_batch() {
        let pipeline = DataPipeline::new()
            .with_loader(|id| DataSource::new(id, format!("Source {}", id)));

        let ids = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = pipeline.process_batch(&ids).await;

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn test_pipeline_invalidation() {
        let pipeline = DataPipeline::new()
            .with_loader(|id| DataSource::new(id, format!("Source {}", id)));

        // Load and cache
        let _ = pipeline.process("test").await.unwrap();
        assert!(pipeline.cache.contains("test"));

        // Invalidate
        pipeline.invalidate("test");
        assert!(!pipeline.cache.contains("test"));
    }
}
