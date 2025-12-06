//! Data pipeline with LRU caching
//!
//! Async data processing with TTL-based cache eviction and parallel batch support.

use crate::metrics::DataMatchingMetrics;
use crate::types::{DataError, DataSource};
use dashmap::DashMap;
use futures::future::join_all;
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
    pub async fn process_batch(&self, source_ids: &[String]) -> Vec<Result<DataSource, DataError>> {
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

/// Negative cache entry - remembers that something was not found
#[derive(Debug, Clone)]
pub struct NegativeCacheEntry {
    /// When this entry was cached
    pub cached_at: Instant,
    /// TTL for negative cache (usually shorter than positive)
    pub ttl: Duration,
}

impl NegativeCacheEntry {
    pub fn new(ttl: Duration) -> Self {
        Self {
            cached_at: Instant::now(),
            ttl,
        }
    }

    pub fn is_expired(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }
}

/// Concurrent cache using DashMap for high-throughput scenarios
pub struct ConcurrentCache {
    /// Main data cache
    cache: DashMap<String, CachedData>,
    /// Negative cache (remembers "not found" results)
    negative_cache: DashMap<String, NegativeCacheEntry>,
    /// Maximum capacity
    capacity: usize,
    /// Default TTL for entries
    default_ttl: Duration,
    /// TTL for negative cache entries (shorter)
    negative_ttl: Duration,
    /// Metrics
    metrics: Arc<DataMatchingMetrics>,
}

impl ConcurrentCache {
    /// Create a new concurrent cache
    pub fn new(capacity: usize, metrics: Arc<DataMatchingMetrics>) -> Self {
        Self {
            cache: DashMap::with_capacity(capacity),
            negative_cache: DashMap::with_capacity(capacity / 4),
            capacity,
            default_ttl: Duration::from_secs(300),
            negative_ttl: Duration::from_secs(60),
            metrics,
        }
    }

    /// Set default TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = ttl;
        self
    }

    /// Set negative cache TTL
    pub fn with_negative_ttl(mut self, ttl: Duration) -> Self {
        self.negative_ttl = ttl;
        self
    }

    /// Get from cache (checks negative cache too)
    pub fn get(&self, key: &str) -> CacheResult {
        // Check negative cache first
        if let Some(entry) = self.negative_cache.get(key) {
            if !entry.is_expired() {
                self.metrics.record_cache(true, true);
                return CacheResult::NegativeHit;
            } else {
                drop(entry);
                self.negative_cache.remove(key);
            }
        }

        // Check main cache
        if let Some(entry) = self.cache.get(key) {
            if !entry.is_expired() {
                self.metrics.record_cache(true, false);
                return CacheResult::Hit(entry.data.clone());
            } else {
                drop(entry);
                self.cache.remove(key);
            }
        }

        self.metrics.record_cache(false, false);
        CacheResult::Miss
    }

    /// Insert into cache
    pub fn insert(&self, key: String, data: DataSource) {
        // Evict if at capacity
        if self.cache.len() >= self.capacity {
            if let Some(entry) = self.cache.iter().next() {
                let key_to_remove = entry.key().clone();
                drop(entry);
                self.cache.remove(&key_to_remove);
                self.metrics.record_eviction();
            }
        }

        // Remove from negative cache if present
        self.negative_cache.remove(&key);

        self.cache
            .insert(key, CachedData::new(data, self.default_ttl));
    }

    /// Insert a negative cache entry
    pub fn insert_negative(&self, key: String) {
        if self.negative_cache.len() >= self.capacity / 4 {
            if let Some(entry) = self.negative_cache.iter().next() {
                let key_to_remove = entry.key().clone();
                drop(entry);
                self.negative_cache.remove(&key_to_remove);
            }
        }

        self.negative_cache
            .insert(key, NegativeCacheEntry::new(self.negative_ttl));
    }

    /// Remove from both caches
    pub fn remove(&self, key: &str) {
        self.cache.remove(key);
        self.negative_cache.remove(key);
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.cache.clear();
        self.negative_cache.clear();
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get negative cache size
    pub fn negative_len(&self) -> usize {
        self.negative_cache.len()
    }
}

/// Result of a cache lookup
#[derive(Debug, Clone)]
pub enum CacheResult {
    /// Data found in cache
    Hit(DataSource),
    /// Key was previously looked up and not found
    NegativeHit,
    /// Key not in cache
    Miss,
}

/// High-performance parallel data pipeline
pub struct ParallelPipeline {
    /// Concurrent cache
    cache: Arc<ConcurrentCache>,
    /// Async data loader
    loader: Option<Arc<dyn Fn(String) -> DataSource + Send + Sync>>,
    /// Metrics
    metrics: Arc<DataMatchingMetrics>,
    /// Maximum concurrent tasks
    max_concurrency: usize,
}

impl ParallelPipeline {
    /// Create a new parallel pipeline
    pub fn new(capacity: usize) -> Self {
        let metrics = Arc::new(DataMatchingMetrics::new());
        Self {
            cache: Arc::new(ConcurrentCache::new(capacity, metrics.clone())),
            loader: None,
            metrics,
            max_concurrency: 10,
        }
    }

    /// Set maximum concurrency
    pub fn with_max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = max;
        self
    }

    /// Set cache TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.cache =
            Arc::new(ConcurrentCache::new(self.cache.capacity, self.metrics.clone()).with_ttl(ttl));
        self
    }

    /// Set custom data loader
    pub fn with_loader<F>(mut self, loader: F) -> Self
    where
        F: Fn(String) -> DataSource + Send + Sync + 'static,
    {
        self.loader = Some(Arc::new(loader));
        self
    }

    /// Process a single source
    pub async fn process(&self, source_id: &str) -> Result<DataSource, DataError> {
        let start = Instant::now();

        match self.cache.get(source_id) {
            CacheResult::Hit(data) => {
                self.metrics.record_query(true, start.elapsed());
                return Ok(data);
            }
            CacheResult::NegativeHit => {
                self.metrics.record_query(false, start.elapsed());
                return Err(DataError::SourceNotFound(source_id.to_string()));
            }
            CacheResult::Miss => {}
        }

        let result = self.load_source(source_id).await;

        match result {
            Ok(data) => {
                self.cache.insert(source_id.to_string(), data.clone());
                self.metrics.record_query(true, start.elapsed());
                Ok(data)
            }
            Err(e) => {
                self.cache.insert_negative(source_id.to_string());
                self.metrics.record_query(false, start.elapsed());
                Err(e)
            }
        }
    }

    /// Process multiple sources in parallel
    pub async fn process_parallel(
        &self,
        source_ids: Vec<String>,
    ) -> Vec<Result<DataSource, DataError>> {
        let chunks: Vec<_> = source_ids
            .chunks(self.max_concurrency)
            .map(|c| c.to_vec())
            .collect();

        let mut all_results = Vec::with_capacity(source_ids.len());

        for chunk in chunks {
            let tasks: Vec<_> = chunk
                .into_iter()
                .map(|id| {
                    let cache = self.cache.clone();
                    let loader = self.loader.clone();
                    let metrics = self.metrics.clone();
                    async move {
                        let start = Instant::now();

                        match cache.get(&id) {
                            CacheResult::Hit(data) => {
                                metrics.record_query(true, start.elapsed());
                                return Ok(data);
                            }
                            CacheResult::NegativeHit => {
                                metrics.record_query(false, start.elapsed());
                                return Err(DataError::SourceNotFound(id));
                            }
                            CacheResult::Miss => {}
                        }

                        tokio::time::sleep(Duration::from_millis(10)).await;

                        if let Some(ref loader) = loader {
                            let data = loader(id.clone());
                            cache.insert(id, data.clone());
                            metrics.record_query(true, start.elapsed());
                            Ok(data)
                        } else {
                            let data = DataSource::new(&id, format!("Source {}", id));
                            cache.insert(id, data.clone());
                            metrics.record_query(true, start.elapsed());
                            Ok(data)
                        }
                    }
                })
                .collect();

            let chunk_results = join_all(tasks).await;
            all_results.extend(chunk_results);
        }

        all_results
    }

    /// Load a data source
    async fn load_source(&self, source_id: &str) -> Result<DataSource, DataError> {
        tokio::time::sleep(Duration::from_millis(10)).await;

        if let Some(ref loader) = self.loader {
            Ok(loader(source_id.to_string()))
        } else {
            Ok(DataSource::new(source_id, format!("Source {}", source_id)))
        }
    }

    /// Get metrics
    pub fn metrics(&self) -> &DataMatchingMetrics {
        &self.metrics
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.clear();
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
        let pipeline =
            DataPipeline::new().with_loader(|id| DataSource::new(id, format!("Source {}", id)));

        let ids = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = pipeline.process_batch(&ids).await;

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn test_pipeline_invalidation() {
        let pipeline =
            DataPipeline::new().with_loader(|id| DataSource::new(id, format!("Source {}", id)));

        // Load and cache
        let _ = pipeline.process("test").await.unwrap();
        assert!(pipeline.cache.contains("test"));

        // Invalidate
        pipeline.invalidate("test");
        assert!(!pipeline.cache.contains("test"));
    }

    #[test]
    fn test_concurrent_cache_basic() {
        let metrics = Arc::new(DataMatchingMetrics::new());
        let cache = ConcurrentCache::new(10, metrics);

        let source = DataSource::new("test", "Test Source");
        cache.insert("a".to_string(), source);

        match cache.get("a") {
            CacheResult::Hit(data) => assert_eq!(data.id, "test"),
            _ => panic!("Expected cache hit"),
        }

        match cache.get("nonexistent") {
            CacheResult::Miss => {}
            _ => panic!("Expected cache miss"),
        }
    }

    #[test]
    fn test_concurrent_cache_negative() {
        let metrics = Arc::new(DataMatchingMetrics::new());
        let cache = ConcurrentCache::new(10, metrics);

        // Insert negative entry
        cache.insert_negative("missing".to_string());

        // Should get negative hit
        match cache.get("missing") {
            CacheResult::NegativeHit => {}
            _ => panic!("Expected negative cache hit"),
        }

        // Insert actual data - should remove from negative cache
        cache.insert("missing".to_string(), DataSource::new("missing", "Found"));

        match cache.get("missing") {
            CacheResult::Hit(data) => assert_eq!(data.id, "missing"),
            _ => panic!("Expected cache hit after insert"),
        }
    }

    #[tokio::test]
    async fn test_parallel_pipeline() {
        let pipeline = ParallelPipeline::new(100)
            .with_max_concurrency(5)
            .with_loader(|id| DataSource::new(&id, format!("Source {}", id)));

        let ids: Vec<String> = (0..20).map(|i| format!("source_{}", i)).collect();
        let results = pipeline.process_parallel(ids).await;

        assert_eq!(results.len(), 20);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn test_parallel_pipeline_caching() {
        let pipeline = ParallelPipeline::new(100)
            .with_loader(|id| DataSource::new(&id, format!("Source {}", id)));

        // First call
        let _ = pipeline.process("test").await.unwrap();

        // Second call should be cached
        let start = Instant::now();
        let _ = pipeline.process("test").await.unwrap();
        let cached_duration = start.elapsed();

        // Should be very fast (cached)
        assert!(cached_duration < Duration::from_millis(5));
    }
}
