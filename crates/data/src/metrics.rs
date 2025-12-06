//! Data matching metrics and observability
//!
//! Comprehensive metrics for monitoring data matching performance.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

/// Metrics for data matching operations
#[derive(Debug, Default)]
pub struct DataMatchingMetrics {
    /// Total number of queries processed
    total_queries: AtomicU64,
    /// Number of successful matches
    successful_matches: AtomicU64,
    /// Number of failed/no matches
    failed_matches: AtomicU64,
    /// Cache hits
    cache_hits: AtomicU64,
    /// Cache misses
    cache_misses: AtomicU64,
    /// Cache evictions
    cache_evictions: AtomicU64,
    /// Negative cache hits (cached "not found")
    negative_cache_hits: AtomicU64,
    /// Processing time in microseconds (exponential moving average)
    processing_time_ema: AtomicU64,
    /// Peak processing time in microseconds
    peak_processing_time: AtomicU64,
    /// Total records scanned
    records_scanned: AtomicU64,
    /// Memory usage estimate in bytes
    memory_usage: AtomicUsize,
    /// CPF matches found
    cpf_matches: AtomicU64,
    /// CNPJ matches found
    cnpj_matches: AtomicU64,
    /// Fuzzy name matches found
    fuzzy_matches: AtomicU64,
    /// Cross-source matches (entity found in multiple sources)
    cross_source_matches: AtomicU64,
}

impl DataMatchingMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a query result
    pub fn record_query(&self, success: bool, duration: Duration) {
        self.total_queries.fetch_add(1, Ordering::Relaxed);

        if success {
            self.successful_matches.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_matches.fetch_add(1, Ordering::Relaxed);
        }

        // Update exponential moving average (EMA) for processing time
        // α = 0.3 gives more weight to recent values
        let duration_us = duration.as_micros() as u64;
        let prev = self.processing_time_ema.load(Ordering::Relaxed);
        let new_ema = if prev == 0 {
            duration_us
        } else {
            // EMA = α * current + (1 - α) * previous
            // Using integer math: (3 * current + 7 * previous) / 10
            (duration_us * 3 + prev * 7) / 10
        };
        self.processing_time_ema.store(new_ema, Ordering::Relaxed);

        // Update peak if this is the highest
        self.peak_processing_time
            .fetch_max(duration_us, Ordering::Relaxed);
    }

    /// Record cache access
    pub fn record_cache(&self, hit: bool, negative: bool) {
        if hit {
            if negative {
                self.negative_cache_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record cache eviction
    pub fn record_eviction(&self) {
        self.cache_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record records scanned
    pub fn record_scan(&self, count: u64) {
        self.records_scanned.fetch_add(count, Ordering::Relaxed);
    }

    /// Record match type
    pub fn record_match_type(&self, cpf: bool, cnpj: bool, fuzzy: bool, cross_source: bool) {
        if cpf {
            self.cpf_matches.fetch_add(1, Ordering::Relaxed);
        }
        if cnpj {
            self.cnpj_matches.fetch_add(1, Ordering::Relaxed);
        }
        if fuzzy {
            self.fuzzy_matches.fetch_add(1, Ordering::Relaxed);
        }
        if cross_source {
            self.cross_source_matches.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Update memory usage estimate
    pub fn set_memory_usage(&self, bytes: usize) {
        self.memory_usage.store(bytes, Ordering::Relaxed);
    }

    /// Get cache hit rate (0.0 - 1.0)
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f64;
        let negative_hits = self.negative_cache_hits.load(Ordering::Relaxed) as f64;
        let misses = self.cache_misses.load(Ordering::Relaxed) as f64;
        let total = hits + negative_hits + misses;

        if total == 0.0 {
            0.0
        } else {
            (hits + negative_hits) / total
        }
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let successful = self.successful_matches.load(Ordering::Relaxed) as f64;
        let total = self.total_queries.load(Ordering::Relaxed) as f64;

        if total == 0.0 {
            0.0
        } else {
            successful / total
        }
    }

    /// Get average processing time in milliseconds
    pub fn avg_processing_time_ms(&self) -> f64 {
        self.processing_time_ema.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Get peak processing time in milliseconds
    pub fn peak_processing_time_ms(&self) -> f64 {
        self.peak_processing_time.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Get throughput (records per second based on EMA)
    pub fn throughput_rps(&self) -> f64 {
        let avg_time_us = self.processing_time_ema.load(Ordering::Relaxed) as f64;
        if avg_time_us == 0.0 {
            0.0
        } else {
            1_000_000.0 / avg_time_us
        }
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_queries: self.total_queries.load(Ordering::Relaxed),
            successful_matches: self.successful_matches.load(Ordering::Relaxed),
            failed_matches: self.failed_matches.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cache_evictions: self.cache_evictions.load(Ordering::Relaxed),
            negative_cache_hits: self.negative_cache_hits.load(Ordering::Relaxed),
            cache_hit_rate: self.cache_hit_rate(),
            success_rate: self.success_rate(),
            avg_processing_time_ms: self.avg_processing_time_ms(),
            peak_processing_time_ms: self.peak_processing_time_ms(),
            throughput_rps: self.throughput_rps(),
            records_scanned: self.records_scanned.load(Ordering::Relaxed),
            memory_usage_bytes: self.memory_usage.load(Ordering::Relaxed),
            cpf_matches: self.cpf_matches.load(Ordering::Relaxed),
            cnpj_matches: self.cnpj_matches.load(Ordering::Relaxed),
            fuzzy_matches: self.fuzzy_matches.load(Ordering::Relaxed),
            cross_source_matches: self.cross_source_matches.load(Ordering::Relaxed),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.total_queries.store(0, Ordering::Relaxed);
        self.successful_matches.store(0, Ordering::Relaxed);
        self.failed_matches.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.cache_evictions.store(0, Ordering::Relaxed);
        self.negative_cache_hits.store(0, Ordering::Relaxed);
        self.processing_time_ema.store(0, Ordering::Relaxed);
        self.peak_processing_time.store(0, Ordering::Relaxed);
        self.records_scanned.store(0, Ordering::Relaxed);
        self.cpf_matches.store(0, Ordering::Relaxed);
        self.cnpj_matches.store(0, Ordering::Relaxed);
        self.fuzzy_matches.store(0, Ordering::Relaxed);
        self.cross_source_matches.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub total_queries: u64,
    pub successful_matches: u64,
    pub failed_matches: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_evictions: u64,
    pub negative_cache_hits: u64,
    pub cache_hit_rate: f64,
    pub success_rate: f64,
    pub avg_processing_time_ms: f64,
    pub peak_processing_time_ms: f64,
    pub throughput_rps: f64,
    pub records_scanned: u64,
    pub memory_usage_bytes: usize,
    pub cpf_matches: u64,
    pub cnpj_matches: u64,
    pub fuzzy_matches: u64,
    pub cross_source_matches: u64,
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Data Matching Metrics:")?;
        writeln!(
            f,
            "  Queries: {} total, {} successful ({:.1}%)",
            self.total_queries,
            self.successful_matches,
            self.success_rate * 100.0
        )?;
        writeln!(
            f,
            "  Cache: {:.1}% hit rate ({} hits, {} misses, {} negative)",
            self.cache_hit_rate * 100.0,
            self.cache_hits,
            self.cache_misses,
            self.negative_cache_hits
        )?;
        writeln!(
            f,
            "  Performance: {:.2}ms avg, {:.2}ms peak, {:.1} queries/sec",
            self.avg_processing_time_ms, self.peak_processing_time_ms, self.throughput_rps
        )?;
        writeln!(
            f,
            "  Matches: {} CPF, {} CNPJ, {} fuzzy, {} cross-source",
            self.cpf_matches, self.cnpj_matches, self.fuzzy_matches, self.cross_source_matches
        )?;
        writeln!(
            f,
            "  Records scanned: {}, Memory: {} bytes",
            self.records_scanned, self.memory_usage_bytes
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_basic() {
        let metrics = DataMatchingMetrics::new();

        metrics.record_query(true, Duration::from_micros(1000));
        metrics.record_query(true, Duration::from_micros(2000));
        metrics.record_query(false, Duration::from_micros(500));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_queries, 3);
        assert_eq!(snapshot.successful_matches, 2);
        assert_eq!(snapshot.failed_matches, 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let metrics = DataMatchingMetrics::new();

        metrics.record_cache(true, false); // hit
        metrics.record_cache(true, false); // hit
        metrics.record_cache(false, false); // miss

        let rate = metrics.cache_hit_rate();
        assert!((rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_negative_cache() {
        let metrics = DataMatchingMetrics::new();

        metrics.record_cache(true, true); // negative hit
        metrics.record_cache(true, false); // regular hit
        metrics.record_cache(false, false); // miss

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.negative_cache_hits, 1);
        assert_eq!(snapshot.cache_hits, 1);
        assert_eq!(snapshot.cache_misses, 1);
        // Both negative and regular hits count toward hit rate
        assert!((snapshot.cache_hit_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_ema_processing_time() {
        let metrics = DataMatchingMetrics::new();

        // First value sets the baseline
        metrics.record_query(true, Duration::from_micros(1000));
        assert_eq!(metrics.processing_time_ema.load(Ordering::Relaxed), 1000);

        // EMA should move toward new value
        metrics.record_query(true, Duration::from_micros(2000));
        let ema = metrics.processing_time_ema.load(Ordering::Relaxed);
        assert!(ema > 1000 && ema < 2000);
    }

    #[test]
    fn test_peak_tracking() {
        let metrics = DataMatchingMetrics::new();

        metrics.record_query(true, Duration::from_micros(100));
        metrics.record_query(true, Duration::from_micros(500));
        metrics.record_query(true, Duration::from_micros(200));

        assert_eq!(metrics.peak_processing_time.load(Ordering::Relaxed), 500);
    }

    #[test]
    fn test_match_types() {
        let metrics = DataMatchingMetrics::new();

        metrics.record_match_type(true, false, false, false);
        metrics.record_match_type(false, true, false, true);
        metrics.record_match_type(false, false, true, true);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.cpf_matches, 1);
        assert_eq!(snapshot.cnpj_matches, 1);
        assert_eq!(snapshot.fuzzy_matches, 1);
        assert_eq!(snapshot.cross_source_matches, 2);
    }

    #[test]
    fn test_reset() {
        let metrics = DataMatchingMetrics::new();

        metrics.record_query(true, Duration::from_micros(1000));
        metrics.record_cache(true, false);

        metrics.reset();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_queries, 0);
        assert_eq!(snapshot.cache_hits, 0);
    }
}
