//! Cost tracking for LLM API usage

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Cost tracking statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostStats {
    /// Total cost in USD
    pub total_cost: f64,

    /// Total requests made
    pub total_requests: u64,

    /// Total input tokens
    pub total_input_tokens: u64,

    /// Total output tokens
    pub total_output_tokens: u64,

    /// Cached tokens (if applicable)
    pub cached_tokens: u64,

    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,

    /// Average latency in milliseconds
    pub avg_latency_ms: f64,

    /// Cost savings from caching
    pub cache_savings: f64,

    /// First request timestamp
    pub started_at: DateTime<Utc>,

    /// Last request timestamp
    pub last_request_at: Option<DateTime<Utc>>,
}

impl Default for CostStats {
    fn default() -> Self {
        Self {
            total_cost: 0.0,
            total_requests: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            cached_tokens: 0,
            cache_hit_rate: 0.0,
            avg_latency_ms: 0.0,
            cache_savings: 0.0,
            started_at: Utc::now(),
            last_request_at: None,
        }
    }
}

/// Cost tracker
pub struct CostTracker {
    stats: Arc<RwLock<CostStats>>,
    latencies: Arc<RwLock<Vec<f64>>>,
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(CostStats::default())),
            latencies: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record an API request
    pub fn record_request(
        &self,
        cost: f64,
        input_tokens: usize,
        output_tokens: usize,
        cached_tokens: Option<usize>,
        latency_ms: f64,
    ) {
        let mut stats = self.stats.write();

        stats.total_cost += cost;
        stats.total_requests += 1;
        stats.total_input_tokens += input_tokens as u64;
        stats.total_output_tokens += output_tokens as u64;
        stats.last_request_at = Some(Utc::now());

        if let Some(cached) = cached_tokens {
            stats.cached_tokens += cached as u64;

            // Estimate cache savings (cached tokens typically cost 10% of full price)
            let cache_cost_multiplier = 0.1;
            let full_cost_per_token = cost / (input_tokens + output_tokens) as f64;
            let savings = cached as f64 * full_cost_per_token * (1.0 - cache_cost_multiplier);
            stats.cache_savings += savings;
        }

        // Update cache hit rate
        if stats.total_input_tokens > 0 {
            stats.cache_hit_rate = stats.cached_tokens as f64 / stats.total_input_tokens as f64;
        }

        // Update average latency
        let mut latencies = self.latencies.write();
        latencies.push(latency_ms);
        stats.avg_latency_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
    }

    /// Get current statistics
    pub fn stats(&self) -> CostStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset(&self) {
        let mut stats = self.stats.write();
        *stats = CostStats::default();

        let mut latencies = self.latencies.write();
        latencies.clear();
    }

    /// Get cost per request
    pub fn cost_per_request(&self) -> f64 {
        let stats = self.stats.read();
        if stats.total_requests > 0 {
            stats.total_cost / stats.total_requests as f64
        } else {
            0.0
        }
    }

    /// Get tokens per request
    pub fn tokens_per_request(&self) -> f64 {
        let stats = self.stats.read();
        if stats.total_requests > 0 {
            (stats.total_input_tokens + stats.total_output_tokens) as f64
                / stats.total_requests as f64
        } else {
            0.0
        }
    }

    /// Print summary
    pub fn print_summary(&self) {
        let stats = self.stats();

        println!("\n═══════════════════════════════════════════");
        println!("           COST TRACKER SUMMARY");
        println!("═══════════════════════════════════════════");
        println!("Total Cost:           ${:.6}", stats.total_cost);
        println!("Cache Savings:        ${:.6}", stats.cache_savings);
        println!(
            "Net Cost:             ${:.6}",
            stats.total_cost - stats.cache_savings
        );
        println!("───────────────────────────────────────────");
        println!("Total Requests:       {}", stats.total_requests);
        println!("Input Tokens:         {}", stats.total_input_tokens);
        println!("Output Tokens:        {}", stats.total_output_tokens);
        println!("Cached Tokens:        {}", stats.cached_tokens);
        println!("───────────────────────────────────────────");
        println!("Cache Hit Rate:       {:.1}%", stats.cache_hit_rate * 100.0);
        println!("Avg Latency:          {:.0} ms", stats.avg_latency_ms);
        println!("Cost per Request:     ${:.6}", self.cost_per_request());
        println!("Tokens per Request:   {:.0}", self.tokens_per_request());
        println!("═══════════════════════════════════════════\n");
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CostTracker {
    fn clone(&self) -> Self {
        Self {
            stats: self.stats.clone(),
            latencies: self.latencies.clone(),
        }
    }
}
