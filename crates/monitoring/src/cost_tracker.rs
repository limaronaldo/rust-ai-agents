//! Cost tracking for LLM API usage with advanced ANSI dashboard

use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// ANSI color codes for terminal rendering
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const RED: &str = "\x1b[31m";
    pub const BG_GREEN: &str = "\x1b[42m";
    pub const BG_YELLOW: &str = "\x1b[43m";
    pub const BG_RED: &str = "\x1b[41m";
}

/// Model-specific pricing (per 1K tokens)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_price: f64,
    pub output_price: f64,
    pub cached_input_price: Option<f64>,
}

impl ModelPricing {
    pub fn new(input: f64, output: f64) -> Self {
        Self {
            input_price: input,
            output_price: output,
            cached_input_price: Some(input * 0.1), // 90% discount for cached
        }
    }

    pub fn with_cache_price(mut self, cached: f64) -> Self {
        self.cached_input_price = Some(cached);
        self
    }
}

/// Default pricing for common models
pub fn default_pricing() -> HashMap<String, ModelPricing> {
    let mut pricing = HashMap::new();

    // OpenAI models
    pricing.insert("gpt-4o".to_string(), ModelPricing::new(0.0025, 0.01));
    pricing.insert(
        "gpt-4o-mini".to_string(),
        ModelPricing::new(0.00015, 0.0006),
    );
    pricing.insert("gpt-4-turbo".to_string(), ModelPricing::new(0.01, 0.03));
    pricing.insert(
        "gpt-3.5-turbo".to_string(),
        ModelPricing::new(0.0005, 0.0015),
    );

    // Anthropic models
    pricing.insert(
        "claude-3-5-sonnet-20241022".to_string(),
        ModelPricing::new(0.003, 0.015).with_cache_price(0.0003),
    );
    pricing.insert(
        "claude-3-opus-20240229".to_string(),
        ModelPricing::new(0.015, 0.075).with_cache_price(0.0015),
    );
    pricing.insert(
        "claude-3-haiku-20240307".to_string(),
        ModelPricing::new(0.00025, 0.00125).with_cache_price(0.000025),
    );

    // DeepSeek models
    pricing.insert(
        "deepseek-chat".to_string(),
        ModelPricing::new(0.00014, 0.00028),
    );
    pricing.insert(
        "deepseek-reasoner".to_string(),
        ModelPricing::new(0.00055, 0.00219),
    );

    pricing
}

/// Request record for detailed tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    pub timestamp: DateTime<Utc>,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub cost: f64,
    pub latency_ms: f64,
    pub agent_id: Option<String>,
    pub success: bool,
}

/// Per-model statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelStats {
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub total_cost: f64,
    pub total_latency_ms: f64,
    pub errors: u64,
}

/// Per-agent statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentStats {
    pub requests: u64,
    pub total_cost: f64,
    pub total_tokens: u64,
    pub avg_latency_ms: f64,
}

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

    /// Per-model breakdown
    pub by_model: HashMap<String, ModelStats>,

    /// Per-agent breakdown
    pub by_agent: HashMap<String, AgentStats>,

    /// Error count
    pub total_errors: u64,

    /// Requests per minute (rolling)
    pub requests_per_minute: f64,
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
            by_model: HashMap::new(),
            by_agent: HashMap::new(),
            total_errors: 0,
            requests_per_minute: 0.0,
        }
    }
}

/// Budget alert configuration
pub struct BudgetAlert {
    pub threshold: f64,
    pub triggered: bool,
    pub callback: Option<Arc<dyn Fn(f64) + Send + Sync>>,
}

impl std::fmt::Debug for BudgetAlert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetAlert")
            .field("threshold", &self.threshold)
            .field("triggered", &self.triggered)
            .field("callback", &self.callback.as_ref().map(|_| "<callback>"))
            .finish()
    }
}

impl Clone for BudgetAlert {
    fn clone(&self) -> Self {
        Self {
            threshold: self.threshold,
            triggered: self.triggered,
            callback: self.callback.clone(),
        }
    }
}

/// Cost tracker with advanced features
pub struct CostTracker {
    stats: Arc<RwLock<CostStats>>,
    latencies: Arc<RwLock<Vec<f64>>>,
    records: Arc<RwLock<Vec<RequestRecord>>>,
    pricing: Arc<RwLock<HashMap<String, ModelPricing>>>,
    budget_alerts: Arc<RwLock<Vec<BudgetAlert>>>,
    max_records: usize,
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(CostStats::default())),
            latencies: Arc::new(RwLock::new(Vec::new())),
            records: Arc::new(RwLock::new(Vec::new())),
            pricing: Arc::new(RwLock::new(default_pricing())),
            budget_alerts: Arc::new(RwLock::new(Vec::new())),
            max_records: 10000,
        }
    }

    /// Create with custom pricing
    pub fn with_pricing(self, pricing: HashMap<String, ModelPricing>) -> Self {
        *self.pricing.write() = pricing;
        self
    }

    /// Set maximum records to keep
    pub fn with_max_records(mut self, max: usize) -> Self {
        self.max_records = max;
        self
    }

    /// Add a budget alert
    pub fn add_budget_alert(
        &self,
        threshold: f64,
        callback: Option<Arc<dyn Fn(f64) + Send + Sync>>,
    ) {
        self.budget_alerts.write().push(BudgetAlert {
            threshold,
            triggered: false,
            callback,
        });
    }

    /// Calculate cost for a request
    pub fn calculate_cost(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
    ) -> f64 {
        let pricing = self.pricing.read();
        if let Some(p) = pricing.get(model) {
            let regular_input = input_tokens.saturating_sub(cached_tokens);
            let input_cost = (regular_input as f64 / 1000.0) * p.input_price;
            let cached_cost = if let Some(cached_price) = p.cached_input_price {
                (cached_tokens as f64 / 1000.0) * cached_price
            } else {
                (cached_tokens as f64 / 1000.0) * p.input_price * 0.1
            };
            let output_cost = (output_tokens as f64 / 1000.0) * p.output_price;
            input_cost + cached_cost + output_cost
        } else {
            // Default pricing if model not found
            let input_cost = (input_tokens as f64 / 1000.0) * 0.001;
            let output_cost = (output_tokens as f64 / 1000.0) * 0.002;
            input_cost + output_cost
        }
    }

    /// Record an API request with full details
    #[allow(clippy::too_many_arguments)]
    pub fn record_request_detailed(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
        latency_ms: f64,
        agent_id: Option<&str>,
        success: bool,
    ) {
        let cost = self.calculate_cost(model, input_tokens, output_tokens, cached_tokens);

        let record = RequestRecord {
            timestamp: Utc::now(),
            model: model.to_string(),
            input_tokens,
            output_tokens,
            cached_tokens,
            cost,
            latency_ms,
            agent_id: agent_id.map(|s| s.to_string()),
            success,
        };

        // Update records
        {
            let mut records = self.records.write();
            records.push(record);
            let current_len = records.len();
            if current_len > self.max_records {
                let drain_count = current_len - self.max_records;
                records.drain(0..drain_count);
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_cost += cost;
            stats.total_requests += 1;
            stats.total_input_tokens += input_tokens;
            stats.total_output_tokens += output_tokens;
            stats.cached_tokens += cached_tokens;
            stats.last_request_at = Some(Utc::now());

            if !success {
                stats.total_errors += 1;
            }

            // Calculate cache savings
            if cached_tokens > 0 {
                let pricing = self.pricing.read();
                if let Some(p) = pricing.get(model) {
                    let full_cost = (cached_tokens as f64 / 1000.0) * p.input_price;
                    let cached_cost = if let Some(cached_price) = p.cached_input_price {
                        (cached_tokens as f64 / 1000.0) * cached_price
                    } else {
                        full_cost * 0.1
                    };
                    stats.cache_savings += full_cost - cached_cost;
                }
            }

            // Update cache hit rate
            if stats.total_input_tokens > 0 {
                stats.cache_hit_rate = stats.cached_tokens as f64 / stats.total_input_tokens as f64;
            }

            // Update model stats
            let model_stats = stats.by_model.entry(model.to_string()).or_default();
            model_stats.requests += 1;
            model_stats.input_tokens += input_tokens;
            model_stats.output_tokens += output_tokens;
            model_stats.cached_tokens += cached_tokens;
            model_stats.total_cost += cost;
            model_stats.total_latency_ms += latency_ms;
            if !success {
                model_stats.errors += 1;
            }

            // Update agent stats
            if let Some(agent) = agent_id {
                let agent_stats = stats.by_agent.entry(agent.to_string()).or_default();
                agent_stats.requests += 1;
                agent_stats.total_cost += cost;
                agent_stats.total_tokens += input_tokens + output_tokens;
                agent_stats.avg_latency_ms =
                    (agent_stats.avg_latency_ms * (agent_stats.requests - 1) as f64 + latency_ms)
                        / agent_stats.requests as f64;
            }

            // Calculate requests per minute
            let elapsed = Utc::now()
                .signed_duration_since(stats.started_at)
                .num_seconds() as f64;
            if elapsed > 0.0 {
                stats.requests_per_minute = stats.total_requests as f64 / (elapsed / 60.0);
            }
        }

        // Update latencies
        {
            let mut latencies = self.latencies.write();
            latencies.push(latency_ms);
            let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
            self.stats.write().avg_latency_ms = avg;
        }

        // Check budget alerts
        self.check_budget_alerts();
    }

    /// Legacy record_request for backwards compatibility
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
        drop(stats);
        let mut latencies = self.latencies.write();
        latencies.push(latency_ms);
        let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
        drop(latencies);
        self.stats.write().avg_latency_ms = avg;
    }

    fn check_budget_alerts(&self) {
        let current_cost = self.stats.read().total_cost;
        let mut alerts = self.budget_alerts.write();

        for alert in alerts.iter_mut() {
            if !alert.triggered && current_cost >= alert.threshold {
                alert.triggered = true;
                if let Some(ref callback) = alert.callback {
                    callback(current_cost);
                }
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> CostStats {
        self.stats.read().clone()
    }

    /// Get recent records
    pub fn recent_records(&self, count: usize) -> Vec<RequestRecord> {
        let records = self.records.read();
        records.iter().rev().take(count).cloned().collect()
    }

    /// Get records for time window
    pub fn records_since(&self, since: DateTime<Utc>) -> Vec<RequestRecord> {
        let records = self.records.read();
        records
            .iter()
            .filter(|r| r.timestamp >= since)
            .cloned()
            .collect()
    }

    /// Reset statistics
    pub fn reset(&self) {
        let mut stats = self.stats.write();
        *stats = CostStats::default();

        let mut latencies = self.latencies.write();
        latencies.clear();

        let mut records = self.records.write();
        records.clear();

        let mut alerts = self.budget_alerts.write();
        for alert in alerts.iter_mut() {
            alert.triggered = false;
        }
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

    /// Generate progress bar
    fn progress_bar(&self, value: f64, max: f64, width: usize) -> String {
        let ratio = (value / max).min(1.0);
        let filled = (ratio * width as f64) as usize;
        let empty = width - filled;

        let color = if ratio < 0.5 {
            colors::GREEN
        } else if ratio < 0.8 {
            colors::YELLOW
        } else {
            colors::RED
        };

        format!(
            "{}{}{}{}",
            color,
            "█".repeat(filled),
            colors::DIM,
            "░".repeat(empty)
        ) + colors::RESET
    }

    /// Format currency
    fn format_currency(amount: f64) -> String {
        if amount < 0.01 {
            format!("${:.6}", amount)
        } else if amount < 1.0 {
            format!("${:.4}", amount)
        } else {
            format!("${:.2}", amount)
        }
    }

    /// Format duration
    fn format_duration(duration: Duration) -> String {
        let secs = duration.num_seconds();
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
        }
    }

    /// Print advanced ANSI dashboard
    pub fn print_dashboard(&self) {
        let stats = self.stats();
        let elapsed = Utc::now().signed_duration_since(stats.started_at);

        // Clear screen and move cursor to top
        print!("\x1b[2J\x1b[H");

        // Header
        println!(
            "{}{}╔══════════════════════════════════════════════════════════════════╗{}",
            colors::BOLD,
            colors::CYAN,
            colors::RESET
        );
        println!(
            "{}{}║          🚀 RUST AI AGENTS - COST DASHBOARD                      ║{}",
            colors::BOLD,
            colors::CYAN,
            colors::RESET
        );
        println!(
            "{}{}╚══════════════════════════════════════════════════════════════════╝{}",
            colors::BOLD,
            colors::CYAN,
            colors::RESET
        );
        println!();

        // Session info
        println!(
            "{}  Session Duration:{} {}",
            colors::DIM,
            colors::RESET,
            Self::format_duration(elapsed)
        );
        println!(
            "{}  Last Activity:{}    {}",
            colors::DIM,
            colors::RESET,
            stats
                .last_request_at
                .map(|t| t.format("%H:%M:%S").to_string())
                .unwrap_or_else(|| "N/A".to_string())
        );
        println!();

        // Cost summary box
        println!(
            "{}┌─────────────────────────────────────────────────────────────────┐{}",
            colors::BLUE,
            colors::RESET
        );
        println!(
            "{}│{}  💰 COST SUMMARY                                                {}│{}",
            colors::BLUE,
            colors::BOLD,
            colors::RESET,
            colors::BLUE
        );
        println!(
            "{}├─────────────────────────────────────────────────────────────────┤{}",
            colors::BLUE,
            colors::RESET
        );
        println!(
            "{}│{}  Total Cost:        {}{}{}                              {}│{}",
            colors::BLUE,
            colors::RESET,
            colors::GREEN,
            Self::format_currency(stats.total_cost),
            " ".repeat(20 - Self::format_currency(stats.total_cost).len()),
            colors::BLUE,
            colors::RESET
        );
        println!(
            "{}│{}  Cache Savings:     {}{}{}                              {}│{}",
            colors::BLUE,
            colors::RESET,
            colors::MAGENTA,
            Self::format_currency(stats.cache_savings),
            " ".repeat(20 - Self::format_currency(stats.cache_savings).len()),
            colors::BLUE,
            colors::RESET
        );
        println!(
            "{}│{}  Net Cost:          {}{}{}                              {}│{}",
            colors::BLUE,
            colors::RESET,
            colors::YELLOW,
            Self::format_currency(stats.total_cost - stats.cache_savings),
            " ".repeat(20 - Self::format_currency(stats.total_cost - stats.cache_savings).len()),
            colors::BLUE,
            colors::RESET
        );
        println!(
            "{}└─────────────────────────────────────────────────────────────────┘{}",
            colors::BLUE,
            colors::RESET
        );
        println!();

        // Token usage
        println!(
            "{}┌─────────────────────────────────────────────────────────────────┐{}",
            colors::MAGENTA,
            colors::RESET
        );
        println!(
            "{}│{}  📊 TOKEN USAGE                                                  {}│{}",
            colors::MAGENTA,
            colors::BOLD,
            colors::RESET,
            colors::MAGENTA
        );
        println!(
            "{}├─────────────────────────────────────────────────────────────────┤{}",
            colors::MAGENTA,
            colors::RESET
        );
        let total_tokens = stats.total_input_tokens + stats.total_output_tokens;
        println!(
            "{}│{}  Input Tokens:      {:>12}                              {}│{}",
            colors::MAGENTA,
            colors::RESET,
            stats.total_input_tokens,
            colors::MAGENTA,
            colors::RESET
        );
        println!(
            "{}│{}  Output Tokens:     {:>12}                              {}│{}",
            colors::MAGENTA,
            colors::RESET,
            stats.total_output_tokens,
            colors::MAGENTA,
            colors::RESET
        );
        println!(
            "{}│{}  Cached Tokens:     {:>12}                              {}│{}",
            colors::MAGENTA,
            colors::RESET,
            stats.cached_tokens,
            colors::MAGENTA,
            colors::RESET
        );
        println!(
            "{}│{}  Total Tokens:      {:>12}                              {}│{}",
            colors::MAGENTA,
            colors::RESET,
            total_tokens,
            colors::MAGENTA,
            colors::RESET
        );
        println!(
            "{}└─────────────────────────────────────────────────────────────────┘{}",
            colors::MAGENTA,
            colors::RESET
        );
        println!();

        // Performance metrics
        println!(
            "{}┌─────────────────────────────────────────────────────────────────┐{}",
            colors::GREEN,
            colors::RESET
        );
        println!(
            "{}│{}  ⚡ PERFORMANCE                                                  {}│{}",
            colors::GREEN,
            colors::BOLD,
            colors::RESET,
            colors::GREEN
        );
        println!(
            "{}├─────────────────────────────────────────────────────────────────┤{}",
            colors::GREEN,
            colors::RESET
        );
        println!(
            "{}│{}  Total Requests:    {:>12}                              {}│{}",
            colors::GREEN,
            colors::RESET,
            stats.total_requests,
            colors::GREEN,
            colors::RESET
        );
        println!(
            "{}│{}  Requests/min:      {:>12.2}                              {}│{}",
            colors::GREEN,
            colors::RESET,
            stats.requests_per_minute,
            colors::GREEN,
            colors::RESET
        );
        println!(
            "{}│{}  Avg Latency:       {:>10.0} ms                              {}│{}",
            colors::GREEN,
            colors::RESET,
            stats.avg_latency_ms,
            colors::GREEN,
            colors::RESET
        );
        println!(
            "{}│{}  Errors:            {:>12}                              {}│{}",
            colors::GREEN,
            colors::RESET,
            stats.total_errors,
            colors::GREEN,
            colors::RESET
        );
        println!(
            "{}└─────────────────────────────────────────────────────────────────┘{}",
            colors::GREEN,
            colors::RESET
        );
        println!();

        // Cache efficiency bar
        println!("  {}Cache Efficiency:{}", colors::BOLD, colors::RESET);
        print!("  ");
        print!("{}", self.progress_bar(stats.cache_hit_rate, 1.0, 40));
        println!("  {:.1}%", stats.cache_hit_rate * 100.0);
        println!();

        // Model breakdown
        if !stats.by_model.is_empty() {
            println!(
                "{}┌─────────────────────────────────────────────────────────────────┐{}",
                colors::YELLOW,
                colors::RESET
            );
            println!(
                "{}│{}  🤖 MODEL BREAKDOWN                                             {}│{}",
                colors::YELLOW,
                colors::BOLD,
                colors::RESET,
                colors::YELLOW
            );
            println!(
                "{}├─────────────────────────────────────────────────────────────────┤{}",
                colors::YELLOW,
                colors::RESET
            );

            let mut models: Vec<_> = stats.by_model.iter().collect();
            models.sort_by(|a, b| b.1.total_cost.partial_cmp(&a.1.total_cost).unwrap());

            for (model, model_stats) in models.iter().take(5) {
                let avg_latency = if model_stats.requests > 0 {
                    model_stats.total_latency_ms / model_stats.requests as f64
                } else {
                    0.0
                };
                println!(
                    "{}│{}  {:<25} {} │ {:>6} reqs │ {:>6.0}ms    {}│{}",
                    colors::YELLOW,
                    colors::RESET,
                    if model.len() > 25 {
                        &model[..25]
                    } else {
                        model
                    },
                    Self::format_currency(model_stats.total_cost),
                    model_stats.requests,
                    avg_latency,
                    colors::YELLOW,
                    colors::RESET
                );
            }

            println!(
                "{}└─────────────────────────────────────────────────────────────────┘{}",
                colors::YELLOW,
                colors::RESET
            );
        }

        println!();
        println!(
            "{}  Press Ctrl+C to exit dashboard{}",
            colors::DIM,
            colors::RESET
        );
    }

    /// Print simple summary (non-dashboard version)
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

    /// Export stats to JSON
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&self.stats()).unwrap_or_default()
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
            records: self.records.clone(),
            pricing: self.pricing.clone(),
            budget_alerts: Arc::new(RwLock::new(Vec::new())), // Don't clone callbacks
            max_records: self.max_records,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_calculation() {
        let tracker = CostTracker::new();

        // Test GPT-4o pricing
        let cost = tracker.calculate_cost("gpt-4o", 1000, 500, 0);
        // Input: 1000/1000 * 0.0025 = 0.0025
        // Output: 500/1000 * 0.01 = 0.005
        // Total: 0.0075
        assert!((cost - 0.0075).abs() < 0.0001);
    }

    #[test]
    fn test_cache_savings() {
        let tracker = CostTracker::new();

        tracker.record_request_detailed(
            "claude-3-5-sonnet-20241022",
            1000,
            500,
            800, // 800 cached tokens
            100.0,
            Some("test-agent"),
            true,
        );

        let stats = tracker.stats();
        assert!(stats.cache_savings > 0.0);
        assert!(stats.cache_hit_rate > 0.0);
    }

    #[test]
    fn test_budget_alert() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let tracker = CostTracker::new();
        let triggered = Arc::new(AtomicBool::new(false));
        let triggered_clone = triggered.clone();

        tracker.add_budget_alert(
            0.01,
            Some(Arc::new(move |_| {
                triggered_clone.store(true, Ordering::SeqCst);
            })),
        );

        // Record expensive request
        tracker.record_request_detailed("gpt-4-turbo", 10000, 5000, 0, 100.0, None, true);

        assert!(triggered.load(Ordering::SeqCst));
    }
}
