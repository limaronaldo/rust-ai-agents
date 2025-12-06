//! Real-time terminal dashboard

use crossterm::{
    cursor::MoveTo,
    execute,
    terminal::{Clear, ClearType},
};
use std::io::{stdout, Write};

use crate::cost_tracker::CostStats;

/// Terminal dashboard for real-time monitoring
pub struct Dashboard {
    enabled: bool,
}

impl Dashboard {
    pub fn new() -> Self {
        Self { enabled: true }
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Render dashboard with current stats
    pub fn render(&self, stats: &CostStats, agent_count: usize, messages_processed: u64) {
        if !self.enabled {
            return;
        }

        let mut stdout = stdout();

        // Clear and move to top
        let _ = execute!(stdout, Clear(ClearType::All), MoveTo(0, 0));

        // Header
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║           RUST AI AGENTS - LIVE DASHBOARD                ║");
        println!("╠═══════════════════════════════════════════════════════════╣");

        // System metrics
        println!(
            "║ 🤖 Agents Running:     {:>6}                           ║",
            agent_count
        );
        println!(
            "║ 📨 Messages Processed: {:>6}                           ║",
            messages_processed
        );
        println!("╠═══════════════════════════════════════════════════════════╣");

        // Cost metrics
        println!("║ 💰 COST METRICS                                           ║");
        println!("║ ───────────────────────────────────────────────────────── ║");
        println!(
            "║   Total Cost:         ${:>10.6}                     ║",
            stats.total_cost
        );
        println!(
            "║   Cache Savings:      ${:>10.6}                     ║",
            stats.cache_savings
        );
        println!(
            "║   Net Cost:           ${:>10.6}                     ║",
            stats.total_cost - stats.cache_savings
        );
        println!("╠═══════════════════════════════════════════════════════════╣");

        // Token metrics
        println!("║ 🎯 TOKEN METRICS                                          ║");
        println!("║ ───────────────────────────────────────────────────────── ║");
        println!(
            "║   Input Tokens:       {:>12}                         ║",
            format_number(stats.total_input_tokens)
        );
        println!(
            "║   Output Tokens:      {:>12}                         ║",
            format_number(stats.total_output_tokens)
        );
        println!(
            "║   Cached Tokens:      {:>12}                         ║",
            format_number(stats.cached_tokens)
        );
        println!("╠═══════════════════════════════════════════════════════════╣");

        // Performance metrics
        println!("║ ⚡ PERFORMANCE                                             ║");
        println!("║ ───────────────────────────────────────────────────────── ║");
        println!(
            "║   Requests:           {:>12}                         ║",
            format_number(stats.total_requests)
        );
        println!(
            "║   Cache Hit Rate:     {:>11.1}%                        ║",
            stats.cache_hit_rate * 100.0
        );
        println!(
            "║   Avg Latency:        {:>9.0} ms                        ║",
            stats.avg_latency_ms
        );

        // Cache efficiency bar
        print!("║   Cache Efficiency:   ");
        print_progress_bar(stats.cache_hit_rate, 30);
        println!(" ║");

        println!("╚═══════════════════════════════════════════════════════════╝");

        let _ = stdout.flush();
    }

    /// Render compact status line
    pub fn render_status_line(&self, stats: &CostStats) {
        if !self.enabled {
            return;
        }

        print!(
            "\r💰 ${:.6} | 🎯 {} tokens | ♻️ {:.1}% cache | ⚡ {:.0}ms | 📊 {} reqs",
            stats.total_cost,
            format_number(stats.total_input_tokens + stats.total_output_tokens),
            stats.cache_hit_rate * 100.0,
            stats.avg_latency_ms,
            format_number(stats.total_requests)
        );
        let _ = stdout().flush();
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Format large numbers with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(c);
        count += 1;
    }

    result.chars().rev().collect()
}

/// Print a progress bar
fn print_progress_bar(percentage: f64, width: usize) {
    let filled = (percentage * width as f64) as usize;
    let empty = width - filled;

    print!("[");
    for _ in 0..filled {
        print!("█");
    }
    for _ in 0..empty {
        print!("░");
    }
    print!("]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(123), "123");
    }
}
