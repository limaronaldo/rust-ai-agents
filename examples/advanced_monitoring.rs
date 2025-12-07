//! Advanced Monitoring Example
//!
//! This example demonstrates the advanced monitoring capabilities:
//! - Cost tracking with real-time dashboard
//! - Circuit breaker pattern for tool resilience
//! - Alert manager for Slack/Discord notifications
//!
//! Run with: cargo run --example advanced_monitoring

use async_trait::async_trait;
use rust_ai_agents_core::tool::{ExecutionContext, Tool, ToolSchema};
use rust_ai_agents_core::types::AgentId;
use rust_ai_agents_monitoring::{
    alerts::{Alert, AlertManager, AlertSeverity, RateLimitConfig},
    cost_tracker::{CostTracker, ModelPricing},
};
use rust_ai_agents_tools::registry::{CircuitBreakerConfig, EnhancedToolRegistry, RetryConfig};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Example tool that simulates API calls with configurable failure rate
struct UnreliableApiTool {
    call_count: AtomicU32,
    fail_every_n: u32,
}

impl UnreliableApiTool {
    fn new(fail_every_n: u32) -> Self {
        Self {
            call_count: AtomicU32::new(0),
            fail_every_n,
        }
    }
}

#[async_trait]
impl Tool for UnreliableApiTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("unreliable_api", "An API that sometimes fails").with_parameters(
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to send"
                    }
                },
                "required": ["query"]
            }),
        )
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, rust_ai_agents_core::errors::ToolError> {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);

        // Simulate some processing time
        tokio::time::sleep(Duration::from_millis(50 + (count as u64 % 100))).await;

        // Fail every N calls
        if count > 0 && count.is_multiple_of(self.fail_every_n) {
            return Err(rust_ai_agents_core::errors::ToolError::ExecutionFailed(
                "Simulated API failure".to_string(),
            ));
        }

        let query = arguments
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        Ok(serde_json::json!({
            "result": format!("Processed query: {}", query),
            "call_number": count
        }))
    }
}

/// Example tool that always succeeds
struct ReliableCalculator;

#[async_trait]
impl Tool for ReliableCalculator {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("calculator", "A reliable calculator").with_parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" },
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                }
            },
            "required": ["a", "b", "operation"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, rust_ai_agents_core::errors::ToolError> {
        let a = arguments.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b = arguments.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let op = arguments
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("add");

        let result = match op {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Err(rust_ai_agents_core::errors::ToolError::ExecutionFailed(
                        "Division by zero".to_string(),
                    ));
                }
                a / b
            }
            _ => {
                return Err(rust_ai_agents_core::errors::ToolError::InvalidArguments(
                    "Unknown operation".to_string(),
                ))
            }
        };

        Ok(serde_json::json!({ "result": result }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Rust AI Agents - Advanced Monitoring Example\n");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // 1. Setup Cost Tracker with Custom Pricing
    // =========================================================================
    println!("📊 Setting up Cost Tracker...\n");

    let mut custom_pricing = HashMap::new();
    custom_pricing.insert("gpt-4o".to_string(), ModelPricing::new(0.0025, 0.01));
    custom_pricing.insert(
        "claude-3-5-sonnet".to_string(),
        ModelPricing::new(0.003, 0.015).with_cache_price(0.0003),
    );

    let cost_tracker = CostTracker::new()
        .with_pricing(custom_pricing)
        .with_max_records(1000);

    // Add budget alert
    cost_tracker.add_budget_alert(
        0.10,
        Some(Arc::new(|cost| {
            println!("⚠️  Budget alert! Current cost: ${:.4}", cost);
        })),
    );

    // =========================================================================
    // 2. Setup Alert Manager
    // =========================================================================
    println!("🔔 Setting up Alert Manager...\n");

    let alert_manager = AlertManager::new().with_rate_limit(RateLimitConfig {
        max_alerts: 5,
        window: Duration::from_secs(60),
        cooldown: Duration::from_secs(30),
    });

    // Add console output for all alerts
    alert_manager.add_console(Some(AlertSeverity::Info));

    // In production, you would add Slack/Discord:
    // alert_manager.add_slack("https://hooks.slack.com/...", Some(AlertSeverity::Warning));
    // alert_manager.add_discord("https://discord.com/api/webhooks/...", Some(AlertSeverity::Error));

    // =========================================================================
    // 3. Setup Enhanced Tool Registry with Circuit Breaker
    // =========================================================================
    println!("🔧 Setting up Enhanced Tool Registry with Circuit Breaker...\n");

    let registry = EnhancedToolRegistry::new()
        .with_circuit_breaker(CircuitBreakerConfig {
            failure_threshold: 3,
            reset_timeout: Duration::from_secs(5),
            success_threshold: 2,
        })
        .with_retry(RetryConfig {
            max_retries: 2,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
            multiplier: 2.0,
            jitter: true,
        })
        .with_timeout(Duration::from_secs(10));

    // Register tools
    registry.register(Arc::new(UnreliableApiTool::new(4))); // Fails every 4th call
    registry.register(Arc::new(ReliableCalculator));

    // =========================================================================
    // 4. Simulate Workload
    // =========================================================================
    println!("🔄 Simulating API workload...\n");

    let ctx = ExecutionContext::new(AgentId::new("example-agent"));

    // Simulate multiple API calls
    for i in 0..15 {
        let result = registry
            .execute(
                "unreliable_api",
                &ctx,
                serde_json::json!({ "query": format!("request_{}", i) }),
            )
            .await;

        match &result {
            Ok(value) => {
                println!("  ✓ Request {}: {:?}", i, value);

                // Record successful request to cost tracker
                cost_tracker.record_request_detailed(
                    "gpt-4o",
                    500 + (i as u64 * 10),            // Input tokens
                    200 + (i as u64 * 5),             // Output tokens
                    if i % 3 == 0 { 300 } else { 0 }, // Some cached
                    50.0 + (i as f64 * 2.0),          // Latency
                    Some("example-agent"),
                    true,
                );
            }
            Err(e) => {
                println!("  ✗ Request {}: {}", i, e);

                // Check if circuit is open
                if let Some(state) = registry.get_circuit_state("unreliable_api") {
                    if state == rust_ai_agents_tools::registry::CircuitState::Open {
                        alert_manager.circuit_opened("unreliable_api", 3).await.ok();
                    }
                }
            }
        }

        // Small delay between requests
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // =========================================================================
    // 5. Calculator calls (always succeed)
    // =========================================================================
    println!("\n🔢 Running calculator operations...\n");

    let operations = vec![
        ("add", 10.0, 5.0),
        ("multiply", 7.0, 8.0),
        ("divide", 100.0, 4.0),
        ("subtract", 50.0, 23.0),
    ];

    for (op, a, b) in operations {
        let result = registry
            .execute(
                "calculator",
                &ctx,
                serde_json::json!({ "a": a, "b": b, "operation": op }),
            )
            .await?;

        println!("  {} {} {} = {:?}", a, op, b, result.get("result"));

        // Record to cost tracker (using a cheaper model for simple calculations)
        cost_tracker.record_request_detailed(
            "gpt-4o",
            100,
            50,
            80, // High cache rate for repeated calculations
            20.0,
            Some("calculator-agent"),
            true,
        );
    }

    // =========================================================================
    // 6. Display Results
    // =========================================================================
    println!("\n");

    // Print tool health report
    registry.print_health_report();

    // Print cost summary
    cost_tracker.print_summary();

    // Show detailed stats
    let stats = cost_tracker.stats();
    println!("📈 Detailed Statistics:");
    println!("  • Total Requests: {}", stats.total_requests);
    println!("  • Cache Hit Rate: {:.1}%", stats.cache_hit_rate * 100.0);
    println!("  • Total Cost: ${:.6}", stats.total_cost);
    println!("  • Cache Savings: ${:.6}", stats.cache_savings);
    println!("  • Requests/min: {:.2}", stats.requests_per_minute);

    // Model breakdown
    println!("\n📊 Cost by Model:");
    for (model, model_stats) in &stats.by_model {
        println!(
            "  • {}: ${:.6} ({} requests)",
            model, model_stats.total_cost, model_stats.requests
        );
    }

    // Agent breakdown
    println!("\n🤖 Cost by Agent:");
    for (agent, agent_stats) in &stats.by_agent {
        println!(
            "  • {}: ${:.6} ({} requests)",
            agent, agent_stats.total_cost, agent_stats.requests
        );
    }

    // =========================================================================
    // 7. Send Summary Alert
    // =========================================================================
    println!("\n");
    alert_manager
        .send(
            Alert::info(
                "Workload Complete",
                format!(
                    "Processed {} requests with ${:.4} total cost",
                    stats.total_requests, stats.total_cost
                ),
            )
            .with_metadata("cache_savings", format!("${:.4}", stats.cache_savings))
            .with_metadata(
                "error_rate",
                format!(
                    "{:.1}%",
                    (stats.total_errors as f64 / stats.total_requests as f64) * 100.0
                ),
            ),
        )
        .await
        .ok();

    // Export stats as JSON
    println!("\n📋 Exported Stats (JSON):");
    println!("{}", cost_tracker.export_json());

    println!("\n✅ Example complete!\n");

    Ok(())
}
