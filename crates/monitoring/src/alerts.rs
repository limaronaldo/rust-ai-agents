//! Alert system for monitoring thresholds

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Cost threshold in USD
    pub cost_threshold: Option<f64>,

    /// Token threshold
    pub token_threshold: Option<u64>,

    /// Latency threshold in ms
    pub latency_threshold: Option<f64>,

    /// Error rate threshold (0.0 - 1.0)
    pub error_rate_threshold: Option<f64>,
}

impl AlertConfig {
    pub fn new() -> Self {
        Self {
            cost_threshold: None,
            token_threshold: None,
            latency_threshold: None,
            error_rate_threshold: None,
        }
    }

    pub fn with_cost_threshold(mut self, threshold: f64) -> Self {
        self.cost_threshold = Some(threshold);
        self
    }

    pub fn with_token_threshold(mut self, threshold: u64) -> Self {
        self.token_threshold = Some(threshold);
        self
    }

    pub fn with_latency_threshold(mut self, threshold: f64) -> Self {
        self.latency_threshold = Some(threshold);
        self
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub level: AlertLevel,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Alert manager
pub struct AlertManager {
    config: AlertConfig,
    alerts: Arc<RwLock<Vec<Alert>>>,
}

impl AlertManager {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            alerts: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Check metrics and trigger alerts if needed
    pub fn check_metrics(&self, total_cost: f64, total_tokens: u64, avg_latency: f64) {
        // Check cost threshold
        if let Some(threshold) = self.config.cost_threshold {
            if total_cost >= threshold {
                self.trigger_alert(
                    AlertLevel::Warning,
                    format!(
                        "Cost threshold exceeded: ${:.6} >= ${:.6}",
                        total_cost, threshold
                    ),
                );
            }
        }

        // Check token threshold
        if let Some(threshold) = self.config.token_threshold {
            if total_tokens >= threshold {
                self.trigger_alert(
                    AlertLevel::Warning,
                    format!(
                        "Token threshold exceeded: {} >= {}",
                        total_tokens, threshold
                    ),
                );
            }
        }

        // Check latency threshold
        if let Some(threshold) = self.config.latency_threshold {
            if avg_latency >= threshold {
                self.trigger_alert(
                    AlertLevel::Warning,
                    format!(
                        "Latency threshold exceeded: {:.0}ms >= {:.0}ms",
                        avg_latency, threshold
                    ),
                );
            }
        }
    }

    /// Trigger an alert
    pub fn trigger_alert(&self, level: AlertLevel, message: String) {
        let alert = Alert {
            level,
            message: message.clone(),
            timestamp: chrono::Utc::now(),
        };

        let mut alerts = self.alerts.write();
        alerts.push(alert);

        // Print alert to console
        let level_str = match level {
            AlertLevel::Info => "ℹ️  INFO",
            AlertLevel::Warning => "⚠️  WARNING",
            AlertLevel::Error => "❌ ERROR",
            AlertLevel::Critical => "🚨 CRITICAL",
        };

        eprintln!("[{}] {}", level_str, message);
    }

    /// Get all alerts
    pub fn get_alerts(&self) -> Vec<Alert> {
        self.alerts.read().clone()
    }

    /// Clear all alerts
    pub fn clear_alerts(&self) {
        self.alerts.write().clear();
    }

    /// Get alert count by level
    pub fn count_by_level(&self, level: AlertLevel) -> usize {
        self.alerts
            .read()
            .iter()
            .filter(|a| a.level == level)
            .count()
    }
}
