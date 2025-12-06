//! Alert manager for Slack, Discord, and webhook notifications

use parking_lot::RwLock;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl AlertSeverity {
    pub fn emoji(&self) -> &'static str {
        match self {
            AlertSeverity::Info => "ℹ️",
            AlertSeverity::Warning => "⚠️",
            AlertSeverity::Error => "❌",
            AlertSeverity::Critical => "🚨",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            AlertSeverity::Info => "#36a64f",
            AlertSeverity::Warning => "#ffcc00",
            AlertSeverity::Error => "#ff6600",
            AlertSeverity::Critical => "#ff0000",
        }
    }

    pub fn discord_color(&self) -> u32 {
        match self {
            AlertSeverity::Info => 0x36a64f,
            AlertSeverity::Warning => 0xffcc00,
            AlertSeverity::Error => 0xff6600,
            AlertSeverity::Critical => 0xff0000,
        }
    }
}

/// Alert message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub severity: AlertSeverity,
    pub title: String,
    pub message: String,
    pub source: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

impl Alert {
    pub fn new(
        severity: AlertSeverity,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            title: title.into(),
            message: message.into(),
            source: "rust-ai-agents".to_string(),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn info(title: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(AlertSeverity::Info, title, message)
    }

    pub fn warning(title: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(AlertSeverity::Warning, title, message)
    }

    pub fn error(title: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(AlertSeverity::Error, title, message)
    }

    pub fn critical(title: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(AlertSeverity::Critical, title, message)
    }
}

/// Slack webhook payload
#[derive(Debug, Serialize)]
struct SlackPayload {
    text: String,
    attachments: Vec<SlackAttachment>,
}

#[derive(Debug, Serialize)]
struct SlackAttachment {
    color: String,
    title: String,
    text: String,
    fields: Vec<SlackField>,
    footer: String,
    ts: i64,
}

#[derive(Debug, Serialize)]
struct SlackField {
    title: String,
    value: String,
    short: bool,
}

/// Discord webhook payload
#[derive(Debug, Serialize)]
struct DiscordPayload {
    content: Option<String>,
    embeds: Vec<DiscordEmbed>,
}

#[derive(Debug, Serialize)]
struct DiscordEmbed {
    title: String,
    description: String,
    color: u32,
    fields: Vec<DiscordField>,
    footer: DiscordFooter,
    timestamp: String,
}

#[derive(Debug, Serialize)]
struct DiscordField {
    name: String,
    value: String,
    inline: bool,
}

#[derive(Debug, Serialize)]
struct DiscordFooter {
    text: String,
}

/// Generic webhook payload
#[derive(Debug, Serialize)]
struct WebhookPayload {
    alert: Alert,
}

/// Alert channel configuration
#[derive(Debug, Clone)]
pub enum AlertChannel {
    Slack {
        webhook_url: String,
        channel: Option<String>,
    },
    Discord {
        webhook_url: String,
    },
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    Console,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum alerts per window
    pub max_alerts: u32,
    /// Time window duration
    pub window: Duration,
    /// Cooldown after hitting limit
    pub cooldown: Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_alerts: 10,
            window: Duration::from_secs(60),
            cooldown: Duration::from_secs(300),
        }
    }
}

/// Rate limiter state
struct RateLimiter {
    config: RateLimitConfig,
    alert_times: Vec<Instant>,
    cooldown_until: Option<Instant>,
}

impl RateLimiter {
    fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            alert_times: Vec::new(),
            cooldown_until: None,
        }
    }

    fn can_send(&mut self) -> bool {
        let now = Instant::now();

        // Check cooldown
        if let Some(until) = self.cooldown_until {
            if now < until {
                return false;
            }
            self.cooldown_until = None;
        }

        // Clean old entries
        self.alert_times
            .retain(|t| now.duration_since(*t) < self.config.window);

        // Check limit
        if self.alert_times.len() >= self.config.max_alerts as usize {
            self.cooldown_until = Some(now + self.config.cooldown);
            return false;
        }

        self.alert_times.push(now);
        true
    }
}

/// Alert manager for sending notifications
pub struct AlertManager {
    client: Client,
    channels: Arc<RwLock<Vec<(AlertChannel, Option<AlertSeverity>)>>>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    enabled: Arc<RwLock<bool>>,
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            channels: Arc::new(RwLock::new(Vec::new())),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(RateLimitConfig::default()))),
            enabled: Arc::new(RwLock::new(true)),
        }
    }

    /// Configure rate limiting
    pub fn with_rate_limit(self, config: RateLimitConfig) -> Self {
        *self.rate_limiter.write() = RateLimiter::new(config);
        self
    }

    /// Add a Slack channel
    pub fn add_slack(&self, webhook_url: impl Into<String>, min_severity: Option<AlertSeverity>) {
        self.channels.write().push((
            AlertChannel::Slack {
                webhook_url: webhook_url.into(),
                channel: None,
            },
            min_severity,
        ));
    }

    /// Add a Discord channel
    pub fn add_discord(&self, webhook_url: impl Into<String>, min_severity: Option<AlertSeverity>) {
        self.channels.write().push((
            AlertChannel::Discord {
                webhook_url: webhook_url.into(),
            },
            min_severity,
        ));
    }

    /// Add a generic webhook
    pub fn add_webhook(
        &self,
        url: impl Into<String>,
        headers: HashMap<String, String>,
        min_severity: Option<AlertSeverity>,
    ) {
        self.channels.write().push((
            AlertChannel::Webhook {
                url: url.into(),
                headers,
            },
            min_severity,
        ));
    }

    /// Add console output
    pub fn add_console(&self, min_severity: Option<AlertSeverity>) {
        self.channels
            .write()
            .push((AlertChannel::Console, min_severity));
    }

    /// Enable or disable alerts
    pub fn set_enabled(&self, enabled: bool) {
        *self.enabled.write() = enabled;
    }

    /// Check if severity meets minimum threshold
    fn meets_threshold(severity: AlertSeverity, min: Option<AlertSeverity>) -> bool {
        match min {
            None => true,
            Some(min_sev) => {
                let severity_ord = match severity {
                    AlertSeverity::Info => 0,
                    AlertSeverity::Warning => 1,
                    AlertSeverity::Error => 2,
                    AlertSeverity::Critical => 3,
                };
                let min_ord = match min_sev {
                    AlertSeverity::Info => 0,
                    AlertSeverity::Warning => 1,
                    AlertSeverity::Error => 2,
                    AlertSeverity::Critical => 3,
                };
                severity_ord >= min_ord
            }
        }
    }

    /// Send an alert to all configured channels
    pub async fn send(&self, alert: Alert) -> Result<(), AlertError> {
        if !*self.enabled.read() {
            return Ok(());
        }

        if !self.rate_limiter.write().can_send() {
            return Err(AlertError::RateLimited);
        }

        let channels = self.channels.read().clone();
        let mut errors = Vec::new();

        for (channel, min_severity) in channels {
            if !Self::meets_threshold(alert.severity, min_severity) {
                continue;
            }

            let result = match channel {
                AlertChannel::Slack { webhook_url, .. } => {
                    self.send_slack(&webhook_url, &alert).await
                }
                AlertChannel::Discord { webhook_url } => {
                    self.send_discord(&webhook_url, &alert).await
                }
                AlertChannel::Webhook { url, headers } => {
                    self.send_webhook(&url, &headers, &alert).await
                }
                AlertChannel::Console => {
                    self.send_console(&alert);
                    Ok(())
                }
            };

            if let Err(e) = result {
                errors.push(e);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AlertError::PartialFailure(errors))
        }
    }

    async fn send_slack(&self, webhook_url: &str, alert: &Alert) -> Result<(), AlertError> {
        let mut fields: Vec<SlackField> = alert
            .metadata
            .iter()
            .map(|(k, v)| SlackField {
                title: k.clone(),
                value: v.clone(),
                short: true,
            })
            .collect();

        fields.insert(
            0,
            SlackField {
                title: "Severity".to_string(),
                value: format!("{} {:?}", alert.severity.emoji(), alert.severity),
                short: true,
            },
        );

        let payload = SlackPayload {
            text: format!("{} {}", alert.severity.emoji(), alert.title),
            attachments: vec![SlackAttachment {
                color: alert.severity.color().to_string(),
                title: alert.title.clone(),
                text: alert.message.clone(),
                fields,
                footer: alert.source.clone(),
                ts: alert.timestamp.timestamp(),
            }],
        };

        self.client
            .post(webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| AlertError::SendFailed(format!("Slack: {}", e)))?
            .error_for_status()
            .map_err(|e| AlertError::SendFailed(format!("Slack: {}", e)))?;

        Ok(())
    }

    async fn send_discord(&self, webhook_url: &str, alert: &Alert) -> Result<(), AlertError> {
        let fields: Vec<DiscordField> = alert
            .metadata
            .iter()
            .map(|(k, v)| DiscordField {
                name: k.clone(),
                value: v.clone(),
                inline: true,
            })
            .collect();

        let payload = DiscordPayload {
            content: None,
            embeds: vec![DiscordEmbed {
                title: format!("{} {}", alert.severity.emoji(), alert.title),
                description: alert.message.clone(),
                color: alert.severity.discord_color(),
                fields,
                footer: DiscordFooter {
                    text: alert.source.clone(),
                },
                timestamp: alert.timestamp.to_rfc3339(),
            }],
        };

        self.client
            .post(webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| AlertError::SendFailed(format!("Discord: {}", e)))?
            .error_for_status()
            .map_err(|e| AlertError::SendFailed(format!("Discord: {}", e)))?;

        Ok(())
    }

    async fn send_webhook(
        &self,
        url: &str,
        headers: &HashMap<String, String>,
        alert: &Alert,
    ) -> Result<(), AlertError> {
        let payload = WebhookPayload {
            alert: alert.clone(),
        };

        let mut request = self.client.post(url).json(&payload);

        for (key, value) in headers {
            request = request.header(key.as_str(), value.as_str());
        }

        request
            .send()
            .await
            .map_err(|e| AlertError::SendFailed(format!("Webhook: {}", e)))?
            .error_for_status()
            .map_err(|e| AlertError::SendFailed(format!("Webhook: {}", e)))?;

        Ok(())
    }

    fn send_console(&self, alert: &Alert) {
        let color = match alert.severity {
            AlertSeverity::Info => "\x1b[36m",     // Cyan
            AlertSeverity::Warning => "\x1b[33m",  // Yellow
            AlertSeverity::Error => "\x1b[31m",    // Red
            AlertSeverity::Critical => "\x1b[35m", // Magenta
        };
        let reset = "\x1b[0m";

        println!(
            "\n{}[{}] {} {}{}",
            color,
            alert.timestamp.format("%Y-%m-%d %H:%M:%S"),
            alert.severity.emoji(),
            alert.title,
            reset
        );
        println!("  {}", alert.message);
        if !alert.metadata.is_empty() {
            for (k, v) in &alert.metadata {
                println!("  {}: {}", k, v);
            }
        }
        println!();
    }

    /// Send convenience methods
    pub async fn info(
        &self,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), AlertError> {
        self.send(Alert::info(title, message)).await
    }

    pub async fn warning(
        &self,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), AlertError> {
        self.send(Alert::warning(title, message)).await
    }

    pub async fn error(
        &self,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), AlertError> {
        self.send(Alert::error(title, message)).await
    }

    pub async fn critical(
        &self,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), AlertError> {
        self.send(Alert::critical(title, message)).await
    }

    /// Budget alert helper
    pub async fn budget_exceeded(&self, current: f64, threshold: f64) -> Result<(), AlertError> {
        self.send(
            Alert::warning(
                "Budget Threshold Exceeded",
                format!(
                    "Current spend ${:.4} has exceeded threshold ${:.4}",
                    current, threshold
                ),
            )
            .with_metadata("current_spend", format!("${:.4}", current))
            .with_metadata("threshold", format!("${:.4}", threshold))
            .with_metadata("overage", format!("${:.4}", current - threshold)),
        )
        .await
    }

    /// Circuit breaker alert helper
    pub async fn circuit_opened(
        &self,
        tool_name: &str,
        failure_count: u32,
    ) -> Result<(), AlertError> {
        self.send(
            Alert::error(
                "Circuit Breaker Opened",
                format!(
                    "Tool '{}' circuit breaker opened after {} failures",
                    tool_name, failure_count
                ),
            )
            .with_metadata("tool", tool_name.to_string())
            .with_metadata("failures", failure_count.to_string()),
        )
        .await
    }

    /// Error rate alert helper
    pub async fn high_error_rate(
        &self,
        rate: f64,
        tool_name: Option<&str>,
    ) -> Result<(), AlertError> {
        let title = if let Some(name) = tool_name {
            format!("High Error Rate: {}", name)
        } else {
            "High Error Rate".to_string()
        };

        self.send(
            Alert::warning(title, format!("Error rate at {:.1}%", rate * 100.0))
                .with_metadata("error_rate", format!("{:.1}%", rate * 100.0)),
        )
        .await
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for AlertManager {
    fn clone(&self) -> Self {
        Self {
            client: Client::new(),
            channels: self.channels.clone(),
            rate_limiter: self.rate_limiter.clone(),
            enabled: self.enabled.clone(),
        }
    }
}

/// Alert errors
#[derive(Debug)]
pub enum AlertError {
    SendFailed(String),
    RateLimited,
    PartialFailure(Vec<AlertError>),
}

impl std::fmt::Display for AlertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertError::SendFailed(msg) => write!(f, "Failed to send alert: {}", msg),
            AlertError::RateLimited => write!(f, "Alert rate limited"),
            AlertError::PartialFailure(errors) => {
                write!(f, "Partial failure: {} channels failed", errors.len())
            }
        }
    }
}

impl std::error::Error for AlertError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let alert = Alert::critical("Test Alert", "This is a test")
            .with_source("test-source")
            .with_metadata("key", "value");

        assert_eq!(alert.severity, AlertSeverity::Critical);
        assert_eq!(alert.title, "Test Alert");
        assert_eq!(alert.source, "test-source");
        assert_eq!(alert.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(AlertManager::meets_threshold(
            AlertSeverity::Critical,
            Some(AlertSeverity::Warning)
        ));
        assert!(AlertManager::meets_threshold(
            AlertSeverity::Error,
            Some(AlertSeverity::Warning)
        ));
        assert!(AlertManager::meets_threshold(
            AlertSeverity::Warning,
            Some(AlertSeverity::Warning)
        ));
        assert!(!AlertManager::meets_threshold(
            AlertSeverity::Info,
            Some(AlertSeverity::Warning)
        ));
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(RateLimitConfig {
            max_alerts: 2,
            window: Duration::from_secs(60),
            cooldown: Duration::from_millis(100),
        });

        assert!(limiter.can_send());
        assert!(limiter.can_send());
        assert!(!limiter.can_send()); // Rate limited

        std::thread::sleep(Duration::from_millis(150));
        assert!(limiter.can_send()); // Cooldown expired
    }
}
