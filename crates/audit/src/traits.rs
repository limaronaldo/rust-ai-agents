//! Audit logger traits and configuration.
//!
//! This module defines the core `AuditLogger` trait that all backends implement,
//! along with configuration types and utilities.

use crate::error::AuditError;
use crate::types::{AuditEvent, AuditLevel};
use async_trait::async_trait;
use std::sync::Arc;

/// Configuration for audit logging.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Minimum level to log (events below this are filtered)
    pub min_level: AuditLevel,
    /// Whether to include debug information
    pub include_debug: bool,
    /// Maximum size for parameter/result payloads (bytes)
    pub max_payload_size: usize,
    /// Whether to redact sensitive fields
    pub redact_sensitive: bool,
    /// Fields to redact
    pub redact_fields: Vec<String>,
    /// Whether audit logging is enabled
    pub enabled: bool,
    /// Buffer size for async logging
    pub buffer_size: usize,
    /// Flush interval in seconds
    pub flush_interval_secs: u64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            min_level: AuditLevel::Info,
            include_debug: false,
            max_payload_size: 10 * 1024, // 10KB
            redact_sensitive: true,
            redact_fields: vec![
                "password".to_string(),
                "secret".to_string(),
                "token".to_string(),
                "api_key".to_string(),
                "apiKey".to_string(),
                "authorization".to_string(),
            ],
            enabled: true,
            buffer_size: 1000,
            flush_interval_secs: 5,
        }
    }
}

impl AuditConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a development configuration with debug enabled.
    pub fn development() -> Self {
        Self {
            min_level: AuditLevel::Debug,
            include_debug: true,
            redact_sensitive: false,
            ..Default::default()
        }
    }

    /// Create a production configuration with stricter settings.
    pub fn production() -> Self {
        Self {
            min_level: AuditLevel::Info,
            include_debug: false,
            redact_sensitive: true,
            max_payload_size: 5 * 1024, // 5KB in production
            ..Default::default()
        }
    }

    /// Set the minimum log level.
    pub fn with_min_level(mut self, level: AuditLevel) -> Self {
        self.min_level = level;
        self
    }

    /// Enable or disable debug information.
    pub fn with_debug(mut self, include: bool) -> Self {
        self.include_debug = include;
        self
    }

    /// Set maximum payload size.
    pub fn with_max_payload_size(mut self, size: usize) -> Self {
        self.max_payload_size = size;
        self
    }

    /// Enable or disable sensitive field redaction.
    pub fn with_redaction(mut self, enabled: bool) -> Self {
        self.redact_sensitive = enabled;
        self
    }

    /// Add fields to redact.
    pub fn with_redact_fields(mut self, fields: Vec<String>) -> Self {
        self.redact_fields.extend(fields);
        self
    }

    /// Enable or disable audit logging.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set buffer size.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Check if an event should be logged based on level.
    pub fn should_log(&self, level: AuditLevel) -> bool {
        self.enabled && level >= self.min_level
    }
}

/// The core audit logger trait.
///
/// All audit backends implement this trait to provide consistent logging behavior.
#[async_trait]
pub trait AuditLogger: Send + Sync {
    /// Log an audit event.
    async fn log(&self, event: AuditEvent) -> Result<(), AuditError>;

    /// Log multiple events in a batch.
    async fn log_batch(&self, events: Vec<AuditEvent>) -> Result<(), AuditError> {
        for event in events {
            self.log(event).await?;
        }
        Ok(())
    }

    /// Flush any buffered events.
    async fn flush(&self) -> Result<(), AuditError>;

    /// Get the logger name for identification.
    fn name(&self) -> &str;

    /// Check if the logger is healthy.
    async fn health_check(&self) -> Result<(), AuditError> {
        Ok(())
    }

    /// Get statistics about logged events.
    async fn stats(&self) -> AuditStats {
        AuditStats::default()
    }
}

/// Statistics about audit logging.
#[derive(Debug, Clone, Default)]
pub struct AuditStats {
    /// Total events logged
    pub total_events: u64,
    /// Events by level
    pub events_by_level: std::collections::HashMap<AuditLevel, u64>,
    /// Failed log attempts
    pub failed_events: u64,
    /// Bytes written
    pub bytes_written: u64,
    /// Last event timestamp
    pub last_event_time: Option<chrono::DateTime<chrono::Utc>>,
}

/// A composite logger that writes to multiple backends.
pub struct CompositeLogger {
    loggers: Vec<Arc<dyn AuditLogger>>,
    config: AuditConfig,
}

impl CompositeLogger {
    /// Create a new composite logger.
    pub fn new(config: AuditConfig) -> Self {
        Self {
            loggers: Vec::new(),
            config,
        }
    }

    /// Add a logger backend.
    pub fn add_logger(mut self, logger: Arc<dyn AuditLogger>) -> Self {
        self.loggers.push(logger);
        self
    }

    /// Add a logger backend by reference.
    pub fn with_logger(&mut self, logger: Arc<dyn AuditLogger>) {
        self.loggers.push(logger);
    }
}

#[async_trait]
impl AuditLogger for CompositeLogger {
    async fn log(&self, event: AuditEvent) -> Result<(), AuditError> {
        if !self.config.should_log(event.level) {
            return Ok(());
        }

        let mut errors = Vec::new();
        for logger in &self.loggers {
            if let Err(e) = logger.log(event.clone()).await {
                errors.push(format!("{}: {}", logger.name(), e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AuditError::Multiple(errors))
        }
    }

    async fn flush(&self) -> Result<(), AuditError> {
        let mut errors = Vec::new();
        for logger in &self.loggers {
            if let Err(e) = logger.flush().await {
                errors.push(format!("{}: {}", logger.name(), e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AuditError::Multiple(errors))
        }
    }

    fn name(&self) -> &str {
        "composite"
    }

    async fn health_check(&self) -> Result<(), AuditError> {
        for logger in &self.loggers {
            logger.health_check().await?;
        }
        Ok(())
    }
}

/// A no-op logger for testing or when audit is disabled.
pub struct NoOpLogger;

#[async_trait]
impl AuditLogger for NoOpLogger {
    async fn log(&self, _event: AuditEvent) -> Result<(), AuditError> {
        Ok(())
    }

    async fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }

    fn name(&self) -> &str {
        "noop"
    }
}

/// A logger that collects events in memory for testing.
#[derive(Default)]
pub struct MemoryLogger {
    events: tokio::sync::RwLock<Vec<AuditEvent>>,
}

impl MemoryLogger {
    /// Create a new memory logger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all logged events.
    pub async fn events(&self) -> Vec<AuditEvent> {
        self.events.read().await.clone()
    }

    /// Get event count.
    pub async fn count(&self) -> usize {
        self.events.read().await.len()
    }

    /// Clear all events.
    pub async fn clear(&self) {
        self.events.write().await.clear();
    }
}

#[async_trait]
impl AuditLogger for MemoryLogger {
    async fn log(&self, event: AuditEvent) -> Result<(), AuditError> {
        self.events.write().await.push(event);
        Ok(())
    }

    async fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }

    fn name(&self) -> &str {
        "memory"
    }

    async fn stats(&self) -> AuditStats {
        let events = self.events.read().await;
        let mut stats = AuditStats {
            total_events: events.len() as u64,
            ..Default::default()
        };

        for event in events.iter() {
            *stats.events_by_level.entry(event.level).or_insert(0) += 1;
        }

        if let Some(last) = events.last() {
            stats.last_event_time = Some(last.timestamp);
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EventKind;

    #[test]
    fn test_audit_config_defaults() {
        let config = AuditConfig::default();
        assert_eq!(config.min_level, AuditLevel::Info);
        assert!(config.enabled);
        assert!(config.redact_sensitive);
    }

    #[test]
    fn test_audit_config_should_log() {
        let config = AuditConfig::new().with_min_level(AuditLevel::Warn);

        assert!(!config.should_log(AuditLevel::Debug));
        assert!(!config.should_log(AuditLevel::Info));
        assert!(config.should_log(AuditLevel::Warn));
        assert!(config.should_log(AuditLevel::Error));
        assert!(config.should_log(AuditLevel::Critical));
    }

    #[test]
    fn test_audit_config_disabled() {
        let config = AuditConfig::new().enabled(false);
        assert!(!config.should_log(AuditLevel::Critical));
    }

    #[tokio::test]
    async fn test_memory_logger() {
        let logger = MemoryLogger::new();

        let event = AuditEvent::info(EventKind::Custom {
            name: "test".to_string(),
            payload: serde_json::json!({}),
        });

        logger.log(event).await.unwrap();
        assert_eq!(logger.count().await, 1);

        let events = logger.events().await;
        assert_eq!(events.len(), 1);
    }

    #[tokio::test]
    async fn test_noop_logger() {
        let logger = NoOpLogger;
        let event = AuditEvent::info(EventKind::Custom {
            name: "test".to_string(),
            payload: serde_json::json!({}),
        });

        assert!(logger.log(event).await.is_ok());
        assert!(logger.flush().await.is_ok());
    }

    #[tokio::test]
    async fn test_composite_logger() {
        let memory1 = Arc::new(MemoryLogger::new());
        let memory2 = Arc::new(MemoryLogger::new());

        let composite = CompositeLogger::new(AuditConfig::default())
            .add_logger(memory1.clone())
            .add_logger(memory2.clone());

        let event = AuditEvent::info(EventKind::Custom {
            name: "test".to_string(),
            payload: serde_json::json!({}),
        });

        composite.log(event).await.unwrap();

        assert_eq!(memory1.count().await, 1);
        assert_eq!(memory2.count().await, 1);
    }
}
