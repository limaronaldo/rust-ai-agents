//! Plain text file logger backend.
//!
//! Simple file-based logging with human-readable output format.

use crate::error::AuditError;
use crate::traits::{AuditConfig, AuditLogger, AuditStats};
use crate::types::AuditEvent;
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

/// A simple file-based audit logger.
///
/// Writes audit events as human-readable text lines to a file.
pub struct FileLogger {
    path: PathBuf,
    file: Mutex<Option<File>>,
    config: AuditConfig,
    stats: FileLoggerStats,
}

struct FileLoggerStats {
    total_events: AtomicU64,
    failed_events: AtomicU64,
    bytes_written: AtomicU64,
}

impl FileLogger {
    /// Create a new file logger.
    pub async fn new(path: impl Into<PathBuf>, config: AuditConfig) -> Result<Self, AuditError> {
        let path = path.into();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;

        Ok(Self {
            path,
            file: Mutex::new(Some(file)),
            config,
            stats: FileLoggerStats {
                total_events: AtomicU64::new(0),
                failed_events: AtomicU64::new(0),
                bytes_written: AtomicU64::new(0),
            },
        })
    }

    /// Create with default configuration.
    pub async fn with_path(path: impl Into<PathBuf>) -> Result<Self, AuditError> {
        Self::new(path, AuditConfig::default()).await
    }

    /// Format an event as a human-readable line.
    fn format_event(&self, event: &AuditEvent) -> String {
        let timestamp = event.timestamp.format("%Y-%m-%d %H:%M:%S%.3f UTC");
        let level = event.level.to_string();

        let context_parts: Vec<String> = [
            event
                .context
                .trace_id
                .as_ref()
                .map(|s| format!("trace={}", s)),
            event
                .context
                .session_id
                .as_ref()
                .map(|s| format!("session={}", s)),
            event
                .context
                .agent_id
                .as_ref()
                .map(|s| format!("agent={}", s)),
            event
                .context
                .user_id
                .as_ref()
                .map(|s| format!("user={}", s)),
        ]
        .into_iter()
        .flatten()
        .collect();

        let context_str = if context_parts.is_empty() {
            String::new()
        } else {
            format!(" [{}]", context_parts.join(" "))
        };

        let event_str = match &event.kind {
            crate::types::EventKind::ToolCall {
                tool_name,
                approved,
                duration_ms,
                ..
            } => {
                let duration = duration_ms
                    .map(|d| format!(" ({}ms)", d))
                    .unwrap_or_default();
                let status = if *approved { "approved" } else { "denied" };
                format!("TOOL_CALL tool={} status={}{}", tool_name, status, duration)
            }
            crate::types::EventKind::LlmRequest {
                provider,
                model,
                streaming,
                duration_ms,
                input_tokens,
                output_tokens,
                ..
            } => {
                let duration = duration_ms
                    .map(|d| format!(" ({}ms)", d))
                    .unwrap_or_default();
                let tokens = match (input_tokens, output_tokens) {
                    (Some(i), Some(o)) => format!(" tokens={}/{}", i, o),
                    _ => String::new(),
                };
                let stream = if *streaming { " streaming" } else { "" };
                format!(
                    "LLM_REQUEST provider={} model={}{}{}{}",
                    provider, model, stream, tokens, duration
                )
            }
            crate::types::EventKind::LlmResponse {
                provider,
                model,
                finish_reason,
                tool_calls_count,
            } => {
                let reason = finish_reason
                    .as_ref()
                    .map(|r| format!(" reason={}", r))
                    .unwrap_or_default();
                let tools = if *tool_calls_count > 0 {
                    format!(" tool_calls={}", tool_calls_count)
                } else {
                    String::new()
                };
                format!(
                    "LLM_RESPONSE provider={} model={}{}{}",
                    provider, model, reason, tools
                )
            }
            crate::types::EventKind::AgentLifecycle { agent_id, action } => {
                format!("AGENT_LIFECYCLE agent={} action={:?}", agent_id, action)
            }
            crate::types::EventKind::ApprovalDecision {
                tool_name,
                approved,
                approver,
                reason,
            } => {
                let status = if *approved { "approved" } else { "denied" };
                let reason_str = reason
                    .as_ref()
                    .map(|r| format!(" reason=\"{}\"", r))
                    .unwrap_or_default();
                format!(
                    "APPROVAL tool={} status={} by={}{}",
                    tool_name, status, approver, reason_str
                )
            }
            crate::types::EventKind::Error {
                error_type,
                message,
                ..
            } => {
                format!("ERROR type={} message=\"{}\"", error_type, message)
            }
            crate::types::EventKind::Security {
                event_type,
                description,
            } => {
                format!(
                    "SECURITY type={:?} description=\"{}\"",
                    event_type, description
                )
            }
            crate::types::EventKind::Custom { name, .. } => {
                format!("CUSTOM name={}", name)
            }
        };

        format!("[{}] {} {}{}\n", timestamp, level, event_str, context_str)
    }
}

#[async_trait]
impl AuditLogger for FileLogger {
    async fn log(&self, event: AuditEvent) -> Result<(), AuditError> {
        if !self.config.should_log(event.level) {
            return Ok(());
        }

        let line = self.format_event(&event);
        let bytes = line.as_bytes();

        let mut file_guard = self.file.lock().await;
        if let Some(file) = file_guard.as_mut() {
            match file.write_all(bytes).await {
                Ok(_) => {
                    self.stats.total_events.fetch_add(1, Ordering::Relaxed);
                    self.stats
                        .bytes_written
                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                    Ok(())
                }
                Err(e) => {
                    self.stats.failed_events.fetch_add(1, Ordering::Relaxed);
                    Err(AuditError::Io(e))
                }
            }
        } else {
            self.stats.failed_events.fetch_add(1, Ordering::Relaxed);
            Err(AuditError::NotInitialized)
        }
    }

    async fn flush(&self) -> Result<(), AuditError> {
        let mut file_guard = self.file.lock().await;
        if let Some(file) = file_guard.as_mut() {
            file.flush().await?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "file"
    }

    async fn health_check(&self) -> Result<(), AuditError> {
        let file_guard = self.file.lock().await;
        if file_guard.is_some() {
            Ok(())
        } else {
            Err(AuditError::NotInitialized)
        }
    }

    async fn stats(&self) -> AuditStats {
        AuditStats {
            total_events: self.stats.total_events.load(Ordering::Relaxed),
            failed_events: self.stats.failed_events.load(Ordering::Relaxed),
            bytes_written: self.stats.bytes_written.load(Ordering::Relaxed),
            ..Default::default()
        }
    }
}

impl std::fmt::Debug for FileLogger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileLogger")
            .field("path", &self.path)
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AuditContext, EventKind};
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_file_logger_creation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("audit.log");

        let logger = FileLogger::with_path(&path).await.unwrap();
        assert!(path.exists());
        assert_eq!(logger.name(), "file");
    }

    #[tokio::test]
    async fn test_file_logger_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("audit.log");

        let logger = FileLogger::with_path(&path).await.unwrap();

        let event = AuditEvent::tool_call("read_file", serde_json::json!({"path": "/tmp"}), true)
            .with_context(AuditContext::new().with_trace_id("trace-123"));

        logger.log(event).await.unwrap();
        logger.flush().await.unwrap();

        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(content.contains("TOOL_CALL"));
        assert!(content.contains("read_file"));
        assert!(content.contains("trace=trace-123"));
    }

    #[tokio::test]
    async fn test_file_logger_stats() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("audit.log");

        let logger = FileLogger::with_path(&path).await.unwrap();

        for i in 0..5 {
            let event = AuditEvent::tool_call(format!("tool_{}", i), serde_json::json!({}), true);
            logger.log(event).await.unwrap();
        }

        let stats = logger.stats().await;
        assert_eq!(stats.total_events, 5);
        assert!(stats.bytes_written > 0);
    }

    #[tokio::test]
    async fn test_file_logger_level_filtering() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("audit.log");

        let config = AuditConfig::new().with_min_level(crate::types::AuditLevel::Warn);
        let logger = FileLogger::new(&path, config).await.unwrap();

        // Info event should be filtered
        let info_event = AuditEvent::info(EventKind::Custom {
            name: "test".to_string(),
            payload: serde_json::json!({}),
        });
        logger.log(info_event).await.unwrap();

        // Warn event should be logged
        let warn_event = AuditEvent::warn(EventKind::Custom {
            name: "warning".to_string(),
            payload: serde_json::json!({}),
        });
        logger.log(warn_event).await.unwrap();
        logger.flush().await.unwrap();

        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(!content.contains("name=test"));
        assert!(content.contains("name=warning"));
    }
}
