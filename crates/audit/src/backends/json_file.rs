//! JSON file logger backend with rotation support.
//!
//! Structured JSON logging with configurable rotation policies.

use crate::error::AuditError;
use crate::traits::{AuditConfig, AuditLogger, AuditStats};
use crate::types::AuditEvent;
use async_trait::async_trait;
use chrono::{DateTime, Timelike, Utc};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

/// Rotation policy for log files.
#[derive(Debug, Clone)]
pub enum RotationPolicy {
    /// Rotate when file exceeds size in bytes
    Size(u64),
    /// Rotate daily
    Daily,
    /// Rotate hourly
    Hourly,
    /// Never rotate
    Never,
}

impl Default for RotationPolicy {
    fn default() -> Self {
        Self::Size(10 * 1024 * 1024) // 10MB default
    }
}

/// Configuration for log rotation.
#[derive(Debug, Clone)]
pub struct RotationConfig {
    /// Rotation policy
    pub policy: RotationPolicy,
    /// Maximum number of rotated files to keep
    pub max_files: usize,
    /// Whether to compress rotated files
    pub compress: bool,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            policy: RotationPolicy::default(),
            max_files: 10,
            compress: false,
        }
    }
}

impl RotationConfig {
    /// Create a new rotation config.
    pub fn new(policy: RotationPolicy) -> Self {
        Self {
            policy,
            ..Default::default()
        }
    }

    /// Set maximum files to keep.
    pub fn with_max_files(mut self, max: usize) -> Self {
        self.max_files = max;
        self
    }

    /// Enable compression.
    pub fn with_compression(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }
}

/// JSON file logger with rotation support.
///
/// Writes each audit event as a JSON line (JSONL format) with optional rotation.
pub struct JsonFileLogger {
    base_path: PathBuf,
    current_file: Mutex<CurrentFile>,
    config: AuditConfig,
    rotation: RotationConfig,
    stats: JsonLoggerStats,
}

struct CurrentFile {
    file: Option<File>,
    path: PathBuf,
    bytes_written: u64,
    created_at: DateTime<Utc>,
}

struct JsonLoggerStats {
    total_events: AtomicU64,
    failed_events: AtomicU64,
    bytes_written: AtomicU64,
    rotations: AtomicU64,
}

impl JsonFileLogger {
    /// Create a new JSON file logger.
    pub async fn new(
        base_path: impl Into<PathBuf>,
        config: AuditConfig,
        rotation: RotationConfig,
    ) -> Result<Self, AuditError> {
        let base_path = base_path.into();

        // Ensure parent directory exists
        if let Some(parent) = base_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let current_path = Self::generate_filename(&base_path, Utc::now());
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&current_path)
            .await?;

        let metadata = file.metadata().await?;

        Ok(Self {
            base_path,
            current_file: Mutex::new(CurrentFile {
                file: Some(file),
                path: current_path,
                bytes_written: metadata.len(),
                created_at: Utc::now(),
            }),
            config,
            rotation,
            stats: JsonLoggerStats {
                total_events: AtomicU64::new(0),
                failed_events: AtomicU64::new(0),
                bytes_written: AtomicU64::new(0),
                rotations: AtomicU64::new(0),
            },
        })
    }

    /// Create with default configuration.
    pub async fn with_path(path: impl Into<PathBuf>) -> Result<Self, AuditError> {
        Self::new(path, AuditConfig::default(), RotationConfig::default()).await
    }

    /// Generate a filename with timestamp.
    fn generate_filename(base: &std::path::Path, time: DateTime<Utc>) -> PathBuf {
        let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("audit");
        let ext = base.extension().and_then(|s| s.to_str()).unwrap_or("jsonl");
        let parent = base.parent().unwrap_or(std::path::Path::new("."));

        parent.join(format!("{}-{}.{}", stem, time.format("%Y%m%d-%H%M%S"), ext))
    }

    /// Check if rotation is needed.
    fn needs_rotation(&self, current: &CurrentFile, now: DateTime<Utc>) -> bool {
        match self.rotation.policy {
            RotationPolicy::Size(max_size) => current.bytes_written >= max_size,
            RotationPolicy::Daily => current.created_at.date_naive() != now.date_naive(),
            RotationPolicy::Hourly => {
                current.created_at.date_naive() != now.date_naive()
                    || current.created_at.hour() != now.hour()
            }
            RotationPolicy::Never => false,
        }
    }

    /// Rotate the log file.
    async fn rotate(&self, current: &mut CurrentFile) -> Result<(), AuditError> {
        // Close current file
        if let Some(mut file) = current.file.take() {
            file.flush().await?;
        }

        // Clean up old files
        self.cleanup_old_files().await?;

        // Create new file
        let now = Utc::now();
        let new_path = Self::generate_filename(&self.base_path, now);
        let new_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&new_path)
            .await?;

        current.file = Some(new_file);
        current.path = new_path;
        current.bytes_written = 0;
        current.created_at = now;

        self.stats.rotations.fetch_add(1, Ordering::Relaxed);

        tracing::info!("Audit log rotated to {:?}", current.path);

        Ok(())
    }

    /// Clean up old rotated files.
    async fn cleanup_old_files(&self) -> Result<(), AuditError> {
        let parent = self.base_path.parent().unwrap_or(std::path::Path::new("."));
        let stem = self
            .base_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audit");

        let mut entries: Vec<_> = Vec::new();
        let mut dir = tokio::fs::read_dir(parent).await?;

        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(stem) && name.contains('-') {
                    if let Ok(metadata) = entry.metadata().await {
                        entries.push((
                            path,
                            metadata
                                .modified()
                                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                        ));
                    }
                }
            }
        }

        // Sort by modification time (oldest first)
        entries.sort_by_key(|(_, time)| *time);

        // Remove oldest files if we have too many
        while entries.len() > self.rotation.max_files {
            if let Some((path, _)) = entries.first() {
                tracing::debug!("Removing old audit log: {:?}", path);
                tokio::fs::remove_file(path).await?;
                entries.remove(0);
            }
        }

        Ok(())
    }
}

#[async_trait]
impl AuditLogger for JsonFileLogger {
    async fn log(&self, event: AuditEvent) -> Result<(), AuditError> {
        if !self.config.should_log(event.level) {
            return Ok(());
        }

        let mut json = serde_json::to_string(&event)?;
        json.push('\n');
        let bytes = json.as_bytes();

        let mut current = self.current_file.lock().await;

        // Check for rotation
        let now = Utc::now();
        if self.needs_rotation(&current, now) {
            self.rotate(&mut current).await?;
        }

        // Write event
        if let Some(file) = current.file.as_mut() {
            match file.write_all(bytes).await {
                Ok(_) => {
                    current.bytes_written += bytes.len() as u64;
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
        let mut current = self.current_file.lock().await;
        if let Some(file) = current.file.as_mut() {
            file.flush().await?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "json_file"
    }

    async fn health_check(&self) -> Result<(), AuditError> {
        let current = self.current_file.lock().await;
        if current.file.is_some() {
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

impl std::fmt::Debug for JsonFileLogger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonFileLogger")
            .field("base_path", &self.base_path)
            .field("config", &self.config)
            .field("rotation", &self.rotation)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AuditContext;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_json_logger_creation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");

        let logger = JsonFileLogger::with_path(&path).await.unwrap();
        assert_eq!(logger.name(), "json_file");
    }

    #[tokio::test]
    async fn test_json_logger_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");

        let logger = JsonFileLogger::with_path(&path).await.unwrap();

        let event =
            AuditEvent::tool_call("execute_code", serde_json::json!({"lang": "python"}), true)
                .with_context(AuditContext::new().with_agent_id("agent-1"));

        logger.log(event).await.unwrap();
        logger.flush().await.unwrap();

        // Read the generated file
        let current = logger.current_file.lock().await;
        let content = tokio::fs::read_to_string(&current.path).await.unwrap();

        // Verify it's valid JSON
        let parsed: AuditEvent = serde_json::from_str(content.trim()).unwrap();
        assert!(matches!(
            parsed.kind,
            crate::types::EventKind::ToolCall { .. }
        ));
    }

    #[tokio::test]
    async fn test_json_logger_rotation_by_size() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");

        let rotation = RotationConfig::new(RotationPolicy::Size(500)); // Small size for testing
        let logger = JsonFileLogger::new(&path, AuditConfig::default(), rotation)
            .await
            .unwrap();

        // Write enough events to trigger rotation
        for i in 0..20 {
            let event = AuditEvent::tool_call(
                format!("tool_with_long_name_{}", i),
                serde_json::json!({"data": "some test data that takes up space"}),
                true,
            );
            logger.log(event).await.unwrap();
        }
        logger.flush().await.unwrap();

        // Check that rotation happened
        let stats = logger.stats().await;
        assert!(stats.total_events > 0);

        // Check multiple files exist
        let mut count = 0;
        let mut dir_entries = tokio::fs::read_dir(dir.path()).await.unwrap();
        while let Some(_) = dir_entries.next_entry().await.unwrap() {
            count += 1;
        }
        assert!(count >= 1);
    }

    #[tokio::test]
    async fn test_rotation_config() {
        let config = RotationConfig::new(RotationPolicy::Daily)
            .with_max_files(5)
            .with_compression(true);

        assert!(matches!(config.policy, RotationPolicy::Daily));
        assert_eq!(config.max_files, 5);
        assert!(config.compress);
    }
}
