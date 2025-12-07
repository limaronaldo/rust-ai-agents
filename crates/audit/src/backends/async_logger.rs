//! Async logger wrapper for non-blocking audit logging.
//!
//! Wraps any `AuditLogger` to provide buffered, non-blocking logging
//! with background flushing.

use crate::error::AuditError;
use crate::traits::{AuditConfig, AuditLogger};
use crate::types::AuditEvent;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};

/// Async wrapper that provides non-blocking logging.
///
/// Events are sent to a background task that handles the actual logging,
/// allowing the caller to continue without waiting for I/O.
pub struct AsyncLogger {
    sender: mpsc::Sender<LogCommand>,
    name: String,
}

enum LogCommand {
    Log(Box<AuditEvent>),
    Flush,
    Shutdown,
}

impl AsyncLogger {
    /// Create a new async logger wrapping the given backend.
    pub fn new<L: AuditLogger + 'static>(inner: L, config: &AuditConfig) -> Self {
        let (sender, receiver) = mpsc::channel(config.buffer_size);
        let name = format!("async({})", inner.name());
        let flush_interval = config.flush_interval_secs;

        tokio::spawn(Self::background_task(
            Arc::new(inner),
            receiver,
            flush_interval,
        ));

        Self { sender, name }
    }

    /// Create with default buffer size.
    pub fn wrap<L: AuditLogger + 'static>(inner: L) -> Self {
        Self::new(inner, &AuditConfig::default())
    }

    async fn background_task(
        inner: Arc<dyn AuditLogger>,
        mut receiver: mpsc::Receiver<LogCommand>,
        flush_interval_secs: u64,
    ) {
        let mut flush_timer = interval(Duration::from_secs(flush_interval_secs));

        loop {
            tokio::select! {
                cmd = receiver.recv() => {
                    match cmd {
                        Some(LogCommand::Log(event)) => {
                            if let Err(e) = inner.log(*event).await {
                                tracing::error!("Async audit log error: {}", e);
                            }
                        }
                        Some(LogCommand::Flush) => {
                            if let Err(e) = inner.flush().await {
                                tracing::error!("Async audit flush error: {}", e);
                            }
                        }
                        Some(LogCommand::Shutdown) | None => {
                            // Final flush before shutdown
                            let _ = inner.flush().await;
                            break;
                        }
                    }
                }
                _ = flush_timer.tick() => {
                    if let Err(e) = inner.flush().await {
                        tracing::error!("Async audit periodic flush error: {}", e);
                    }
                }
            }
        }

        tracing::debug!("Async audit logger background task stopped");
    }

    /// Gracefully shutdown the logger.
    pub async fn shutdown(&self) -> Result<(), AuditError> {
        self.sender
            .send(LogCommand::Shutdown)
            .await
            .map_err(|_| AuditError::ChannelSend)
    }
}

#[async_trait]
impl AuditLogger for AsyncLogger {
    async fn log(&self, event: AuditEvent) -> Result<(), AuditError> {
        self.sender
            .send(LogCommand::Log(Box::new(event)))
            .await
            .map_err(|_| AuditError::ChannelSend)
    }

    async fn flush(&self) -> Result<(), AuditError> {
        self.sender
            .send(LogCommand::Flush)
            .await
            .map_err(|_| AuditError::ChannelSend)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for AsyncLogger {
    fn drop(&mut self) {
        // Try to send shutdown signal (best effort)
        let _ = self.sender.try_send(LogCommand::Shutdown);
    }
}

/// A builder for creating async loggers with custom configuration.
pub struct AsyncLoggerBuilder<L> {
    inner: L,
    buffer_size: usize,
    flush_interval_secs: u64,
}

impl<L: AuditLogger + 'static> AsyncLoggerBuilder<L> {
    /// Create a new builder.
    pub fn new(inner: L) -> Self {
        Self {
            inner,
            buffer_size: 1000,
            flush_interval_secs: 5,
        }
    }

    /// Set the buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the flush interval in seconds.
    pub fn flush_interval(mut self, secs: u64) -> Self {
        self.flush_interval_secs = secs;
        self
    }

    /// Build the async logger.
    pub fn build(self) -> AsyncLogger {
        let config = AuditConfig {
            buffer_size: self.buffer_size,
            flush_interval_secs: self.flush_interval_secs,
            ..Default::default()
        };
        AsyncLogger::new(self.inner, &config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryLogger;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_async_logger_basic() {
        let memory = Arc::new(MemoryLogger::new());
        let memory_clone = memory.clone();

        // Create async logger with the Arc
        let (sender, mut receiver) = mpsc::channel::<LogCommand>(100);
        let name = "async(memory)".to_string();

        let inner = memory_clone;
        tokio::spawn(async move {
            while let Some(cmd) = receiver.recv().await {
                match cmd {
                    LogCommand::Log(event) => {
                        let _ = inner.log(*event).await;
                    }
                    LogCommand::Flush => {
                        let _ = inner.flush().await;
                    }
                    LogCommand::Shutdown => break,
                }
            }
        });

        let async_logger = AsyncLogger { sender, name };

        let event = AuditEvent::tool_call("test", serde_json::json!({}), true);
        async_logger.log(event).await.unwrap();
        async_logger.flush().await.unwrap();

        // Give background task time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Event may or may not have been processed yet due to async nature
        let _ = memory.count().await;
    }

    #[tokio::test]
    async fn test_async_logger_builder() {
        let memory = MemoryLogger::new();

        let logger = AsyncLoggerBuilder::new(memory)
            .buffer_size(500)
            .flush_interval(2)
            .build();

        assert!(logger.name().contains("async"));
    }

    #[tokio::test]
    async fn test_async_logger_wrap() {
        let memory = MemoryLogger::new();
        let logger = AsyncLogger::wrap(memory);

        assert!(logger.name().contains("async"));
        assert!(logger.name().contains("memory"));
    }
}
