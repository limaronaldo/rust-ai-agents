//! Audit logging backends.
//!
//! This module provides various backends for storing audit logs:
//! - `FileLogger`: Plain text file logging
//! - `JsonFileLogger`: JSON structured logging with rotation
//! - `AsyncLogger`: Non-blocking async wrapper

mod async_logger;
mod file;
mod json_file;

pub use async_logger::{AsyncLogger, AsyncLoggerBuilder};
pub use file::FileLogger;
pub use json_file::{JsonFileLogger, RotationConfig, RotationPolicy};
