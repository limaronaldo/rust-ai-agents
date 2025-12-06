//! # Built-in Tools
//!
//! Common tools that can be used by agents, with enhanced registry
//! featuring circuit breaker, retry, and timeout patterns.
//!
//! ## Available Tools
//!
//! ### Math
//! - `CalculatorTool` - Full expression evaluation with scientific functions
//! - `UnitConverterTool` - Convert between units (length, weight, temperature)
//! - `StatisticsTool` - Calculate mean, median, std dev, etc.
//!
//! ### DateTime
//! - `CurrentTimeTool` - Get current date/time
//! - `DateParserTool` - Parse dates in various formats
//! - `DateCalculatorTool` - Date arithmetic and differences
//!
//! ### Encoding
//! - `JsonTool` - Parse, format, merge JSON
//! - `Base64Tool` - Encode/decode Base64
//! - `HashTool` - SHA256/SHA512 hashing
//! - `UrlEncodeTool` - URL encode/decode
//!
//! ### File System
//! - `ReadFileTool` - Read file contents
//! - `WriteFileTool` - Write to files
//! - `ListDirectoryTool` - List directory contents
//!
//! ### Web
//! - `HttpRequestTool` - Make HTTP requests
//! - `WebSearchTool` - Web search (stub)
//!
//! ### SQL (requires `postgres` feature)
//! - `ExecuteSqlTool` - Execute read-only SQL queries
//! - `ListTablesTool` - List database tables and columns
//! - `DescribeTableTool` - Get detailed table schema information
//!
//! ### Cross-Reference (requires `crossref` feature)
//! - `CrossReferenceEntityTool` - Cross-reference entities across data sources with PT-BR narratives

pub use rust_ai_agents_core::tool::*;
pub use rust_ai_agents_macros::Tool;

use std::sync::Arc;

#[cfg(feature = "crossref")]
pub mod crossref;
pub mod datetime;
pub mod encoding;
pub mod file;
pub mod macro_tools;
pub mod math;
pub mod registry;
#[cfg(feature = "postgres")]
pub mod sql;
pub mod web;

#[cfg(feature = "crossref")]
pub use crossref::*;
pub use datetime::*;
pub use encoding::*;
pub use file::*;
pub use macro_tools::*;
pub use math::*;
pub use registry::*;
#[cfg(feature = "postgres")]
pub use sql::*;
pub use web::*;

/// Create a registry with all built-in tools
pub fn create_default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // Math tools
    registry.register(Arc::new(CalculatorTool::new()));
    registry.register(Arc::new(UnitConverterTool::new()));
    registry.register(Arc::new(StatisticsTool::new()));

    // DateTime tools
    registry.register(Arc::new(CurrentTimeTool::new()));
    registry.register(Arc::new(DateParserTool::new()));
    registry.register(Arc::new(DateCalculatorTool::new()));

    // Encoding tools
    registry.register(Arc::new(JsonTool::new()));
    registry.register(Arc::new(Base64Tool::new()));
    registry.register(Arc::new(HashTool::new()));
    registry.register(Arc::new(UrlEncodeTool::new()));

    // File tools
    registry.register(Arc::new(ReadFileTool::new()));
    registry.register(Arc::new(WriteFileTool::new()));
    registry.register(Arc::new(ListDirectoryTool::new()));

    // Web tools
    registry.register(Arc::new(HttpRequestTool::new()));
    registry.register(Arc::new(WebSearchTool::new()));

    registry
}
