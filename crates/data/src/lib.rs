//! # Data Matching Module
//!
//! High-performance data matching, CPF validation, and cross-source
//! consolidation for Brazilian data sources.
//!
//! ## Features
//!
//! - **CPF Matcher**: Normalize and validate Brazilian CPF numbers
//! - **Name Matcher**: Fuzzy matching with Brazilian name conventions
//! - **Data Matcher**: Cross-source entity resolution and consolidation
//! - **Data Pipeline**: Async processing with LRU caching
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_data::{DataMatcher, CpfMatcher, NameMatcher};
//!
//! let matcher = DataMatcher::new();
//! let results = matcher.match_across_sources(&sources, "Lucas Oliveira", Some("123.456.789-00"));
//! ```

pub mod cpf;
pub mod matcher;
pub mod name;
pub mod pipeline;
pub mod types;

pub use cpf::CpfMatcher;
pub use matcher::DataMatcher;
pub use name::NameMatcher;
pub use pipeline::{DataCache, DataPipeline};
pub use types::*;
