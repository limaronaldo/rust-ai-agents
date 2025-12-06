//! # Data Matching Module
//!
//! High-performance data matching, CPF/CNPJ validation, and cross-source
//! consolidation for Brazilian data sources.
//!
//! ## Features
//!
//! - **CPF Matcher**: Normalize and validate Brazilian CPF numbers
//! - **CNPJ Matcher**: Normalize and validate Brazilian CNPJ numbers
//! - **Name Matcher**: Fuzzy matching with Brazilian name conventions
//! - **Data Matcher**: Cross-source entity resolution and consolidation
//! - **Data Pipeline**: Async processing with LRU caching
//! - **Parallel Pipeline**: High-throughput concurrent processing with DashMap
//! - **Metrics**: Comprehensive observability with EMA processing times
//! - **SQL Extractor**: PostgreSQL data extraction with dynamic schema (requires `postgres` feature)
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_data::{DataMatcher, CpfMatcher, CnpjMatcher, NameMatcher};
//!
//! let matcher = DataMatcher::new();
//! let results = matcher.match_across_sources(&sources, "Lucas Oliveira", Some("123.456.789-00"));
//! ```
//!
//! ## SQL Extraction (with `postgres` feature)
//!
//! ```rust,ignore
//! use rust_ai_agents_data::sql::{PostgresExtractor, PostgresConfig};
//!
//! let config = PostgresConfig::new("postgres://user:pass@localhost/db");
//! let pool = config.create_pool().await?;
//!
//! let extractor = PostgresExtractor::new("sales", "Recent Sales", pool)
//!     .with_query("SELECT * FROM sales WHERE created_at > NOW() - INTERVAL '30 days'");
//!
//! let data_source = extractor.extract().await?;
//! ```

pub mod cnpj;
pub mod cpf;
pub mod matcher;
pub mod metrics;
pub mod name;
pub mod pipeline;
#[cfg(feature = "postgres")]
pub mod sql;
pub mod types;

pub use cnpj::CnpjMatcher;
pub use cpf::CpfMatcher;
pub use matcher::DataMatcher;
pub use metrics::{DataMatchingMetrics, MetricsSnapshot};
pub use name::NameMatcher;
pub use pipeline::{CacheResult, ConcurrentCache, DataCache, DataPipeline, ParallelPipeline};
#[cfg(feature = "postgres")]
pub use sql::{DataExtractor, PostgresConfig, PostgresExtractor, QueryBuilder};
pub use types::*;
