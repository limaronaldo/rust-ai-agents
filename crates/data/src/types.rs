//! Core types for data matching

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A data source containing records
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    /// Unique identifier for this source
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Schema definition (field names and types)
    pub schema: DataSchema,
    /// Records in this source
    pub records: Vec<DataRecord>,
}

impl DataSource {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            schema: DataSchema::default(),
            records: Vec::new(),
        }
    }

    pub fn with_records(mut self, records: Vec<DataRecord>) -> Self {
        self.records = records;
        self
    }
}

/// Schema definition for a data source
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataSchema {
    /// Field definitions
    pub fields: HashMap<String, FieldType>,
    /// Primary key fields
    pub primary_keys: Vec<String>,
}

/// Field type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    Text,
    Integer,
    Float,
    Boolean,
    Date,
    Cpf,
    Cnpj,
    Email,
    Phone,
}

/// A single record from a data source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRecord {
    /// Source this record came from
    pub source_id: String,
    /// Field values
    pub fields: HashMap<String, FieldValue>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl DataRecord {
    pub fn new(source_id: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            fields: HashMap::new(),
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    pub fn with_field(mut self, key: impl Into<String>, value: FieldValue) -> Self {
        self.fields.insert(key.into(), value);
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Get a text field value
    pub fn get_text(&self, key: &str) -> Option<&str> {
        match self.fields.get(key) {
            Some(FieldValue::Text(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get any field that might contain a name
    pub fn get_name_field(&self) -> Option<&str> {
        // Try common name field variations
        for key in &["nome", "name", "nome_completo", "full_name", "razao_social"] {
            if let Some(FieldValue::Text(s)) = self.fields.get(*key) {
                return Some(s.as_str());
            }
        }
        None
    }

    /// Get any field that might contain a CPF
    pub fn get_cpf_field(&self) -> Option<&str> {
        for key in &["cpf", "documento", "document", "cpf_cnpj", "tax_id"] {
            if let Some(FieldValue::Text(s)) = self.fields.get(*key) {
                // Check if it looks like a CPF (11 digits when normalized)
                let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
                if digits.len() == 11 {
                    return Some(s.as_str());
                }
            }
        }
        None
    }
}

/// Field value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldValue {
    Text(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Date(String),
    Null,
}

impl From<String> for FieldValue {
    fn from(s: String) -> Self {
        FieldValue::Text(s)
    }
}

impl From<&str> for FieldValue {
    fn from(s: &str) -> Self {
        FieldValue::Text(s.to_string())
    }
}

impl From<i64> for FieldValue {
    fn from(n: i64) -> Self {
        FieldValue::Integer(n)
    }
}

impl From<f64> for FieldValue {
    fn from(n: f64) -> Self {
        FieldValue::Float(n)
    }
}

impl From<bool> for FieldValue {
    fn from(b: bool) -> Self {
        FieldValue::Boolean(b)
    }
}

/// Result of matching across sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    /// Normalized entity identifier
    pub entity_id: String,
    /// Sources that matched
    pub sources: Vec<SourceMatch>,
    /// Overall confidence score
    pub confidence: f64,
    /// Consolidated fields from all sources
    pub consolidated_fields: HashMap<String, FieldValue>,
    /// Match metadata
    pub metadata: MatchMetadata,
}

impl MatchResult {
    pub fn new(entity_id: impl Into<String>) -> Self {
        Self {
            entity_id: entity_id.into(),
            sources: Vec::new(),
            confidence: 0.0,
            consolidated_fields: HashMap::new(),
            metadata: MatchMetadata::default(),
        }
    }

    pub fn add_source(&mut self, source: SourceMatch) {
        self.sources.push(source);
    }

    pub fn calculate_confidence(&mut self) {
        if self.sources.is_empty() {
            self.confidence = 0.0;
            return;
        }

        // Weighted average based on individual match scores
        let total: f64 = self.sources.iter().map(|s| s.score).sum();
        self.confidence = total / self.sources.len() as f64;

        // Boost confidence if matched in multiple sources
        if self.sources.len() > 1 {
            self.confidence = (self.confidence * 1.1).min(1.0);
        }
    }
}

/// A match from a specific source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMatch {
    /// Source identifier
    pub source_id: String,
    /// Source name
    pub source_name: String,
    /// Match score (0.0 - 1.0)
    pub score: f64,
    /// Matched record
    pub record: DataRecord,
    /// Match type (exact, fuzzy, etc.)
    pub match_type: MatchType,
}

/// Type of match
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchType {
    /// Exact match on identifier (CPF, etc.)
    ExactId,
    /// Exact text match
    ExactText,
    /// Fuzzy text match
    Fuzzy,
    /// Phonetic match
    Phonetic,
    /// Combined match (multiple criteria)
    Combined,
}

/// Metadata about the matching process
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatchMetadata {
    /// Time taken to match (milliseconds)
    pub match_time_ms: u64,
    /// Number of records scanned
    pub records_scanned: usize,
    /// Match criteria used
    pub criteria: Vec<String>,
    /// Warnings or notes
    pub warnings: Vec<String>,
}

/// Errors that can occur during data operations
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("Source not found: {0}")]
    SourceNotFound(String),

    #[error("Invalid CPF: {0}")]
    InvalidCpf(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
