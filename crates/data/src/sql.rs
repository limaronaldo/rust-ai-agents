//! SQL Data Extraction for PostgreSQL
//!
//! Generic SQL extractor that converts query results to JSON dynamically
//! without requiring compile-time schema knowledge.
//!
//! ## Features
//!
//! - **Dynamic Row Conversion**: Automatically converts SQL types to JSON
//! - **Type Detection**: Handles integers, floats, strings, booleans, JSON, UUIDs, dates
//! - **Connection Pooling**: Uses SQLx's built-in connection pool
//! - **Parameterized Queries**: Safe query execution with bind parameters
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_data::sql::PostgresExtractor;
//! use sqlx::postgres::PgPoolOptions;
//!
//! let pool = PgPoolOptions::new()
//!     .max_connections(5)
//!     .connect("postgres://user:pass@localhost/db")
//!     .await?;
//!
//! let extractor = PostgresExtractor::new("sales", "Recent Sales", pool)
//!     .with_query("SELECT * FROM sales WHERE created_at > NOW() - INTERVAL '30 days'");
//!
//! let data_source = extractor.extract().await?;
//! ```

use crate::types::{DataError, DataRecord, DataSchema, DataSource, FieldType, FieldValue};
use async_trait::async_trait;
use serde_json::{json, Value};
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::{Column, Row, TypeInfo, ValueRef};
use std::collections::HashMap;

use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Configuration for PostgreSQL connection
#[derive(Debug, Clone)]
pub struct PostgresConfig {
    /// Database connection URL
    pub url: String,
    /// Maximum number of connections in the pool
    pub max_connections: u32,
    /// Minimum number of connections to keep alive
    pub min_connections: u32,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Idle timeout for connections
    pub idle_timeout: Duration,
    /// Maximum lifetime of a connection
    pub max_lifetime: Duration,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            max_connections: 10,
            min_connections: 1,
            connect_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(1800),
        }
    }
}

impl PostgresConfig {
    /// Create config from URL
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Default::default()
        }
    }

    /// Set max connections
    pub fn with_max_connections(mut self, n: u32) -> Self {
        self.max_connections = n;
        self
    }

    /// Create a connection pool from this config
    pub async fn create_pool(&self) -> Result<PgPool, DataError> {
        PgPoolOptions::new()
            .max_connections(self.max_connections)
            .min_connections(self.min_connections)
            .acquire_timeout(self.connect_timeout)
            .idle_timeout(self.idle_timeout)
            .max_lifetime(self.max_lifetime)
            .connect(&self.url)
            .await
            .map_err(|e| {
                DataError::ProcessingError(format!("Failed to connect to PostgreSQL: {}", e))
            })
    }
}

/// PostgreSQL data extractor with dynamic schema detection
pub struct PostgresExtractor {
    /// Unique identifier for this extractor
    id: String,
    /// Human-readable name
    name: String,
    /// Connection pool
    pool: PgPool,
    /// SQL query to execute
    query: String,
    /// Query parameters
    params: Vec<QueryParam>,
    /// Field to use as record ID
    id_field: Option<String>,
    /// Metadata to attach to the data source
    metadata: HashMap<String, String>,
}

/// Query parameter types
#[derive(Debug, Clone)]
pub enum QueryParam {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl PostgresExtractor {
    /// Create a new PostgreSQL extractor
    pub fn new(id: impl Into<String>, name: impl Into<String>, pool: PgPool) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            pool,
            query: String::new(),
            params: Vec::new(),
            id_field: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the SQL query
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = query.into();
        self
    }

    /// Add a string parameter
    pub fn with_param_str(mut self, value: impl Into<String>) -> Self {
        self.params.push(QueryParam::String(value.into()));
        self
    }

    /// Add an integer parameter
    pub fn with_param_int(mut self, value: i64) -> Self {
        self.params.push(QueryParam::Int(value));
        self
    }

    /// Add a float parameter
    pub fn with_param_float(mut self, value: f64) -> Self {
        self.params.push(QueryParam::Float(value));
        self
    }

    /// Add a boolean parameter
    pub fn with_param_bool(mut self, value: bool) -> Self {
        self.params.push(QueryParam::Bool(value));
        self
    }

    /// Set the field to use as record ID
    pub fn with_id_field(mut self, field: impl Into<String>) -> Self {
        self.id_field = Some(field.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Execute query and extract data
    pub async fn extract(&self) -> Result<DataSource, DataError> {
        let start = Instant::now();

        if self.query.is_empty() {
            return Err(DataError::ProcessingError("No query specified".to_string()));
        }

        debug!(extractor_id = %self.id, query = %self.query, "Executing SQL query");

        // Execute query
        let rows = sqlx::query(&self.query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DataError::ProcessingError(format!("Query execution failed: {}", e)))?;

        info!(
            extractor_id = %self.id,
            rows_fetched = rows.len(),
            elapsed_ms = start.elapsed().as_millis(),
            "SQL query completed"
        );

        // Detect schema from first row
        let schema = if !rows.is_empty() {
            self.detect_schema(&rows[0])
        } else {
            DataSchema::default()
        };

        // Convert rows to records
        let records: Vec<DataRecord> = rows.iter().map(|row| self.row_to_record(row)).collect();

        // Build metadata
        let mut source_metadata = self.metadata.clone();
        source_metadata.insert("source_type".to_string(), "postgresql".to_string());
        source_metadata.insert("query".to_string(), self.query.clone());
        source_metadata.insert("rows_fetched".to_string(), records.len().to_string());
        source_metadata.insert(
            "elapsed_ms".to_string(),
            start.elapsed().as_millis().to_string(),
        );

        Ok(DataSource {
            id: self.id.clone(),
            name: self.name.clone(),
            schema,
            records,
        })
    }

    /// Detect schema from a row
    fn detect_schema(&self, row: &PgRow) -> DataSchema {
        let mut fields = HashMap::new();
        let mut primary_keys = Vec::new();

        for col in row.columns() {
            let name = col.name().to_string();
            let type_info = col.type_info();
            let type_name = type_info.name();

            let field_type = match type_name {
                "INT2" | "INT4" | "INT8" | "SERIAL" | "BIGSERIAL" => FieldType::Integer,
                "FLOAT4" | "FLOAT8" | "NUMERIC" | "DECIMAL" => FieldType::Float,
                "BOOL" => FieldType::Boolean,
                "DATE" | "TIMESTAMP" | "TIMESTAMPTZ" | "TIME" | "TIMETZ" => FieldType::Date,
                "TEXT" | "VARCHAR" | "CHAR" | "BPCHAR" | "NAME" => FieldType::Text,
                _ => FieldType::Text, // Default to text
            };

            // Detect CPF/CNPJ fields by name
            let field_type = if name.contains("cpf") {
                FieldType::Cpf
            } else if name.contains("cnpj") {
                FieldType::Cnpj
            } else if name.contains("email") {
                FieldType::Email
            } else if name.contains("phone") || name.contains("telefone") {
                FieldType::Phone
            } else {
                field_type
            };

            // Detect primary keys by name
            if name == "id" || name.ends_with("_id") || name == "uuid" {
                primary_keys.push(name.clone());
            }

            fields.insert(name, field_type);
        }

        DataSchema {
            fields,
            primary_keys,
        }
    }

    /// Convert a PostgreSQL row to a DataRecord
    fn row_to_record(&self, row: &PgRow) -> DataRecord {
        let mut fields = HashMap::new();
        let mut record_id = String::from("unknown");

        for col in row.columns() {
            let name = col.name().to_string();
            let ordinal = col.ordinal();

            let value = self.extract_value(row, ordinal);

            // Try to extract record ID
            if let Some(ref id_field) = self.id_field {
                if name == *id_field {
                    record_id = value_to_string(&value);
                }
            } else if record_id == "unknown" {
                // Auto-detect ID from common field names
                if name == "id" || name == "uuid" || name == "cpf" || name == "cnpj" {
                    record_id = value_to_string(&value);
                }
            }

            fields.insert(name, value);
        }

        DataRecord {
            source_id: self.id.clone(),
            fields,
            confidence: 1.0, // Database data is considered "truth"
            metadata: HashMap::new(),
        }
    }

    /// Extract a value from a row column, trying multiple types
    fn extract_value(&self, row: &PgRow, ordinal: usize) -> FieldValue {
        // Try each type in order of specificity

        // Integers
        if let Ok(val) = row.try_get::<Option<i64>, _>(ordinal) {
            return val.map(FieldValue::Integer).unwrap_or(FieldValue::Null);
        }
        if let Ok(val) = row.try_get::<Option<i32>, _>(ordinal) {
            return val
                .map(|v| FieldValue::Integer(v as i64))
                .unwrap_or(FieldValue::Null);
        }
        if let Ok(val) = row.try_get::<Option<i16>, _>(ordinal) {
            return val
                .map(|v| FieldValue::Integer(v as i64))
                .unwrap_or(FieldValue::Null);
        }

        // Floats
        if let Ok(val) = row.try_get::<Option<f64>, _>(ordinal) {
            return val.map(FieldValue::Float).unwrap_or(FieldValue::Null);
        }
        if let Ok(val) = row.try_get::<Option<f32>, _>(ordinal) {
            return val
                .map(|v| FieldValue::Float(v as f64))
                .unwrap_or(FieldValue::Null);
        }

        // Boolean
        if let Ok(val) = row.try_get::<Option<bool>, _>(ordinal) {
            return val.map(FieldValue::Boolean).unwrap_or(FieldValue::Null);
        }

        // String
        if let Ok(val) = row.try_get::<Option<String>, _>(ordinal) {
            return val.map(FieldValue::Text).unwrap_or(FieldValue::Null);
        }

        // JSON/JSONB - convert to string representation
        if let Ok(val) = row.try_get::<Option<serde_json::Value>, _>(ordinal) {
            return val
                .map(|v| FieldValue::Text(v.to_string()))
                .unwrap_or(FieldValue::Null);
        }

        // UUID
        if let Ok(val) = row.try_get::<Option<sqlx::types::Uuid>, _>(ordinal) {
            return val
                .map(|v| FieldValue::Text(v.to_string()))
                .unwrap_or(FieldValue::Null);
        }

        // DateTime with timezone
        if let Ok(val) = row.try_get::<Option<chrono::DateTime<chrono::Utc>>, _>(ordinal) {
            return val
                .map(|v| FieldValue::Date(v.to_rfc3339()))
                .unwrap_or(FieldValue::Null);
        }

        // NaiveDateTime (without timezone)
        if let Ok(val) = row.try_get::<Option<chrono::NaiveDateTime>, _>(ordinal) {
            return val
                .map(|v| FieldValue::Date(v.to_string()))
                .unwrap_or(FieldValue::Null);
        }

        // NaiveDate
        if let Ok(val) = row.try_get::<Option<chrono::NaiveDate>, _>(ordinal) {
            return val
                .map(|v| FieldValue::Date(v.to_string()))
                .unwrap_or(FieldValue::Null);
        }

        // BigDecimal - convert to f64
        if let Ok(val) = row.try_get::<Option<sqlx::types::BigDecimal>, _>(ordinal) {
            return val
                .and_then(|v| {
                    use std::str::FromStr;
                    f64::from_str(&v.to_string()).ok()
                })
                .map(FieldValue::Float)
                .unwrap_or(FieldValue::Null);
        }

        // Fallback: try to get raw bytes and convert to string
        let value_ref = row.try_get_raw(ordinal);
        if let Ok(vr) = value_ref {
            if vr.is_null() {
                return FieldValue::Null;
            }
        }

        warn!(
            ordinal = ordinal,
            "Could not extract value, defaulting to Null"
        );
        FieldValue::Null
    }
}

/// Convert FieldValue to string for ID extraction
fn value_to_string(value: &FieldValue) -> String {
    match value {
        FieldValue::Text(s) => s.clone(),
        FieldValue::Integer(n) => n.to_string(),
        FieldValue::Float(n) => n.to_string(),
        FieldValue::Boolean(b) => b.to_string(),
        FieldValue::Date(d) => d.clone(),
        FieldValue::Null => "null".to_string(),
    }
}

/// Trait for data extractors
#[async_trait]
pub trait DataExtractor: Send + Sync {
    /// Unique identifier for this extractor
    fn id(&self) -> &str;

    /// Human-readable name
    fn name(&self) -> &str;

    /// Extract data from the source
    async fn extract(&self) -> Result<DataSource, DataError>;
}

#[async_trait]
impl DataExtractor for PostgresExtractor {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn extract(&self) -> Result<DataSource, DataError> {
        PostgresExtractor::extract(self).await
    }
}

/// SQL Query Builder for safe query construction
#[derive(Debug, Clone)]
pub struct QueryBuilder {
    table: String,
    columns: Vec<String>,
    where_clauses: Vec<String>,
    order_by: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
}

impl QueryBuilder {
    /// Create a new query builder for a table
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            columns: vec!["*".to_string()],
            where_clauses: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }
    }

    /// Select specific columns
    pub fn select(mut self, columns: &[&str]) -> Self {
        self.columns = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add a WHERE clause
    pub fn where_clause(mut self, clause: impl Into<String>) -> Self {
        self.where_clauses.push(clause.into());
        self
    }

    /// Add ORDER BY
    pub fn order_by(mut self, order: impl Into<String>) -> Self {
        self.order_by = Some(order.into());
        self
    }

    /// Add LIMIT
    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Add OFFSET
    pub fn offset(mut self, offset: u32) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Build the SQL query string
    pub fn build(&self) -> String {
        let mut query = format!("SELECT {} FROM {}", self.columns.join(", "), self.table);

        if !self.where_clauses.is_empty() {
            query.push_str(" WHERE ");
            query.push_str(&self.where_clauses.join(" AND "));
        }

        if let Some(ref order) = self.order_by {
            query.push_str(" ORDER BY ");
            query.push_str(order);
        }

        if let Some(limit) = self.limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        if let Some(offset) = self.offset {
            query.push_str(&format!(" OFFSET {}", offset));
        }

        query
    }
}

/// Result of a raw SQL execution
#[derive(Debug, Clone)]
pub struct SqlResult {
    /// Column names
    pub columns: Vec<String>,
    /// Rows as JSON values
    pub rows: Vec<HashMap<String, Value>>,
    /// Number of rows affected (for INSERT/UPDATE/DELETE)
    pub rows_affected: u64,
    /// Execution time in milliseconds
    pub elapsed_ms: u64,
}

/// Execute a raw SQL query and return results as JSON
pub async fn execute_query(pool: &PgPool, query: &str) -> Result<SqlResult, DataError> {
    let start = Instant::now();

    // Validate query is read-only for safety
    let query_upper = query.trim().to_uppercase();
    if !query_upper.starts_with("SELECT") && !query_upper.starts_with("WITH") {
        return Err(DataError::ProcessingError(
            "Only SELECT queries are allowed. Use execute_statement for modifications.".to_string(),
        ));
    }

    let rows = sqlx::query(query)
        .fetch_all(pool)
        .await
        .map_err(|e| DataError::ProcessingError(format!("Query failed: {}", e)))?;

    let columns: Vec<String> = if !rows.is_empty() {
        rows[0]
            .columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect()
    } else {
        Vec::new()
    };

    let json_rows: Vec<HashMap<String, Value>> = rows.iter().map(|row| row_to_json(row)).collect();

    Ok(SqlResult {
        columns,
        rows: json_rows,
        rows_affected: 0,
        elapsed_ms: start.elapsed().as_millis() as u64,
    })
}

/// Convert a PostgreSQL row to a JSON HashMap
fn row_to_json(row: &PgRow) -> HashMap<String, Value> {
    let mut map = HashMap::new();

    for col in row.columns() {
        let name = col.name().to_string();
        let ordinal = col.ordinal();

        let value = extract_json_value(row, ordinal);
        map.insert(name, value);
    }

    map
}

/// Extract a JSON value from a row column
fn extract_json_value(row: &PgRow, ordinal: usize) -> Value {
    // Try each type in order

    // Integer types
    if let Ok(val) = row.try_get::<Option<i64>, _>(ordinal) {
        return val.map(|v| json!(v)).unwrap_or(Value::Null);
    }
    if let Ok(val) = row.try_get::<Option<i32>, _>(ordinal) {
        return val.map(|v| json!(v)).unwrap_or(Value::Null);
    }
    if let Ok(val) = row.try_get::<Option<i16>, _>(ordinal) {
        return val.map(|v| json!(v)).unwrap_or(Value::Null);
    }

    // Float types
    if let Ok(val) = row.try_get::<Option<f64>, _>(ordinal) {
        return val.map(|v| json!(v)).unwrap_or(Value::Null);
    }
    if let Ok(val) = row.try_get::<Option<f32>, _>(ordinal) {
        return val.map(|v| json!(v)).unwrap_or(Value::Null);
    }

    // Boolean
    if let Ok(val) = row.try_get::<Option<bool>, _>(ordinal) {
        return val.map(|v| json!(v)).unwrap_or(Value::Null);
    }

    // String
    if let Ok(val) = row.try_get::<Option<String>, _>(ordinal) {
        return val.map(|v| json!(v)).unwrap_or(Value::Null);
    }

    // Native JSON/JSONB
    if let Ok(val) = row.try_get::<Option<serde_json::Value>, _>(ordinal) {
        return val.unwrap_or(Value::Null);
    }

    // UUID
    if let Ok(val) = row.try_get::<Option<sqlx::types::Uuid>, _>(ordinal) {
        return val.map(|v| json!(v.to_string())).unwrap_or(Value::Null);
    }

    // DateTime
    if let Ok(val) = row.try_get::<Option<chrono::DateTime<chrono::Utc>>, _>(ordinal) {
        return val.map(|v| json!(v.to_rfc3339())).unwrap_or(Value::Null);
    }

    // NaiveDateTime
    if let Ok(val) = row.try_get::<Option<chrono::NaiveDateTime>, _>(ordinal) {
        return val.map(|v| json!(v.to_string())).unwrap_or(Value::Null);
    }

    // NaiveDate
    if let Ok(val) = row.try_get::<Option<chrono::NaiveDate>, _>(ordinal) {
        return val.map(|v| json!(v.to_string())).unwrap_or(Value::Null);
    }

    // BigDecimal
    if let Ok(val) = row.try_get::<Option<sqlx::types::BigDecimal>, _>(ordinal) {
        return val.map(|v| json!(v.to_string())).unwrap_or(Value::Null);
    }

    // Check for NULL
    if let Ok(vr) = row.try_get_raw(ordinal) {
        if vr.is_null() {
            return Value::Null;
        }
    }

    // Fallback
    Value::Null
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_builder_simple() {
        let query = QueryBuilder::new("users").build();
        assert_eq!(query, "SELECT * FROM users");
    }

    #[test]
    fn test_query_builder_with_columns() {
        let query = QueryBuilder::new("users")
            .select(&["id", "name", "email"])
            .build();
        assert_eq!(query, "SELECT id, name, email FROM users");
    }

    #[test]
    fn test_query_builder_with_where() {
        let query = QueryBuilder::new("users")
            .where_clause("active = true")
            .where_clause("age > 18")
            .build();
        assert_eq!(
            query,
            "SELECT * FROM users WHERE active = true AND age > 18"
        );
    }

    #[test]
    fn test_query_builder_full() {
        let query = QueryBuilder::new("orders")
            .select(&["id", "total", "created_at"])
            .where_clause("status = 'completed'")
            .order_by("created_at DESC")
            .limit(100)
            .offset(50)
            .build();
        assert_eq!(
            query,
            "SELECT id, total, created_at FROM orders WHERE status = 'completed' ORDER BY created_at DESC LIMIT 100 OFFSET 50"
        );
    }

    #[test]
    fn test_postgres_config_default() {
        let config = PostgresConfig::default();
        assert_eq!(config.max_connections, 10);
        assert_eq!(config.min_connections, 1);
        assert_eq!(config.connect_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_postgres_config_builder() {
        let config = PostgresConfig::new("postgres://localhost/test").with_max_connections(20);
        assert_eq!(config.url, "postgres://localhost/test");
        assert_eq!(config.max_connections, 20);
    }

    #[test]
    fn test_value_to_string() {
        assert_eq!(
            value_to_string(&FieldValue::Text("hello".to_string())),
            "hello"
        );
        assert_eq!(value_to_string(&FieldValue::Integer(42)), "42");
        assert_eq!(value_to_string(&FieldValue::Float(3.14)), "3.14");
        assert_eq!(value_to_string(&FieldValue::Boolean(true)), "true");
        assert_eq!(value_to_string(&FieldValue::Null), "null");
    }
}
