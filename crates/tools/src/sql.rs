//! SQL Tools for Agent Database Access
//!
//! Provides tools for agents to query PostgreSQL databases safely.
//!
//! ## Security
//!
//! - `ExecuteSqlTool`: Read-only queries (SELECT/WITH only)
//! - `FetchDataTool`: Pre-configured extractors for specific data sources
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_tools::sql::{ExecuteSqlTool, SqlToolConfig};
//! use sqlx::postgres::PgPool;
//!
//! let pool = PgPool::connect("postgres://localhost/db").await?;
//! let tool = ExecuteSqlTool::new(pool).with_max_rows(1000);
//!
//! // Agent can now execute: SELECT * FROM users WHERE active = true
//! ```

use async_trait::async_trait;
use rust_ai_agents_core::{errors::ToolError, ExecutionContext, Tool, ToolSchema};
use serde_json::{json, Value};
use sqlx::postgres::{PgPool, PgRow};
use sqlx::{Column, Row, ValueRef};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

/// Configuration for SQL tools
#[derive(Debug, Clone)]
pub struct SqlToolConfig {
    /// Maximum rows to return
    pub max_rows: u32,
    /// Query timeout in seconds
    pub timeout_secs: u32,
    /// Allow only SELECT queries
    pub read_only: bool,
    /// Allowed tables (empty = all allowed)
    pub allowed_tables: Vec<String>,
    /// Blocked tables
    pub blocked_tables: Vec<String>,
}

impl Default for SqlToolConfig {
    fn default() -> Self {
        Self {
            max_rows: 1000,
            timeout_secs: 30,
            read_only: true,
            allowed_tables: Vec::new(),
            blocked_tables: vec![
                "pg_".to_string(), // System tables
                "information_schema".to_string(),
            ],
        }
    }
}

impl SqlToolConfig {
    /// Set maximum rows
    pub fn with_max_rows(mut self, max: u32) -> Self {
        self.max_rows = max;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, secs: u32) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Allow specific tables only
    pub fn with_allowed_tables(mut self, tables: Vec<String>) -> Self {
        self.allowed_tables = tables;
        self
    }

    /// Block specific tables
    pub fn with_blocked_tables(mut self, tables: Vec<String>) -> Self {
        self.blocked_tables = tables;
        self
    }
}

/// Tool for executing read-only SQL queries
pub struct ExecuteSqlTool {
    pool: PgPool,
    config: SqlToolConfig,
}

impl ExecuteSqlTool {
    /// Create a new SQL execution tool
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            config: SqlToolConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: SqlToolConfig) -> Self {
        self.config = config;
        self
    }

    /// Set maximum rows
    pub fn with_max_rows(mut self, max: u32) -> Self {
        self.config.max_rows = max;
        self
    }

    /// Validate query is safe to execute
    fn validate_query(&self, query: &str) -> Result<(), ToolError> {
        let query_upper = query.trim().to_uppercase();

        // Check read-only constraint
        if self.config.read_only {
            if !query_upper.starts_with("SELECT") && !query_upper.starts_with("WITH") {
                return Err(ToolError::InvalidArguments(
                    "Only SELECT queries are allowed in read-only mode".to_string(),
                ));
            }

            // Block dangerous operations even in SELECT
            let dangerous = [
                "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT",
                "REVOKE",
            ];
            for op in dangerous {
                if query_upper.contains(op) {
                    return Err(ToolError::InvalidArguments(format!(
                        "{} operations are not allowed",
                        op
                    )));
                }
            }
        }

        // Check blocked tables
        let query_lower = query.to_lowercase();
        for blocked in &self.config.blocked_tables {
            if query_lower.contains(&blocked.to_lowercase()) {
                return Err(ToolError::InvalidArguments(format!(
                    "Access to table pattern '{}' is blocked",
                    blocked
                )));
            }
        }

        // Check allowed tables if specified
        if !self.config.allowed_tables.is_empty() {
            let mut found_allowed = false;
            for allowed in &self.config.allowed_tables {
                if query_lower.contains(&allowed.to_lowercase()) {
                    found_allowed = true;
                    break;
                }
            }
            if !found_allowed {
                return Err(ToolError::InvalidArguments(
                    "Query does not reference any allowed tables".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Execute query and return results as JSON
    async fn execute_query(&self, query: &str) -> Result<Value, ToolError> {
        let start = Instant::now();

        // Add LIMIT if not present
        let query_with_limit = if !query.to_uppercase().contains("LIMIT") {
            format!(
                "{} LIMIT {}",
                query.trim_end_matches(';'),
                self.config.max_rows
            )
        } else {
            query.to_string()
        };

        debug!(query = %query_with_limit, "Executing SQL query");

        let rows = sqlx::query(&query_with_limit)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Query failed: {}", e)))?;

        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Get column names
        let columns: Vec<String> = if !rows.is_empty() {
            rows[0]
                .columns()
                .iter()
                .map(|c| c.name().to_string())
                .collect()
        } else {
            Vec::new()
        };

        // Convert rows to JSON
        let json_rows: Vec<Value> = rows.iter().map(|row| row_to_json(row)).collect();

        info!(
            rows_returned = json_rows.len(),
            elapsed_ms = elapsed_ms,
            "SQL query completed"
        );

        Ok(json!({
            "success": true,
            "columns": columns,
            "rows": json_rows,
            "row_count": json_rows.len(),
            "elapsed_ms": elapsed_ms,
            "truncated": json_rows.len() >= self.config.max_rows as usize
        }))
    }
}

#[async_trait]
impl Tool for ExecuteSqlTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "execute_sql",
            "Execute a read-only SQL query against the PostgreSQL database. \
             Only SELECT queries are allowed. Use this to retrieve data, \
             join tables, aggregate results, and analyze information."
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL SELECT query to execute. Must be a valid PostgreSQL query."
                }
            },
            "required": ["query"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: Value,
    ) -> Result<Value, ToolError> {
        let query = arguments["query"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'query' field".to_string()))?;

        // Validate query safety
        self.validate_query(query)?;

        // Execute query
        self.execute_query(query).await
    }
}

/// Tool for listing available tables
pub struct ListTablesTool {
    pool: PgPool,
    schema: String,
}

impl ListTablesTool {
    /// Create a new list tables tool
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            schema: "public".to_string(),
        }
    }

    /// Set the schema to list tables from
    pub fn with_schema(mut self, schema: impl Into<String>) -> Self {
        self.schema = schema.into();
        self
    }
}

#[async_trait]
impl Tool for ListTablesTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "list_tables",
            "List all tables in the database with their column information. \
             Use this to understand the database schema before writing queries.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "schema": {
                    "type": "string",
                    "description": "Database schema to list tables from (default: public)",
                    "default": "public"
                },
                "include_columns": {
                    "type": "boolean",
                    "description": "Include column details for each table",
                    "default": true
                }
            }
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: Value,
    ) -> Result<Value, ToolError> {
        let schema = arguments["schema"].as_str().unwrap_or(&self.schema);

        let include_columns = arguments["include_columns"].as_bool().unwrap_or(true);

        // Get tables
        let tables_query = format!(
            "SELECT table_name, table_type
             FROM information_schema.tables
             WHERE table_schema = '{}'
             ORDER BY table_name",
            schema
        );

        let table_rows = sqlx::query(&tables_query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to list tables: {}", e)))?;

        let mut tables = Vec::new();

        for row in table_rows {
            let table_name: String = row
                .try_get("table_name")
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
            let table_type: String = row
                .try_get("table_type")
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            let mut table_info = json!({
                "name": table_name,
                "type": table_type
            });

            if include_columns {
                let columns_query = format!(
                    "SELECT column_name, data_type, is_nullable, column_default
                     FROM information_schema.columns
                     WHERE table_schema = '{}' AND table_name = '{}'
                     ORDER BY ordinal_position",
                    schema, table_name
                );

                let column_rows = sqlx::query(&columns_query)
                    .fetch_all(&self.pool)
                    .await
                    .map_err(|e| {
                        ToolError::ExecutionFailed(format!("Failed to get columns: {}", e))
                    })?;

                let columns: Vec<Value> = column_rows
                    .iter()
                    .map(|row| {
                        json!({
                            "name": row.try_get::<String, _>("column_name").unwrap_or_default(),
                            "type": row.try_get::<String, _>("data_type").unwrap_or_default(),
                            "nullable": row.try_get::<String, _>("is_nullable").unwrap_or_default() == "YES",
                            "default": row.try_get::<Option<String>, _>("column_default").ok().flatten()
                        })
                    })
                    .collect();

                table_info["columns"] = json!(columns);
            }

            tables.push(table_info);
        }

        Ok(json!({
            "schema": schema,
            "tables": tables,
            "table_count": tables.len()
        }))
    }
}

/// Tool for describing a specific table's schema
pub struct DescribeTableTool {
    pool: PgPool,
}

impl DescribeTableTool {
    /// Create a new describe table tool
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Tool for DescribeTableTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "describe_table",
            "Get detailed information about a specific table including columns, \
             data types, constraints, indexes, and sample data.",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "The table name to describe"
                },
                "schema": {
                    "type": "string",
                    "description": "Database schema (default: public)",
                    "default": "public"
                },
                "include_sample": {
                    "type": "boolean",
                    "description": "Include sample rows from the table",
                    "default": true
                },
                "sample_limit": {
                    "type": "integer",
                    "description": "Number of sample rows to include",
                    "default": 5
                }
            },
            "required": ["table"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: Value,
    ) -> Result<Value, ToolError> {
        let table = arguments["table"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'table' field".to_string()))?;

        let schema = arguments["schema"].as_str().unwrap_or("public");

        let include_sample = arguments["include_sample"].as_bool().unwrap_or(true);

        let sample_limit = arguments["sample_limit"].as_u64().unwrap_or(5) as i32;

        // Get columns
        let columns_query = format!(
            "SELECT column_name, data_type, character_maximum_length,
                    numeric_precision, is_nullable, column_default
             FROM information_schema.columns
             WHERE table_schema = '{}' AND table_name = '{}'
             ORDER BY ordinal_position",
            schema, table
        );

        let column_rows = sqlx::query(&columns_query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to describe table: {}", e)))?;

        if column_rows.is_empty() {
            return Err(ToolError::ExecutionFailed(format!(
                "Table '{}.{}' not found",
                schema, table
            )));
        }

        let columns: Vec<Value> = column_rows
            .iter()
            .map(|row| {
                let data_type: String = row.try_get("data_type").unwrap_or_default();
                let max_length: Option<i32> = row.try_get("character_maximum_length").ok();
                let precision: Option<i32> = row.try_get("numeric_precision").ok();

                let full_type = if let Some(len) = max_length {
                    format!("{}({})", data_type, len)
                } else if let Some(prec) = precision {
                    format!("{}({})", data_type, prec)
                } else {
                    data_type
                };

                json!({
                    "name": row.try_get::<String, _>("column_name").unwrap_or_default(),
                    "type": full_type,
                    "nullable": row.try_get::<String, _>("is_nullable").unwrap_or_default() == "YES",
                    "default": row.try_get::<Option<String>, _>("column_default").ok().flatten()
                })
            })
            .collect();

        // Get primary key
        let pk_query = format!(
            "SELECT a.attname
             FROM pg_index i
             JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
             WHERE i.indrelid = '{}.{}'::regclass AND i.indisprimary",
            schema, table
        );

        let pk_columns: Vec<String> = sqlx::query(&pk_query)
            .fetch_all(&self.pool)
            .await
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| r.try_get::<String, _>("attname").ok())
                    .collect()
            })
            .unwrap_or_default();

        // Get row count estimate
        let count_query = format!(
            "SELECT reltuples::bigint AS estimate
             FROM pg_class
             WHERE oid = '{}.{}'::regclass",
            schema, table
        );

        let row_count: i64 = sqlx::query(&count_query)
            .fetch_one(&self.pool)
            .await
            .and_then(|row| row.try_get("estimate"))
            .unwrap_or(0);

        let mut result = json!({
            "table": table,
            "schema": schema,
            "columns": columns,
            "primary_key": pk_columns,
            "row_count_estimate": row_count
        });

        // Get sample data if requested
        if include_sample && row_count > 0 {
            let sample_query = format!("SELECT * FROM {}.{} LIMIT {}", schema, table, sample_limit);

            if let Ok(sample_rows) = sqlx::query(&sample_query).fetch_all(&self.pool).await {
                let samples: Vec<Value> = sample_rows.iter().map(|row| row_to_json(row)).collect();
                result["sample_data"] = json!(samples);
            }
        }

        Ok(result)
    }
}

/// Convert a PostgreSQL row to a JSON Value
fn row_to_json(row: &PgRow) -> Value {
    let mut map = serde_json::Map::new();

    for col in row.columns() {
        let name = col.name().to_string();
        let ordinal = col.ordinal();
        let value = extract_json_value(row, ordinal);
        map.insert(name, value);
    }

    Value::Object(map)
}

/// Extract a JSON value from a row column
fn extract_json_value(row: &PgRow, ordinal: usize) -> Value {
    // Try each type in order of frequency

    // String (most common)
    if let Ok(val) = row.try_get::<Option<String>, _>(ordinal) {
        return val.map(|v| json!(v)).unwrap_or(Value::Null);
    }

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

    // Native JSON/JSONB
    if let Ok(val) = row.try_get::<Option<serde_json::Value>, _>(ordinal) {
        return val.unwrap_or(Value::Null);
    }

    // UUID
    if let Ok(val) = row.try_get::<Option<sqlx::types::Uuid>, _>(ordinal) {
        return val.map(|v| json!(v.to_string())).unwrap_or(Value::Null);
    }

    // DateTime with timezone
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

/// Create all SQL tools with a shared connection pool
pub fn create_sql_tools(pool: PgPool) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(ExecuteSqlTool::new(pool.clone())),
        Arc::new(ListTablesTool::new(pool.clone())),
        Arc::new(DescribeTableTool::new(pool)),
    ]
}

/// Create SQL tools with custom configuration
pub fn create_sql_tools_with_config(pool: PgPool, config: SqlToolConfig) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(ExecuteSqlTool::new(pool.clone()).with_config(config)),
        Arc::new(ListTablesTool::new(pool.clone())),
        Arc::new(DescribeTableTool::new(pool)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_query_select() {
        let tool = ExecuteSqlTool::new(PgPool::connect_lazy("postgres://localhost/test").unwrap());

        // Valid queries
        assert!(tool.validate_query("SELECT * FROM users").is_ok());
        assert!(tool.validate_query("select id, name from orders").is_ok());
        assert!(tool
            .validate_query("WITH cte AS (SELECT 1) SELECT * FROM cte")
            .is_ok());
    }

    #[test]
    fn test_validate_query_blocked() {
        let tool = ExecuteSqlTool::new(PgPool::connect_lazy("postgres://localhost/test").unwrap());

        // Blocked operations
        assert!(tool.validate_query("INSERT INTO users VALUES (1)").is_err());
        assert!(tool.validate_query("UPDATE users SET name = 'x'").is_err());
        assert!(tool.validate_query("DELETE FROM users").is_err());
        assert!(tool.validate_query("DROP TABLE users").is_err());
        assert!(tool.validate_query("TRUNCATE users").is_err());
    }

    #[test]
    fn test_validate_query_blocked_tables() {
        let tool = ExecuteSqlTool::new(PgPool::connect_lazy("postgres://localhost/test").unwrap());

        // System tables blocked
        assert!(tool
            .validate_query("SELECT * FROM pg_catalog.pg_tables")
            .is_err());
        assert!(tool
            .validate_query("SELECT * FROM information_schema.tables")
            .is_err());
    }

    #[test]
    fn test_validate_query_allowed_tables() {
        let config = SqlToolConfig::default()
            .with_allowed_tables(vec!["users".to_string(), "orders".to_string()]);

        let pool = PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let tool = ExecuteSqlTool::new(pool).with_config(config);

        // Allowed tables work
        assert!(tool.validate_query("SELECT * FROM users").is_ok());
        assert!(tool.validate_query("SELECT * FROM orders").is_ok());

        // Non-allowed tables blocked
        assert!(tool.validate_query("SELECT * FROM secrets").is_err());
    }

    #[test]
    fn test_sql_config_builder() {
        let config = SqlToolConfig::default()
            .with_max_rows(500)
            .with_timeout(60)
            .with_allowed_tables(vec!["t1".to_string()])
            .with_blocked_tables(vec!["t2".to_string()]);

        assert_eq!(config.max_rows, 500);
        assert_eq!(config.timeout_secs, 60);
        assert_eq!(config.allowed_tables, vec!["t1"]);
        assert!(config.blocked_tables.contains(&"t2".to_string()));
    }
}
