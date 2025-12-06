//! SQLite-based persistent storage for agents
//!
//! High-performance local storage using SQLite with WAL mode.
//! Perfect for autonomous agents that need embedded, zero-dependency persistence.
//!
//! # Features
//! - **WAL Mode**: Write-Ahead Logging for concurrent reads during writes
//! - **Session Storage**: Persistent agent sessions with resume tokens
//! - **Memory Storage**: Long-term semantic memory for agents
//! - **Vector Storage**: Optional vector embeddings storage
//! - **Zero Dependencies**: Single file database, no external services
//!
//! # Example
//! ```ignore
//! use rust_ai_agents_agents::sqlite_store::*;
//!
//! // Create SQLite store with WAL mode
//! let store = SqliteStore::new("agent_data.db").await?;
//!
//! // Use as session backend
//! let session_manager = SessionManager::new(Arc::new(store.session_backend()));
//! ```

use async_trait::async_trait;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use rust_ai_agents_core::errors::MemoryError;
use rust_ai_agents_core::Message;

use crate::persistence::{MemoryBackend, Session};
use crate::vector_store::{SearchResult, VectorDocument, VectorStore};

/// SQLite store configuration
#[derive(Debug, Clone)]
pub struct SqliteConfig {
    /// Database file path
    pub path: String,
    /// Enable WAL mode (recommended for performance)
    pub wal_mode: bool,
    /// Maximum connections in pool
    pub max_connections: u32,
    /// Busy timeout in seconds
    pub busy_timeout_secs: u64,
    /// Create database if missing
    pub create_if_missing: bool,
    /// Enable foreign keys
    pub foreign_keys: bool,
}

impl Default for SqliteConfig {
    fn default() -> Self {
        Self {
            path: "agent_data.db".to_string(),
            wal_mode: true,
            max_connections: 5,
            busy_timeout_secs: 5,
            create_if_missing: true,
            foreign_keys: true,
        }
    }
}

impl SqliteConfig {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }

    pub fn in_memory() -> Self {
        Self {
            path: ":memory:".to_string(),
            wal_mode: false, // WAL doesn't work with :memory:
            ..Default::default()
        }
    }
}

/// Main SQLite store for agent persistence
pub struct SqliteStore {
    pool: SqlitePool,
    #[allow(dead_code)]
    config: SqliteConfig,
}

impl SqliteStore {
    /// Create a new SQLite store with the given configuration
    pub async fn new(config: SqliteConfig) -> Result<Self, MemoryError> {
        let options = SqliteConnectOptions::from_str(&format!("sqlite://{}?mode=rwc", config.path))
            .map_err(|e| MemoryError::StorageError(e.to_string()))?
            .journal_mode(if config.wal_mode {
                SqliteJournalMode::Wal
            } else {
                SqliteJournalMode::Delete
            })
            .create_if_missing(config.create_if_missing)
            .foreign_keys(config.foreign_keys)
            .busy_timeout(std::time::Duration::from_secs(config.busy_timeout_secs));

        let pool = SqlitePoolOptions::new()
            .max_connections(config.max_connections)
            .connect_with(options)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let store = Self { pool, config };
        store.initialize_schema().await?;

        Ok(store)
    }

    /// Create a store with default configuration at the given path
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self, MemoryError> {
        let config = SqliteConfig::new(path.as_ref().to_string_lossy().to_string());
        Self::new(config).await
    }

    /// Create an in-memory store (for testing)
    pub async fn in_memory() -> Result<Self, MemoryError> {
        Self::new(SqliteConfig::in_memory()).await
    }

    /// Initialize database schema
    async fn initialize_schema(&self) -> Result<(), MemoryError> {
        // Sessions table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                messages TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                resume_token TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        // Resume tokens index
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_sessions_resume_token
            ON sessions(resume_token) WHERE resume_token IS NOT NULL
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        // Agent sessions index
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_sessions_agent_id
            ON sessions(agent_id)
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        // Vector documents table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS vector_documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT NOT NULL,
                source_id TEXT,
                chunk_index INTEGER
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        // Source index for vector documents
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_vector_docs_source
            ON vector_documents(source_id) WHERE source_id IS NOT NULL
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        // Agent memory table (for semantic memory)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS agent_memory (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                tags TEXT,
                created_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_agent_memory_agent
            ON agent_memory(agent_id)
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        tracing::info!("SQLite schema initialized");
        Ok(())
    }

    /// Get the connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Create a session backend from this store
    pub fn session_backend(self: &Arc<Self>) -> SqliteSessionBackend {
        SqliteSessionBackend {
            store: Arc::clone(self),
        }
    }

    /// Create a vector store from this store
    pub fn vector_store(self: &Arc<Self>, dimension: usize) -> SqliteVectorStore {
        SqliteVectorStore {
            store: Arc::clone(self),
            dimension,
        }
    }
}

/// SQLite-based session storage backend
pub struct SqliteSessionBackend {
    store: Arc<SqliteStore>,
}

#[async_trait]
impl MemoryBackend for SqliteSessionBackend {
    async fn save_session(&self, session: &Session) -> Result<(), MemoryError> {
        let messages_json = serde_json::to_string(&session.messages)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
        let metadata_json = serde_json::to_string(&session.metadata)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO sessions
            (id, agent_id, messages, metadata, created_at, updated_at, resume_token)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&session.id)
        .bind(&session.agent_id)
        .bind(&messages_json)
        .bind(&metadata_json)
        .bind(session.created_at)
        .bind(session.updated_at)
        .bind(&session.resume_token)
        .execute(&self.store.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(())
    }

    async fn load_session(&self, session_id: &str) -> Result<Option<Session>, MemoryError> {
        let row: Option<SessionRow> = sqlx::query_as(
            "SELECT id, agent_id, messages, metadata, created_at, updated_at, resume_token FROM sessions WHERE id = ?",
        )
        .bind(session_id)
        .fetch_optional(&self.store.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        row.map(|r| r.into_session()).transpose()
    }

    async fn load_by_resume_token(&self, token: &str) -> Result<Option<Session>, MemoryError> {
        let row: Option<SessionRow> = sqlx::query_as(
            "SELECT id, agent_id, messages, metadata, created_at, updated_at, resume_token FROM sessions WHERE resume_token = ?",
        )
        .bind(token)
        .fetch_optional(&self.store.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        row.map(|r| r.into_session()).transpose()
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        sqlx::query("DELETE FROM sessions WHERE id = ?")
            .bind(session_id)
            .execute(&self.store.pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(())
    }

    async fn list_sessions(&self, agent_id: &str) -> Result<Vec<Session>, MemoryError> {
        let rows: Vec<SessionRow> = sqlx::query_as(
            "SELECT id, agent_id, messages, metadata, created_at, updated_at, resume_token FROM sessions WHERE agent_id = ?",
        )
        .bind(agent_id)
        .fetch_all(&self.store.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        rows.into_iter().map(|r| r.into_session()).collect()
    }

    async fn list_session_ids(&self, agent_id: &str) -> Result<Vec<String>, MemoryError> {
        let rows: Vec<(String,)> = sqlx::query_as("SELECT id FROM sessions WHERE agent_id = ?")
            .bind(agent_id)
            .fetch_all(&self.store.pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(rows.into_iter().map(|(id,)| id).collect())
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool, MemoryError> {
        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM sessions WHERE id = ?")
            .bind(session_id)
            .fetch_one(&self.store.pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(count.0 > 0)
    }

    async fn clear_agent_sessions(&self, agent_id: &str) -> Result<usize, MemoryError> {
        let result = sqlx::query("DELETE FROM sessions WHERE agent_id = ?")
            .bind(agent_id)
            .execute(&self.store.pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(result.rows_affected() as usize)
    }

    fn backend_name(&self) -> &'static str {
        "sqlite"
    }
}

/// Internal row type for session queries
#[derive(sqlx::FromRow)]
struct SessionRow {
    id: String,
    agent_id: String,
    messages: String,
    metadata: String,
    created_at: i64,
    updated_at: i64,
    resume_token: Option<String>,
}

impl SessionRow {
    fn into_session(self) -> Result<Session, MemoryError> {
        let messages: Vec<Message> = serde_json::from_str(&self.messages)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
        let metadata: HashMap<String, serde_json::Value> = serde_json::from_str(&self.metadata)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

        Ok(Session {
            id: self.id,
            agent_id: self.agent_id,
            messages,
            metadata,
            created_at: self.created_at,
            updated_at: self.updated_at,
            resume_token: self.resume_token,
        })
    }
}

/// SQLite-based vector store
pub struct SqliteVectorStore {
    store: Arc<SqliteStore>,
    dimension: usize,
}

impl SqliteVectorStore {
    /// Serialize embedding to bytes
    fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
        embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Deserialize embedding from bytes
    fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(arr)
            })
            .collect()
    }

    /// Compute cosine similarity
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

#[async_trait]
impl VectorStore for SqliteVectorStore {
    async fn insert(&self, doc: VectorDocument) -> Result<(), MemoryError> {
        if doc.embedding.len() != self.dimension {
            return Err(MemoryError::StorageError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                doc.embedding.len()
            )));
        }

        let embedding_bytes = Self::embedding_to_bytes(&doc.embedding);
        let metadata_json = serde_json::to_string(&doc.metadata)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO vector_documents
            (id, content, embedding, metadata, source_id, chunk_index)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&doc.id)
        .bind(&doc.content)
        .bind(&embedding_bytes)
        .bind(&metadata_json)
        .bind(&doc.source_id)
        .bind(doc.chunk_index.map(|i| i as i64))
        .execute(&self.store.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(())
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, MemoryError> {
        if query_embedding.len() != self.dimension {
            return Err(MemoryError::StorageError(format!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                query_embedding.len()
            )));
        }

        // Fetch all documents (for small datasets; consider approximate search for large ones)
        let rows: Vec<VectorDocRow> = sqlx::query_as(
            "SELECT id, content, embedding, metadata, source_id, chunk_index FROM vector_documents",
        )
        .fetch_all(&self.store.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let mut results: Vec<SearchResult> = rows
            .into_iter()
            .filter_map(|row| {
                let doc = row.into_document().ok()?;
                let score = Self::cosine_similarity(query_embedding, &doc.embedding);
                Some(SearchResult {
                    document: doc,
                    score,
                })
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<VectorDocument>, MemoryError> {
        let row: Option<VectorDocRow> = sqlx::query_as(
            "SELECT id, content, embedding, metadata, source_id, chunk_index FROM vector_documents WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.store.pool)
        .await
        .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        row.map(|r| r.into_document()).transpose()
    }

    async fn delete(&self, id: &str) -> Result<bool, MemoryError> {
        let result = sqlx::query("DELETE FROM vector_documents WHERE id = ?")
            .bind(id)
            .execute(&self.store.pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }

    async fn delete_by_source(&self, source_id: &str) -> Result<usize, MemoryError> {
        let result = sqlx::query("DELETE FROM vector_documents WHERE source_id = ?")
            .bind(source_id)
            .execute(&self.store.pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(result.rows_affected() as usize)
    }

    async fn count(&self) -> Result<usize, MemoryError> {
        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM vector_documents")
            .fetch_one(&self.store.pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(count.0 as usize)
    }

    async fn clear(&self) -> Result<(), MemoryError> {
        sqlx::query("DELETE FROM vector_documents")
            .execute(&self.store.pool)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(())
    }

    fn name(&self) -> &'static str {
        "sqlite"
    }
}

/// Internal row type for vector document queries
#[derive(sqlx::FromRow)]
struct VectorDocRow {
    id: String,
    content: String,
    embedding: Vec<u8>,
    metadata: String,
    source_id: Option<String>,
    chunk_index: Option<i64>,
}

impl VectorDocRow {
    fn into_document(self) -> Result<VectorDocument, MemoryError> {
        let embedding = SqliteVectorStore::bytes_to_embedding(&self.embedding);
        let metadata: HashMap<String, serde_json::Value> = serde_json::from_str(&self.metadata)
            .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

        Ok(VectorDocument {
            id: self.id,
            content: self.content,
            embedding,
            metadata,
            source_id: self.source_id,
            chunk_index: self.chunk_index.map(|i| i as usize),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_store_creation() {
        let store = SqliteStore::in_memory().await.unwrap();
        assert!(store.pool().acquire().await.is_ok());
    }

    #[tokio::test]
    async fn test_sqlite_session_backend() {
        let store = Arc::new(SqliteStore::in_memory().await.unwrap());
        let backend = store.session_backend();

        // Create session
        let mut session = Session::new("test-agent");
        session.set_metadata("key", serde_json::json!("value"));

        // Save
        backend.save_session(&session).await.unwrap();

        // Load
        let loaded = backend.load_session(&session.id).await.unwrap().unwrap();
        assert_eq!(loaded.agent_id, "test-agent");
        assert_eq!(
            loaded.get_metadata("key"),
            Some(&serde_json::json!("value"))
        );

        // List
        let sessions = backend.list_sessions("test-agent").await.unwrap();
        assert_eq!(sessions.len(), 1);

        // Delete
        backend.delete_session(&session.id).await.unwrap();
        assert!(!backend.session_exists(&session.id).await.unwrap());
    }

    #[tokio::test]
    async fn test_sqlite_session_resume_token() {
        let store = Arc::new(SqliteStore::in_memory().await.unwrap());
        let backend = store.session_backend();

        let mut session = Session::new("test-agent");
        let token = session.generate_resume_token();
        backend.save_session(&session).await.unwrap();

        // Resume by token
        let resumed = backend.load_by_resume_token(&token).await.unwrap().unwrap();
        assert_eq!(resumed.id, session.id);
    }

    #[tokio::test]
    async fn test_sqlite_vector_store() {
        let store = Arc::new(SqliteStore::in_memory().await.unwrap());
        let vector_store = store.vector_store(4);

        // Insert documents
        let doc1 = VectorDocument::new("doc1", "Hello world", vec![1.0, 0.0, 0.0, 0.0]);
        let doc2 = VectorDocument::new("doc2", "Goodbye", vec![0.0, 1.0, 0.0, 0.0]);

        vector_store.insert(doc1).await.unwrap();
        vector_store.insert(doc2).await.unwrap();

        assert_eq!(vector_store.count().await.unwrap(), 2);

        // Search
        let results = vector_store.search(&[1.0, 0.0, 0.0, 0.0], 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "doc1");

        // Delete
        vector_store.delete("doc1").await.unwrap();
        assert_eq!(vector_store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_sqlite_vector_store_by_source() {
        let store = Arc::new(SqliteStore::in_memory().await.unwrap());
        let vector_store = store.vector_store(4);

        let doc1 = VectorDocument::new("chunk1", "Part 1", vec![1.0, 0.0, 0.0, 0.0])
            .with_source("doc1", 0);
        let doc2 = VectorDocument::new("chunk2", "Part 2", vec![0.0, 1.0, 0.0, 0.0])
            .with_source("doc1", 1);
        let doc3 =
            VectorDocument::new("other", "Other", vec![0.0, 0.0, 1.0, 0.0]).with_source("doc2", 0);

        vector_store.insert(doc1).await.unwrap();
        vector_store.insert(doc2).await.unwrap();
        vector_store.insert(doc3).await.unwrap();

        // Delete by source
        let deleted = vector_store.delete_by_source("doc1").await.unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(vector_store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_sqlite_clear_agent_sessions() {
        let store = Arc::new(SqliteStore::in_memory().await.unwrap());
        let backend = store.session_backend();

        // Create multiple sessions
        let session1 = Session::new("agent-1");
        let session2 = Session::new("agent-1");
        let session3 = Session::new("agent-2");

        backend.save_session(&session1).await.unwrap();
        backend.save_session(&session2).await.unwrap();
        backend.save_session(&session3).await.unwrap();

        // Clear agent-1
        let cleared = backend.clear_agent_sessions("agent-1").await.unwrap();
        assert_eq!(cleared, 2);

        // Verify
        let remaining = backend.list_sessions("agent-1").await.unwrap();
        assert_eq!(remaining.len(), 0);

        let agent2_sessions = backend.list_sessions("agent-2").await.unwrap();
        assert_eq!(agent2_sessions.len(), 1);
    }
}
