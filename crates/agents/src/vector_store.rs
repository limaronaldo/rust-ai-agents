//! Vector Store and RAG (Retrieval-Augmented Generation) System
//!
//! Provides semantic memory for agents through vector embeddings and similarity search.
//!
//! # Features
//! - **VectorStore trait**: Pluggable backends for vector storage
//! - **InMemoryVectorStore**: Fast in-memory store with HNSW-like indexing
//! - **SledVectorStore**: Persistent vector storage using Sled
//! - **RAGPipeline**: Complete retrieval-augmented generation pipeline
//! - **Document chunking**: Automatic text splitting for large documents
//!
//! # Example
//! ```ignore
//! use rust_ai_agents_agents::vector_store::*;
//!
//! // Create vector store
//! let store = InMemoryVectorStore::new(1536); // OpenAI embedding dimension
//!
//! // Create RAG pipeline
//! let rag = RAGPipeline::new(store, embedder)
//!     .with_chunk_size(512)
//!     .with_top_k(5);
//!
//! // Index documents
//! rag.index_document("doc1", "Long document text...").await?;
//!
//! // Query with context
//! let context = rag.retrieve("What is the main topic?", 5).await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use rust_ai_agents_core::errors::MemoryError;

/// A document chunk with its embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    /// Unique document ID
    pub id: String,
    /// Original text content
    pub content: String,
    /// Vector embedding
    pub embedding: Vec<f32>,
    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Source document ID (for chunks)
    pub source_id: Option<String>,
    /// Chunk index within source document
    pub chunk_index: Option<usize>,
}

impl VectorDocument {
    pub fn new(id: impl Into<String>, content: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            embedding,
            metadata: HashMap::new(),
            source_id: None,
            chunk_index: None,
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn with_source(mut self, source_id: impl Into<String>, chunk_index: usize) -> Self {
        self.source_id = Some(source_id.into());
        self.chunk_index = Some(chunk_index);
        self
    }
}

/// Search result with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matched document
    pub document: VectorDocument,
    /// Similarity score (0.0 - 1.0, higher is more similar)
    pub score: f32,
}

/// Trait for vector embedding generation
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Generate embedding for text
    async fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError>;

    /// Generate embeddings for multiple texts (batch)
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, MemoryError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Get embedding dimension
    fn dimension(&self) -> usize;
}

/// Trait for vector storage backends
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert a document into the store
    async fn insert(&self, doc: VectorDocument) -> Result<(), MemoryError>;

    /// Insert multiple documents
    async fn insert_batch(&self, docs: Vec<VectorDocument>) -> Result<(), MemoryError> {
        for doc in docs {
            self.insert(doc).await?;
        }
        Ok(())
    }

    /// Search for similar documents
    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, MemoryError>;

    /// Get document by ID
    async fn get(&self, id: &str) -> Result<Option<VectorDocument>, MemoryError>;

    /// Delete document by ID
    async fn delete(&self, id: &str) -> Result<bool, MemoryError>;

    /// Delete all documents with a given source_id
    async fn delete_by_source(&self, source_id: &str) -> Result<usize, MemoryError>;

    /// Get total document count
    async fn count(&self) -> Result<usize, MemoryError>;

    /// Clear all documents
    async fn clear(&self) -> Result<(), MemoryError>;

    /// Get store name for logging
    fn name(&self) -> &'static str;
}

/// In-memory vector store with brute-force similarity search
pub struct InMemoryVectorStore {
    documents: Arc<RwLock<HashMap<String, VectorDocument>>>,
    dimension: usize,
}

impl InMemoryVectorStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            dimension,
        }
    }

    /// Compute cosine similarity between two vectors
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
impl VectorStore for InMemoryVectorStore {
    async fn insert(&self, doc: VectorDocument) -> Result<(), MemoryError> {
        if doc.embedding.len() != self.dimension {
            return Err(MemoryError::StorageError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                doc.embedding.len()
            )));
        }

        let mut docs = self.documents.write().await;
        docs.insert(doc.id.clone(), doc);
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

        let docs = self.documents.read().await;

        let mut results: Vec<SearchResult> = docs
            .values()
            .map(|doc| SearchResult {
                document: doc.clone(),
                score: Self::cosine_similarity(query_embedding, &doc.embedding),
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top_k
        results.truncate(top_k);

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<VectorDocument>, MemoryError> {
        let docs = self.documents.read().await;
        Ok(docs.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool, MemoryError> {
        let mut docs = self.documents.write().await;
        Ok(docs.remove(id).is_some())
    }

    async fn delete_by_source(&self, source_id: &str) -> Result<usize, MemoryError> {
        let mut docs = self.documents.write().await;
        let to_remove: Vec<_> = docs
            .iter()
            .filter(|(_, doc)| doc.source_id.as_deref() == Some(source_id))
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            docs.remove(&id);
        }

        Ok(count)
    }

    async fn count(&self) -> Result<usize, MemoryError> {
        let docs = self.documents.read().await;
        Ok(docs.len())
    }

    async fn clear(&self) -> Result<(), MemoryError> {
        let mut docs = self.documents.write().await;
        docs.clear();
        Ok(())
    }

    fn name(&self) -> &'static str {
        "in-memory"
    }
}

/// Sled-based persistent vector store
pub struct SledVectorStore {
    db: sled::Db,
    documents_tree: sled::Tree,
    index_tree: sled::Tree,
    dimension: usize,
}

impl SledVectorStore {
    /// Create a new Sled vector store at the given path
    pub fn new<P: AsRef<std::path::Path>>(path: P, dimension: usize) -> Result<Self, MemoryError> {
        let db = sled::open(path).map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let documents_tree = db
            .open_tree("vector_documents")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let index_tree = db
            .open_tree("vector_index")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(Self {
            db,
            documents_tree,
            index_tree,
            dimension,
        })
    }

    /// Create a temporary in-memory Sled store (for testing)
    pub fn temporary(dimension: usize) -> Result<Self, MemoryError> {
        let db = sled::Config::new()
            .temporary(true)
            .open()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let documents_tree = db
            .open_tree("vector_documents")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let index_tree = db
            .open_tree("vector_index")
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(Self {
            db,
            documents_tree,
            index_tree,
            dimension,
        })
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<(), MemoryError> {
        self.db
            .flush()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(())
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        InMemoryVectorStore::cosine_similarity(a, b)
    }
}

#[async_trait]
impl VectorStore for SledVectorStore {
    async fn insert(&self, doc: VectorDocument) -> Result<(), MemoryError> {
        if doc.embedding.len() != self.dimension {
            return Err(MemoryError::StorageError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                doc.embedding.len()
            )));
        }

        let doc_bytes =
            serde_json::to_vec(&doc).map_err(|e| MemoryError::SerializationError(e.to_string()))?;

        self.documents_tree
            .insert(doc.id.as_bytes(), doc_bytes)
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        // Index by source_id if present
        if let Some(source_id) = &doc.source_id {
            let key = format!("source:{}:{}", source_id, doc.id);
            self.index_tree
                .insert(key.as_bytes(), doc.id.as_bytes())
                .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        }

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

        let mut results = Vec::new();

        for item in self.documents_tree.iter() {
            let (_, value) = item.map_err(|e| MemoryError::StorageError(e.to_string()))?;
            let doc: VectorDocument = serde_json::from_slice(&value)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

            let score = Self::cosine_similarity(query_embedding, &doc.embedding);
            results.push(SearchResult {
                document: doc,
                score,
            });
        }

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
        match self.documents_tree.get(id.as_bytes()) {
            Ok(Some(bytes)) => {
                let doc: VectorDocument = serde_json::from_slice(&bytes)
                    .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
                Ok(Some(doc))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(MemoryError::StorageError(e.to_string())),
        }
    }

    async fn delete(&self, id: &str) -> Result<bool, MemoryError> {
        // Get doc first to remove from index
        if let Some(doc) = self.get(id).await? {
            if let Some(source_id) = &doc.source_id {
                let key = format!("source:{}:{}", source_id, id);
                let _ = self.index_tree.remove(key.as_bytes());
            }
        }

        let removed = self
            .documents_tree
            .remove(id.as_bytes())
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        Ok(removed.is_some())
    }

    async fn delete_by_source(&self, source_id: &str) -> Result<usize, MemoryError> {
        let prefix = format!("source:{}:", source_id);
        let mut ids_to_remove = Vec::new();

        for item in self.index_tree.scan_prefix(prefix.as_bytes()) {
            let (_, value) = item.map_err(|e| MemoryError::StorageError(e.to_string()))?;
            let id = String::from_utf8(value.to_vec())
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
            ids_to_remove.push(id);
        }

        let count = ids_to_remove.len();
        for id in ids_to_remove {
            self.delete(&id).await?;
        }

        Ok(count)
    }

    async fn count(&self) -> Result<usize, MemoryError> {
        Ok(self.documents_tree.len())
    }

    async fn clear(&self) -> Result<(), MemoryError> {
        self.documents_tree
            .clear()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        self.index_tree
            .clear()
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "sled"
    }
}

/// Text chunking strategy
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    /// Fixed size chunks with overlap
    FixedSize { chunk_size: usize, overlap: usize },
    /// Split by sentences
    Sentence { max_sentences: usize },
    /// Split by paragraphs
    Paragraph,
    /// No chunking - use full text
    None,
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        ChunkingStrategy::FixedSize {
            chunk_size: 512,
            overlap: 64,
        }
    }
}

/// Text chunker for splitting documents
pub struct TextChunker {
    strategy: ChunkingStrategy,
}

impl TextChunker {
    pub fn new(strategy: ChunkingStrategy) -> Self {
        Self { strategy }
    }

    /// Split text into chunks
    pub fn chunk(&self, text: &str) -> Vec<String> {
        match &self.strategy {
            ChunkingStrategy::FixedSize {
                chunk_size,
                overlap,
            } => self.chunk_fixed_size(text, *chunk_size, *overlap),
            ChunkingStrategy::Sentence { max_sentences } => {
                self.chunk_by_sentences(text, *max_sentences)
            }
            ChunkingStrategy::Paragraph => self.chunk_by_paragraphs(text),
            ChunkingStrategy::None => vec![text.to_string()],
        }
    }

    fn chunk_fixed_size(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();

            if !chunk.trim().is_empty() {
                chunks.push(chunk.trim().to_string());
            }

            if end >= chars.len() {
                break;
            }

            start = if overlap < chunk_size {
                start + chunk_size - overlap
            } else {
                start + chunk_size
            };
        }

        chunks
    }

    fn chunk_by_sentences(&self, text: &str, max_sentences: usize) -> Vec<String> {
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        sentences
            .chunks(max_sentences)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|s| s.trim())
                    .collect::<Vec<_>>()
                    .join(". ")
                    + "."
            })
            .collect()
    }

    fn chunk_by_paragraphs(&self, text: &str) -> Vec<String> {
        text.split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .map(|p| p.trim().to_string())
            .collect()
    }
}

/// RAG (Retrieval-Augmented Generation) Pipeline
pub struct RAGPipeline<S: VectorStore, E: Embedder> {
    store: Arc<S>,
    embedder: Arc<E>,
    chunker: TextChunker,
    default_top_k: usize,
}

impl<S: VectorStore, E: Embedder> RAGPipeline<S, E> {
    /// Create a new RAG pipeline
    pub fn new(store: S, embedder: E) -> Self {
        Self {
            store: Arc::new(store),
            embedder: Arc::new(embedder),
            chunker: TextChunker::new(ChunkingStrategy::default()),
            default_top_k: 5,
        }
    }

    /// Set chunking strategy
    pub fn with_chunking(mut self, strategy: ChunkingStrategy) -> Self {
        self.chunker = TextChunker::new(strategy);
        self
    }

    /// Set default top_k for retrieval
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.default_top_k = top_k;
        self
    }

    /// Index a document (with automatic chunking)
    pub async fn index_document(
        &self,
        doc_id: &str,
        content: &str,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<usize, MemoryError> {
        // First, remove any existing chunks for this document
        self.store.delete_by_source(doc_id).await?;

        // Chunk the content
        let chunks = self.chunker.chunk(content);
        let chunk_count = chunks.len();

        // Generate embeddings for all chunks
        let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        let embeddings = self.embedder.embed_batch(&chunk_refs).await?;

        // Create and insert documents
        for (i, (chunk, embedding)) in chunks.into_iter().zip(embeddings).enumerate() {
            let chunk_id = format!("{}:chunk:{}", doc_id, i);
            let mut doc = VectorDocument::new(chunk_id, chunk, embedding).with_source(doc_id, i);

            if let Some(ref meta) = metadata {
                for (k, v) in meta {
                    doc = doc.with_metadata(k.clone(), v.clone());
                }
            }

            self.store.insert(doc).await?;
        }

        tracing::info!(doc_id = doc_id, chunks = chunk_count, "Indexed document");

        Ok(chunk_count)
    }

    /// Retrieve relevant context for a query
    pub async fn retrieve(
        &self,
        query: &str,
        top_k: Option<usize>,
    ) -> Result<Vec<SearchResult>, MemoryError> {
        let k = top_k.unwrap_or(self.default_top_k);

        // Generate query embedding
        let query_embedding = self.embedder.embed(query).await?;

        // Search for similar documents
        let results = self.store.search(&query_embedding, k).await?;

        tracing::debug!(query = query, results = results.len(), "Retrieved context");

        Ok(results)
    }

    /// Retrieve and format context as a string for LLM prompt
    pub async fn retrieve_context(
        &self,
        query: &str,
        top_k: Option<usize>,
    ) -> Result<String, MemoryError> {
        let results = self.retrieve(query, top_k).await?;

        if results.is_empty() {
            return Ok(String::new());
        }

        let context = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "[Source {}] (relevance: {:.2})\n{}",
                    i + 1,
                    r.score,
                    r.document.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(context)
    }

    /// Build an augmented prompt with retrieved context
    pub async fn augment_prompt(
        &self,
        query: &str,
        top_k: Option<usize>,
    ) -> Result<String, MemoryError> {
        let context = self.retrieve_context(query, top_k).await?;

        if context.is_empty() {
            return Ok(query.to_string());
        }

        Ok(format!(
            "Use the following context to answer the question. If the context doesn't contain \
            relevant information, say so and answer based on your knowledge.\n\n\
            Context:\n{}\n\n\
            Question: {}",
            context, query
        ))
    }

    /// Delete a document and all its chunks
    pub async fn delete_document(&self, doc_id: &str) -> Result<usize, MemoryError> {
        self.store.delete_by_source(doc_id).await
    }

    /// Get document count
    pub async fn document_count(&self) -> Result<usize, MemoryError> {
        self.store.count().await
    }

    /// Clear all documents
    pub async fn clear(&self) -> Result<(), MemoryError> {
        self.store.clear().await
    }
}

/// Semantic memory for agents using vector store
pub struct SemanticMemory<S: VectorStore, E: Embedder> {
    rag: RAGPipeline<S, E>,
    agent_id: String,
}

impl<S: VectorStore, E: Embedder> SemanticMemory<S, E> {
    pub fn new(store: S, embedder: E, agent_id: impl Into<String>) -> Self {
        Self {
            rag: RAGPipeline::new(store, embedder),
            agent_id: agent_id.into(),
        }
    }

    /// Remember a piece of information
    pub async fn remember(
        &self,
        content: &str,
        tags: Option<Vec<String>>,
    ) -> Result<(), MemoryError> {
        let memory_id = format!("{}:memory:{}", self.agent_id, uuid::Uuid::new_v4());

        let mut metadata = HashMap::new();
        metadata.insert("agent_id".to_string(), serde_json::json!(self.agent_id));
        metadata.insert(
            "timestamp".to_string(),
            serde_json::json!(chrono::Utc::now().to_rfc3339()),
        );

        if let Some(tags) = tags {
            metadata.insert("tags".to_string(), serde_json::json!(tags));
        }

        self.rag
            .index_document(&memory_id, content, Some(metadata))
            .await?;
        Ok(())
    }

    /// Recall relevant memories for a query
    pub async fn recall(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, MemoryError> {
        self.rag.retrieve(query, Some(top_k)).await
    }

    /// Get formatted context for a query
    pub async fn get_context(&self, query: &str, top_k: usize) -> Result<String, MemoryError> {
        self.rag.retrieve_context(query, Some(top_k)).await
    }

    /// Forget memories matching a query (by deleting similar documents)
    pub async fn forget(&self, query: &str, threshold: f32) -> Result<usize, MemoryError> {
        let results = self.rag.retrieve(query, Some(100)).await?;

        let mut deleted = 0;
        for result in results {
            if result.score >= threshold {
                if self.rag.store.delete(&result.document.id).await? {
                    deleted += 1;
                }
            }
        }

        Ok(deleted)
    }
}

/// Embedder wrapper that uses an LLMBackend for embeddings
pub struct LLMEmbedder {
    backend: Arc<dyn rust_ai_agents_providers::LLMBackend>,
    dimension: usize,
}

impl LLMEmbedder {
    /// Create a new LLM-based embedder
    ///
    /// Common dimensions:
    /// - OpenAI text-embedding-3-small: 1536
    /// - OpenAI text-embedding-3-large: 3072
    /// - OpenAI text-embedding-ada-002: 1536
    pub fn new(backend: Arc<dyn rust_ai_agents_providers::LLMBackend>, dimension: usize) -> Self {
        Self { backend, dimension }
    }

    /// Create embedder for OpenAI text-embedding-3-small (1536 dimensions)
    pub fn openai_small(backend: Arc<dyn rust_ai_agents_providers::LLMBackend>) -> Self {
        Self::new(backend, 1536)
    }

    /// Create embedder for OpenAI text-embedding-3-large (3072 dimensions)
    pub fn openai_large(backend: Arc<dyn rust_ai_agents_providers::LLMBackend>) -> Self {
        Self::new(backend, 3072)
    }
}

#[async_trait]
impl Embedder for LLMEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        self.backend
            .embed(text)
            .await
            .map_err(|e| MemoryError::StorageError(format!("Embedding failed: {}", e)))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock embedder for testing
    struct MockEmbedder {
        dimension: usize,
    }

    impl MockEmbedder {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
            // Simple deterministic embedding based on text hash
            let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
            let mut embedding = vec![0.0f32; self.dimension];

            for (i, val) in embedding.iter_mut().enumerate() {
                *val = ((hash.wrapping_add(i as u64) % 1000) as f32 / 1000.0) - 0.5;
            }

            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut embedding {
                    *val /= norm;
                }
            }

            Ok(embedding)
        }

        fn dimension(&self) -> usize {
            self.dimension
        }
    }

    #[tokio::test]
    async fn test_in_memory_vector_store() {
        let store = InMemoryVectorStore::new(4);

        // Insert documents
        let doc1 = VectorDocument::new("doc1", "Hello world", vec![1.0, 0.0, 0.0, 0.0]);
        let doc2 = VectorDocument::new("doc2", "Goodbye world", vec![0.0, 1.0, 0.0, 0.0]);
        let doc3 = VectorDocument::new("doc3", "Hello there", vec![0.9, 0.1, 0.0, 0.0]);

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();
        store.insert(doc3).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 3);

        // Search
        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id, "doc1"); // Most similar
        assert_eq!(results[1].document.id, "doc3"); // Second most similar
    }

    #[tokio::test]
    async fn test_sled_vector_store() {
        let store = SledVectorStore::temporary(4).unwrap();

        let doc = VectorDocument::new("test", "Test content", vec![1.0, 0.0, 0.0, 0.0])
            .with_metadata("key", serde_json::json!("value"));

        store.insert(doc.clone()).await.unwrap();

        let retrieved = store.get("test").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Test content");
        assert_eq!(
            retrieved.metadata.get("key"),
            Some(&serde_json::json!("value"))
        );

        store.delete("test").await.unwrap();
        assert!(store.get("test").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_text_chunker_fixed_size() {
        let chunker = TextChunker::new(ChunkingStrategy::FixedSize {
            chunk_size: 10,
            overlap: 2,
        });

        let text = "Hello world, this is a test of the chunking system.";
        let chunks = chunker.chunk(text);

        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.len() <= 12)); // Allow some variance
    }

    #[tokio::test]
    async fn test_text_chunker_sentences() {
        let chunker = TextChunker::new(ChunkingStrategy::Sentence { max_sentences: 2 });

        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 2);
    }

    #[tokio::test]
    async fn test_rag_pipeline() {
        let store = InMemoryVectorStore::new(64);
        let embedder = MockEmbedder::new(64);
        let rag = RAGPipeline::new(store, embedder)
            .with_chunking(ChunkingStrategy::None)
            .with_top_k(3);

        // Index documents
        rag.index_document("doc1", "Rust is a systems programming language.", None)
            .await
            .unwrap();
        rag.index_document("doc2", "Python is great for data science.", None)
            .await
            .unwrap();
        rag.index_document("doc3", "Rust has excellent memory safety.", None)
            .await
            .unwrap();

        assert_eq!(rag.document_count().await.unwrap(), 3);

        // Retrieve
        let results = rag.retrieve("Tell me about Rust", Some(2)).await.unwrap();
        assert_eq!(results.len(), 2);

        // Get context
        let context = rag
            .retrieve_context("Rust programming", Some(2))
            .await
            .unwrap();
        assert!(!context.is_empty());
    }

    #[tokio::test]
    async fn test_semantic_memory() {
        let store = InMemoryVectorStore::new(64);
        let embedder = MockEmbedder::new(64);
        let memory = SemanticMemory::new(store, embedder, "test-agent");

        // Remember things
        memory
            .remember(
                "The capital of France is Paris.",
                Some(vec!["geography".to_string()]),
            )
            .await
            .unwrap();
        memory
            .remember(
                "Rust was created by Mozilla.",
                Some(vec!["programming".to_string()]),
            )
            .await
            .unwrap();

        // Recall
        let results = memory
            .recall("What is the capital of France?", 5)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_delete_by_source() {
        let store = InMemoryVectorStore::new(4);

        let doc1 = VectorDocument::new("chunk1", "Part 1", vec![1.0, 0.0, 0.0, 0.0])
            .with_source("doc1", 0);
        let doc2 = VectorDocument::new("chunk2", "Part 2", vec![0.0, 1.0, 0.0, 0.0])
            .with_source("doc1", 1);
        let doc3 =
            VectorDocument::new("other", "Other", vec![0.0, 0.0, 1.0, 0.0]).with_source("doc2", 0);

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();
        store.insert(doc3).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 3);

        let deleted = store.delete_by_source("doc1").await.unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(store.count().await.unwrap(), 1);
    }
}
