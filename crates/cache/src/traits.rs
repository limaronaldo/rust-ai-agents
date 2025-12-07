//! Cache trait definitions

use async_trait::async_trait;

use crate::{CacheEntry, CacheError, CacheStats};

/// Trait for cache implementations
#[async_trait]
pub trait Cache: Send + Sync {
    /// Get an entry from the cache
    async fn get(&self, query: &str, context: &str) -> Result<Option<CacheEntry>, CacheError>;

    /// Store an entry in the cache
    async fn store(
        &self,
        query: &str,
        context: &str,
        response: &str,
        function_calls: Vec<String>,
    ) -> Result<(), CacheError>;

    /// Delete an entry from the cache
    async fn delete(&self, query: &str, context: &str) -> Result<bool, CacheError>;

    /// Clear all entries
    async fn clear(&self) -> Result<usize, CacheError>;

    /// Get cache statistics
    async fn stats(&self) -> Result<CacheStats, CacheError>;

    /// Check if cache contains an entry
    async fn contains(&self, query: &str, context: &str) -> Result<bool, CacheError> {
        Ok(self.get(query, context).await?.is_some())
    }
}

/// Trait for semantic cache with similarity matching
///
/// Unlike regular caching, semantic caching finds similar queries
/// based on embedding vectors rather than exact string matches.
#[async_trait]
pub trait SemanticCache: Cache {
    /// Find similar entries based on embedding similarity
    ///
    /// Takes a pre-computed embedding vector and finds cached entries
    /// with embeddings above the similarity threshold.
    ///
    /// Returns the best matching entry and its similarity score (0.0 to 1.0).
    async fn find_similar_by_embedding(
        &self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Result<Option<(CacheEntry, f32)>, CacheError>;

    /// Store with embedding for semantic matching
    ///
    /// Stores the response along with its embedding vector for
    /// future similarity searches.
    async fn store_with_embedding(
        &self,
        query: &str,
        context: &str,
        response: &str,
        function_calls: Vec<String>,
        embedding: Vec<f32>,
    ) -> Result<(), CacheError>;
}
