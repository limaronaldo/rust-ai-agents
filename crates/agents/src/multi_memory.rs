//! # Multi-Memory System
//!
//! Hierarchical memory system with short-term, long-term, and entity memory.
//!
//! Inspired by CrewAI's memory architecture.
//!
//! ## Memory Types
//!
//! - **Short-term Memory**: Recent conversation context, auto-expires
//! - **Long-term Memory**: Persistent facts and learnings, vector-searchable
//! - **Entity Memory**: Knowledge about specific entities (people, places, concepts)
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents::multi_memory::{MultiMemory, MemoryConfig};
//!
//! let memory = MultiMemory::new(MemoryConfig::default());
//!
//! // Store in short-term (recent context)
//! memory.short_term.add("User asked about Rust").await?;
//!
//! // Store in long-term (persistent knowledge)
//! memory.long_term.store("Rust is a systems programming language", embedding).await?;
//!
//! // Store entity information
//! memory.entity.update("Rust", "category", "programming_language").await?;
//! memory.entity.update("Rust", "creator", "Mozilla").await?;
//!
//! // Query across all memory types
//! let context = memory.recall("Tell me about Rust", query_embedding).await?;
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Configuration for the multi-memory system
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum items in short-term memory
    pub short_term_capacity: usize,
    /// TTL for short-term memories
    pub short_term_ttl: Duration,
    /// Maximum items in long-term memory
    pub long_term_capacity: usize,
    /// Similarity threshold for long-term recall (0.0 to 1.0)
    pub similarity_threshold: f32,
    /// Maximum entities to track
    pub entity_capacity: usize,
    /// Maximum attributes per entity
    pub max_attributes_per_entity: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            short_term_capacity: 100,
            short_term_ttl: Duration::from_secs(3600), // 1 hour
            long_term_capacity: 10000,
            similarity_threshold: 0.7,
            entity_capacity: 1000,
            max_attributes_per_entity: 50,
        }
    }
}

/// A memory item with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Unique ID
    pub id: String,
    /// The content
    pub content: String,
    /// When it was created
    pub created_at: u64,
    /// When it was last accessed
    pub last_accessed: u64,
    /// Access count
    pub access_count: u32,
    /// Optional importance score (0.0 to 1.0)
    pub importance: f32,
    /// Source of the memory
    pub source: MemorySource,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl MemoryItem {
    pub fn new(content: impl Into<String>, source: MemorySource) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance: 0.5,
            source,
            metadata: HashMap::new(),
        }
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.access_count += 1;
    }
}

/// Source of a memory
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemorySource {
    /// From user input
    User,
    /// From agent response
    Agent,
    /// From tool execution
    Tool,
    /// From external knowledge base
    External,
    /// System-generated
    System,
}

// ============================================================================
// Short-Term Memory
// ============================================================================

/// Short-term memory for recent conversation context
pub struct ShortTermMemory {
    items: RwLock<VecDeque<(MemoryItem, Instant)>>,
    capacity: usize,
    ttl: Duration,
}

impl ShortTermMemory {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            items: RwLock::new(VecDeque::with_capacity(capacity)),
            capacity,
            ttl,
        }
    }

    /// Add a new memory item
    pub fn add(&self, content: impl Into<String>, source: MemorySource) {
        let item = MemoryItem::new(content, source);
        self.add_item(item);
    }

    /// Add a memory item
    pub fn add_item(&self, item: MemoryItem) {
        let mut items = self.items.write();

        // Remove expired items
        let now = Instant::now();
        while let Some((_, created)) = items.front() {
            if now.duration_since(*created) > self.ttl {
                items.pop_front();
            } else {
                break;
            }
        }

        // Enforce capacity
        while items.len() >= self.capacity {
            items.pop_front();
        }

        items.push_back((item, now));
    }

    /// Get all current memories (not expired)
    pub fn get_all(&self) -> Vec<MemoryItem> {
        let items = self.items.read();
        let now = Instant::now();

        items
            .iter()
            .filter(|(_, created)| now.duration_since(*created) <= self.ttl)
            .map(|(item, _)| item.clone())
            .collect()
    }

    /// Get the N most recent memories
    pub fn get_recent(&self, n: usize) -> Vec<MemoryItem> {
        let items = self.items.read();
        let now = Instant::now();

        items
            .iter()
            .rev()
            .filter(|(_, created)| now.duration_since(*created) <= self.ttl)
            .take(n)
            .map(|(item, _)| item.clone())
            .collect()
    }

    /// Search for memories containing text
    pub fn search(&self, query: &str) -> Vec<MemoryItem> {
        let query_lower = query.to_lowercase();
        let items = self.items.read();
        let now = Instant::now();

        items
            .iter()
            .filter(|(_, created)| now.duration_since(*created) <= self.ttl)
            .filter(|(item, _)| item.content.to_lowercase().contains(&query_lower))
            .map(|(item, _)| item.clone())
            .collect()
    }

    /// Clear all memories
    pub fn clear(&self) {
        self.items.write().clear();
    }

    /// Get current count of memories
    pub fn len(&self) -> usize {
        let items = self.items.read();
        let now = Instant::now();
        items
            .iter()
            .filter(|(_, created)| now.duration_since(*created) <= self.ttl)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get as formatted context string
    pub fn as_context(&self, max_items: usize) -> String {
        let items = self.get_recent(max_items);
        items
            .iter()
            .map(|item| format!("[{}] {}", format_source(&item.source), item.content))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn format_source(source: &MemorySource) -> &'static str {
    match source {
        MemorySource::User => "User",
        MemorySource::Agent => "Agent",
        MemorySource::Tool => "Tool",
        MemorySource::External => "External",
        MemorySource::System => "System",
    }
}

// ============================================================================
// Long-Term Memory
// ============================================================================

/// Entry in long-term memory with embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTermEntry {
    pub item: MemoryItem,
    pub embedding: Vec<f32>,
}

/// Long-term memory with semantic search
pub struct LongTermMemory {
    entries: RwLock<Vec<LongTermEntry>>,
    capacity: usize,
    similarity_threshold: f32,
}

impl LongTermMemory {
    pub fn new(capacity: usize, similarity_threshold: f32) -> Self {
        Self {
            entries: RwLock::new(Vec::with_capacity(capacity)),
            capacity,
            similarity_threshold,
        }
    }

    /// Store a memory with its embedding
    pub fn store(
        &self,
        content: impl Into<String>,
        embedding: Vec<f32>,
        source: MemorySource,
    ) -> String {
        let item = MemoryItem::new(content, source);
        self.store_item(item, embedding)
    }

    /// Store a memory item with embedding
    pub fn store_item(&self, item: MemoryItem, embedding: Vec<f32>) -> String {
        let id = item.id.clone();
        let entry = LongTermEntry { item, embedding };

        let mut entries = self.entries.write();

        // Enforce capacity - remove oldest by access time
        while entries.len() >= self.capacity {
            if let Some(idx) = entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.item.last_accessed)
                .map(|(idx, _)| idx)
            {
                entries.remove(idx);
            }
        }

        entries.push(entry);
        debug!(id = %id, "Long-term memory stored");
        id
    }

    /// Search for similar memories using embedding
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Vec<(MemoryItem, f32)> {
        let mut entries = self.entries.write();

        let mut results: Vec<_> = entries
            .iter_mut()
            .map(|entry| {
                let similarity = cosine_similarity(&entry.embedding, query_embedding);
                (entry, similarity)
            })
            .filter(|(_, sim)| *sim >= self.similarity_threshold)
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N and update access info
        results
            .into_iter()
            .take(limit)
            .map(|(entry, sim)| {
                entry.item.touch();
                (entry.item.clone(), sim)
            })
            .collect()
    }

    /// Get all entries (for export/debugging)
    pub fn get_all(&self) -> Vec<LongTermEntry> {
        self.entries.read().clone()
    }

    /// Remove a specific memory
    pub fn remove(&self, id: &str) -> bool {
        let mut entries = self.entries.write();
        let initial_len = entries.len();
        entries.retain(|e| e.item.id != id);
        entries.len() < initial_len
    }

    /// Clear all memories
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Get memories by importance threshold
    pub fn get_important(&self, min_importance: f32) -> Vec<MemoryItem> {
        self.entries
            .read()
            .iter()
            .filter(|e| e.item.importance >= min_importance)
            .map(|e| e.item.clone())
            .collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}

// ============================================================================
// Entity Memory
// ============================================================================

/// An entity with attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity name/identifier
    pub name: String,
    /// Entity type (person, place, concept, etc.)
    pub entity_type: String,
    /// Attributes as key-value pairs
    pub attributes: HashMap<String, String>,
    /// When the entity was first seen
    pub created_at: u64,
    /// When the entity was last updated
    pub updated_at: u64,
    /// Confidence in the entity's existence (0.0 to 1.0)
    pub confidence: f32,
    /// Related entity names
    pub relations: Vec<(String, String)>, // (relation_type, target_entity)
}

impl Entity {
    pub fn new(name: impl Into<String>, entity_type: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            name: name.into(),
            entity_type: entity_type.into(),
            attributes: HashMap::new(),
            created_at: now,
            updated_at: now,
            confidence: 1.0,
            relations: Vec::new(),
        }
    }

    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
        self.touch();
    }

    pub fn get_attribute(&self, key: &str) -> Option<&String> {
        self.attributes.get(key)
    }

    pub fn add_relation(&mut self, relation_type: impl Into<String>, target: impl Into<String>) {
        self.relations.push((relation_type.into(), target.into()));
        self.touch();
    }

    fn touch(&mut self) {
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

/// Entity memory for tracking knowledge about specific entities
pub struct EntityMemory {
    entities: RwLock<HashMap<String, Entity>>,
    capacity: usize,
    max_attributes: usize,
}

impl EntityMemory {
    pub fn new(capacity: usize, max_attributes: usize) -> Self {
        Self {
            entities: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            max_attributes,
        }
    }

    /// Get or create an entity
    pub fn get_or_create(&self, name: &str, entity_type: &str) -> Entity {
        let mut entities = self.entities.write();

        if let Some(entity) = entities.get(name) {
            return entity.clone();
        }

        // Enforce capacity
        while entities.len() >= self.capacity {
            // Remove least recently updated entity
            if let Some(oldest) = entities
                .iter()
                .min_by_key(|(_, e)| e.updated_at)
                .map(|(k, _)| k.clone())
            {
                entities.remove(&oldest);
            }
        }

        let entity = Entity::new(name, entity_type);
        entities.insert(name.to_string(), entity.clone());
        entity
    }

    /// Update an entity's attribute
    pub fn update_attribute(&self, name: &str, key: &str, value: &str) -> bool {
        let mut entities = self.entities.write();

        if let Some(entity) = entities.get_mut(name) {
            if entity.attributes.len() < self.max_attributes || entity.attributes.contains_key(key)
            {
                entity.set_attribute(key, value);
                return true;
            }
        }
        false
    }

    /// Get an entity by name
    pub fn get(&self, name: &str) -> Option<Entity> {
        self.entities.read().get(name).cloned()
    }

    /// Check if an entity exists
    pub fn exists(&self, name: &str) -> bool {
        self.entities.read().contains_key(name)
    }

    /// Add a relation between entities
    pub fn add_relation(&self, source: &str, relation_type: &str, target: &str) -> bool {
        let mut entities = self.entities.write();

        if let Some(entity) = entities.get_mut(source) {
            entity.add_relation(relation_type, target);
            return true;
        }
        false
    }

    /// Get all entities of a specific type
    pub fn get_by_type(&self, entity_type: &str) -> Vec<Entity> {
        self.entities
            .read()
            .values()
            .filter(|e| e.entity_type == entity_type)
            .cloned()
            .collect()
    }

    /// Search entities by attribute
    pub fn search_by_attribute(&self, key: &str, value: &str) -> Vec<Entity> {
        self.entities
            .read()
            .values()
            .filter(|e| e.attributes.get(key).map(|v| v == value).unwrap_or(false))
            .cloned()
            .collect()
    }

    /// Get all entities
    pub fn get_all(&self) -> Vec<Entity> {
        self.entities.read().values().cloned().collect()
    }

    /// Remove an entity
    pub fn remove(&self, name: &str) -> bool {
        self.entities.write().remove(name).is_some()
    }

    /// Clear all entities
    pub fn clear(&self) {
        self.entities.write().clear();
    }

    pub fn len(&self) -> usize {
        self.entities.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.read().is_empty()
    }

    /// Get entity as context string
    pub fn entity_context(&self, name: &str) -> Option<String> {
        self.get(name).map(|entity| {
            let mut lines = vec![format!("{} ({})", entity.name, entity.entity_type)];

            for (key, value) in &entity.attributes {
                lines.push(format!("  - {}: {}", key, value));
            }

            for (relation, target) in &entity.relations {
                lines.push(format!("  - {} -> {}", relation, target));
            }

            lines.join("\n")
        })
    }
}

// ============================================================================
// Multi-Memory System
// ============================================================================

/// Unified multi-memory system combining all memory types
pub struct MultiMemory {
    /// Short-term memory for recent context
    pub short_term: Arc<ShortTermMemory>,
    /// Long-term memory for persistent knowledge
    pub long_term: Arc<LongTermMemory>,
    /// Entity memory for structured knowledge
    pub entity: Arc<EntityMemory>,
    /// Configuration
    #[allow(dead_code)]
    config: MemoryConfig,
}

impl MultiMemory {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            short_term: Arc::new(ShortTermMemory::new(
                config.short_term_capacity,
                config.short_term_ttl,
            )),
            long_term: Arc::new(LongTermMemory::new(
                config.long_term_capacity,
                config.similarity_threshold,
            )),
            entity: Arc::new(EntityMemory::new(
                config.entity_capacity,
                config.max_attributes_per_entity,
            )),
            config,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(MemoryConfig::default())
    }

    /// Recall relevant memories for a query
    pub fn recall(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
    ) -> RecallResult {
        let mut result = RecallResult::default();

        // Search short-term memory by text
        result.short_term = self.short_term.search(query);

        // Search long-term memory by embedding if available
        if let Some(embedding) = query_embedding {
            result.long_term = self.long_term.search(embedding, limit);
        }

        // Extract potential entity names from query (simple word extraction)
        let words: Vec<&str> = query.split_whitespace().collect();
        for word in words {
            if let Some(entity) = self.entity.get(word) {
                result.entities.push(entity);
            }
        }

        result
    }

    /// Build context string from memories
    pub fn build_context(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        max_short_term: usize,
        max_long_term: usize,
    ) -> String {
        let mut sections = Vec::new();

        // Recent context
        let recent = self.short_term.get_recent(max_short_term);
        if !recent.is_empty() {
            let context = recent
                .iter()
                .map(|item| format!("- {}", item.content))
                .collect::<Vec<_>>()
                .join("\n");
            sections.push(format!("## Recent Context\n{}", context));
        }

        // Relevant long-term memories
        if let Some(embedding) = query_embedding {
            let long_term = self.long_term.search(embedding, max_long_term);
            if !long_term.is_empty() {
                let context = long_term
                    .iter()
                    .map(|(item, score)| format!("- {} (relevance: {:.2})", item.content, score))
                    .collect::<Vec<_>>()
                    .join("\n");
                sections.push(format!("## Relevant Knowledge\n{}", context));
            }
        }

        // Related entities
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut entity_contexts = Vec::new();
        for word in words {
            if let Some(ctx) = self.entity.entity_context(word) {
                entity_contexts.push(ctx);
            }
        }
        if !entity_contexts.is_empty() {
            sections.push(format!(
                "## Known Entities\n{}",
                entity_contexts.join("\n\n")
            ));
        }

        sections.join("\n\n")
    }

    /// Clear all memories
    pub fn clear_all(&self) {
        self.short_term.clear();
        self.long_term.clear();
        self.entity.clear();
        info!("All memories cleared");
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            short_term_count: self.short_term.len(),
            long_term_count: self.long_term.len(),
            entity_count: self.entity.len(),
        }
    }
}

/// Result of a memory recall
#[derive(Debug, Default)]
pub struct RecallResult {
    /// Memories from short-term
    pub short_term: Vec<MemoryItem>,
    /// Memories from long-term with similarity scores
    pub long_term: Vec<(MemoryItem, f32)>,
    /// Related entities
    pub entities: Vec<Entity>,
}

impl RecallResult {
    pub fn is_empty(&self) -> bool {
        self.short_term.is_empty() && self.long_term.is_empty() && self.entities.is_empty()
    }

    pub fn total_count(&self) -> usize {
        self.short_term.len() + self.long_term.len() + self.entities.len()
    }
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub short_term_count: usize,
    pub long_term_count: usize,
    pub entity_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_term_memory_add_and_retrieve() {
        let memory = ShortTermMemory::new(10, Duration::from_secs(3600));

        memory.add("First message", MemorySource::User);
        memory.add("Second message", MemorySource::Agent);

        let all = memory.get_all();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_short_term_memory_capacity() {
        let memory = ShortTermMemory::new(3, Duration::from_secs(3600));

        memory.add("1", MemorySource::User);
        memory.add("2", MemorySource::User);
        memory.add("3", MemorySource::User);
        memory.add("4", MemorySource::User);

        let all = memory.get_all();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].content, "2");
    }

    #[test]
    fn test_short_term_memory_search() {
        let memory = ShortTermMemory::new(10, Duration::from_secs(3600));

        memory.add("Hello world", MemorySource::User);
        memory.add("Goodbye world", MemorySource::User);
        memory.add("Something else", MemorySource::User);

        let results = memory.search("world");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_long_term_memory_store_and_search() {
        let memory = LongTermMemory::new(100, 0.5);

        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.0, 1.0, 0.0];
        let embedding3 = vec![0.9, 0.1, 0.0]; // Similar to 1

        memory.store("First fact", embedding1.clone(), MemorySource::External);
        memory.store("Second fact", embedding2, MemorySource::External);
        memory.store("Third fact", embedding3, MemorySource::External);

        let results = memory.search(&embedding1, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].1 > results[1].1); // First should be more similar
    }

    #[test]
    fn test_long_term_memory_threshold() {
        let memory = LongTermMemory::new(100, 0.9); // High threshold

        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.0, 1.0, 0.0]; // Orthogonal, similarity = 0

        memory.store("First fact", embedding1.clone(), MemorySource::External);
        memory.store("Second fact", embedding2, MemorySource::External);

        let results = memory.search(&embedding1, 10);
        assert_eq!(results.len(), 1); // Only exact match passes threshold
    }

    #[test]
    fn test_entity_memory_create_and_update() {
        let memory = EntityMemory::new(100, 50);

        let entity = memory.get_or_create("Rust", "programming_language");
        assert_eq!(entity.name, "Rust");

        memory.update_attribute("Rust", "creator", "Mozilla");
        memory.update_attribute("Rust", "year", "2010");

        let updated = memory.get("Rust").unwrap();
        assert_eq!(updated.attributes.get("creator").unwrap(), "Mozilla");
        assert_eq!(updated.attributes.get("year").unwrap(), "2010");
    }

    #[test]
    fn test_entity_memory_relations() {
        let memory = EntityMemory::new(100, 50);

        memory.get_or_create("Rust", "programming_language");
        memory.get_or_create("Mozilla", "organization");

        memory.add_relation("Rust", "created_by", "Mozilla");

        let rust = memory.get("Rust").unwrap();
        assert_eq!(rust.relations.len(), 1);
        assert_eq!(
            rust.relations[0],
            ("created_by".to_string(), "Mozilla".to_string())
        );
    }

    #[test]
    fn test_entity_memory_search() {
        let memory = EntityMemory::new(100, 50);

        memory.get_or_create("Rust", "programming_language");
        memory.get_or_create("Python", "programming_language");
        memory.get_or_create("Mozilla", "organization");

        memory.update_attribute("Rust", "type", "compiled");
        memory.update_attribute("Python", "type", "interpreted");

        let langs = memory.get_by_type("programming_language");
        assert_eq!(langs.len(), 2);

        let compiled = memory.search_by_attribute("type", "compiled");
        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].name, "Rust");
    }

    #[test]
    fn test_multi_memory_recall() {
        let memory = MultiMemory::default_config();

        // Add short-term memories - query must be substring of content
        memory
            .short_term
            .add("User asked about Rust programming", MemorySource::User);

        // Add long-term memories
        let embedding = vec![1.0, 0.0, 0.0];
        memory.long_term.store(
            "Rust is a systems programming language",
            embedding.clone(),
            MemorySource::External,
        );

        // Add entity
        memory.entity.get_or_create("Rust", "programming_language");
        memory.entity.update_attribute("Rust", "creator", "Mozilla");

        // Recall - use "Rust" as query since it's the entity name
        let result = memory.recall("Rust", Some(&embedding), 10);

        assert_eq!(result.short_term.len(), 1); // Contains "Rust"
        assert_eq!(result.long_term.len(), 1); // Matches embedding
        assert_eq!(result.entities.len(), 1); // "Rust" entity found by exact name
    }

    #[test]
    fn test_multi_memory_build_context() {
        let memory = MultiMemory::default_config();

        memory
            .short_term
            .add("Previous message", MemorySource::User);

        let embedding = vec![1.0, 0.0, 0.0];
        memory
            .long_term
            .store("Relevant fact", embedding.clone(), MemorySource::External);

        memory.entity.get_or_create("Test", "concept");
        memory
            .entity
            .update_attribute("Test", "description", "A test entity");

        let context = memory.build_context("Test query", Some(&embedding), 5, 5);

        assert!(context.contains("Previous message"));
        assert!(context.contains("Relevant fact"));
        assert!(context.contains("Test"));
    }

    #[test]
    fn test_memory_stats() {
        let memory = MultiMemory::default_config();

        memory.short_term.add("Message 1", MemorySource::User);
        memory.short_term.add("Message 2", MemorySource::User);

        memory
            .long_term
            .store("Fact 1", vec![1.0], MemorySource::External);

        memory.entity.get_or_create("Entity1", "type");
        memory.entity.get_or_create("Entity2", "type");
        memory.entity.get_or_create("Entity3", "type");

        let stats = memory.stats();
        assert_eq!(stats.short_term_count, 2);
        assert_eq!(stats.long_term_count, 1);
        assert_eq!(stats.entity_count, 3);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.001);

        // Orthogonal vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 0.001);

        // Opposite vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_item_builder() {
        let item = MemoryItem::new("Test content", MemorySource::User)
            .with_importance(0.8)
            .with_metadata("key", "value");

        assert_eq!(item.content, "Test content");
        assert_eq!(item.importance, 0.8);
        assert_eq!(item.metadata.get("key").unwrap(), "value");
    }
}
