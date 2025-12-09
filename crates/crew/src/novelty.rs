//! # Answer Novelty Detection
//!
//! MassGen-inspired novelty system ensuring agents provide diverse,
//! non-redundant answers to improve overall result quality.
//!
//! ## Features
//!
//! - **Similarity Detection**: Compare answers for semantic similarity
//! - **Novelty Scoring**: Rate how novel an answer is compared to existing ones
//! - **Diversity Enforcement**: Reject or flag redundant answers
//! - **Topic Clustering**: Group similar answers by topic
//! - **Coverage Tracking**: Ensure all aspects of a query are covered
//!
//! ## Example
//!
//! ```ignore
//! use rust_ai_agents_crew::novelty::*;
//!
//! let detector = NoveltyDetector::new(NoveltyConfig::default());
//!
//! // Add first answer
//! detector.add_answer("agent1", "The property is worth $500,000").await;
//!
//! // Check novelty of second answer
//! let result = detector.check_novelty(
//!     "agent2",
//!     "I estimate the value at $500K"
//! ).await;
//!
//! println!("Novelty score: {}", result.novelty_score);
//! println!("Is novel: {}", result.is_novel);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Result of novelty check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyResult {
    /// Novelty score (0.0 = duplicate, 1.0 = completely novel)
    pub novelty_score: f32,
    /// Whether the answer meets novelty threshold
    pub is_novel: bool,
    /// Most similar existing answer
    pub most_similar: Option<SimilarAnswer>,
    /// Topics covered by this answer
    pub topics_covered: Vec<String>,
    /// New topics introduced
    pub new_topics: Vec<String>,
    /// Redundant topics (already well-covered)
    pub redundant_topics: Vec<String>,
    /// Suggestions for improving novelty
    pub suggestions: Vec<String>,
}

/// Information about a similar existing answer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarAnswer {
    /// Agent that provided the similar answer
    pub agent_id: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// The similar content
    pub content: String,
    /// Specific overlapping elements
    pub overlapping_elements: Vec<String>,
}

/// An answer stored for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredAnswer {
    /// Agent ID
    pub agent_id: String,
    /// Original content
    pub content: String,
    /// Normalized tokens for comparison
    pub tokens: Vec<String>,
    /// Key phrases extracted
    pub key_phrases: Vec<String>,
    /// Numeric values found
    pub numeric_values: Vec<f64>,
    /// Topics covered
    pub topics: Vec<String>,
    /// Timestamp when added
    pub added_at: u64,
}

/// Topic coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicCoverage {
    /// Topic name
    pub topic: String,
    /// Number of answers covering this topic
    pub coverage_count: usize,
    /// Agents that covered this topic
    pub covered_by: Vec<String>,
    /// Quality of coverage (0.0 to 1.0)
    pub coverage_quality: f32,
}

/// Configuration for novelty detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyConfig {
    /// Minimum novelty score to accept (0.0 to 1.0)
    pub novelty_threshold: f32,
    /// Similarity threshold for considering answers similar
    pub similarity_threshold: f32,
    /// Minimum topics for a comprehensive answer
    pub min_topics: usize,
    /// Whether to extract numeric values for comparison
    pub compare_numeric_values: bool,
    /// Tolerance for numeric value comparison (percentage)
    pub numeric_tolerance: f32,
    /// Common words to ignore in comparison
    pub stop_words: Vec<String>,
    /// Domain-specific synonyms
    pub synonyms: HashMap<String, Vec<String>>,
}

impl Default for NoveltyConfig {
    fn default() -> Self {
        Self {
            novelty_threshold: 0.3,
            similarity_threshold: 0.7,
            min_topics: 2,
            compare_numeric_values: true,
            numeric_tolerance: 0.05, // 5% tolerance
            stop_words: default_stop_words(),
            synonyms: HashMap::new(),
        }
    }
}

impl NoveltyConfig {
    /// Create config for strict novelty requirements
    pub fn strict() -> Self {
        Self {
            novelty_threshold: 0.5,
            similarity_threshold: 0.5,
            min_topics: 3,
            ..Default::default()
        }
    }

    /// Create config for lenient novelty requirements
    pub fn lenient() -> Self {
        Self {
            novelty_threshold: 0.2,
            similarity_threshold: 0.8,
            min_topics: 1,
            ..Default::default()
        }
    }

    /// Add domain-specific synonyms
    pub fn with_synonyms(mut self, word: impl Into<String>, synonyms: Vec<String>) -> Self {
        self.synonyms.insert(word.into(), synonyms);
        self
    }
}

/// Default stop words for English
fn default_stop_words() -> Vec<String> {
    vec![
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall",
        "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "and", "but", "if", "or", "because", "until", "while",
        "this", "that", "these", "those", "it", "its",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Novelty detector for comparing answers
pub struct NoveltyDetector {
    config: NoveltyConfig,
    answers: Arc<RwLock<Vec<StoredAnswer>>>,
    topic_coverage: Arc<RwLock<HashMap<String, TopicCoverage>>>,
}

impl NoveltyDetector {
    /// Create a new novelty detector
    pub fn new(config: NoveltyConfig) -> Self {
        Self {
            config,
            answers: Arc::new(RwLock::new(Vec::new())),
            topic_coverage: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add an answer to the comparison pool
    pub async fn add_answer(&self, agent_id: impl Into<String>, content: impl Into<String>) {
        let agent_id = agent_id.into();
        let content = content.into();

        let tokens = self.tokenize(&content);
        let key_phrases = self.extract_key_phrases(&content);
        let numeric_values = self.extract_numeric_values(&content);
        let topics = self.extract_topics(&content);

        let stored = StoredAnswer {
            agent_id: agent_id.clone(),
            content,
            tokens,
            key_phrases,
            numeric_values,
            topics: topics.clone(),
            added_at: current_timestamp(),
        };

        self.answers.write().await.push(stored);

        // Update topic coverage
        let mut coverage = self.topic_coverage.write().await;
        for topic in topics {
            let entry = coverage.entry(topic.clone()).or_insert(TopicCoverage {
                topic,
                coverage_count: 0,
                covered_by: Vec::new(),
                coverage_quality: 0.0,
            });
            entry.coverage_count += 1;
            entry.covered_by.push(agent_id.clone());
        }
    }

    /// Check novelty of a new answer
    pub async fn check_novelty(
        &self,
        _agent_id: impl Into<String>,
        content: impl Into<String>,
    ) -> NoveltyResult {
        let content = content.into();

        let tokens = self.tokenize(&content);
        let key_phrases = self.extract_key_phrases(&content);
        let numeric_values = self.extract_numeric_values(&content);
        let topics = self.extract_topics(&content);

        let answers = self.answers.read().await;

        if answers.is_empty() {
            // First answer is always novel
            return NoveltyResult {
                novelty_score: 1.0,
                is_novel: true,
                most_similar: None,
                topics_covered: topics.clone(),
                new_topics: topics,
                redundant_topics: Vec::new(),
                suggestions: Vec::new(),
            };
        }

        // Find most similar answer
        let mut most_similar: Option<(f32, &StoredAnswer)> = None;
        let mut max_similarity = 0.0f32;

        for answer in answers.iter() {
            let similarity = self.calculate_similarity(
                &tokens,
                &key_phrases,
                &numeric_values,
                &answer.tokens,
                &answer.key_phrases,
                &answer.numeric_values,
            );

            if similarity > max_similarity {
                max_similarity = similarity;
                most_similar = Some((similarity, answer));
            }
        }

        // Calculate novelty score (inverse of max similarity)
        let novelty_score = 1.0 - max_similarity;
        let is_novel = novelty_score >= self.config.novelty_threshold;

        // Find new and redundant topics
        let coverage = self.topic_coverage.read().await;
        let mut new_topics = Vec::new();
        let mut redundant_topics = Vec::new();

        for topic in &topics {
            if let Some(cov) = coverage.get(topic) {
                if cov.coverage_count >= 2 {
                    redundant_topics.push(topic.clone());
                }
            } else {
                new_topics.push(topic.clone());
            }
        }

        // Generate suggestions
        let suggestions = self.generate_suggestions(
            novelty_score,
            &new_topics,
            &redundant_topics,
            most_similar.as_ref().map(|(_, a)| *a),
        );

        let similar_answer = most_similar.map(|(sim, answer)| {
            let overlapping = self.find_overlapping_elements(&tokens, &answer.tokens);
            SimilarAnswer {
                agent_id: answer.agent_id.clone(),
                similarity: sim,
                content: answer.content.clone(),
                overlapping_elements: overlapping,
            }
        });

        NoveltyResult {
            novelty_score,
            is_novel,
            most_similar: similar_answer,
            topics_covered: topics,
            new_topics,
            redundant_topics,
            suggestions,
        }
    }

    /// Check and add answer (combines check_novelty and add_answer)
    pub async fn check_and_add(
        &self,
        agent_id: impl Into<String>,
        content: impl Into<String>,
    ) -> NoveltyResult {
        let agent_id = agent_id.into();
        let content = content.into();

        let result = self.check_novelty(&agent_id, &content).await;

        if result.is_novel {
            self.add_answer(&agent_id, &content).await;
        }

        result
    }

    /// Get current topic coverage
    pub async fn get_topic_coverage(&self) -> Vec<TopicCoverage> {
        self.topic_coverage.read().await.values().cloned().collect()
    }

    /// Get topics that need more coverage
    pub async fn get_uncovered_topics(&self, all_expected_topics: &[String]) -> Vec<String> {
        let coverage = self.topic_coverage.read().await;
        all_expected_topics
            .iter()
            .filter(|t| !coverage.contains_key(*t))
            .cloned()
            .collect()
    }

    /// Get diversity score for all answers
    pub async fn get_diversity_score(&self) -> f32 {
        let answers = self.answers.read().await;

        if answers.len() <= 1 {
            return 1.0;
        }

        let mut total_novelty = 0.0;
        let mut count = 0;

        for i in 0..answers.len() {
            for j in (i + 1)..answers.len() {
                let similarity = self.calculate_similarity(
                    &answers[i].tokens,
                    &answers[i].key_phrases,
                    &answers[i].numeric_values,
                    &answers[j].tokens,
                    &answers[j].key_phrases,
                    &answers[j].numeric_values,
                );
                total_novelty += 1.0 - similarity;
                count += 1;
            }
        }

        if count == 0 {
            1.0
        } else {
            total_novelty / count as f32
        }
    }

    /// Clear all stored answers
    pub async fn clear(&self) {
        self.answers.write().await.clear();
        self.topic_coverage.write().await.clear();
    }

    // Internal methods

    fn tokenize(&self, content: &str) -> Vec<String> {
        content
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .filter(|s| !self.config.stop_words.contains(&s.to_string()))
            .map(|s| self.normalize_synonym(s))
            .collect()
    }

    fn normalize_synonym(&self, word: &str) -> String {
        // Check if this word is a synonym of something
        for (canonical, synonyms) in &self.config.synonyms {
            if synonyms.contains(&word.to_string()) || canonical == word {
                return canonical.clone();
            }
        }
        word.to_string()
    }

    fn extract_key_phrases(&self, content: &str) -> Vec<String> {
        // Simple key phrase extraction: 2-3 word sequences
        let words: Vec<&str> = content
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .collect();

        let mut phrases = Vec::new();

        // Bigrams
        for window in words.windows(2) {
            let phrase = window.join(" ").to_lowercase();
            if !phrase
                .split_whitespace()
                .all(|w| self.config.stop_words.contains(&w.to_string()))
            {
                phrases.push(phrase);
            }
        }

        // Trigrams
        for window in words.windows(3) {
            let phrase = window.join(" ").to_lowercase();
            if !phrase
                .split_whitespace()
                .all(|w| self.config.stop_words.contains(&w.to_string()))
            {
                phrases.push(phrase);
            }
        }

        phrases
    }

    fn extract_numeric_values(&self, content: &str) -> Vec<f64> {
        if !self.config.compare_numeric_values {
            return Vec::new();
        }

        let mut values = Vec::new();
        let re_patterns = [
            r"\$[\d,]+(?:\.\d+)?",               // Currency: $500,000.00
            r"[\d,]+(?:\.\d+)?%",                // Percentage: 15.5%
            r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", // Numbers: 500,000.50
        ];

        for pattern in re_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.find_iter(content) {
                    let num_str: String = cap
                        .as_str()
                        .chars()
                        .filter(|c| c.is_ascii_digit() || *c == '.')
                        .collect();
                    if let Ok(num) = num_str.parse::<f64>() {
                        values.push(num);
                    }
                }
            }
        }

        values
    }

    fn extract_topics(&self, content: &str) -> Vec<String> {
        // Simple topic extraction based on key nouns
        let topic_indicators = [
            (
                "price",
                vec!["price", "cost", "value", "worth", "estimate", "valuation"],
            ),
            (
                "location",
                vec!["location", "address", "area", "neighborhood", "region"],
            ),
            (
                "size",
                vec![
                    "size", "square", "meters", "feet", "area", "rooms", "bedrooms",
                ],
            ),
            (
                "condition",
                vec!["condition", "quality", "renovated", "new", "old"],
            ),
            (
                "amenities",
                vec!["amenities", "pool", "garage", "garden", "balcony"],
            ),
            (
                "market",
                vec!["market", "trend", "demand", "supply", "comparable"],
            ),
        ];

        let content_lower = content.to_lowercase();
        let mut topics = Vec::new();

        for (topic, keywords) in topic_indicators {
            if keywords.iter().any(|k| content_lower.contains(k)) {
                topics.push(topic.to_string());
            }
        }

        topics
    }

    fn calculate_similarity(
        &self,
        tokens1: &[String],
        phrases1: &[String],
        nums1: &[f64],
        tokens2: &[String],
        phrases2: &[String],
        nums2: &[f64],
    ) -> f32 {
        let token_sim = self.jaccard_similarity(tokens1, tokens2);
        let phrase_sim = self.jaccard_similarity(phrases1, phrases2);
        let num_sim = self.numeric_similarity(nums1, nums2);

        // Weighted average
        let weights =
            if self.config.compare_numeric_values && !nums1.is_empty() && !nums2.is_empty() {
                (0.4, 0.3, 0.3) // tokens, phrases, numbers
            } else {
                (0.6, 0.4, 0.0) // tokens, phrases, no numbers
            };

        token_sim * weights.0 + phrase_sim * weights.1 + num_sim * weights.2
    }

    fn jaccard_similarity(&self, set1: &[String], set2: &[String]) -> f32 {
        if set1.is_empty() && set2.is_empty() {
            return 1.0;
        }
        if set1.is_empty() || set2.is_empty() {
            return 0.0;
        }

        let set1: std::collections::HashSet<_> = set1.iter().collect();
        let set2: std::collections::HashSet<_> = set2.iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        intersection as f32 / union as f32
    }

    fn numeric_similarity(&self, nums1: &[f64], nums2: &[f64]) -> f32 {
        if nums1.is_empty() || nums2.is_empty() {
            return 0.0;
        }

        let mut matches = 0;
        let total = nums1.len().max(nums2.len());

        for n1 in nums1 {
            for n2 in nums2 {
                let diff = (n1 - n2).abs() / n1.abs().max(1.0);
                if diff <= self.config.numeric_tolerance as f64 {
                    matches += 1;
                    break;
                }
            }
        }

        matches as f32 / total as f32
    }

    fn find_overlapping_elements(&self, tokens1: &[String], tokens2: &[String]) -> Vec<String> {
        let set1: std::collections::HashSet<_> = tokens1.iter().collect();
        let set2: std::collections::HashSet<_> = tokens2.iter().collect();

        set1.intersection(&set2)
            .take(10)
            .map(|s| (*s).clone())
            .collect()
    }

    fn generate_suggestions(
        &self,
        novelty_score: f32,
        new_topics: &[String],
        redundant_topics: &[String],
        most_similar: Option<&StoredAnswer>,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if novelty_score < self.config.novelty_threshold {
            suggestions.push("Consider providing a different perspective or approach".to_string());

            if let Some(similar) = most_similar {
                suggestions.push(format!(
                    "Your answer is very similar to the one from {}",
                    similar.agent_id
                ));
            }
        }

        if new_topics.is_empty() && !redundant_topics.is_empty() {
            suggestions.push(format!(
                "Topics {} are already well covered, consider exploring other aspects",
                redundant_topics.join(", ")
            ));
        }

        if !new_topics.is_empty() {
            suggestions.push(format!(
                "Good coverage of new topics: {}",
                new_topics.join(", ")
            ));
        }

        suggestions
    }
}

impl Default for NoveltyDetector {
    fn default() -> Self {
        Self::new(NoveltyConfig::default())
    }
}

/// Helper function to get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_first_answer_is_novel() {
        let detector = NoveltyDetector::default();

        let result = detector
            .check_novelty("agent1", "The property is worth $500,000")
            .await;

        assert!(result.is_novel);
        assert!((result.novelty_score - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_duplicate_answer_not_novel() {
        let detector = NoveltyDetector::default();

        detector
            .add_answer("agent1", "The property is worth $500,000")
            .await;

        let result = detector
            .check_novelty("agent2", "The property is worth $500,000")
            .await;

        assert!(!result.is_novel);
        assert!(result.novelty_score < 0.3);
    }

    #[tokio::test]
    async fn test_similar_answer_detected() {
        let detector = NoveltyDetector::default();

        detector
            .add_answer(
                "agent1",
                "The property is worth $500,000 based on market analysis",
            )
            .await;

        let result = detector
            .check_novelty(
                "agent2",
                "Based on my analysis, the property value is $500,000",
            )
            .await;

        assert!(result.most_similar.is_some());
        let similar = result.most_similar.unwrap();
        assert_eq!(similar.agent_id, "agent1");
        assert!(similar.similarity > 0.3);
    }

    #[tokio::test]
    async fn test_different_answer_is_novel() {
        let detector = NoveltyDetector::default();

        detector
            .add_answer("agent1", "The property is worth $500,000")
            .await;

        let result = detector
            .check_novelty(
                "agent2",
                "The neighborhood has excellent schools and low crime rates",
            )
            .await;

        assert!(result.is_novel);
        assert!(result.novelty_score > 0.5);
    }

    #[tokio::test]
    async fn test_topic_extraction() {
        let detector = NoveltyDetector::default();

        detector
            .add_answer(
                "agent1",
                "The price of this property is $500,000 for 200 square meters",
            )
            .await;

        let coverage = detector.get_topic_coverage().await;
        let topics: Vec<_> = coverage.iter().map(|c| c.topic.as_str()).collect();

        assert!(topics.contains(&"price"));
        assert!(topics.contains(&"size"));
    }

    #[tokio::test]
    async fn test_numeric_value_comparison() {
        let detector = NoveltyDetector::new(NoveltyConfig {
            compare_numeric_values: true,
            numeric_tolerance: 0.05, // 5%
            ..Default::default()
        });

        detector.add_answer("agent1", "The value is $500,000").await;

        // Similar value (within 5%)
        let result = detector
            .check_novelty("agent2", "The value is $510,000")
            .await;
        assert!(result.most_similar.is_some());

        // Different value (outside 5%)
        let result = detector
            .check_novelty("agent3", "The value is $600,000")
            .await;
        // Should be more novel due to different numeric value
        assert!(result.novelty_score > 0.2);
    }

    #[tokio::test]
    async fn test_diversity_score() {
        let detector = NoveltyDetector::default();

        // Add diverse answers
        detector
            .add_answer("agent1", "The property is worth $500,000")
            .await;
        detector
            .add_answer("agent2", "The neighborhood has great schools")
            .await;
        detector
            .add_answer("agent3", "The property has 3 bedrooms and 2 bathrooms")
            .await;

        let diversity = detector.get_diversity_score().await;
        assert!(diversity > 0.5);
    }

    #[tokio::test]
    async fn test_check_and_add() {
        let detector = NoveltyDetector::default();

        // First answer should be added
        let result = detector
            .check_and_add("agent1", "The property is worth $500,000")
            .await;
        assert!(result.is_novel);

        // Duplicate should not be added
        let result = detector
            .check_and_add("agent2", "The property is worth $500,000")
            .await;
        assert!(!result.is_novel);

        // Verify only one answer was added
        let coverage = detector.get_topic_coverage().await;
        let price_coverage = coverage.iter().find(|c| c.topic == "price").unwrap();
        assert_eq!(price_coverage.coverage_count, 1);
    }

    #[tokio::test]
    async fn test_synonyms() {
        let config = NoveltyConfig::default()
            .with_synonyms("price", vec!["cost".to_string(), "value".to_string()]);

        let detector = NoveltyDetector::new(config);

        detector.add_answer("agent1", "The price is $500,000").await;

        let result = detector
            .check_novelty("agent2", "The cost is $500,000")
            .await;

        // Should detect similarity due to synonyms
        assert!(result.most_similar.is_some());
        assert!(result.most_similar.unwrap().similarity > 0.5);
    }

    #[tokio::test]
    async fn test_suggestions() {
        let detector = NoveltyDetector::default();

        detector
            .add_answer("agent1", "The property price is $500,000")
            .await;
        detector
            .add_answer("agent2", "The price estimate is around $500,000")
            .await;

        let result = detector
            .check_novelty("agent3", "The property is valued at $500,000")
            .await;

        // Should get suggestions since similar to existing
        assert!(!result.suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_strict_config() {
        let detector = NoveltyDetector::new(NoveltyConfig::strict());

        detector
            .add_answer("agent1", "The property is worth $500,000")
            .await;

        // With strict config, nearly identical answers won't pass
        let result = detector
            .check_novelty("agent2", "The property is worth $500,000")
            .await;

        // Stricter threshold means this duplicate answer won't be novel
        assert!(!result.is_novel);
        assert!(result.novelty_score < 0.5);
    }

    #[tokio::test]
    async fn test_clear() {
        let detector = NoveltyDetector::default();

        detector
            .add_answer("agent1", "The property is worth $500,000")
            .await;

        detector.clear().await;

        let result = detector
            .check_novelty("agent2", "The property is worth $500,000")
            .await;

        // After clear, should be novel again
        assert!(result.is_novel);
    }
}
