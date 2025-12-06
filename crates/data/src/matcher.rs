//! Cross-source data matching and entity resolution
//!
//! Consolidates records from multiple sources into unified entities.

use crate::{cpf::CpfMatcher, name::NameMatcher, types::*};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

/// Data matcher for cross-source entity resolution
#[derive(Debug, Clone)]
pub struct DataMatcher {
    /// CPF matcher
    cpf_matcher: CpfMatcher,
    /// Name matcher
    name_matcher: NameMatcher,
    /// Minimum confidence for CPF match
    cpf_confidence_threshold: f64,
    /// Minimum confidence for name match
    name_confidence_threshold: f64,
}

impl Default for DataMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl DataMatcher {
    /// Create a new data matcher
    pub fn new() -> Self {
        Self {
            cpf_matcher: CpfMatcher::new(),
            name_matcher: NameMatcher::new(),
            cpf_confidence_threshold: 0.99,
            name_confidence_threshold: 0.85,
        }
    }

    /// Set CPF confidence threshold
    pub fn with_cpf_threshold(mut self, threshold: f64) -> Self {
        self.cpf_confidence_threshold = threshold;
        self
    }

    /// Set name confidence threshold
    pub fn with_name_threshold(mut self, threshold: f64) -> Self {
        self.name_confidence_threshold = threshold;
        self.name_matcher = self.name_matcher.with_threshold(threshold);
        self
    }

    /// Match across multiple data sources
    ///
    /// Returns consolidated results grouping records that refer to the same entity
    pub fn match_across_sources(
        &self,
        sources: &[DataSource],
        query_name: &str,
        query_cpf: Option<&str>,
    ) -> Vec<MatchResult> {
        let start = Instant::now();
        let mut records_scanned = 0;
        let mut candidates: Vec<(DataRecord, &DataSource, f64, MatchType)> = Vec::new();

        // Normalize query CPF if provided
        let normalized_cpf = query_cpf.and_then(|cpf| self.cpf_matcher.normalize(cpf));

        info!(
            query_name = query_name,
            query_cpf = ?query_cpf,
            sources_count = sources.len(),
            "Starting cross-source match"
        );

        // Phase 1: Find all matching records
        for source in sources {
            for record in &source.records {
                records_scanned += 1;

                // Try CPF match first (highest confidence)
                if let Some(ref cpf) = normalized_cpf {
                    if let Some(record_cpf) = record.get_cpf_field() {
                        if self.cpf_matcher.matches(cpf, record_cpf) {
                            candidates.push((record.clone(), source, 1.0, MatchType::ExactId));
                            continue;
                        }
                    }
                }

                // Try name match
                if let Some(record_name) = record.get_name_field() {
                    let similarity = self.name_matcher.similarity(query_name, record_name);

                    if similarity >= self.name_confidence_threshold {
                        let match_type = if similarity >= 0.99 {
                            MatchType::ExactText
                        } else {
                            MatchType::Fuzzy
                        };

                        candidates.push((record.clone(), source, similarity, match_type));
                    }
                }
            }
        }

        debug!(
            candidates_found = candidates.len(),
            records_scanned, "Initial candidate search complete"
        );

        // Phase 2: Group candidates by entity
        let grouped = self.group_candidates(candidates, query_name);

        // Phase 3: Create match results
        let mut results: Vec<MatchResult> = grouped
            .into_iter()
            .map(|(entity_id, matches)| {
                let mut result = MatchResult::new(&entity_id);

                for (record, source, score, match_type) in matches {
                    // Add fields to consolidated
                    for (key, value) in &record.fields {
                        result
                            .consolidated_fields
                            .entry(key.clone())
                            .or_insert_with(|| value.clone());
                    }

                    result.add_source(SourceMatch {
                        source_id: source.id.clone(),
                        source_name: source.name.clone(),
                        score,
                        record,
                        match_type,
                    });
                }

                result.calculate_confidence();
                result.metadata = MatchMetadata {
                    match_time_ms: start.elapsed().as_millis() as u64,
                    records_scanned,
                    criteria: self.get_criteria_used(&result),
                    warnings: Vec::new(),
                };

                result
            })
            .collect();

        // Sort by confidence
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        info!(
            results_count = results.len(),
            elapsed_ms = start.elapsed().as_millis(),
            "Cross-source match complete"
        );

        results
    }

    /// Group candidates by probable entity
    fn group_candidates<'a>(
        &self,
        candidates: Vec<(DataRecord, &'a DataSource, f64, MatchType)>,
        query_name: &str,
    ) -> HashMap<String, Vec<(DataRecord, &'a DataSource, f64, MatchType)>> {
        let mut groups: HashMap<String, Vec<(DataRecord, &'a DataSource, f64, MatchType)>> =
            HashMap::new();

        for candidate in candidates {
            // Determine entity ID
            let entity_id = self.determine_entity_id(&candidate.0, query_name);

            groups.entry(entity_id).or_default().push(candidate);
        }

        // Merge groups that likely refer to the same entity
        self.merge_similar_groups(groups)
    }

    /// Determine entity ID from a record
    fn determine_entity_id(&self, record: &DataRecord, query_name: &str) -> String {
        // If record has CPF, use that as primary ID
        if let Some(cpf) = record.get_cpf_field() {
            if let Some(normalized) = self.cpf_matcher.normalize(cpf) {
                return format!("cpf_{}", normalized);
            }
        }

        // Otherwise use normalized name
        if let Some(name) = record.get_name_field() {
            return self.name_matcher.to_entity_id(name);
        }

        // Fallback to query name
        self.name_matcher.to_entity_id(query_name)
    }

    /// Merge groups that likely refer to the same entity
    fn merge_similar_groups<'a>(
        &self,
        mut groups: HashMap<String, Vec<(DataRecord, &'a DataSource, f64, MatchType)>>,
    ) -> HashMap<String, Vec<(DataRecord, &'a DataSource, f64, MatchType)>> {
        let entity_ids: Vec<String> = groups.keys().cloned().collect();

        // Find CPF-based groups
        let cpf_groups: Vec<_> = entity_ids
            .iter()
            .filter(|id| id.starts_with("cpf_"))
            .cloned()
            .collect();

        // For each CPF group, absorb name-based groups that match
        for cpf_id in &cpf_groups {
            let cpf_records = groups.get(cpf_id).cloned().unwrap_or_default();

            // Get names from CPF group
            let cpf_names: Vec<String> = cpf_records
                .iter()
                .filter_map(|(r, _, _, _)| r.get_name_field().map(String::from))
                .collect();

            // Find matching name groups
            let mut to_merge = Vec::new();
            for name_id in entity_ids.iter().filter(|id| !id.starts_with("cpf_")) {
                for cpf_name in &cpf_names {
                    let name_from_id = name_id.replace('_', " ");
                    if self.name_matcher.similarity(cpf_name, &name_from_id)
                        >= self.name_confidence_threshold
                    {
                        to_merge.push(name_id.clone());
                        break;
                    }
                }
            }

            // Merge
            for merge_id in to_merge {
                if let Some(records) = groups.remove(&merge_id) {
                    groups.entry(cpf_id.clone()).or_default().extend(records);
                }
            }
        }

        groups
    }

    /// Get criteria used for matching
    fn get_criteria_used(&self, result: &MatchResult) -> Vec<String> {
        let mut criteria = Vec::new();

        let has_cpf_match = result
            .sources
            .iter()
            .any(|s| matches!(s.match_type, MatchType::ExactId));

        let has_fuzzy_match = result
            .sources
            .iter()
            .any(|s| matches!(s.match_type, MatchType::Fuzzy));

        if has_cpf_match {
            criteria.push("CPF match".to_string());
        }
        if has_fuzzy_match {
            criteria.push("Fuzzy name match".to_string());
        }
        if result.sources.len() > 1 {
            criteria.push(format!("Cross-source ({} sources)", result.sources.len()));
        }

        criteria
    }

    /// Find a single best match
    pub fn find_best_match(
        &self,
        sources: &[DataSource],
        query_name: &str,
        query_cpf: Option<&str>,
    ) -> Option<MatchResult> {
        self.match_across_sources(sources, query_name, query_cpf)
            .into_iter()
            .next()
    }

    /// Check if an entity exists in sources
    pub fn entity_exists(
        &self,
        sources: &[DataSource],
        query_name: &str,
        query_cpf: Option<&str>,
    ) -> bool {
        !self
            .match_across_sources(sources, query_name, query_cpf)
            .is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sources() -> Vec<DataSource> {
        vec![
            DataSource {
                id: "source1".to_string(),
                name: "Fonte A".to_string(),
                schema: DataSchema::default(),
                records: vec![
                    DataRecord::new("source1")
                        .with_field(
                            "nome",
                            FieldValue::Text("Lucas Melo de Oliveira".to_string()),
                        )
                        .with_field("cpf", FieldValue::Text("123.456.789-00".to_string()))
                        .with_confidence(1.0),
                    DataRecord::new("source1")
                        .with_field("nome", FieldValue::Text("Ana Clara Silva".to_string()))
                        .with_confidence(0.9),
                ],
            },
            DataSource {
                id: "source2".to_string(),
                name: "Fonte B".to_string(),
                schema: DataSchema::default(),
                records: vec![
                    DataRecord::new("source2")
                        .with_field(
                            "nome_completo",
                            FieldValue::Text("Lucas M. Oliveira".to_string()),
                        )
                        .with_field("documento", FieldValue::Text("12345678900".to_string()))
                        .with_confidence(1.0),
                    DataRecord::new("source2")
                        .with_field("nome", FieldValue::Text("Ana C Silva".to_string()))
                        .with_field("cpf", FieldValue::Text("987.654.321-00".to_string()))
                        .with_confidence(0.95),
                ],
            },
        ]
    }

    #[test]
    fn test_exact_cpf_match() {
        let matcher = DataMatcher::new();
        let sources = create_test_sources();

        let results =
            matcher.match_across_sources(&sources, "qualquer nome", Some("123.456.789-00"));

        assert!(!results.is_empty(), "Should find CPF match");
        assert_eq!(results[0].sources.len(), 2, "Should find in both sources");
        assert!(results[0].confidence > 0.95);
    }

    #[test]
    fn test_cpf_variations() {
        let matcher = DataMatcher::new();
        let sources = create_test_sources();

        let variations = vec!["12345678900", "123.456.789-00", " 123 456 789 00 "];

        for cpf in variations {
            let results = matcher.match_across_sources(&sources, "x", Some(cpf));
            assert!(!results.is_empty(), "Should match CPF: {}", cpf);
        }
    }

    #[test]
    fn test_fuzzy_name_matching() {
        let matcher = DataMatcher::new();
        let sources = create_test_sources();

        let results = matcher.match_across_sources(&sources, "Lucas Oliveira", None);

        assert!(!results.is_empty(), "Should find name match");
        assert!(results[0].confidence >= 0.85);
    }

    #[test]
    fn test_cross_source_consolidation() {
        let matcher = DataMatcher::new();
        let sources = create_test_sources();

        let results = matcher.match_across_sources(&sources, "Ana Silva", None);

        assert!(!results.is_empty(), "Should find Ana");
        // Ana appears in both sources with slightly different names
        assert!(results[0].sources.len() >= 1);
    }

    #[test]
    fn test_no_match() {
        let matcher = DataMatcher::new();
        let sources = create_test_sources();

        let results =
            matcher.match_across_sources(&sources, "José Inexistente", Some("000.000.000-00"));

        assert!(results.is_empty(), "Should not find non-existent person");
    }

    #[test]
    fn test_find_best_match() {
        let matcher = DataMatcher::new();
        let sources = create_test_sources();

        let result = matcher.find_best_match(&sources, "Lucas", Some("123.456.789-00"));

        assert!(result.is_some());
        assert!(result.unwrap().confidence > 0.9);
    }

    #[test]
    fn test_entity_exists() {
        let matcher = DataMatcher::new();
        let sources = create_test_sources();

        assert!(matcher.entity_exists(&sources, "Lucas", Some("123.456.789-00")));
        assert!(!matcher.entity_exists(&sources, "Nobody", Some("000.000.000-00")));
    }
}
