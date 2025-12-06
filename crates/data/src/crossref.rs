//! Cross-reference narrative generation
//!
//! Generates human-readable PT-BR narratives for entity cross-referencing
//! across multiple data sources.

use crate::types::{DataRecord, DataSource, FieldValue, MatchResult};
use crate::DataMatcher;
use std::collections::HashMap;

/// Cross-reference result with narrative
#[derive(Debug, Clone)]
pub struct CrossReferenceResult {
    /// The entity being cross-referenced
    pub entity_id: String,
    /// Query used for matching
    pub query_name: String,
    /// Query CPF if provided
    pub query_cpf: Option<String>,
    /// Narrative description in PT-BR
    pub narrative: String,
    /// Summary of each source match
    pub source_summaries: Vec<SourceSummary>,
    /// Total sources matched
    pub total_sources: usize,
    /// Overall confidence
    pub confidence: f64,
    /// Underlying match result
    pub match_result: Option<MatchResult>,
}

/// Summary of a single source match
#[derive(Debug, Clone)]
pub struct SourceSummary {
    /// Source identifier
    pub source_id: String,
    /// Source name
    pub source_name: String,
    /// Human-readable summary of the record
    pub summary: String,
    /// Match confidence
    pub confidence: f64,
    /// Key fields extracted
    pub key_fields: HashMap<String, String>,
}

/// Cross-reference engine for narrative generation
#[derive(Debug, Clone)]
pub struct CrossReferencer {
    matcher: DataMatcher,
}

impl Default for CrossReferencer {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossReferencer {
    /// Create a new cross-referencer
    pub fn new() -> Self {
        Self {
            matcher: DataMatcher::new(),
        }
    }

    /// Create with custom matcher
    pub fn with_matcher(matcher: DataMatcher) -> Self {
        Self { matcher }
    }

    /// Cross-reference an entity across sources and generate narrative
    pub fn cross_reference(
        &self,
        sources: &[DataSource],
        query_name: &str,
        query_cpf: Option<&str>,
    ) -> CrossReferenceResult {
        let results = self
            .matcher
            .match_across_sources(sources, query_name, query_cpf);

        if results.is_empty() {
            return CrossReferenceResult {
                entity_id: String::new(),
                query_name: query_name.to_string(),
                query_cpf: query_cpf.map(String::from),
                narrative: format!(
                    "Nenhum registro encontrado para '{}' nas fontes consultadas.",
                    query_name
                ),
                source_summaries: Vec::new(),
                total_sources: 0,
                confidence: 0.0,
                match_result: None,
            };
        }

        // Take best match
        let best = &results[0];
        let source_summaries = self.build_source_summaries(best);
        let narrative = self.build_narrative(query_name, query_cpf, best, &source_summaries);

        CrossReferenceResult {
            entity_id: best.entity_id.clone(),
            query_name: query_name.to_string(),
            query_cpf: query_cpf.map(String::from),
            narrative,
            source_summaries,
            total_sources: best.sources.len(),
            confidence: best.confidence,
            match_result: Some(best.clone()),
        }
    }

    /// Build summaries for each source match
    fn build_source_summaries(&self, result: &MatchResult) -> Vec<SourceSummary> {
        result
            .sources
            .iter()
            .map(|source_match| {
                let key_fields = self.extract_key_fields(&source_match.record);
                let summary =
                    self.summarize_record(&source_match.record, &source_match.source_name);

                SourceSummary {
                    source_id: source_match.source_id.clone(),
                    source_name: source_match.source_name.clone(),
                    summary,
                    confidence: source_match.score,
                    key_fields,
                }
            })
            .collect()
    }

    /// Extract key fields from a record as strings
    fn extract_key_fields(&self, record: &DataRecord) -> HashMap<String, String> {
        let mut key_fields = HashMap::new();

        for (key, value) in &record.fields {
            let str_value = match value {
                FieldValue::Text(s) => s.clone(),
                FieldValue::Integer(n) => n.to_string(),
                FieldValue::Float(f) => format!("{:.2}", f),
                FieldValue::Boolean(b) => if *b { "sim" } else { "não" }.to_string(),
                FieldValue::Date(d) => d.clone(),
                FieldValue::Null => continue,
            };

            if !str_value.is_empty() {
                key_fields.insert(key.clone(), str_value);
            }
        }

        key_fields
    }

    /// Summarize a single record
    fn summarize_record(&self, record: &DataRecord, source_name: &str) -> String {
        let mut parts = Vec::new();

        // Name
        if let Some(name) = record.get_name_field() {
            parts.push(format!("nome: {}", name));
        }

        // CPF
        if let Some(cpf) = record.get_cpf_field() {
            parts.push(format!("CPF: {}", cpf));
        }

        // Other interesting fields
        let interesting_fields = [
            "email",
            "telefone",
            "phone",
            "endereco",
            "address",
            "cidade",
            "city",
            "estado",
            "state",
            "valor",
            "value",
            "status",
            "tipo",
            "type",
            "data",
            "date",
            "created_at",
            "updated_at",
        ];

        for field_name in &interesting_fields {
            if let Some(FieldValue::Text(value)) = record.fields.get(*field_name) {
                if !value.is_empty() && parts.len() < 6 {
                    parts.push(format!("{}: {}", field_name, value));
                }
            }
        }

        if parts.is_empty() {
            format!("registro encontrado em {}", source_name)
        } else {
            parts.join(", ")
        }
    }

    /// Build the PT-BR narrative
    fn build_narrative(
        &self,
        query_name: &str,
        query_cpf: Option<&str>,
        result: &MatchResult,
        summaries: &[SourceSummary],
    ) -> String {
        let mut narrative = String::new();

        // Opening
        let cpf_info = query_cpf
            .map(|cpf| format!(" (CPF: {})", cpf))
            .unwrap_or_default();

        narrative.push_str(&format!(
            "**{}**{} foi encontrado em {} fonte(s) com {:.0}% de confiança.\n\n",
            query_name,
            cpf_info,
            result.sources.len(),
            result.confidence * 100.0
        ));

        // Source details
        for (i, summary) in summaries.iter().enumerate() {
            let confidence_str = match summary.confidence {
                c if c >= 0.95 => "correspondência exata",
                c if c >= 0.85 => "alta correspondência",
                c if c >= 0.70 => "correspondência moderada",
                _ => "baixa correspondência",
            };

            narrative.push_str(&format!(
                "{}. **{}** ({}):\n   {}\n\n",
                i + 1,
                summary.source_name,
                confidence_str,
                summary.summary
            ));
        }

        // Summary line
        let source_names: Vec<&str> = summaries.iter().map(|s| s.source_name.as_str()).collect();

        if source_names.len() == 1 {
            narrative.push_str(&format!("Aparece somente em **{}**.", source_names[0]));
        } else if source_names.len() == 2 {
            narrative.push_str(&format!(
                "Aparece em **{}** e **{}**.",
                source_names[0], source_names[1]
            ));
        } else {
            let last = source_names.last().unwrap();
            let rest = &source_names[..source_names.len() - 1];
            narrative.push_str(&format!(
                "Aparece em **{}** e **{}**.",
                rest.join("**, **"),
                last
            ));
        }

        narrative
    }

    /// Generate a compact one-line narrative
    pub fn compact_narrative(
        &self,
        sources: &[DataSource],
        query_name: &str,
        query_cpf: Option<&str>,
    ) -> String {
        let result = self.cross_reference(sources, query_name, query_cpf);

        if result.total_sources == 0 {
            return format!("'{}': não encontrado", query_name);
        }

        let source_names: Vec<&str> = result
            .source_summaries
            .iter()
            .map(|s| s.source_name.as_str())
            .collect();

        let source_list = if source_names.len() == 1 {
            source_names[0].to_string()
        } else if source_names.len() == 2 {
            format!("{} e {}", source_names[0], source_names[1])
        } else {
            let last = source_names.last().unwrap();
            let rest = &source_names[..source_names.len() - 1];
            format!("{} e {}", rest.join(", "), last)
        };

        format!(
            "'{}': aparece em {} ({:.0}% confiança)",
            query_name,
            source_list,
            result.confidence * 100.0
        )
    }
}

/// Build a cross-reference narrative from match results (standalone function)
pub fn build_cross_reference_narrative(
    query_name: &str,
    query_cpf: Option<&str>,
    match_result: &MatchResult,
) -> String {
    let crossref = CrossReferencer::new();
    let summaries = crossref.build_source_summaries(match_result);
    crossref.build_narrative(query_name, query_cpf, match_result, &summaries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSchema, FieldValue};

    fn create_test_sources() -> Vec<DataSource> {
        vec![
            DataSource {
                id: "parties".to_string(),
                name: "Parties".to_string(),
                schema: DataSchema::default(),
                records: vec![DataRecord::new("parties")
                    .with_field(
                        "nome",
                        FieldValue::Text("Lucas Melo de Oliveira".to_string()),
                    )
                    .with_field("cpf", FieldValue::Text("123.456.789-00".to_string()))
                    .with_field("email", FieldValue::Text("lucas@email.com".to_string()))
                    .with_confidence(1.0)],
            },
            DataSource {
                id: "iptu".to_string(),
                name: "IPTU".to_string(),
                schema: DataSchema::default(),
                records: vec![DataRecord::new("iptu")
                    .with_field("nome", FieldValue::Text("LUCAS M OLIVEIRA".to_string()))
                    .with_field("documento", FieldValue::Text("12345678900".to_string()))
                    .with_field(
                        "endereco",
                        FieldValue::Text("Rua das Flores, 123".to_string()),
                    )
                    .with_confidence(1.0)],
            },
            DataSource {
                id: "transactions".to_string(),
                name: "Transações".to_string(),
                schema: DataSchema::default(),
                records: vec![DataRecord::new("transactions")
                    .with_field("nome", FieldValue::Text("Lucas Oliveira".to_string()))
                    .with_field("cpf", FieldValue::Text("123.456.789-00".to_string()))
                    .with_field("valor", FieldValue::Text("R$ 500.000,00".to_string()))
                    .with_field("tipo", FieldValue::Text("Compra".to_string()))
                    .with_confidence(1.0)],
            },
        ]
    }

    #[test]
    fn test_cross_reference_with_cpf() {
        let crossref = CrossReferencer::new();
        let sources = create_test_sources();

        let result = crossref.cross_reference(&sources, "Lucas Oliveira", Some("123.456.789-00"));

        assert!(
            result.total_sources >= 2,
            "Should match in multiple sources"
        );
        assert!(result.confidence > 0.85);
        assert!(result.narrative.contains("Lucas Oliveira"));
        assert!(result.narrative.contains("encontrado"));
    }

    #[test]
    fn test_cross_reference_by_name() {
        let crossref = CrossReferencer::new();
        let sources = create_test_sources();

        let result = crossref.cross_reference(&sources, "Lucas Melo Oliveira", None);

        assert!(result.total_sources >= 1, "Should find by name");
        assert!(!result.narrative.is_empty());
    }

    #[test]
    fn test_cross_reference_not_found() {
        let crossref = CrossReferencer::new();
        let sources = create_test_sources();

        let result = crossref.cross_reference(&sources, "Pessoa Inexistente", None);

        assert_eq!(result.total_sources, 0);
        assert!(result.narrative.contains("Nenhum registro"));
    }

    #[test]
    fn test_compact_narrative() {
        let crossref = CrossReferencer::new();
        let sources = create_test_sources();

        let narrative = crossref.compact_narrative(&sources, "Lucas", Some("123.456.789-00"));

        assert!(narrative.contains("aparece em"));
        assert!(narrative.contains("confiança"));
    }

    #[test]
    fn test_narrative_contains_source_names() {
        let crossref = CrossReferencer::new();
        let sources = create_test_sources();

        let result = crossref.cross_reference(&sources, "Lucas", Some("123.456.789-00"));

        // Should mention at least one source name
        let contains_source = result.narrative.contains("Parties")
            || result.narrative.contains("IPTU")
            || result.narrative.contains("Transações");

        assert!(contains_source, "Narrative should mention source names");
    }

    #[test]
    fn test_source_summaries() {
        let crossref = CrossReferencer::new();
        let sources = create_test_sources();

        let result = crossref.cross_reference(&sources, "Lucas", Some("123.456.789-00"));

        for summary in &result.source_summaries {
            assert!(!summary.source_name.is_empty());
            assert!(!summary.summary.is_empty());
            assert!(summary.confidence > 0.0);
        }
    }
}
