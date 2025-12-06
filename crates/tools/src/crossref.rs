//! Cross-reference entity tool for agents
//!
//! Allows agents to cross-reference entities across data sources
//! and generate PT-BR narratives.

use async_trait::async_trait;
use rust_ai_agents_core::errors::ToolError;
use rust_ai_agents_core::tool::{ExecutionContext, Tool, ToolSchema};
use rust_ai_agents_data::{CrossReferencer, DataSource};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Tool for cross-referencing entities across data sources
pub struct CrossReferenceEntityTool {
    /// Registered data sources
    sources: Arc<RwLock<Vec<DataSource>>>,
    /// Cross-referencer instance
    crossref: CrossReferencer,
}

impl CrossReferenceEntityTool {
    /// Create a new cross-reference tool
    pub fn new() -> Self {
        Self {
            sources: Arc::new(RwLock::new(Vec::new())),
            crossref: CrossReferencer::new(),
        }
    }

    /// Create with pre-registered sources
    pub fn with_sources(sources: Vec<DataSource>) -> Self {
        Self {
            sources: Arc::new(RwLock::new(sources)),
            crossref: CrossReferencer::new(),
        }
    }

    /// Add a data source
    pub async fn add_source(&self, source: DataSource) {
        let mut sources = self.sources.write().await;
        sources.push(source);
    }

    /// Get shared sources reference for external updates
    pub fn sources(&self) -> Arc<RwLock<Vec<DataSource>>> {
        Arc::clone(&self.sources)
    }
}

impl Default for CrossReferenceEntityTool {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CrossReferenceEntityTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossReferenceEntityTool").finish()
    }
}

#[async_trait]
impl Tool for CrossReferenceEntityTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "cross_reference_entity".to_string(),
            description: "Busca uma entidade (pessoa ou empresa) em múltiplas fontes de dados e gera uma narrativa em português descrevendo onde foi encontrada. Útil para investigação de dados e verificação de cadastros.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome da pessoa ou empresa a ser pesquisada"
                    },
                    "cpf": {
                        "type": "string",
                        "description": "CPF opcional para correspondência exata (formato: XXX.XXX.XXX-XX ou apenas números)"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["full", "compact"],
                        "description": "Formato da resposta: 'full' para narrativa completa, 'compact' para resumo em uma linha",
                        "default": "full"
                    }
                },
                "required": ["name"]
            }),
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: Value,
    ) -> Result<Value, ToolError> {
        let name = arguments["name"].as_str().ok_or_else(|| {
            ToolError::InvalidArguments("Parâmetro 'name' é obrigatório".to_string())
        })?;

        let cpf = arguments["cpf"].as_str();

        let format = arguments["format"].as_str().unwrap_or("full");

        let sources = self.sources.read().await;

        if sources.is_empty() {
            return Ok(json!({
                "success": false,
                "error": "Nenhuma fonte de dados registrada. Adicione fontes antes de pesquisar.",
                "narrative": format!("Não foi possível pesquisar '{}': nenhuma fonte de dados disponível.", name)
            }));
        }

        let result = self.crossref.cross_reference(&sources, name, cpf);

        if format == "compact" {
            let compact = self.crossref.compact_narrative(&sources, name, cpf);
            return Ok(json!({
                "success": true,
                "found": result.total_sources > 0,
                "narrative": compact,
                "sources_matched": result.total_sources,
                "confidence": result.confidence
            }));
        }

        // Full format
        let source_details: Vec<Value> = result
            .source_summaries
            .iter()
            .map(|s| {
                json!({
                    "source_id": s.source_id,
                    "source_name": s.source_name,
                    "summary": s.summary,
                    "confidence": s.confidence,
                    "key_fields": s.key_fields
                })
            })
            .collect();

        Ok(json!({
            "success": true,
            "found": result.total_sources > 0,
            "entity_id": result.entity_id,
            "query": {
                "name": name,
                "cpf": cpf
            },
            "narrative": result.narrative,
            "sources_matched": result.total_sources,
            "confidence": result.confidence,
            "source_details": source_details
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_ai_agents_data::types::{DataRecord, DataSchema, FieldValue};

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
        ]
    }

    fn create_test_context() -> ExecutionContext {
        use rust_ai_agents_core::types::AgentId;
        ExecutionContext::new(AgentId::new("test-agent"))
    }

    #[tokio::test]
    async fn test_cross_reference_tool_execution() {
        let tool = CrossReferenceEntityTool::with_sources(create_test_sources());
        let ctx = create_test_context();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "name": "Lucas Oliveira",
                    "cpf": "123.456.789-00"
                }),
            )
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert!(result["found"].as_bool().unwrap());
        assert!(result["sources_matched"].as_u64().unwrap() >= 1);
        assert!(!result["narrative"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_cross_reference_compact_format() {
        let tool = CrossReferenceEntityTool::with_sources(create_test_sources());
        let ctx = create_test_context();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "name": "Lucas",
                    "cpf": "123.456.789-00",
                    "format": "compact"
                }),
            )
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        let narrative = result["narrative"].as_str().unwrap();
        assert!(narrative.contains("aparece em") || narrative.contains("não encontrado"));
    }

    #[tokio::test]
    async fn test_cross_reference_not_found() {
        let tool = CrossReferenceEntityTool::with_sources(create_test_sources());
        let ctx = create_test_context();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "name": "Pessoa Inexistente"
                }),
            )
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert!(!result["found"].as_bool().unwrap());
        assert_eq!(result["sources_matched"].as_u64().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_cross_reference_no_sources() {
        let tool = CrossReferenceEntityTool::new();
        let ctx = create_test_context();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "name": "Lucas"
                }),
            )
            .await
            .unwrap();

        assert!(!result["success"].as_bool().unwrap());
        assert!(result["error"].as_str().unwrap().contains("fonte"));
    }

    #[tokio::test]
    async fn test_add_source_dynamically() {
        let tool = CrossReferenceEntityTool::new();
        let ctx = create_test_context();

        // Initially no sources
        let result = tool.execute(&ctx, json!({"name": "Test"})).await.unwrap();
        assert!(!result["success"].as_bool().unwrap());

        // Add a source
        tool.add_source(DataSource {
            id: "test".to_string(),
            name: "Test Source".to_string(),
            schema: DataSchema::default(),
            records: vec![DataRecord::new("test")
                .with_field("nome", FieldValue::Text("Test Person".to_string()))
                .with_confidence(1.0)],
        })
        .await;

        // Now should work
        let result = tool
            .execute(&ctx, json!({"name": "Test Person"}))
            .await
            .unwrap();
        assert!(result["success"].as_bool().unwrap());
        assert!(result["found"].as_bool().unwrap());
    }

    #[test]
    fn test_tool_schema() {
        let tool = CrossReferenceEntityTool::new();

        let schema = tool.schema();
        assert_eq!(schema.name, "cross_reference_entity");
        assert!(schema.description.contains("português"));
        assert!(schema.parameters["properties"]["name"].is_object());
        assert!(schema.parameters["properties"]["cpf"].is_object());
        assert!(schema.parameters["properties"]["format"].is_object());
        assert!(!schema.dangerous);
    }
}
