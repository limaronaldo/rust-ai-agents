//! Data Integration Demo
//!
//! Demonstrates how to use the cross-reference entity tool to search for
//! entities across multiple data sources and generate PT-BR narratives.
//!
//! This example shows:
//! - Creating multiple data sources
//! - Using the CrossReferenceEntityTool
//! - Generating narratives in Portuguese
//! - Integrating with an AI agent for data investigation

use rust_ai_agents_core::tool::{ExecutionContext, Tool};
use rust_ai_agents_core::types::AgentId;
use rust_ai_agents_data::{
    types::{DataRecord, DataSchema, DataSource, FieldValue},
    CrossReferencer,
};
use rust_ai_agents_tools::CrossReferenceEntityTool;

/// Create sample data sources for demonstration
fn create_sample_sources() -> Vec<DataSource> {
    vec![
        // Parties table (cadastro de pessoas)
        DataSource {
            id: "parties".to_string(),
            name: "Parties (Cadastro)".to_string(),
            schema: DataSchema::default(),
            records: vec![
                DataRecord::new("parties")
                    .with_field(
                        "nome",
                        FieldValue::Text("Lucas Melo de Oliveira".to_string()),
                    )
                    .with_field("cpf", FieldValue::Text("123.456.789-00".to_string()))
                    .with_field(
                        "email",
                        FieldValue::Text("lucas.melo@email.com".to_string()),
                    )
                    .with_field("telefone", FieldValue::Text("(11) 99999-1234".to_string()))
                    .with_field("cidade", FieldValue::Text("São Paulo".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("parties")
                    .with_field("nome", FieldValue::Text("Ana Clara da Silva".to_string()))
                    .with_field("cpf", FieldValue::Text("987.654.321-00".to_string()))
                    .with_field(
                        "email",
                        FieldValue::Text("ana.silva@empresa.com".to_string()),
                    )
                    .with_field("cidade", FieldValue::Text("Rio de Janeiro".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("parties")
                    .with_field("nome", FieldValue::Text("João Pedro Santos".to_string()))
                    .with_field("cpf", FieldValue::Text("111.222.333-44".to_string()))
                    .with_field(
                        "email",
                        FieldValue::Text("joao.santos@mail.com".to_string()),
                    )
                    .with_confidence(1.0),
            ],
        },
        // IPTU records (property tax)
        DataSource {
            id: "iptu".to_string(),
            name: "IPTU (Contribuintes)".to_string(),
            schema: DataSchema::default(),
            records: vec![
                DataRecord::new("iptu")
                    .with_field("nome", FieldValue::Text("LUCAS M OLIVEIRA".to_string()))
                    .with_field("documento", FieldValue::Text("12345678900".to_string()))
                    .with_field(
                        "endereco",
                        FieldValue::Text("Rua das Flores, 123 - Jardim Paulista".to_string()),
                    )
                    .with_field("valor", FieldValue::Text("R$ 2.500,00".to_string()))
                    .with_field("status", FieldValue::Text("Pago".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("iptu")
                    .with_field("nome", FieldValue::Text("ANA C SILVA".to_string()))
                    .with_field("documento", FieldValue::Text("98765432100".to_string()))
                    .with_field(
                        "endereco",
                        FieldValue::Text("Av. Atlântica, 456 - Copacabana".to_string()),
                    )
                    .with_field("valor", FieldValue::Text("R$ 8.000,00".to_string()))
                    .with_field("status", FieldValue::Text("Em aberto".to_string()))
                    .with_confidence(1.0),
            ],
        },
        // Property transactions
        DataSource {
            id: "transactions".to_string(),
            name: "Transações Imobiliárias".to_string(),
            schema: DataSchema::default(),
            records: vec![
                DataRecord::new("transactions")
                    .with_field("nome", FieldValue::Text("Lucas Oliveira".to_string()))
                    .with_field("cpf", FieldValue::Text("123.456.789-00".to_string()))
                    .with_field("tipo", FieldValue::Text("Compra".to_string()))
                    .with_field("valor", FieldValue::Text("R$ 850.000,00".to_string()))
                    .with_field("data", FieldValue::Text("2024-03-15".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("transactions")
                    .with_field("nome", FieldValue::Text("Ana Clara Silva".to_string()))
                    .with_field("cpf", FieldValue::Text("987.654.321-00".to_string()))
                    .with_field("tipo", FieldValue::Text("Venda".to_string()))
                    .with_field("valor", FieldValue::Text("R$ 1.200.000,00".to_string()))
                    .with_field("data", FieldValue::Text("2024-06-20".to_string()))
                    .with_confidence(1.0),
            ],
        },
    ]
}

/// Demo 1: Direct use of CrossReferencer
fn demo_cross_referencer() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Demo 1: Usando CrossReferencer diretamente");
    println!("═══════════════════════════════════════════════════════════════\n");

    let sources = create_sample_sources();
    let crossref = CrossReferencer::new();

    // Search by name and CPF
    println!("🔍 Pesquisando: Lucas Oliveira (CPF: 123.456.789-00)\n");
    let result = crossref.cross_reference(&sources, "Lucas Oliveira", Some("123.456.789-00"));

    println!("{}", result.narrative);
    println!("\n---");

    // Search by name only
    println!("\n🔍 Pesquisando: Ana Silva (sem CPF)\n");
    let result = crossref.cross_reference(&sources, "Ana Silva", None);

    println!("{}", result.narrative);
    println!("\n---");

    // Search for non-existent person
    println!("\n🔍 Pesquisando: Maria Santos (não existe)\n");
    let result = crossref.cross_reference(&sources, "Maria Santos", None);

    println!("{}", result.narrative);
    println!("\n---");

    // Compact narratives
    println!("\n📝 Narrativas compactas:\n");
    for name in &[
        "Lucas Melo",
        "Ana Clara",
        "João Pedro",
        "Pessoa Inexistente",
    ] {
        let compact = crossref.compact_narrative(&sources, name, None);
        println!("   • {}", compact);
    }
}

/// Demo 2: Using CrossReferenceEntityTool
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       Data Integration Demo - Cross-Reference Tool            ║");
    println!("║                                                               ║");
    println!("║  Demonstra busca de entidades em múltiplas fontes de dados    ║");
    println!("║  com geração de narrativas em português.                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!("\n");

    // Demo 1: Direct CrossReferencer usage
    demo_cross_referencer();

    println!("\n\n");

    // Demo 2: Using the tool
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Demo 2: Usando CrossReferenceEntityTool");
    println!("═══════════════════════════════════════════════════════════════\n");

    let sources = create_sample_sources();
    let tool = CrossReferenceEntityTool::with_sources(sources);

    // Create execution context
    let ctx = ExecutionContext::new(AgentId::new("data-investigator"));

    // Full format search
    println!("🔍 Busca completa (formato full):\n");
    let result = tool
        .execute(
            &ctx,
            serde_json::json!({
                "name": "Lucas Melo Oliveira",
                "cpf": "123.456.789-00",
                "format": "full"
            }),
        )
        .await?;

    println!("Resultado JSON:");
    println!("{}", serde_json::to_string_pretty(&result)?);

    println!("\n---\n");

    // Compact format search
    println!("🔍 Busca compacta (formato compact):\n");
    let result = tool
        .execute(
            &ctx,
            serde_json::json!({
                "name": "Ana Clara",
                "format": "compact"
            }),
        )
        .await?;

    println!("Resultado: {}", result["narrative"]);

    println!("\n\n");

    // Demo 3: Agent integration example (without actual LLM call)
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Demo 3: Configuração de Agente Investigador de Dados");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("System prompt sugerido para agente investigador:\n");
    println!("─────────────────────────────────────────────────────────────────");
    print_investigator_prompt();
    println!("─────────────────────────────────────────────────────────────────");

    println!("\n\n");

    // Demo 4: Show tool schema
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Demo 4: Schema da Ferramenta");
    println!("═══════════════════════════════════════════════════════════════\n");

    let schema = tool.schema();
    println!("Nome: {}", schema.name);
    println!("Descrição: {}", schema.description);
    println!("Parâmetros:");
    println!("{}", serde_json::to_string_pretty(&schema.parameters)?);

    println!("\n✨ Demo concluída!");

    Ok(())
}

fn print_investigator_prompt() {
    let prompt = r#"
Você é um Investigador de Dados especializado em análise de cadastros
brasileiros. Sua função é:

1. Receber consultas sobre pessoas ou empresas
2. Usar a ferramenta cross_reference_entity para buscar em múltiplas fontes
3. Analisar os resultados e identificar inconsistências
4. Gerar relatórios detalhados em português

Ao usar cross_reference_entity:
- Sempre forneça o nome completo quando disponível
- Se tiver CPF/CNPJ, inclua para maior precisão
- Use formato "full" para análises detalhadas
- Use formato "compact" para listagens rápidas

Exemplo de uso:
{
  "name": "João da Silva",
  "cpf": "123.456.789-00",
  "format": "full"
}

Ao reportar resultados:
- Destaque correspondências em múltiplas fontes
- Indique o nível de confiança da correspondência
- Aponte possíveis inconsistências nos dados
- Sugira verificações adicionais quando apropriado
"#;
    println!("{}", prompt);
}
