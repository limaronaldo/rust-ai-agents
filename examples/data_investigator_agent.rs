//! Data Investigator Agent
//!
//! An AI agent that uses the cross_reference_entity tool to investigate
//! Brazilian entities across multiple data sources.
//!
//! This example demonstrates:
//! - Creating an agent with data investigation capabilities
//! - Using the CrossReferenceEntityTool with real data sources
//! - Agent loop with tool calls and PT-BR responses
//!
//! ## Running
//!
//! Set your API key:
//! ```bash
//! export OPENROUTER_API_KEY=your_key_here
//! ```
//!
//! Then run:
//! ```bash
//! cargo run --example data_investigator_agent
//! ```

use rust_ai_agents_agents::*;
use rust_ai_agents_core::*;
use rust_ai_agents_data::types::{DataRecord, DataSchema, DataSource, FieldValue};
use rust_ai_agents_providers::*;
use rust_ai_agents_tools::{CrossReferenceEntityTool, ToolRegistry};
use std::sync::Arc;

/// System prompt for the data investigator agent (PT-BR)
const INVESTIGATOR_SYSTEM_PROMPT: &str = r#"Voce e um Investigador de Dados especializado em analise de cadastros brasileiros.

Sua funcao e:
1. Receber consultas sobre pessoas ou empresas
2. Usar a ferramenta cross_reference_entity para buscar em multiplas fontes de dados
3. Analisar os resultados e identificar padroes, inconsistencias ou informacoes relevantes
4. Gerar relatorios detalhados em portugues

## Como usar a ferramenta cross_reference_entity

A ferramenta aceita os seguintes parametros:
- name (obrigatorio): Nome da pessoa ou empresa
- cpf (opcional): CPF para correspondencia mais precisa (formato: XXX.XXX.XXX-XX ou apenas numeros)
- format: "full" para analise detalhada, "compact" para resumo rapido

Exemplo de uso:
{
  "name": "Joao da Silva",
  "cpf": "123.456.789-00",
  "format": "full"
}

## Fontes de Dados Disponiveis

Voce tem acesso as seguintes fontes:
- **Parties (Cadastro)**: Cadastro geral de pessoas com nome, CPF, email, telefone, cidade
- **IPTU (Contribuintes)**: Registros de IPTU com nome, documento, endereco, valor, status
- **Transacoes Imobiliarias**: Historico de compras e vendas de imoveis
- **Cadastro de Empresas**: Empresas com razao social, CNPJ, socios, capital

## Ao Reportar Resultados

- Destaque correspondencias em multiplas fontes (indica pessoa ativa em varios contextos)
- Aponte o nivel de confianca da correspondencia
- Identifique possiveis inconsistencias (nomes diferentes, documentos divergentes)
- Sugira verificacoes adicionais quando apropriado
- Use formatacao clara com marcadores e secoes
"#;

/// Create sample Brazilian data sources
fn create_brazilian_sources() -> Vec<DataSource> {
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
                    .with_field("cidade", FieldValue::Text("Sao Paulo".to_string()))
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
                    .with_field("nome", FieldValue::Text("Joao Pedro Santos".to_string()))
                    .with_field("cpf", FieldValue::Text("111.222.333-44".to_string()))
                    .with_field(
                        "email",
                        FieldValue::Text("joao.santos@mail.com".to_string()),
                    )
                    .with_confidence(1.0),
                DataRecord::new("parties")
                    .with_field("nome", FieldValue::Text("Maria Fernanda Costa".to_string()))
                    .with_field("cpf", FieldValue::Text("555.666.777-88".to_string()))
                    .with_field(
                        "email",
                        FieldValue::Text("maria.costa@outlook.com".to_string()),
                    )
                    .with_field("cidade", FieldValue::Text("Belo Horizonte".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("parties")
                    .with_field(
                        "nome",
                        FieldValue::Text("Carlos Eduardo Mendes".to_string()),
                    )
                    .with_field("cpf", FieldValue::Text("222.333.444-55".to_string()))
                    .with_field(
                        "email",
                        FieldValue::Text("carlos.mendes@corp.com".to_string()),
                    )
                    .with_field("cidade", FieldValue::Text("Curitiba".to_string()))
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
                        FieldValue::Text("Av. Atlantica, 456 - Copacabana".to_string()),
                    )
                    .with_field("valor", FieldValue::Text("R$ 8.000,00".to_string()))
                    .with_field("status", FieldValue::Text("Em aberto".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("iptu")
                    .with_field("nome", FieldValue::Text("MARIA F COSTA".to_string()))
                    .with_field("documento", FieldValue::Text("55566677788".to_string()))
                    .with_field(
                        "endereco",
                        FieldValue::Text("Rua da Bahia, 789 - Centro".to_string()),
                    )
                    .with_field("valor", FieldValue::Text("R$ 3.200,00".to_string()))
                    .with_field("status", FieldValue::Text("Pago".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("iptu")
                    .with_field("nome", FieldValue::Text("CARLOS E MENDES".to_string()))
                    .with_field("documento", FieldValue::Text("22233344455".to_string()))
                    .with_field(
                        "endereco",
                        FieldValue::Text("Rua XV de Novembro, 100 - Centro".to_string()),
                    )
                    .with_field("valor", FieldValue::Text("R$ 4.100,00".to_string()))
                    .with_field("status", FieldValue::Text("Pago".to_string()))
                    .with_confidence(1.0),
            ],
        },
        // Property transactions
        DataSource {
            id: "transactions".to_string(),
            name: "Transacoes Imobiliarias".to_string(),
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
                DataRecord::new("transactions")
                    .with_field("nome", FieldValue::Text("Joao P Santos".to_string()))
                    .with_field("cpf", FieldValue::Text("111.222.333-44".to_string()))
                    .with_field("tipo", FieldValue::Text("Compra".to_string()))
                    .with_field("valor", FieldValue::Text("R$ 450.000,00".to_string()))
                    .with_field("data", FieldValue::Text("2024-09-10".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("transactions")
                    .with_field("nome", FieldValue::Text("Carlos Mendes".to_string()))
                    .with_field("cpf", FieldValue::Text("222.333.444-55".to_string()))
                    .with_field("tipo", FieldValue::Text("Compra".to_string()))
                    .with_field("valor", FieldValue::Text("R$ 720.000,00".to_string()))
                    .with_field("data", FieldValue::Text("2024-01-08".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("transactions")
                    .with_field("nome", FieldValue::Text("Carlos Mendes".to_string()))
                    .with_field("cpf", FieldValue::Text("222.333.444-55".to_string()))
                    .with_field("tipo", FieldValue::Text("Venda".to_string()))
                    .with_field("valor", FieldValue::Text("R$ 890.000,00".to_string()))
                    .with_field("data", FieldValue::Text("2024-11-22".to_string()))
                    .with_confidence(1.0),
            ],
        },
        // Companies registry
        DataSource {
            id: "companies".to_string(),
            name: "Cadastro de Empresas".to_string(),
            schema: DataSchema::default(),
            records: vec![
                DataRecord::new("companies")
                    .with_field(
                        "razao_social",
                        FieldValue::Text("OLIVEIRA TECH LTDA".to_string()),
                    )
                    .with_field("cnpj", FieldValue::Text("12.345.678/0001-90".to_string()))
                    .with_field(
                        "socios",
                        FieldValue::Text("Lucas Melo de Oliveira".to_string()),
                    )
                    .with_field("capital", FieldValue::Text("R$ 100.000,00".to_string()))
                    .with_field("situacao", FieldValue::Text("Ativa".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("companies")
                    .with_field(
                        "razao_social",
                        FieldValue::Text("SILVA CONSULTORIA ME".to_string()),
                    )
                    .with_field("cnpj", FieldValue::Text("98.765.432/0001-10".to_string()))
                    .with_field("socios", FieldValue::Text("Ana Clara da Silva".to_string()))
                    .with_field("capital", FieldValue::Text("R$ 50.000,00".to_string()))
                    .with_field("situacao", FieldValue::Text("Ativa".to_string()))
                    .with_confidence(1.0),
                DataRecord::new("companies")
                    .with_field(
                        "razao_social",
                        FieldValue::Text("MENDES IMOVEIS LTDA".to_string()),
                    )
                    .with_field("cnpj", FieldValue::Text("33.444.555/0001-66".to_string()))
                    .with_field(
                        "socios",
                        FieldValue::Text("Carlos Eduardo Mendes".to_string()),
                    )
                    .with_field("capital", FieldValue::Text("R$ 500.000,00".to_string()))
                    .with_field("situacao", FieldValue::Text("Ativa".to_string()))
                    .with_confidence(1.0),
            ],
        },
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          Data Investigator Agent - HyperAgent Brasil          ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Agente de IA para investigacao de entidades brasileiras      ║");
    println!("║  Usa cross_reference_entity para buscar em multiplas fontes   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Check for API key
    let api_key =
        std::env::var("OPENROUTER_API_KEY").expect("Set OPENROUTER_API_KEY environment variable");

    // Create agent engine
    let engine = Arc::new(AgentEngine::new());

    // Create data sources
    let sources = create_brazilian_sources();
    println!("Fontes de dados carregadas:");
    for source in &sources {
        println!("  - {} ({} registros)", source.name, source.records.len());
    }
    println!();

    // Create the cross-reference tool with data sources
    let crossref_tool = CrossReferenceEntityTool::with_sources(sources);

    // Create tool registry
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(crossref_tool));
    let registry = Arc::new(registry);

    // Create LLM backend
    let backend = Arc::new(OpenRouterProvider::new(
        api_key,
        "anthropic/claude-3.5-sonnet".to_string(), // Good for Portuguese
    )) as Arc<dyn LLMBackend>;

    // Create agent config
    let config = AgentConfig::new("Investigador de Dados", AgentRole::Executor)
        .with_capabilities(vec![Capability::Analysis])
        .with_system_prompt(INVESTIGATOR_SYSTEM_PROMPT)
        .with_temperature(0.3) // Lower temperature for more focused responses
        .with_timeout(60);

    // Spawn agent
    println!("Iniciando agente investigador...");
    let agent_id = engine
        .spawn_agent(config, registry.clone(), backend.clone())
        .await?;
    println!("Agente iniciado: {}\n", agent_id);

    // Sample investigation queries
    let queries = vec![
        "Investigue Lucas Melo de Oliveira, CPF 123.456.789-00. Quero saber em quantas fontes ele aparece e se ha alguma inconsistencia.",
        "Faca uma busca por Carlos Mendes. Ele tem atividade empresarial? Possui imoveis?",
        "Pesquise Ana Clara da Silva. Qual a situacao dela no IPTU?",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Consulta {}: ", i + 1);
        println!("  \"{}\"", query);
        println!("═══════════════════════════════════════════════════════════════");
        println!();

        // Send message to agent
        let user_message = Message::user(agent_id.clone(), *query);
        engine.send_message(user_message)?;

        // Wait for processing
        println!("Processando...\n");
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
    }

    // Get metrics
    let metrics = engine.metrics();
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Metricas do Agente");
    println!("═══════════════════════════════════════════════════════════════");
    println!("   Mensagens processadas: {}", metrics.messages_processed);
    println!("   Chamadas de ferramentas: {}", metrics.total_tool_calls);
    println!("   Timeouts: {}", metrics.timeouts);
    println!();

    // Shutdown
    println!("Encerrando agente...");
    engine.shutdown().await;
    println!("Concluido!");

    Ok(())
}
