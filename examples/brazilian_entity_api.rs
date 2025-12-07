//! Brazilian Entity Investigation API
//!
//! REST API for investigating Brazilian entities across multiple data sources.
//!
//! ## Endpoints
//!
//! - `GET /health` - Health check
//! - `GET /api/v1/metrics` - Pipeline and API metrics
//! - `GET /api/v1/sources` - List available data sources
//! - `DELETE /api/v1/cache` - Clear the cache
//! - `POST /api/v1/investigate` - Investigate a single entity
//! - `POST /api/v1/investigate/batch` - Investigate multiple entities
//! - `GET /api/v1/investigate/status/:id` - Get investigation status
//!
//! ## Running
//!
//! ```bash
//! cargo run --example brazilian_entity_api
//! ```
//!
//! Then access Swagger UI at: http://localhost:3000/swagger-ui

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use rust_ai_agents_data::{
    types::{DataRecord, DataSchema, DataSource, FieldValue},
    CrossReferenceResult, CrossReferencer, DataMatchingMetrics, MetricsSnapshot,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;
use uuid::Uuid;

// ============================================================================
// OpenAPI Documentation
// ============================================================================

#[derive(OpenApi)]
#[openapi(
    info(
        title = "HyperAgent Brasil API",
        version = "1.0.0",
        description = "API de Investigacao de Entidades Brasileiras - Busca cross-reference em multiplas fontes de dados",
        contact(
            name = "Ronaldo Lima",
            url = "https://github.com/limaronaldo/rust-ai-agents"
        )
    ),
    paths(
        health_check,
        get_metrics,
        list_sources,
        clear_cache,
        investigate_entity,
        investigate_batch,
        get_investigation_status,
    ),
    components(schemas(
        HealthResponse,
        MetricsResponse,
        SourceInfo,
        SourcesResponse,
        CacheResponse,
        InvestigationRequest,
        InvestigationResponse,
        EntityDetails,
        SourceMatch,
        BatchRequest,
        BatchResponse,
        StatusResponse,
    )),
    tags(
        (name = "Health", description = "Health check endpoints"),
        (name = "Metrics", description = "Pipeline and API metrics"),
        (name = "Sources", description = "Data source management"),
        (name = "Cache", description = "Cache management"),
        (name = "Investigation", description = "Entity investigation endpoints"),
    )
)]
struct ApiDoc;

// ============================================================================
// Request/Response DTOs
// ============================================================================

#[derive(Debug, Serialize, ToSchema)]
struct HealthResponse {
    status: String,
    version: String,
    uptime_seconds: u64,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, ToSchema)]
struct MetricsResponse {
    pipeline: MetricsSnapshot,
    api: ApiMetrics,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
struct ApiMetrics {
    total_requests: u64,
    active_investigations: usize,
    completed_investigations: usize,
    uptime_seconds: u64,
}

#[derive(Debug, Serialize, ToSchema)]
struct SourceInfo {
    id: String,
    name: String,
    record_count: usize,
}

#[derive(Debug, Serialize, ToSchema)]
struct SourcesResponse {
    sources: Vec<SourceInfo>,
    total_records: usize,
}

#[derive(Debug, Serialize, ToSchema)]
struct CacheResponse {
    status: String,
    message: String,
}

#[derive(Debug, Deserialize, ToSchema)]
struct InvestigationRequest {
    /// Nome da entidade a ser investigada
    #[schema(example = "Lucas Melo de Oliveira")]
    nome: String,

    /// CPF ou CNPJ (opcional, melhora precisao)
    #[schema(example = "123.456.789-00")]
    documento: Option<String>,

    /// Lista de IDs de fontes para buscar (vazio = todas)
    #[serde(default)]
    fontes: Vec<String>,

    /// Limiar minimo de confianca (0.0 - 1.0)
    #[serde(default = "default_threshold")]
    #[schema(example = 0.7)]
    threshold: f64,

    /// Formato de saida: "full" ou "compact"
    #[serde(default = "default_format")]
    #[schema(example = "full")]
    formato: String,
}

fn default_threshold() -> f64 {
    0.7
}

fn default_format() -> String {
    "full".to_string()
}

#[derive(Debug, Clone, Serialize, ToSchema)]
struct InvestigationResponse {
    /// ID unico da investigacao
    request_id: String,

    /// Status: "success" ou "error"
    status: String,

    /// Detalhes da entidade encontrada
    entity: Option<EntityDetails>,

    /// Narrativa em PT-BR
    narrative: String,

    /// Tempo de processamento em ms
    processing_time_ms: u64,

    /// Timestamp
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
struct EntityDetails {
    /// Nome pesquisado
    nome: String,

    /// Documento pesquisado
    documento: Option<String>,

    /// Confianca geral (0.0 - 1.0)
    confianca: f64,

    /// Correspondencias por fonte
    fontes: Vec<SourceMatch>,

    /// Total de registros encontrados
    total_registros: usize,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
struct SourceMatch {
    /// ID da fonte
    source_id: String,

    /// Nome da fonte
    source_name: String,

    /// Numero de registros correspondentes
    record_count: usize,

    /// Confianca da correspondencia
    confidence: f64,

    /// Campos encontrados
    fields: HashMap<String, String>,
}

#[derive(Debug, Deserialize, ToSchema)]
struct BatchRequest {
    /// Lista de entidades a investigar
    requests: Vec<InvestigationRequest>,

    /// Maximo de investigacoes concorrentes
    #[serde(default = "default_max_concurrent")]
    max_concurrent: usize,
}

fn default_max_concurrent() -> usize {
    4
}

#[derive(Debug, Serialize, ToSchema)]
struct BatchResponse {
    /// ID do batch
    batch_id: String,

    /// Resultados individuais
    results: Vec<InvestigationResponse>,

    /// Total processado
    total: usize,

    /// Bem-sucedidos
    successful: usize,

    /// Tempo total em ms
    total_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
struct StatusResponse {
    /// ID da investigacao
    request_id: String,

    /// Status: "pending", "processing", "completed", "not_found"
    status: String,

    /// Resultado (se completed)
    result: Option<InvestigationResponse>,
}

// ============================================================================
// Application State
// ============================================================================

struct AppState {
    /// Cross-referencer para buscas
    crossref: CrossReferencer,

    /// Fontes de dados disponiveis
    sources: Vec<DataSource>,

    /// Metricas do pipeline
    metrics: Arc<DataMatchingMetrics>,

    /// Resultados de investigacoes (para GET /status)
    results: RwLock<HashMap<String, InvestigationResponse>>,

    /// Contador de requests
    request_count: std::sync::atomic::AtomicU64,

    /// Hora de inicio do servidor
    started_at: Instant,
}

impl AppState {
    fn new(sources: Vec<DataSource>) -> Self {
        Self {
            crossref: CrossReferencer::new(),
            sources,
            metrics: Arc::new(DataMatchingMetrics::new()),
            results: RwLock::new(HashMap::new()),
            request_count: std::sync::atomic::AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }

    fn increment_requests(&self) {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn total_requests(&self) -> u64 {
        self.request_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// Health check
#[utoipa::path(
    get,
    path = "/health",
    tag = "Health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.uptime_seconds(),
        timestamp: Utc::now(),
    })
}

/// Get pipeline and API metrics
#[utoipa::path(
    get,
    path = "/api/v1/metrics",
    tag = "Metrics",
    responses(
        (status = 200, description = "Current metrics", body = MetricsResponse)
    )
)]
async fn get_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.increment_requests();

    let results = state.results.read().await;

    Json(MetricsResponse {
        pipeline: state.metrics.snapshot(),
        api: ApiMetrics {
            total_requests: state.total_requests(),
            active_investigations: 0, // Could track in-flight requests
            completed_investigations: results.len(),
            uptime_seconds: state.uptime_seconds(),
        },
    })
}

/// List available data sources
#[utoipa::path(
    get,
    path = "/api/v1/sources",
    tag = "Sources",
    responses(
        (status = 200, description = "List of available sources", body = SourcesResponse)
    )
)]
async fn list_sources(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.increment_requests();

    let sources: Vec<SourceInfo> = state
        .sources
        .iter()
        .map(|s| SourceInfo {
            id: s.id.clone(),
            name: s.name.clone(),
            record_count: s.records.len(),
        })
        .collect();

    let total_records: usize = sources.iter().map(|s| s.record_count).sum();

    Json(SourcesResponse {
        sources,
        total_records,
    })
}

/// Clear the cache
#[utoipa::path(
    delete,
    path = "/api/v1/cache",
    tag = "Cache",
    responses(
        (status = 200, description = "Cache cleared", body = CacheResponse)
    )
)]
async fn clear_cache(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.increment_requests();

    // Clear stored results
    let mut results = state.results.write().await;
    let count = results.len();
    results.clear();

    // Reset metrics
    state.metrics.reset();

    Json(CacheResponse {
        status: "success".to_string(),
        message: format!("Cache cleared. {} investigation results removed.", count),
    })
}

/// Investigate a single entity
#[utoipa::path(
    post,
    path = "/api/v1/investigate",
    tag = "Investigation",
    request_body = InvestigationRequest,
    responses(
        (status = 200, description = "Investigation result", body = InvestigationResponse),
        (status = 400, description = "Invalid request")
    )
)]
async fn investigate_entity(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InvestigationRequest>,
) -> impl IntoResponse {
    state.increment_requests();

    let response = perform_investigation(&state, req).await;

    // Store result for status endpoint
    {
        let mut results = state.results.write().await;
        results.insert(response.request_id.clone(), response.clone());
    }

    Json(response)
}

/// Investigate multiple entities in batch
#[utoipa::path(
    post,
    path = "/api/v1/investigate/batch",
    tag = "Investigation",
    request_body = BatchRequest,
    responses(
        (status = 200, description = "Batch investigation results", body = BatchResponse),
        (status = 400, description = "Invalid request")
    )
)]
async fn investigate_batch(
    State(state): State<Arc<AppState>>,
    Json(batch): Json<BatchRequest>,
) -> impl IntoResponse {
    state.increment_requests();

    let batch_id = Uuid::new_v4().to_string();
    let start = Instant::now();
    let total = batch.requests.len();

    // Process in parallel with concurrency limit
    let results: Vec<InvestigationResponse> = stream::iter(batch.requests)
        .map(|req| {
            let state = state.clone();
            async move { perform_investigation(&state, req).await }
        })
        .buffer_unordered(batch.max_concurrent)
        .collect()
        .await;

    // Store all results
    {
        let mut stored = state.results.write().await;
        for result in &results {
            stored.insert(result.request_id.clone(), result.clone());
        }
    }

    let successful = results.iter().filter(|r| r.status == "success").count();

    Json(BatchResponse {
        batch_id,
        results,
        total,
        successful,
        total_time_ms: start.elapsed().as_millis() as u64,
    })
}

/// Get investigation status by ID
#[utoipa::path(
    get,
    path = "/api/v1/investigate/status/{id}",
    tag = "Investigation",
    params(
        ("id" = String, Path, description = "Investigation request ID")
    ),
    responses(
        (status = 200, description = "Investigation status", body = StatusResponse),
        (status = 404, description = "Investigation not found")
    )
)]
async fn get_investigation_status(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<StatusResponse>, StatusCode> {
    state.increment_requests();

    let results = state.results.read().await;

    if let Some(result) = results.get(&id) {
        Ok(Json(StatusResponse {
            request_id: id,
            status: "completed".to_string(),
            result: Some(result.clone()),
        }))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

// ============================================================================
// Investigation Logic
// ============================================================================

async fn perform_investigation(
    state: &AppState,
    req: InvestigationRequest,
) -> InvestigationResponse {
    let request_id = Uuid::new_v4().to_string();
    let start = Instant::now();

    // Filter sources if specified
    let sources: Vec<&DataSource> = if req.fontes.is_empty() {
        state.sources.iter().collect()
    } else {
        state
            .sources
            .iter()
            .filter(|s| req.fontes.contains(&s.id))
            .collect()
    };

    if sources.is_empty() {
        return InvestigationResponse {
            request_id,
            status: "error".to_string(),
            entity: None,
            narrative: "Nenhuma fonte de dados disponivel.".to_string(),
            processing_time_ms: start.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        };
    }

    // Perform cross-reference search
    let sources_owned: Vec<DataSource> = sources.iter().map(|s| (*s).clone()).collect();
    let result: CrossReferenceResult =
        state
            .crossref
            .cross_reference(&sources_owned, &req.nome, req.documento.as_deref());

    // Record metrics
    let has_matches = result.total_sources > 0;
    state.metrics.record_query(has_matches, start.elapsed());
    state.metrics.record_scan(result.total_sources as u64);

    if has_matches {
        state.metrics.record_match_type(
            req.documento.is_some(),
            false,
            true,
            result.source_summaries.len() > 1,
        );
    }

    // Build response
    let entity = if has_matches {
        let fontes: Vec<SourceMatch> = result
            .source_summaries
            .iter()
            .map(|s| {
                SourceMatch {
                    source_id: s.source_id.clone(),
                    source_name: s.source_name.clone(),
                    record_count: 1, // Each summary represents one match
                    confidence: s.confidence,
                    fields: s.key_fields.clone(),
                }
            })
            .collect();

        Some(EntityDetails {
            nome: req.nome.clone(),
            documento: req.documento.clone(),
            confianca: result.confidence,
            fontes,
            total_registros: result.total_sources,
        })
    } else {
        None
    };

    let narrative = if req.formato == "compact" {
        state
            .crossref
            .compact_narrative(&sources_owned, &req.nome, req.documento.as_deref())
    } else {
        result.narrative.clone()
    };

    InvestigationResponse {
        request_id,
        status: if has_matches { "success" } else { "not_found" }.to_string(),
        entity,
        narrative,
        processing_time_ms: start.elapsed().as_millis() as u64,
        timestamp: Utc::now(),
    }
}

// ============================================================================
// Sample Data
// ============================================================================

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
            ],
        },
    ]
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,tower_http=debug")
        .with_target(false)
        .init();

    // Create sample data
    let sources = create_sample_sources();
    let state = Arc::new(AppState::new(sources));

    // Build router
    let app = Router::new()
        // Swagger UI
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        // Health check
        .route("/health", get(health_check))
        // API v1 routes
        .route("/api/v1/metrics", get(get_metrics))
        .route("/api/v1/sources", get(list_sources))
        .route("/api/v1/cache", delete(clear_cache))
        .route("/api/v1/investigate", post(investigate_entity))
        .route("/api/v1/investigate/batch", post(investigate_batch))
        .route(
            "/api/v1/investigate/status/:id",
            get(get_investigation_status),
        )
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .with_state(state);

    // Start server
    let addr = "0.0.0.0:3000";
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           HyperAgent Brasil - Entity Investigation API        ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║                                                               ║");
    println!("║  Server running at: http://localhost:3000                     ║");
    println!("║  Swagger UI:        http://localhost:3000/swagger-ui          ║");
    println!("║  OpenAPI JSON:      http://localhost:3000/api-docs/openapi.json║");
    println!("║                                                               ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Endpoints:                                                   ║");
    println!("║    GET  /health                    Health check               ║");
    println!("║    GET  /api/v1/metrics            Pipeline metrics           ║");
    println!("║    GET  /api/v1/sources            List data sources          ║");
    println!("║    DELETE /api/v1/cache            Clear cache                ║");
    println!("║    POST /api/v1/investigate        Investigate entity         ║");
    println!("║    POST /api/v1/investigate/batch  Batch investigation        ║");
    println!("║    GET  /api/v1/investigate/status/:id  Get status            ║");
    println!("║                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
