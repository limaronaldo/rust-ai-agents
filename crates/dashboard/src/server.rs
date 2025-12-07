//! Dashboard HTTP server

use axum::{
    routing::{get, post},
    Router,
};
use rust_ai_agents_monitoring::CostTracker;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tracing::info;

use crate::handlers::{
    agent_handler, agents_handler, health_handler, index_handler, metrics_handler,
    prometheus_metrics_handler, restart_agent_handler, session_handler, session_messages_handler,
    session_traces_handler, sessions_handler, start_agent_handler, stats_handler,
    stop_agent_handler, studio_handler, traces_handler, ws_handler,
};
use crate::state::DashboardState;

/// Dashboard web server
pub struct DashboardServer {
    state: Arc<DashboardState>,
}

impl DashboardServer {
    /// Create a new dashboard server with Prometheus metrics
    pub fn new(cost_tracker: Arc<CostTracker>) -> Self {
        // Initialize Prometheus metrics recorder
        let prometheus_handle = rust_ai_agents_monitoring::init_prometheus();

        Self {
            state: Arc::new(DashboardState::new(cost_tracker, prometheus_handle)),
        }
    }

    /// Get the dashboard state for external updates
    pub fn state(&self) -> Arc<DashboardState> {
        self.state.clone()
    }

    /// Build the Axum router
    fn build_router(&self) -> Router {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        Router::new()
            // Pages
            .route("/", get(index_handler))
            .route("/studio", get(studio_handler))
            // WebSocket
            .route("/ws", get(ws_handler))
            // API - Metrics
            .route("/api/metrics", get(metrics_handler))
            .route("/api/stats", get(stats_handler))
            // API - Traces
            .route("/api/traces", get(traces_handler))
            // API - Sessions
            .route("/api/sessions", get(sessions_handler))
            .route("/api/sessions/:id", get(session_handler))
            .route("/api/sessions/:id/traces", get(session_traces_handler))
            .route("/api/sessions/:id/messages", get(session_messages_handler))
            // API - Agents
            .route("/api/agents", get(agents_handler))
            .route("/api/agents/:id", get(agent_handler))
            .route("/api/agents/:id/start", post(start_agent_handler))
            .route("/api/agents/:id/stop", post(stop_agent_handler))
            .route("/api/agents/:id/restart", post(restart_agent_handler))
            // Health
            .route("/health", get(health_handler))
            // Prometheus metrics endpoint
            .route("/metrics", get(prometheus_metrics_handler))
            // Serve static files for studio WASM
            .nest_service(
                "/studio/pkg",
                ServeDir::new("crates/dashboard/static/studio/pkg"),
            )
            .layer(cors)
            .with_state(self.state.clone())
    }

    /// Run the dashboard server
    pub async fn run(self, addr: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let router = self.build_router();
        let listener = TcpListener::bind(addr).await?;

        info!("Dashboard running at http://{}", addr);
        info!("  Legacy dashboard: http://{}/", addr);
        info!("  Agent Studio:     http://{}/studio", addr);

        // Spawn periodic metrics broadcast
        let state = self.state.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                state.broadcast_metrics();
            }
        });

        axum::serve(listener, router).await?;

        Ok(())
    }

    /// Run the dashboard server in a background task
    pub fn spawn(self, addr: String) -> tokio::task::JoinHandle<()> {
        let state = self.state.clone();

        tokio::spawn(async move {
            // Spawn periodic metrics broadcast
            let broadcast_state = state.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(1));
                loop {
                    interval.tick().await;
                    broadcast_state.broadcast_metrics();
                }
            });

            let cors = CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any);

            let router = Router::new()
                // Pages
                .route("/", get(index_handler))
                .route("/studio", get(studio_handler))
                // WebSocket
                .route("/ws", get(ws_handler))
                // API - Metrics
                .route("/api/metrics", get(metrics_handler))
                .route("/api/stats", get(stats_handler))
                // API - Traces
                .route("/api/traces", get(traces_handler))
                // API - Sessions
                .route("/api/sessions", get(sessions_handler))
                .route("/api/sessions/:id", get(session_handler))
                .route("/api/sessions/:id/traces", get(session_traces_handler))
                .route("/api/sessions/:id/messages", get(session_messages_handler))
                // API - Agents
                .route("/api/agents", get(agents_handler))
                .route("/api/agents/:id", get(agent_handler))
                .route("/api/agents/:id/start", post(start_agent_handler))
                .route("/api/agents/:id/stop", post(stop_agent_handler))
                .route("/api/agents/:id/restart", post(restart_agent_handler))
                // Health
                .route("/health", get(health_handler))
                // Serve static files for studio WASM
                .nest_service(
                    "/studio/pkg",
                    ServeDir::new("crates/dashboard/static/studio/pkg"),
                )
                .layer(cors)
                .with_state(state);

            let listener = TcpListener::bind(&addr).await.unwrap();
            info!("Dashboard running at http://{}", addr);
            info!("  Legacy dashboard: http://{}/", addr);
            info!("  Agent Studio:     http://{}/studio", addr);
            axum::serve(listener, router).await.unwrap();
        })
    }
}
