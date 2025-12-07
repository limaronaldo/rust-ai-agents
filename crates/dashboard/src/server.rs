//! Dashboard HTTP server

use axum::{routing::get, Router};
use rust_ai_agents_monitoring::CostTracker;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::handlers::{health_handler, index_handler, metrics_handler, stats_handler, ws_handler};
use crate::state::DashboardState;

/// Dashboard web server
pub struct DashboardServer {
    state: Arc<DashboardState>,
}

impl DashboardServer {
    /// Create a new dashboard server
    pub fn new(cost_tracker: Arc<CostTracker>) -> Self {
        Self {
            state: Arc::new(DashboardState::new(cost_tracker)),
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
            .route("/", get(index_handler))
            .route("/ws", get(ws_handler))
            .route("/api/metrics", get(metrics_handler))
            .route("/api/stats", get(stats_handler))
            .route("/health", get(health_handler))
            .layer(cors)
            .with_state(self.state.clone())
    }

    /// Run the dashboard server
    pub async fn run(self, addr: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let router = self.build_router();
        let listener = TcpListener::bind(addr).await?;

        info!("Dashboard running at http://{}", addr);

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

            let router = Router::new()
                .route("/", get(index_handler))
                .route("/ws", get(ws_handler))
                .route("/api/metrics", get(metrics_handler))
                .route("/api/stats", get(stats_handler))
                .route("/health", get(health_handler))
                .layer(
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_methods(Any)
                        .allow_headers(Any),
                )
                .with_state(state);

            let listener = TcpListener::bind(&addr).await.unwrap();
            info!("Dashboard running at http://{}", addr);
            axum::serve(listener, router).await.unwrap();
        })
    }
}
