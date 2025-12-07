//! HTTP request handlers

use axum::{
    extract::{State, WebSocketUpgrade},
    response::{Html, IntoResponse, Json},
};
use std::sync::Arc;

use crate::state::DashboardState;
use crate::websocket::handle_websocket;

/// Serve the dashboard HTML page
pub async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// Get current metrics as JSON
pub async fn metrics_handler(State(state): State<Arc<DashboardState>>) -> Json<serde_json::Value> {
    let metrics = state.get_metrics();
    Json(serde_json::to_value(metrics).unwrap_or_default())
}

/// Get cost stats as JSON
pub async fn stats_handler(State(state): State<Arc<DashboardState>>) -> Json<serde_json::Value> {
    let stats = state.cost_tracker.stats();
    Json(serde_json::to_value(stats).unwrap_or_default())
}

/// WebSocket upgrade handler
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_websocket(socket, state))
}

/// Health check endpoint
pub async fn health_handler() -> &'static str {
    "OK"
}
