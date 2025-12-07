//! HTTP request handlers

use axum::{
    extract::{Path, State, WebSocketUpgrade},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
};
use std::sync::Arc;

use crate::state::DashboardState;
use crate::websocket::handle_websocket;

/// Serve the legacy dashboard HTML page
pub async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// Serve the new Agent Studio HTML page
pub async fn studio_handler() -> Html<&'static str> {
    Html(include_str!("../static/studio/index.html"))
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

/// Get all traces
pub async fn traces_handler(State(state): State<Arc<DashboardState>>) -> Json<serde_json::Value> {
    let traces = state.get_traces();
    Json(serde_json::to_value(traces).unwrap_or_default())
}

/// Get traces for a specific session
pub async fn session_traces_handler(
    Path(session_id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> Json<serde_json::Value> {
    let traces = state.get_session_traces(&session_id);
    Json(serde_json::to_value(traces).unwrap_or_default())
}

/// Get all sessions
pub async fn sessions_handler(State(state): State<Arc<DashboardState>>) -> Json<serde_json::Value> {
    let sessions = state.get_sessions();
    Json(serde_json::to_value(sessions).unwrap_or_default())
}

/// Get a specific session
pub async fn session_handler(
    Path(id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    match state.get_session(&id) {
        Some(session) => Json(serde_json::to_value(session).unwrap_or_default()).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Get messages for a session
pub async fn session_messages_handler(
    Path(session_id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> Json<serde_json::Value> {
    let messages = state.get_session_messages(&session_id);
    Json(serde_json::to_value(messages).unwrap_or_default())
}

/// Get all agents
pub async fn agents_handler(State(state): State<Arc<DashboardState>>) -> Json<serde_json::Value> {
    let agents = state.get_agents();
    Json(serde_json::to_value(agents).unwrap_or_default())
}

/// Get a specific agent
pub async fn agent_handler(
    Path(id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    match state.get_agent(&id) {
        Some(agent) => Json(serde_json::to_value(agent).unwrap_or_default()).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Start an agent
pub async fn start_agent_handler(
    Path(id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    match state.start_agent(&id) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e).into_response(),
    }
}

/// Stop an agent
pub async fn stop_agent_handler(
    Path(id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    match state.stop_agent(&id) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e).into_response(),
    }
}

/// Restart an agent
pub async fn restart_agent_handler(
    Path(id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    match state.restart_agent(&id) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e).into_response(),
    }
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

/// Prometheus metrics endpoint
///
/// Returns metrics in Prometheus text format for scraping.
/// This endpoint is separate from the JSON metrics API to support Prometheus scraping.
pub async fn prometheus_metrics_handler(State(state): State<Arc<DashboardState>>) -> String {
    state.prometheus_handle.render()
}
