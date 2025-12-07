//! WebSocket handler for real-time updates

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::state::{DashboardState, WsMessage};

/// Handle a WebSocket connection
pub async fn handle_websocket(socket: WebSocket, state: Arc<DashboardState>) {
    let conn_id = Uuid::new_v4();
    info!("WebSocket connected: {}", conn_id);

    let (mut sender, mut receiver) = socket.split();

    // Subscribe to broadcast updates
    let mut rx = state.subscribe();

    // Send initial metrics
    let initial_metrics = state.get_metrics();
    let msg = WsMessage::Metrics(initial_metrics);
    if let Ok(json) = serde_json::to_string(&msg) {
        if sender.send(Message::Text(json.into())).await.is_err() {
            error!("Failed to send initial metrics to {}", conn_id);
            return;
        }
    }

    // Spawn task to forward broadcast messages to this client
    let send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    if let Ok(json) = serde_json::to_string(&msg) {
                        if sender.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    debug!("WebSocket {} lagged by {} messages", conn_id, n);
                }
                Err(broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    });

    // Handle incoming messages from client
    let recv_task = tokio::spawn(async move {
        while let Some(result) = receiver.next().await {
            match result {
                Ok(Message::Text(text)) => {
                    // Handle client messages (e.g., ping)
                    if let Ok(msg) = serde_json::from_str::<WsMessage>(&text) {
                        if let WsMessage::Ping = msg {
                            debug!("Received ping from {}", conn_id);
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket {} closed by client", conn_id);
                    break;
                }
                Err(e) => {
                    error!("WebSocket error for {}: {}", conn_id, e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = send_task => {}
        _ = recv_task => {}
    }

    info!("WebSocket disconnected: {}", conn_id);
}
