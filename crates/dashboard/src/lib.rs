//! # Web Dashboard
//!
//! Real-time web dashboard for monitoring Rust AI Agents.
//!
//! ## Features
//!
//! - WebSocket-based real-time updates
//! - Cost and token metrics visualization
//! - Agent status monitoring
//! - Request history and latency graphs
//!
//! ## Usage
//!
//! ```rust,ignore
//! use rust_ai_agents_dashboard::DashboardServer;
//! use rust_ai_agents_monitoring::CostTracker;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     let cost_tracker = Arc::new(CostTracker::new());
//!     let server = DashboardServer::new(cost_tracker);
//!     server.run("127.0.0.1:3000").await.unwrap();
//! }
//! ```

mod handlers;
mod server;
mod state;
mod websocket;

pub use server::DashboardServer;
pub use state::{AgentStatus, DashboardMetrics, DashboardState};
