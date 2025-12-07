//! # Web Dashboard
//!
//! Real-time web dashboard for monitoring Rust AI Agents.

#![allow(clippy::type_complexity)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_match)]
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

#[cfg(feature = "agents")]
pub mod integration;

#[cfg(feature = "audit")]
pub mod audit;

#[cfg(feature = "streaming")]
pub mod streaming;

pub use server::DashboardServer;
pub use state::{
    AgentStatus, DashboardMetrics, DashboardState, Session, SessionStatus, TraceEntry,
    TraceEntryType,
};

#[cfg(feature = "agents")]
pub use integration::{
    add_demo_data, AgentBridge, DashboardBridge, SessionBridge, TrajectoryBridge,
};

#[cfg(feature = "audit")]
pub use audit::{
    audit_middleware, audit_routes, AuditEventResponse, AuditQuery, AuditState, AuditStats,
};

#[cfg(feature = "streaming")]
pub use streaming::{streaming_routes, StreamEventOutput, StreamInferenceRequest, StreamingState};
