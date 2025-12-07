//! # Agent Studio
//!
//! Web UI for monitoring and debugging AI agents.
//!
//! Built with Leptos 0.8 for a fully Rust frontend experience.
//!
//! ## Features
//!
//! - **Trace Viewer** - Monitor agent executions in real-time
//! - **Session Browser** - View and manage conversation sessions
//! - **Live Updates** - WebSocket connection for real-time data
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                   Agent Studio                       │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
//! │  │   Traces    │  │  Sessions   │  │   Metrics   │ │
//! │  └─────────────┘  └─────────────┘  └─────────────┘ │
//! │                        │                            │
//! │                   WebSocket                         │
//! │                        │                            │
//! └────────────────────────┼────────────────────────────┘
//!                          │
//!                   ┌──────┴──────┐
//!                   │  Dashboard  │
//!                   │   (Axum)    │
//!                   └─────────────┘
//! ```

use leptos::prelude::*;
use leptos_router::components::{Route, Router, Routes};
use leptos_router::path;
use wasm_bindgen::prelude::*;

pub mod api;
pub mod components;
pub mod pages;

// Re-exports for external use
pub use api::{ApiClient, WsClient, WsState};

/// Initialize the Studio application
#[wasm_bindgen(start)]
pub fn main() {
    // Set up panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize tracing
    tracing_wasm::set_as_global_default();

    // Mount the app
    leptos::mount::mount_to_body(App);
}

/// Main application component
#[component]
pub fn App() -> impl IntoView {
    view! {
        <Router>
            <components::Layout>
                <Routes fallback=|| view! { <pages::NotFound /> }>
                    <Route path=path!("/") view=pages::HomePage />
                    <Route path=path!("/traces") view=pages::TracesPage />
                    <Route path=path!("/traces/:id") view=pages::TraceDetailPage />
                    <Route path=path!("/sessions") view=pages::SessionsPage />
                    <Route path=path!("/sessions/:id") view=pages::SessionDetailPage />
                </Routes>
            </components::Layout>
        </Router>
    }
}
