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
//! - **Agent Control** - Start, stop, and restart agents
//! - **Metrics Dashboard** - Real-time charts and statistics
//! - **Settings** - Configure connection and display options
//! - **Live Updates** - WebSocket connection for real-time data
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                   Agent Studio                       │
//! │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐ │
//! │  │ Traces  │ │Sessions │ │ Agents  │ │  Metrics  │ │
//! │  └─────────┘ └─────────┘ └─────────┘ └───────────┘ │
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

use api::WsMessage;

/// Global WebSocket state context
#[derive(Clone)]
pub struct WsContext {
    pub state: RwSignal<WsState>,
    pub last_message: RwSignal<Option<WsMessage>>,
    pub connect: StoredValue<Box<dyn Fn() + Send + Sync>>,
}

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
    // Create global WebSocket state
    let ws_state = RwSignal::new(WsState::Disconnected);
    let ws_message = RwSignal::new(None::<WsMessage>);

    // Create connect function
    let connect_fn = StoredValue::new(Box::new({
        let state = ws_state;
        let message = ws_message;
        move || {
            state.set(WsState::Connecting);
            let mut client = api::WsClient::from_origin();
            let _ = client.connect(
                move |msg| {
                    message.set(Some(msg));
                },
                move |st| {
                    state.set(st);
                },
            );
        }
    }) as Box<dyn Fn() + Send + Sync>);

    // Provide context
    let ws_context = WsContext {
        state: ws_state,
        last_message: ws_message,
        connect: connect_fn,
    };
    provide_context(ws_context.clone());

    // Auto-connect on mount if enabled
    Effect::new(move |_| {
        // Check settings for auto-connect
        let window = web_sys::window().expect("no window");
        if let Ok(Some(storage)) = window.local_storage() {
            if let Ok(Some(json)) = storage.get_item("agent_studio_settings") {
                if let Ok(settings) = serde_json::from_str::<serde_json::Value>(&json) {
                    if settings["auto_connect"].as_bool().unwrap_or(true) {
                        ws_context.connect.with_value(|f| f());
                    }
                    return;
                }
            }
        }
        // Default: auto-connect
        ws_context.connect.with_value(|f| f());
    });

    view! {
        <Router>
            <components::Layout>
                <Routes fallback=|| view! { <pages::NotFound /> }>
                    <Route path=path!("/") view=pages::HomePage />
                    <Route path=path!("/traces") view=pages::TracesPage />
                    <Route path=path!("/traces/:id") view=pages::TraceDetailPage />
                    <Route path=path!("/sessions") view=pages::SessionsPage />
                    <Route path=path!("/sessions/:id") view=pages::SessionDetailPage />
                    <Route path=path!("/agents") view=pages::AgentsPage />
                    <Route path=path!("/metrics") view=pages::MetricsPage />
                    <Route path=path!("/settings") view=pages::SettingsPage />
                </Routes>
            </components::Layout>
        </Router>
    }
}
