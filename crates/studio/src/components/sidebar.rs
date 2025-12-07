//! Sidebar navigation component

use leptos::prelude::*;
use leptos_router::components::A;

/// Sidebar with navigation links
#[component]
pub fn Sidebar() -> impl IntoView {
    view! {
        <aside class="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
            // Logo/Title
            <div class="p-4 border-b border-gray-700">
                <h1 class="text-xl font-bold text-white">"Agent Studio"</h1>
                <p class="text-sm text-gray-400">"Monitor & Debug AI Agents"</p>
            </div>

            // Navigation
            <nav class="flex-1 p-4 space-y-2">
                <NavLink href="/" icon="🏠">"Dashboard"</NavLink>
                <NavLink href="/traces" icon="📊">"Traces"</NavLink>
                <NavLink href="/sessions" icon="💬">"Sessions"</NavLink>
            </nav>

            // Status
            <div class="p-4 border-t border-gray-700">
                <ConnectionStatus />
            </div>
        </aside>
    }
}

/// Navigation link component
#[component]
fn NavLink(href: &'static str, icon: &'static str, children: Children) -> impl IntoView {
    view! {
        <A href=href attr:class="flex items-center gap-3 px-3 py-2 rounded-lg text-gray-300 hover:bg-gray-700 hover:text-white transition-colors">
            <span class="text-lg">{icon}</span>
            <span>{children()}</span>
        </A>
    }
}

/// WebSocket connection status indicator
#[component]
fn ConnectionStatus() -> impl IntoView {
    // TODO: Connect to actual WebSocket state
    let connected = RwSignal::new(false);

    view! {
        <div class="flex items-center gap-2 text-sm">
            <span
                class="w-2 h-2 rounded-full"
                class:bg-green-500=move || connected.get()
                class:bg-red-500=move || !connected.get()
            />
            <span class="text-gray-400">
                {move || if connected.get() { "Connected" } else { "Disconnected" }}
            </span>
        </div>
    }
}
