//! Sidebar navigation component

use leptos::prelude::*;
use leptos_router::components::A;

use crate::api::WsState;
use crate::WsContext;

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
            <nav class="flex-1 p-4 space-y-1">
                <NavSection title="Overview">
                    <NavLink href="/" icon="🏠">"Dashboard"</NavLink>
                    <NavLink href="/metrics" icon="📈">"Metrics"</NavLink>
                </NavSection>

                <NavSection title="Monitor">
                    <NavLink href="/traces" icon="📊">"Traces"</NavLink>
                    <NavLink href="/sessions" icon="💬">"Sessions"</NavLink>
                </NavSection>

                <NavSection title="Control">
                    <NavLink href="/agents" icon="🤖">"Agents"</NavLink>
                </NavSection>

                <NavSection title="System">
                    <NavLink href="/settings" icon="⚙️">"Settings"</NavLink>
                </NavSection>
            </nav>

            // Connection Status
            <div class="p-4 border-t border-gray-700">
                <ConnectionStatus />
            </div>
        </aside>
    }
}

/// Navigation section with title
#[component]
fn NavSection(title: &'static str, children: Children) -> impl IntoView {
    view! {
        <div class="mb-4">
            <p class="px-3 mb-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                {title}
            </p>
            <div class="space-y-1">
                {children()}
            </div>
        </div>
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

/// WebSocket connection status indicator (uses global context)
#[component]
fn ConnectionStatus() -> impl IntoView {
    let ws_context = use_context::<WsContext>();

    view! {
        {move || {
            if let Some(ctx) = ws_context.clone() {
                let state = ctx.state;
                let connect = ctx.connect;

                view! {
                    <div class="space-y-2">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center gap-2 text-sm">
                                <span
                                    class="w-2 h-2 rounded-full"
                                    class:bg-green-500=move || matches!(state.get(), WsState::Connected)
                                    class:bg-yellow-500=move || matches!(state.get(), WsState::Connecting)
                                    class:bg-red-500=move || matches!(state.get(), WsState::Disconnected | WsState::Error(_))
                                />
                                <span class="text-gray-400">
                                    {move || match state.get() {
                                        WsState::Connected => "Connected",
                                        WsState::Connecting => "Connecting...",
                                        WsState::Disconnected => "Disconnected",
                                        WsState::Error(_) => "Error",
                                    }}
                                </span>
                            </div>
                        </div>

                        // Reconnect button when disconnected
                        {move || {
                            let is_disconnected = matches!(state.get(), WsState::Disconnected | WsState::Error(_));
                            is_disconnected.then(|| {
                                view! {
                                    <button
                                        class="w-full px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors"
                                        on:click=move |_| connect.with_value(|f| f())
                                    >
                                        "Reconnect"
                                    </button>
                                }
                            })
                        }}

                        // Error message
                        {move || {
                            if let WsState::Error(msg) = state.get() {
                                let msg_clone = msg.clone();
                                Some(view! {
                                    <p class="text-xs text-red-400 truncate" title=msg_clone>
                                        {msg}
                                    </p>
                                })
                            } else {
                                None
                            }
                        }}
                    </div>
                }.into_any()
            } else {
                // Fallback if no context
                view! {
                    <div class="flex items-center gap-2 text-sm">
                        <span class="w-2 h-2 rounded-full bg-gray-500" />
                        <span class="text-gray-400">"No connection"</span>
                    </div>
                }.into_any()
            }
        }}
    }
}
