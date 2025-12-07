//! Agents page - Agent Control Panel

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::api::{AgentStatus, ApiClient};

/// Agents control panel page
#[component]
pub fn AgentsPage() -> impl IntoView {
    let agents = LocalResource::new(|| async { ApiClient::from_origin().get_agents().await.ok() });

    // Trigger for refreshing agents
    let refresh_trigger = RwSignal::new(0);

    let refresh = move |_| {
        refresh_trigger.update(|n| *n += 1);
    };

    view! {
        <div>
            <div class="flex items-center justify-between mb-6">
                <h1 class="text-2xl font-bold text-white">"Agents"</h1>
                <button
                    class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2"
                    on:click=refresh
                >
                    <span>"🔄"</span>
                    <span>"Refresh"</span>
                </button>
            </div>

            <Suspense fallback=|| view! { <LoadingAgents /> }>
                {move || {
                    // Re-fetch when refresh_trigger changes
                    let _ = refresh_trigger.get();
                    agents.get().map(|data| {
                        match data {
                            Some(agents) if !agents.is_empty() => {
                                view! { <AgentsList agents=agents /> }.into_any()
                            }
                            _ => view! { <EmptyAgents /> }.into_any(),
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

/// List of agent cards
#[component]
fn AgentsList(agents: Vec<AgentStatus>) -> impl IntoView {
    view! {
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {agents.into_iter().map(|agent| {
                view! { <AgentCard agent=agent /> }
            }).collect::<Vec<_>>()}
        </div>
    }
}

/// Single agent card with controls
#[component]
fn AgentCard(agent: AgentStatus) -> impl IntoView {
    let agent_id = agent.id.clone();
    let is_running = agent.status == "running";

    let action_pending = RwSignal::new(false);
    let action_result = RwSignal::new(None::<Result<String, String>>);

    let start_agent = {
        let id = agent_id.clone();
        move |_| {
            let id = id.clone();
            action_pending.set(true);
            action_result.set(None);
            spawn_local(async move {
                let result = ApiClient::from_origin().start_agent(&id).await;
                action_pending.set(false);
                action_result.set(Some(
                    result
                        .map(|_| "Agent started".to_string())
                        .map_err(|e| e.to_string()),
                ));
            });
        }
    };

    let stop_agent = {
        let id = agent_id.clone();
        move |_| {
            let id = id.clone();
            action_pending.set(true);
            action_result.set(None);
            spawn_local(async move {
                let result = ApiClient::from_origin().stop_agent(&id).await;
                action_pending.set(false);
                action_result.set(Some(
                    result
                        .map(|_| "Agent stopped".to_string())
                        .map_err(|e| e.to_string()),
                ));
            });
        }
    };

    let restart_agent = {
        let id = agent_id.clone();
        move |_| {
            let id = id.clone();
            action_pending.set(true);
            action_result.set(None);
            spawn_local(async move {
                let result = ApiClient::from_origin().restart_agent(&id).await;
                action_pending.set(false);
                action_result.set(Some(
                    result
                        .map(|_| "Agent restarted".to_string())
                        .map_err(|e| e.to_string()),
                ));
            });
        }
    };

    let status_class = match agent.status.as_str() {
        "running" => "bg-green-500",
        "idle" => "bg-yellow-500",
        "error" => "bg-red-500",
        "stopped" => "bg-gray-500",
        _ => "bg-gray-500",
    };

    let name = agent.name.clone();
    let role = agent.role.clone();
    let status = agent.status.clone();
    let msgs = agent.messages_processed;
    let task = agent.current_task.clone();
    let last_activity = agent
        .last_activity
        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string());

    view! {
        <div class="bg-gray-800 rounded-lg p-4">
            // Header
            <div class="flex items-center justify-between mb-4">
                <div class="flex items-center gap-3">
                    <span class="text-2xl">"🤖"</span>
                    <div>
                        <h3 class="font-semibold text-white">{name}</h3>
                        <p class="text-sm text-gray-400">{role}</p>
                    </div>
                </div>
                <div class="flex items-center gap-2">
                    <span class=format!("w-3 h-3 rounded-full {}", status_class)></span>
                    <span class="text-sm text-gray-300 capitalize">{status}</span>
                </div>
            </div>

            // Stats
            <div class="grid grid-cols-2 gap-4 mb-4 text-sm">
                <div>
                    <p class="text-gray-400">"Messages Processed"</p>
                    <p class="text-white font-medium">{msgs}</p>
                </div>
                <div>
                    <p class="text-gray-400">"Last Activity"</p>
                    <p class="text-white font-medium">
                        {last_activity.unwrap_or_else(|| "Never".to_string())}
                    </p>
                </div>
            </div>

            // Current task
            {task.map(|t| view! {
                <div class="mb-4 p-3 bg-gray-700/50 rounded-lg">
                    <p class="text-xs text-gray-400 mb-1">"Current Task"</p>
                    <p class="text-sm text-white">{t}</p>
                </div>
            })}

            // Action result
            {move || action_result.get().map(|result| {
                match result {
                    Ok(msg) => view! {
                        <div class="mb-4 p-2 bg-green-900/50 border border-green-700 rounded text-sm text-green-400">
                            {msg}
                        </div>
                    }.into_any(),
                    Err(err) => view! {
                        <div class="mb-4 p-2 bg-red-900/50 border border-red-700 rounded text-sm text-red-400">
                            {err}
                        </div>
                    }.into_any(),
                }
            })}

            // Controls
            <div class="flex items-center gap-2">
                <button
                    class="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-green-900 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm"
                    disabled=move || action_pending.get() || is_running
                    on:click=start_agent
                >
                    {move || if action_pending.get() { "..." } else { "▶ Start" }}
                </button>
                <button
                    class="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-900 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm"
                    disabled=move || action_pending.get() || !is_running
                    on:click=stop_agent
                >
                    {move || if action_pending.get() { "..." } else { "⏹ Stop" }}
                </button>
                <button
                    class="flex-1 px-3 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-yellow-900 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm"
                    disabled=move || action_pending.get()
                    on:click=restart_agent
                >
                    {move || if action_pending.get() { "..." } else { "🔄 Restart" }}
                </button>
            </div>
        </div>
    }
}

#[component]
fn LoadingAgents() -> impl IntoView {
    view! {
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="bg-gray-800 rounded-lg p-4 animate-pulse">
                <div class="h-6 bg-gray-700 rounded w-1/3 mb-4"></div>
                <div class="h-4 bg-gray-700 rounded w-1/2 mb-2"></div>
                <div class="h-4 bg-gray-700 rounded w-2/3"></div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4 animate-pulse">
                <div class="h-6 bg-gray-700 rounded w-1/3 mb-4"></div>
                <div class="h-4 bg-gray-700 rounded w-1/2 mb-2"></div>
                <div class="h-4 bg-gray-700 rounded w-2/3"></div>
            </div>
        </div>
    }
}

#[component]
fn EmptyAgents() -> impl IntoView {
    view! {
        <div class="flex flex-col items-center justify-center p-8 bg-gray-800 rounded-lg">
            <span class="text-4xl mb-4">"🤖"</span>
            <p class="text-gray-400">"No agents registered"</p>
            <p class="text-sm text-gray-500">"Agents will appear here when they connect to the system"</p>
        </div>
    }
}
