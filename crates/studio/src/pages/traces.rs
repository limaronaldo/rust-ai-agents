//! Traces page - Trace Viewer (#8)

use leptos::prelude::*;
use leptos_router::components::A;
use leptos_router::hooks::use_params_map;

use crate::api::{ApiClient, TraceEntry};
use crate::components::TraceTimeline;

/// Traces list page
#[component]
pub fn TracesPage() -> impl IntoView {
    let traces = LocalResource::new(|| async { ApiClient::from_origin().get_traces().await.ok() });

    let search = RwSignal::new(String::new());

    view! {
        <div>
            <div class="flex items-center justify-between mb-6">
                <h1 class="text-2xl font-bold text-white">"Traces"</h1>
                <div class="flex items-center gap-4">
                    // Search input
                    <input
                        type="text"
                        placeholder="Search traces..."
                        class="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                        on:input=move |e| {
                            search.set(event_target_value(&e));
                        }
                    />
                    // Filter dropdown (placeholder)
                    <select class="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                        <option>"All Types"</option>
                        <option>"LLM Requests"</option>
                        <option>"Tool Calls"</option>
                        <option>"Errors"</option>
                    </select>
                </div>
            </div>

            <Suspense fallback=|| view! { <LoadingTraces /> }>
                {move || {
                    traces.get().map(|data| {
                        match data {
                            Some(entries) => {
                                let search_term = search.get();
                                let filtered: Vec<_> = entries
                                    .into_iter()
                                    .filter(|e| {
                                        if search_term.is_empty() {
                                            true
                                        } else {
                                            e.session_id.contains(&search_term)
                                        }
                                    })
                                    .collect();

                                if filtered.is_empty() {
                                    view! { <EmptyTraces /> }.into_any()
                                } else {
                                    view! { <TracesList traces=filtered /> }.into_any()
                                }
                            }
                            None => view! { <EmptyTraces /> }.into_any(),
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

/// Traces list grouped by session
#[component]
fn TracesList(traces: Vec<TraceEntry>) -> impl IntoView {
    // Group by session
    let mut sessions: std::collections::HashMap<String, Vec<TraceEntry>> =
        std::collections::HashMap::new();
    for trace in traces {
        sessions
            .entry(trace.session_id.clone())
            .or_default()
            .push(trace);
    }

    let mut session_list: Vec<_> = sessions.into_iter().collect();
    session_list.sort_by(|a, b| {
        let a_time = a.1.first().map(|t| t.timestamp);
        let b_time = b.1.first().map(|t| t.timestamp);
        b_time.cmp(&a_time)
    });

    view! {
        <div class="space-y-4">
            {session_list.into_iter().map(|(session_id, entries)| {
                let entry_count = entries.len();
                let first_time = entries.first().map(|e| e.timestamp.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or_default();
                let href = format!("/traces/{}", session_id);
                let id_short = format!("{}...", &session_id[..8.min(session_id.len())]);

                view! {
                    <div class="bg-gray-800 rounded-lg overflow-hidden">
                        <A href=href attr:class="block p-4 hover:bg-gray-700/50 transition-colors">
                            <div class="flex items-center justify-between">
                                <div>
                                    <span class="font-medium text-white">"Session "</span>
                                    <code class="text-blue-400">{id_short}</code>
                                </div>
                                <div class="flex items-center gap-4 text-sm text-gray-400">
                                    <span>{entry_count}" entries"</span>
                                    <span>{first_time}</span>
                                </div>
                            </div>
                        </A>
                    </div>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}

/// Trace detail page showing all entries for a session
#[component]
pub fn TraceDetailPage() -> impl IntoView {
    let params = use_params_map();
    let session_id = move || params.read().get("id").unwrap_or_default();

    let traces = LocalResource::new(move || {
        let id = session_id();
        async move { ApiClient::from_origin().get_session_traces(&id).await.ok() }
    });

    view! {
        <div>
            <div class="flex items-center gap-4 mb-6">
                <A href="/traces" attr:class="text-gray-400 hover:text-white">
                    "← Back"
                </A>
                <h1 class="text-2xl font-bold text-white">
                    "Trace Detail"
                </h1>
            </div>

            <div class="bg-gray-800 rounded-lg p-4 mb-6">
                <div class="flex items-center gap-2">
                    <span class="text-gray-400">"Session ID:"</span>
                    <code class="text-blue-400">{session_id}</code>
                </div>
            </div>

            <Suspense fallback=|| view! { <LoadingTraces /> }>
                {move || {
                    traces.get().map(|data| {
                        match data {
                            Some(entries) if !entries.is_empty() => {
                                view! { <TraceTimeline entries=entries /> }.into_any()
                            }
                            _ => view! { <EmptyTraces /> }.into_any(),
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

#[component]
fn LoadingTraces() -> impl IntoView {
    view! {
        <div class="flex items-center justify-center p-8">
            <div class="text-gray-400">"Loading traces..."</div>
        </div>
    }
}

#[component]
fn EmptyTraces() -> impl IntoView {
    view! {
        <div class="flex flex-col items-center justify-center p-8 bg-gray-800 rounded-lg">
            <span class="text-4xl mb-4">"📊"</span>
            <p class="text-gray-400">"No traces found"</p>
            <p class="text-sm text-gray-500">"Traces will appear here when agents execute"</p>
        </div>
    }
}
