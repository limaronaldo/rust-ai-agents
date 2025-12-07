//! Sessions page - Session Browser (#9)

use leptos::prelude::*;
use leptos_router::components::A;
use leptos_router::hooks::use_params_map;

use crate::api::{ApiClient, Session, SessionStatus};
use crate::components::{MessageList, StatusBadge};

/// Sessions list page
#[component]
pub fn SessionsPage() -> impl IntoView {
    let sessions =
        LocalResource::new(|| async { ApiClient::from_origin().get_sessions().await.ok() });

    let search = RwSignal::new(String::new());
    let status_filter = RwSignal::new(String::new());

    view! {
        <div>
            <div class="flex items-center justify-between mb-6">
                <h1 class="text-2xl font-bold text-white">"Sessions"</h1>
                <div class="flex items-center gap-4">
                    // Search input
                    <input
                        type="text"
                        placeholder="Search sessions..."
                        class="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                        on:input=move |e| {
                            search.set(event_target_value(&e));
                        }
                    />
                    // Status filter
                    <select
                        class="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
                        on:change=move |e| {
                            status_filter.set(event_target_value(&e));
                        }
                    >
                        <option value="">"All Status"</option>
                        <option value="active">"Active"</option>
                        <option value="completed">"Completed"</option>
                        <option value="failed">"Failed"</option>
                        <option value="archived">"Archived"</option>
                    </select>
                </div>
            </div>

            <Suspense fallback=|| view! { <LoadingSessions /> }>
                {move || {
                    sessions.get().map(|data| {
                        match data {
                            Some(entries) => {
                                let search_term = search.get().to_lowercase();
                                let status = status_filter.get();

                                let filtered: Vec<_> = entries
                                    .into_iter()
                                    .filter(|s| {
                                        // Search filter
                                        let matches_search = if search_term.is_empty() {
                                            true
                                        } else {
                                            s.id.to_lowercase().contains(&search_term)
                                                || s.name.as_ref().map(|n| n.to_lowercase().contains(&search_term)).unwrap_or(false)
                                        };

                                        // Status filter
                                        let matches_status = if status.is_empty() {
                                            true
                                        } else {
                                            match (&s.status, status.as_str()) {
                                                (SessionStatus::Active, "active") => true,
                                                (SessionStatus::Completed, "completed") => true,
                                                (SessionStatus::Failed, "failed") => true,
                                                (SessionStatus::Archived, "archived") => true,
                                                _ => false,
                                            }
                                        };

                                        matches_search && matches_status
                                    })
                                    .collect();

                                if filtered.is_empty() {
                                    view! { <EmptySessions /> }.into_any()
                                } else {
                                    view! { <SessionsList sessions=filtered /> }.into_any()
                                }
                            }
                            None => view! { <EmptySessions /> }.into_any(),
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

/// Sessions list component
#[component]
fn SessionsList(sessions: Vec<Session>) -> impl IntoView {
    view! {
        <div class="space-y-3">
            {sessions.into_iter().map(|session| {
                view! { <SessionCard session=session /> }
            }).collect::<Vec<_>>()}
        </div>
    }
}

/// Single session card
#[component]
fn SessionCard(session: Session) -> impl IntoView {
    let status = session.status.clone();
    let display_name = session
        .name
        .clone()
        .unwrap_or_else(|| format!("Session {}", &session.id[..8.min(session.id.len())]));
    let id_short = format!("{}...", &session.id[..12.min(session.id.len())]);
    let msg_count = session.message_count;
    let created = session.created_at.format("%Y-%m-%d %H:%M").to_string();
    let href = format!("/sessions/{}", session.id);
    let agent_id = session.agent_id.clone();

    view! {
        <A href=href attr:class="block bg-gray-800 rounded-lg p-4 hover:bg-gray-700/50 transition-colors">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <span class="text-2xl">"💬"</span>
                    <div>
                        <h3 class="font-medium text-white">{display_name}</h3>
                        <p class="text-sm text-gray-400">
                            <code>{id_short}</code>
                        </p>
                    </div>
                </div>
                <div class="flex items-center gap-4">
                    <div class="text-right">
                        <p class="text-sm text-gray-400">{msg_count}" messages"</p>
                        <p class="text-xs text-gray-500">{created}</p>
                    </div>
                    <StatusBadge status=status />
                </div>
            </div>
            {agent_id.map(|aid| {
                view! {
                    <div class="mt-2 pt-2 border-t border-gray-700">
                        <span class="text-xs text-gray-400">"Agent: "</span>
                        <span class="text-xs text-blue-400">{aid}</span>
                    </div>
                }
            })}
        </A>
    }
}

/// Session detail page showing messages
#[component]
pub fn SessionDetailPage() -> impl IntoView {
    let params = use_params_map();
    let session_id = move || params.read().get("id").unwrap_or_default();

    let session = LocalResource::new(move || {
        let id = session_id();
        async move { ApiClient::from_origin().get_session(&id).await.ok() }
    });

    let messages = LocalResource::new(move || {
        let id = session_id();
        async move {
            ApiClient::from_origin()
                .get_session_messages(&id)
                .await
                .ok()
        }
    });

    view! {
        <div>
            <div class="flex items-center gap-4 mb-6">
                <A href="/sessions" attr:class="text-gray-400 hover:text-white">
                    "← Back"
                </A>
                <h1 class="text-2xl font-bold text-white">
                    "Session Detail"
                </h1>
            </div>

            // Session info header
            <Suspense fallback=|| view! { <LoadingSession /> }>
                {move || {
                    session.get().map(|data| {
                        match data {
                            Some(s) => view! { <SessionHeader session=s /> }.into_any(),
                            None => view! {
                                <div class="bg-gray-800 rounded-lg p-4 mb-6">
                                    <p class="text-gray-400">"Session not found"</p>
                                </div>
                            }.into_any(),
                        }
                    })
                }}
            </Suspense>

            // Messages list
            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-lg font-semibold text-white mb-4">"Messages"</h2>
                <Suspense fallback=|| view! { <LoadingMessages /> }>
                    {move || {
                        messages.get().map(|data| {
                            match data {
                                Some(msgs) if !msgs.is_empty() => {
                                    view! { <MessageList messages=msgs /> }.into_any()
                                }
                                _ => view! { <EmptyMessages /> }.into_any(),
                            }
                        })
                    }}
                </Suspense>
            </div>

            // Link to trace view
            <div class="mt-4">
                <A href=move || format!("/traces/{}", session_id()) attr:class="text-blue-400 hover:text-blue-300 text-sm">
                    "View execution traces →"
                </A>
            </div>
        </div>
    }
}

/// Session header with metadata
#[component]
fn SessionHeader(session: Session) -> impl IntoView {
    let status = session.status.clone();
    let name = session
        .name
        .clone()
        .unwrap_or_else(|| "Unnamed Session".to_string());
    let id = session.id.clone();
    let created = session.created_at.format("%Y-%m-%d %H:%M:%S").to_string();
    let updated = session.updated_at.format("%Y-%m-%d %H:%M:%S").to_string();
    let msg_count = session.message_count;
    let agent_id = session.agent_id.clone();

    view! {
        <div class="bg-gray-800 rounded-lg p-4 mb-6">
            <div class="flex items-center justify-between mb-4">
                <div>
                    <h2 class="text-xl font-semibold text-white">{name}</h2>
                    <code class="text-sm text-gray-400">{id}</code>
                </div>
                <StatusBadge status=status />
            </div>

            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                    <p class="text-gray-400">"Created"</p>
                    <p class="text-white">{created}</p>
                </div>
                <div>
                    <p class="text-gray-400">"Updated"</p>
                    <p class="text-white">{updated}</p>
                </div>
                <div>
                    <p class="text-gray-400">"Messages"</p>
                    <p class="text-white">{msg_count}</p>
                </div>
                {agent_id.map(|aid| {
                    view! {
                        <div>
                            <p class="text-gray-400">"Agent"</p>
                            <p class="text-blue-400">{aid}</p>
                        </div>
                    }
                })}
            </div>
        </div>
    }
}

#[component]
fn LoadingSessions() -> impl IntoView {
    view! {
        <div class="flex items-center justify-center p-8">
            <div class="text-gray-400">"Loading sessions..."</div>
        </div>
    }
}

#[component]
fn LoadingSession() -> impl IntoView {
    view! {
        <div class="bg-gray-800 rounded-lg p-4 mb-6 animate-pulse">
            <div class="h-6 bg-gray-700 rounded w-1/3 mb-2"></div>
            <div class="h-4 bg-gray-700 rounded w-1/2"></div>
        </div>
    }
}

#[component]
fn LoadingMessages() -> impl IntoView {
    view! {
        <div class="space-y-3">
            <div class="bg-gray-700/50 rounded-lg p-4 animate-pulse">
                <div class="h-4 bg-gray-600 rounded w-1/4 mb-2"></div>
                <div class="h-3 bg-gray-600 rounded w-3/4"></div>
            </div>
            <div class="bg-gray-700/50 rounded-lg p-4 animate-pulse">
                <div class="h-4 bg-gray-600 rounded w-1/4 mb-2"></div>
                <div class="h-3 bg-gray-600 rounded w-2/3"></div>
            </div>
        </div>
    }
}

#[component]
fn EmptySessions() -> impl IntoView {
    view! {
        <div class="flex flex-col items-center justify-center p-8 bg-gray-800 rounded-lg">
            <span class="text-4xl mb-4">"💬"</span>
            <p class="text-gray-400">"No sessions found"</p>
            <p class="text-sm text-gray-500">"Sessions will appear here when agents start conversations"</p>
        </div>
    }
}

#[component]
fn EmptyMessages() -> impl IntoView {
    view! {
        <div class="flex flex-col items-center justify-center p-4">
            <p class="text-gray-400">"No messages yet"</p>
        </div>
    }
}
