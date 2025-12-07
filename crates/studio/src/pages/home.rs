//! Home/Dashboard page

use leptos::prelude::*;

use crate::api::{ApiClient, DashboardMetrics};
use crate::components::AgentStatusBadge;

/// Home page with overview metrics
#[component]
pub fn HomePage() -> impl IntoView {
    let metrics =
        LocalResource::new(|| async { ApiClient::from_origin().get_metrics().await.ok() });

    view! {
        <div>
            <h1 class="text-2xl font-bold text-white mb-6">"Dashboard"</h1>

            <Suspense fallback=|| view! { <LoadingState /> }>
                {move || {
                    metrics.get().map(|data| {
                        match data {
                            Some(m) => view! { <MetricsDisplay metrics=m /> }.into_any(),
                            None => view! { <EmptyState message="Failed to load metrics" /> }.into_any(),
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

/// Display metrics cards
#[component]
fn MetricsDisplay(metrics: DashboardMetrics) -> impl IntoView {
    view! {
        <div>
            // Stats cards
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <StatCard
                    title="Total Cost"
                    value=format!("${:.4}", metrics.cost_stats.total_cost)
                    icon="💰"
                />
                <StatCard
                    title="Total Tokens"
                    value=format_number(metrics.cost_stats.total_tokens)
                    icon="🔤"
                />
                <StatCard
                    title="Active Agents"
                    value=metrics.active_agents.to_string()
                    icon="🤖"
                />
                <StatCard
                    title="Avg Latency"
                    value=format!("{:.0}ms", metrics.cost_stats.avg_latency_ms)
                    icon="⚡"
                />
            </div>

            // Agents list
            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-lg font-semibold text-white mb-4">"Agents"</h2>
                {if metrics.agents.is_empty() {
                    view! { <p class="text-gray-400">"No agents registered"</p> }.into_any()
                } else {
                    let agents = metrics.agents;
                    view! {
                        <div class="space-y-2">
                            {agents.into_iter().map(|agent| {
                                let name = agent.name.clone();
                                let role = agent.role.clone();
                                let status = agent.status.clone();
                                let msgs = agent.messages_processed;
                                view! {
                                    <div class="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                                        <div>
                                            <span class="font-medium text-white">{name}</span>
                                            <span class="text-sm text-gray-400 ml-2">{role}</span>
                                        </div>
                                        <div class="flex items-center gap-3">
                                            <span class="text-sm text-gray-400">
                                                {msgs}" msgs"
                                            </span>
                                            <AgentStatusBadge status=status />
                                        </div>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }}
            </div>
        </div>
    }
}

/// Single stat card
#[component]
fn StatCard(title: &'static str, value: String, icon: &'static str) -> impl IntoView {
    view! {
        <div class="bg-gray-800 rounded-lg p-4">
            <div class="flex items-center gap-3">
                <span class="text-2xl">{icon}</span>
                <div>
                    <p class="text-sm text-gray-400">{title}</p>
                    <p class="text-xl font-bold text-white">{value}</p>
                </div>
            </div>
        </div>
    }
}

/// Loading state component
#[component]
fn LoadingState() -> impl IntoView {
    view! {
        <div class="flex items-center justify-center p-8">
            <div class="text-gray-400">"Loading..."</div>
        </div>
    }
}

/// Empty state component
#[component]
fn EmptyState(message: &'static str) -> impl IntoView {
    view! {
        <div class="flex items-center justify-center p-8">
            <div class="text-gray-400">{message}</div>
        </div>
    }
}

/// Format large numbers with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}
