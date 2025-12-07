//! Metrics page - Real-time metrics visualization

use leptos::prelude::*;

use crate::api::{ApiClient, CostStats};
use crate::components::{BarChart, DataPoint, LineChart, SparklineCard, Trend};

/// Metrics history for charts
#[derive(Debug, Clone, Default)]
pub struct MetricsHistory {
    pub cost: Vec<f64>,
    pub tokens: Vec<f64>,
    pub latency: Vec<f64>,
    pub requests: Vec<f64>,
}

/// Metrics page with charts
#[component]
pub fn MetricsPage() -> impl IntoView {
    // Current stats
    let stats = LocalResource::new(|| async { ApiClient::from_origin().get_stats().await.ok() });

    // Simulated history (in real app, this would come from API)
    let history = RwSignal::new(MetricsHistory {
        cost: vec![0.12, 0.15, 0.18, 0.14, 0.22, 0.19, 0.25, 0.21, 0.28, 0.24],
        tokens: vec![
            1200.0, 1500.0, 1800.0, 1400.0, 2200.0, 1900.0, 2500.0, 2100.0, 2800.0, 2400.0,
        ],
        latency: vec![
            120.0, 150.0, 130.0, 180.0, 140.0, 160.0, 135.0, 155.0, 145.0, 150.0,
        ],
        requests: vec![10.0, 15.0, 12.0, 18.0, 14.0, 20.0, 16.0, 22.0, 19.0, 25.0],
    });

    // Auto-refresh
    let refresh_count = RwSignal::new(0);

    // Simulate adding new data points periodically
    Effect::new(move |_| {
        let _count = refresh_count.get();
        // In a real app, this would fetch new data
    });

    view! {
        <div>
            <div class="flex items-center justify-between mb-6">
                <h1 class="text-2xl font-bold text-white">"Metrics"</h1>
                <div class="flex items-center gap-4">
                    <select class="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                        <option>"Last Hour"</option>
                        <option>"Last 24 Hours"</option>
                        <option>"Last 7 Days"</option>
                        <option>"Last 30 Days"</option>
                    </select>
                    <button
                        class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                        on:click=move |_| refresh_count.update(|n| *n += 1)
                    >
                        "🔄 Refresh"
                    </button>
                </div>
            </div>

            // Summary cards
            <Suspense fallback=|| view! { <LoadingCards /> }>
                {move || {
                    stats.get().map(|data| {
                        let h = history.get();
                        match data {
                            Some(s) => view! { <SummaryCards stats=s history=h /> }.into_any(),
                            None => view! { <EmptyStats /> }.into_any(),
                        }
                    })
                }}
            </Suspense>

            // Charts
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                // Cost over time
                <ChartCard title="Cost Over Time" subtitle="USD spent on API calls">
                    {move || {
                        let h = history.get();
                        let data: Vec<DataPoint> = h.cost.iter().enumerate()
                            .map(|(i, &v)| DataPoint { timestamp: i as f64, value: v })
                            .collect();
                        view! { <LineChart data=data width=500 height=250 color="#22c55e" y_label="Cost ($)" /> }
                    }}
                </ChartCard>

                // Token usage
                <ChartCard title="Token Usage" subtitle="Tokens consumed per request">
                    {move || {
                        let h = history.get();
                        let data: Vec<DataPoint> = h.tokens.iter().enumerate()
                            .map(|(i, &v)| DataPoint { timestamp: i as f64, value: v })
                            .collect();
                        view! { <LineChart data=data width=500 height=250 color="#3b82f6" y_label="Tokens" /> }
                    }}
                </ChartCard>

                // Latency distribution
                <ChartCard title="Latency" subtitle="Response time in milliseconds">
                    {move || {
                        let h = history.get();
                        let data: Vec<DataPoint> = h.latency.iter().enumerate()
                            .map(|(i, &v)| DataPoint { timestamp: i as f64, value: v })
                            .collect();
                        view! { <LineChart data=data width=500 height=250 color="#eab308" y_label="ms" /> }
                    }}
                </ChartCard>

                // Requests per period
                <ChartCard title="Request Volume" subtitle="Number of API requests">
                    {move || {
                        let h = history.get();
                        let labels: Vec<String> = (1..=h.requests.len()).map(|i| format!("T{}", i)).collect();
                        view! { <BarChart labels=labels values=h.requests.clone() width=500 height=250 color="#8b5cf6" /> }
                    }}
                </ChartCard>
            </div>

            // Model breakdown
            <div class="mt-6">
                <ChartCard title="Usage by Model" subtitle="Token distribution across models">
                    <ModelBreakdown />
                </ChartCard>
            </div>
        </div>
    }
}

/// Summary cards with sparklines
#[component]
fn SummaryCards(stats: CostStats, history: MetricsHistory) -> impl IntoView {
    let cost_trend = if history.cost.len() >= 2 {
        let last = history.cost.last().unwrap_or(&0.0);
        let prev = history.cost.get(history.cost.len() - 2).unwrap_or(&0.0);
        if last > prev {
            Some(Trend::Up)
        } else if last < prev {
            Some(Trend::Down)
        } else {
            Some(Trend::Stable)
        }
    } else {
        None
    };

    let latency_trend = if history.latency.len() >= 2 {
        let last = history.latency.last().unwrap_or(&0.0);
        let prev = history
            .latency
            .get(history.latency.len() - 2)
            .unwrap_or(&0.0);
        if last > prev {
            Some(Trend::Up)
        } else if last < prev {
            Some(Trend::Down)
        } else {
            Some(Trend::Stable)
        }
    } else {
        None
    };

    view! {
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <SparklineCard
                title="Total Cost"
                value=format!("${:.4}", stats.total_cost)
                history=history.cost.clone()
                trend=cost_trend.unwrap_or(Trend::Stable)
                color="#22c55e"
            />
            <SparklineCard
                title="Total Tokens"
                value=format_number(stats.total_tokens)
                history=history.tokens.clone()
                color="#3b82f6"
            />
            <SparklineCard
                title="Total Requests"
                value=format_number(stats.total_requests)
                history=history.requests.clone()
                color="#8b5cf6"
            />
            <SparklineCard
                title="Avg Latency"
                value=format!("{:.0}ms", stats.avg_latency_ms)
                history=history.latency.clone()
                trend=latency_trend.unwrap_or(Trend::Stable)
                color="#eab308"
            />
        </div>
    }
}

/// Chart card wrapper
#[component]
fn ChartCard(title: &'static str, subtitle: &'static str, children: Children) -> impl IntoView {
    view! {
        <div class="bg-gray-800 rounded-lg p-4">
            <div class="mb-4">
                <h3 class="text-lg font-semibold text-white">{title}</h3>
                <p class="text-sm text-gray-400">{subtitle}</p>
            </div>
            <div class="flex justify-center">
                {children()}
            </div>
        </div>
    }
}

/// Model usage breakdown
#[component]
fn ModelBreakdown() -> impl IntoView {
    // Simulated model data
    let models = vec![
        ("GPT-4", 45.0, "#22c55e"),
        ("GPT-3.5", 30.0, "#3b82f6"),
        ("Claude", 15.0, "#8b5cf6"),
        ("Other", 10.0, "#6b7280"),
    ];

    view! {
        <div class="w-full">
            <div class="space-y-3">
                {models.into_iter().map(|(name, percent, color)| {
                    view! {
                        <div>
                            <div class="flex justify-between text-sm mb-1">
                                <span class="text-gray-300">{name}</span>
                                <span class="text-gray-400">{format!("{:.0}%", percent)}</span>
                            </div>
                            <div class="w-full bg-gray-700 rounded-full h-2">
                                <div
                                    class="h-2 rounded-full transition-all duration-500"
                                    style=format!("width: {}%; background-color: {}", percent, color)
                                ></div>
                            </div>
                        </div>
                    }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}

#[component]
fn LoadingCards() -> impl IntoView {
    view! {
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {(0..4).map(|_| view! {
                <div class="bg-gray-800 rounded-lg p-4 animate-pulse">
                    <div class="h-4 bg-gray-700 rounded w-1/2 mb-2"></div>
                    <div class="h-8 bg-gray-700 rounded w-3/4 mb-2"></div>
                    <div class="h-16 bg-gray-700 rounded"></div>
                </div>
            }).collect::<Vec<_>>()}
        </div>
    }
}

#[component]
fn EmptyStats() -> impl IntoView {
    view! {
        <div class="flex flex-col items-center justify-center p-8 bg-gray-800 rounded-lg">
            <span class="text-4xl mb-4">"📊"</span>
            <p class="text-gray-400">"No metrics data available"</p>
            <p class="text-sm text-gray-500">"Metrics will appear once agents start processing requests"</p>
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
