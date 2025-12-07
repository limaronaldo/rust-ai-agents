//! Trace item component for displaying trace entries

use leptos::prelude::*;

use crate::api::{TraceEntry, TraceEntryType};

/// Display a single trace entry
#[component]
pub fn TraceItem(entry: TraceEntry) -> impl IntoView {
    let (icon, title, details, color) = match &entry.entry_type {
        TraceEntryType::LlmRequest {
            model,
            prompt_tokens,
            completion_tokens,
            cost,
        } => (
            "🤖",
            format!("LLM Request - {}", model),
            format!(
                "{} prompt + {} completion tokens (${:.4})",
                prompt_tokens, completion_tokens, cost
            ),
            "blue",
        ),
        TraceEntryType::LlmResponse {
            content,
            finish_reason,
        } => (
            "💬",
            "LLM Response".to_string(),
            format!(
                "{:.100}... ({})",
                content,
                finish_reason.as_deref().unwrap_or("unknown")
            ),
            "green",
        ),
        TraceEntryType::ToolCall {
            tool_name,
            arguments: _,
        } => (
            "🔧",
            format!("Tool Call - {}", tool_name),
            "Executing...".to_string(),
            "yellow",
        ),
        TraceEntryType::ToolResult {
            tool_name,
            success,
            result: _,
        } => (
            if *success { "✅" } else { "❌" },
            format!("Tool Result - {}", tool_name),
            if *success { "Success" } else { "Failed" }.to_string(),
            if *success { "green" } else { "red" },
        ),
        TraceEntryType::AgentThought { thought } => {
            ("💭", "Agent Thought".to_string(), thought.clone(), "purple")
        }
        TraceEntryType::Error {
            message,
            error_type,
        } => (
            "🚨",
            format!("Error - {}", error_type),
            message.clone(),
            "red",
        ),
    };

    let duration_text = entry
        .duration_ms
        .map(|d| format!("{:.1}ms", d))
        .unwrap_or_default();

    let border_class = match color {
        "blue" => "border-blue-500",
        "green" => "border-green-500",
        "yellow" => "border-yellow-500",
        "red" => "border-red-500",
        "purple" => "border-purple-500",
        _ => "border-gray-500",
    };

    view! {
        <div class=format!("border-l-4 {} bg-gray-800 rounded-r-lg p-4 mb-2", border_class)>
            <div class="flex items-start justify-between">
                <div class="flex items-center gap-2">
                    <span class="text-xl">{icon}</span>
                    <span class="font-medium text-white">{title}</span>
                </div>
                <div class="flex items-center gap-2 text-sm text-gray-400">
                    <span>{duration_text}</span>
                    <span>{entry.timestamp.format("%H:%M:%S").to_string()}</span>
                </div>
            </div>
            <p class="mt-2 text-sm text-gray-300 truncate">{details}</p>
        </div>
    }
}

/// Trace timeline showing multiple entries
#[component]
pub fn TraceTimeline(entries: Vec<TraceEntry>) -> impl IntoView {
    view! {
        <div class="space-y-2">
            {entries.into_iter().map(|entry| {
                view! { <TraceItem entry=entry /> }
            }).collect::<Vec<_>>()}
        </div>
    }
}
