//! Status badge component

use leptos::prelude::*;

use crate::api::SessionStatus;

/// Display session status as a colored badge
#[component]
pub fn StatusBadge(status: SessionStatus) -> impl IntoView {
    let (text, class) = match status {
        SessionStatus::Active => (
            "Active",
            "bg-green-500/20 text-green-400 border-green-500/50",
        ),
        SessionStatus::Completed => (
            "Completed",
            "bg-blue-500/20 text-blue-400 border-blue-500/50",
        ),
        SessionStatus::Failed => ("Failed", "bg-red-500/20 text-red-400 border-red-500/50"),
        SessionStatus::Archived => (
            "Archived",
            "bg-gray-500/20 text-gray-400 border-gray-500/50",
        ),
    };

    view! {
        <span class=format!("px-2 py-1 text-xs font-medium rounded border {}", class)>
            {text}
        </span>
    }
}

/// Display agent status as a badge
#[component]
pub fn AgentStatusBadge(status: String) -> impl IntoView {
    let class = match status.as_str() {
        "running" => "bg-green-500/20 text-green-400 border-green-500/50",
        "idle" => "bg-yellow-500/20 text-yellow-400 border-yellow-500/50",
        "error" => "bg-red-500/20 text-red-400 border-red-500/50",
        _ => "bg-gray-500/20 text-gray-400 border-gray-500/50",
    };

    view! {
        <span class=format!("px-2 py-1 text-xs font-medium rounded border capitalize {}", class)>
            {status}
        </span>
    }
}
