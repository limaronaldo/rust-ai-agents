//! Message item component for displaying session messages

use leptos::prelude::*;

use crate::api::SessionMessage;

/// Display a single message in a session
#[component]
pub fn MessageItem(message: SessionMessage) -> impl IntoView {
    let is_user = message.role == "user";
    let is_system = message.role == "system";

    let (icon, bg_class) = if is_user {
        ("👤", "bg-blue-900/50")
    } else if is_system {
        ("⚙️", "bg-gray-700/50")
    } else {
        ("🤖", "bg-green-900/50")
    };

    let container_class = format!("rounded-lg p-4 mb-3 {}", bg_class);
    let role = message.role.clone();
    let content = message.content.clone();
    let time = message.timestamp.format("%H:%M:%S").to_string();

    view! {
        <div class=container_class>
            <div class="flex items-center gap-2 mb-2">
                <span class="text-lg">{icon}</span>
                <span class="font-medium text-white capitalize">{role}</span>
                <span class="text-xs text-gray-400 ml-auto">{time}</span>
            </div>
            <div class="text-gray-200 whitespace-pre-wrap">{content}</div>
        </div>
    }
}

/// Message list showing all messages in a session
#[component]
pub fn MessageList(messages: Vec<SessionMessage>) -> impl IntoView {
    view! {
        <div class="space-y-3">
            {messages.into_iter().map(|msg| {
                view! { <MessageItem message=msg /> }
            }).collect::<Vec<_>>()}
        </div>
    }
}
