//! Main layout component

use leptos::prelude::*;

use super::Sidebar;

/// Main layout with sidebar
#[component]
pub fn Layout(children: Children) -> impl IntoView {
    view! {
        <div class="flex h-screen bg-gray-900 text-gray-100">
            <Sidebar />
            <main class="flex-1 overflow-auto p-6">
                {children()}
            </main>
        </div>
    }
}
