//! 404 Not Found page

use leptos::prelude::*;
use leptos_router::components::A;

/// 404 Not Found page
#[component]
pub fn NotFound() -> impl IntoView {
    view! {
        <div class="flex flex-col items-center justify-center min-h-[60vh]">
            <span class="text-6xl mb-4">"🔍"</span>
            <h1 class="text-4xl font-bold text-white mb-2">"404"</h1>
            <p class="text-xl text-gray-400 mb-6">"Page not found"</p>
            <p class="text-gray-500 mb-8">"The page you're looking for doesn't exist or has been moved."</p>
            <A href="/" attr:class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                "Go to Dashboard"
            </A>
        </div>
    }
}
