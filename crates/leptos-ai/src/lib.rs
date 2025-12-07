//! # Leptos AI
//!
//! AI hooks for Leptos applications - chat, completions, and streaming.
//!
//! ## Features
//!
//! - `use_chat` - Reactive chat with message history and streaming
//! - `use_completion` - Single completion requests
//! - Built on `llm-client` for provider support (OpenAI, Anthropic, OpenRouter)
//!
//! ## Example
//!
//! ```rust,ignore
//! use leptos::*;
//! use leptos_ai::{use_chat, ChatOptions};
//!
//! #[component]
//! fn Chat() -> impl IntoView {
//!     let chat = use_chat(ChatOptions {
//!         provider: "openai".to_string(),
//!         api_key: "sk-...".to_string(),
//!         model: "gpt-4o-mini".to_string(),
//!         ..Default::default()
//!     });
//!
//!     let input = create_node_ref::<html::Input>();
//!
//!     let on_submit = move |ev: ev::SubmitEvent| {
//!         ev.prevent_default();
//!         if let Some(input_el) = input.get() {
//!             let value = input_el.value();
//!             if !value.is_empty() {
//!                 chat.send(&value);
//!                 input_el.set_value("");
//!             }
//!         }
//!     };
//!
//!     view! {
//!         <div class="chat">
//!             <For
//!                 each=move || chat.messages.get()
//!                 key=|msg| msg.id.clone()
//!                 children=move |msg| {
//!                     view! {
//!                         <div class=format!("message {}", msg.role)>
//!                             {msg.content.clone()}
//!                         </div>
//!                     }
//!                 }
//!             />
//!             <Show when=move || chat.is_loading.get()>
//!                 <div class="loading">"Thinking..."</div>
//!             </Show>
//!             <form on:submit=on_submit>
//!                 <input type="text" node_ref=input placeholder="Type a message..." />
//!                 <button type="submit">"Send"</button>
//!             </form>
//!         </div>
//!     }
//! }
//! ```

mod error;
mod hooks;
mod types;

pub use error::*;
pub use hooks::*;
pub use types::*;

// Re-export useful types from llm-client
pub use rust_ai_agents_llm_client::{Provider, Role};
