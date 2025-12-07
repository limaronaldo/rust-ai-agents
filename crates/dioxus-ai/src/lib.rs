//! # Dioxus AI
//!
//! AI hooks for Dioxus applications - chat, completions, and streaming.
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
//! use dioxus::prelude::*;
//! use dioxus_ai::{use_chat, ChatOptions};
//!
//! #[component]
//! fn Chat() -> Element {
//!     let mut chat = use_chat(ChatOptions {
//!         provider: "openai".to_string(),
//!         api_key: "sk-...".to_string(),
//!         model: "gpt-4o-mini".to_string(),
//!         ..Default::default()
//!     });
//!
//!     let mut input = use_signal(String::new);
//!
//!     rsx! {
//!         div {
//!             for msg in chat.messages().iter() {
//!                 p { class: "{msg.role}", "{msg.content}" }
//!             }
//!
//!             if chat.is_loading() {
//!                 p { "Thinking..." }
//!             }
//!
//!             input {
//!                 value: "{input}",
//!                 oninput: move |e| input.set(e.value().clone())
//!             }
//!             button {
//!                 onclick: move |_| {
//!                     chat.send(&input());
//!                     input.set(String::new());
//!                 },
//!                 "Send"
//!             }
//!         }
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
