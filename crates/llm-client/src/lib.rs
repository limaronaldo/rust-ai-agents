//! # LLM Client - Shared Logic
//!
//! Runtime-agnostic LLM client logic for building requests and parsing responses.
//! This crate has NO runtime dependencies (no async, no HTTP client).
//!
//! ## Supported Providers
//!
//! - OpenAI (GPT-4, GPT-3.5, etc.)
//! - Anthropic (Claude 3, etc.)
//! - OpenRouter (100+ models)
//!
//! ## Usage
//!
//! ```rust
//! use rust_ai_agents_llm_client::{
//!     Provider, Message, RequestBuilder, ResponseParser,
//! };
//!
//! // Build a request
//! let messages = vec![
//!     Message::system("You are a helpful assistant."),
//!     Message::user("Hello!"),
//! ];
//!
//! let request = RequestBuilder::new(Provider::OpenAI)
//!     .model("gpt-4o-mini")
//!     .messages(&messages)
//!     .api_key("sk-...")
//!     .temperature(0.7)
//!     .max_tokens(1024)
//!     .stream(false)
//!     .build()
//!     .unwrap();
//!
//! // Use your runtime's HTTP client to send request.url, request.headers, request.body
//! // Then parse the response:
//!
//! let response_json = r#"{"choices":[{"message":{"content":"Hello!"}}]}"#;
//! let response = ResponseParser::parse(Provider::OpenAI, response_json).unwrap();
//! println!("{}", response.content);
//! ```

mod error;
mod message;
mod provider;
mod request;
mod response;

pub use error::{LlmClientError, Result};
pub use message::{Message, Role};
pub use provider::Provider;
pub use request::{HttpRequest, RequestBuilder};
pub use response::{LlmResponse, ResponseParser, StreamChunk, ToolCall, ToolCallChunk, Usage};
