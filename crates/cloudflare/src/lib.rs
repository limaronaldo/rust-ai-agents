//! # Cloudflare Workers Support for Rust AI Agents
//!
//! This crate provides Cloudflare Workers-native AI agent functionality with:
//! - Multiple LLM provider support (OpenAI, Anthropic, OpenRouter)
//! - KV store integration for conversation persistence
//! - Streaming responses using Response streams
//! - Request/response handling for Worker routes
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rust_ai_agents_cloudflare::{CloudflareAgent, CloudflareConfig, Provider};
//! use worker::*;
//!
//! #[event(fetch)]
//! async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
//!     let config = CloudflareConfig::builder()
//!         .provider(Provider::Anthropic)
//!         .api_key(env.secret("ANTHROPIC_API_KEY")?.to_string())
//!         .model("claude-3-5-sonnet-20241022")
//!         .build();
//!
//!     let agent = CloudflareAgent::new(config);
//!
//!     let response = agent.chat("Hello!").await?;
//!     Response::ok(response.content)
//! }
//! ```
//!
//! ## With KV Persistence
//!
//! ```rust,ignore
//! use rust_ai_agents_cloudflare::{CloudflareAgent, KvStore};
//!
//! let kv = env.kv("CONVERSATIONS")?;
//! let agent = CloudflareAgent::new(config)
//!     .with_kv(KvStore::new(kv));
//!
//! // Conversation history is automatically persisted
//! let response = agent.chat_with_session("session-123", "Hello!").await?;
//! ```

mod agent;
mod config;
mod error;
mod http;
mod kv;
mod streaming;

pub use agent::CloudflareAgent;
pub use config::{CloudflareConfig, CloudflareConfigBuilder};
pub use error::CloudflareError;
pub use http::CloudflareHttpClient;
pub use kv::KvStore;
pub use rust_ai_agents_llm_client::{Message, Provider, Role};
pub use streaming::{StreamChunk, StreamingResponse};

/// Re-export worker types for convenience
pub use worker::{Env, Request, Response, Result};
