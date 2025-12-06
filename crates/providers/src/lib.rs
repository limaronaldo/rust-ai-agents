//! # LLM Provider Implementations
//!
//! This crate provides integrations with various LLM providers:
//! - OpenAI (GPT-4, GPT-3.5)
//! - Anthropic (Claude)
//! - OpenRouter (200+ models)

pub mod anthropic;
pub mod backend;
pub mod openai;
pub mod openrouter;

pub use anthropic::AnthropicProvider;
pub use backend::*;
pub use openai::OpenAIProvider;
pub use openrouter::OpenRouterProvider;
