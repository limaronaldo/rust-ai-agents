//! # LLM Provider Implementations
//!
//! This crate provides integrations with various LLM providers:
//! - OpenAI (GPT-4, GPT-3.5)
//! - Anthropic (Claude)
//! - OpenRouter (200+ models)
//!
//! Features:
//! - Governor-based rate limiting with token bucket algorithm
//! - Exponential backoff retry for transient errors
//! - Per-provider rate limit presets

pub mod anthropic;
pub mod backend;
pub mod openai;
pub mod openrouter;
pub mod rate_limit;
pub mod retry;

pub use anthropic::AnthropicProvider;
pub use backend::*;
pub use openai::OpenAIProvider;
pub use openrouter::OpenRouterProvider;
pub use rate_limit::{GovernorRateLimiter, RateLimitConfig};
pub use retry::{with_retry, RetryConfig};
