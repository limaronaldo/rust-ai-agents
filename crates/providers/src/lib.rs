//! # LLM Provider Implementations
//!
//! This crate provides integrations with various LLM providers:
//! - **OpenAI** (GPT-4, GPT-4o, GPT-3.5)
//! - **Anthropic** (Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku)
//! - **OpenRouter** (200+ models with unified API)
//!
//! ## Features
//! - Governor-based rate limiting with token bucket algorithm
//! - Exponential backoff retry for transient errors
//! - Per-provider rate limit presets
//! - Tool/function calling support
//! - Vision support (where available)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rust_ai_agents_providers::{AnthropicProvider, LLMBackend};
//!
//! // Claude 3.5 Sonnet (recommended)
//! let claude = AnthropicProvider::claude_35_sonnet(api_key);
//!
//! // Or with a specific model
//! let opus = AnthropicProvider::claude_3_opus(api_key);
//! let haiku = AnthropicProvider::claude_3_haiku(api_key);
//! ```

pub mod anthropic;
pub mod backend;
pub mod openai;
pub mod openrouter;
pub mod rate_limit;
pub mod retry;

pub use anthropic::{AnthropicProvider, ClaudeModel};
pub use backend::{
    InferenceOutput, LLMBackend, ModelInfo, RateLimiter, StreamEvent, StreamResponse, TokenUsage,
};
pub use openai::OpenAIProvider;
pub use openrouter::OpenRouterProvider;
pub use rate_limit::{GovernorRateLimiter, RateLimitConfig};
pub use retry::{with_retry, RetryConfig};
