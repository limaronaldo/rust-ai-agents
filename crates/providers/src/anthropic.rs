//! Anthropic provider (Claude models)

use crate::backend::{InferenceOutput, LLMBackend, ModelInfo};
use async_trait::async_trait;
use rust_ai_agents_core::{errors::LLMError, LLMMessage, ToolSchema};

pub struct AnthropicProvider {
    #[allow(dead_code)]
    api_key: String,
    model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
    }
}

#[async_trait]
impl LLMBackend for AnthropicProvider {
    async fn infer(
        &self,
        _messages: &[LLMMessage],
        _tools: &[ToolSchema],
        _temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        // TODO: Implement Anthropic API integration
        Err(LLMError::ApiError(
            "Anthropic provider not yet implemented".to_string(),
        ))
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, LLMError> {
        Err(LLMError::ApiError(
            "Anthropic embeddings not supported".to_string(),
        ))
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            model: self.model.clone(),
            provider: "anthropic".to_string(),
            max_tokens: 200_000,
            input_cost_per_1m: 3.0,
            output_cost_per_1m: 15.0,
            supports_functions: true,
            supports_vision: true,
        }
    }
}
