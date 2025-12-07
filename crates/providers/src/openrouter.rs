//! OpenRouter provider - Access to 200+ models with unified API

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, warn};

use crate::backend::{InferenceOutput, LLMBackend, ModelInfo, RateLimiter, TokenUsage};
use rust_ai_agents_core::{errors::LLMError, LLMMessage, MessageRole, ToolCall, ToolSchema};

/// OpenRouter API provider
pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    model: String,
    rate_limiter: Arc<RateLimiter>,
    base_url: String,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .pool_max_idle_per_host(10)
                .tcp_keepalive(std::time::Duration::from_secs(30))
                .local_address(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED))
                .build()
                .expect("Failed to create HTTP client"),
            api_key,
            model,
            rate_limiter: Arc::new(RateLimiter::new(60, 100_000)), // 60 RPM, 100k TPM
            base_url: "https://openrouter.ai/api/v1".to_string(),
        }
    }

    /// Create with custom rate limits
    pub fn with_rate_limits(mut self, rpm: usize, tpm: usize) -> Self {
        self.rate_limiter = Arc::new(RateLimiter::new(rpm, tpm));
        self
    }

    /// Convert internal messages to OpenRouter format
    fn convert_messages(&self, messages: &[LLMMessage]) -> Vec<OpenRouterMessage> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };

                OpenRouterMessage {
                    role: role.to_string(),
                    content: Some(msg.content.clone()),
                    tool_calls: msg.tool_calls.clone().map(|calls| {
                        calls
                            .iter()
                            .map(|call| OpenRouterToolCall {
                                id: call.id.clone(),
                                r#type: "function".to_string(),
                                function: OpenRouterFunction {
                                    name: call.name.clone(),
                                    arguments: serde_json::to_string(&call.arguments)
                                        .unwrap_or_default(),
                                },
                            })
                            .collect()
                    }),
                    tool_call_id: msg.tool_call_id.clone(),
                    name: msg.name.clone(),
                }
            })
            .collect()
    }

    /// Convert tool schemas to OpenRouter format
    fn convert_tools(&self, tools: &[ToolSchema]) -> Vec<OpenRouterTool> {
        tools
            .iter()
            .map(|tool| OpenRouterTool {
                r#type: "function".to_string(),
                function: OpenRouterToolFunction {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.parameters.clone(),
                },
            })
            .collect()
    }
}

#[async_trait]
impl LLMBackend for OpenRouterProvider {
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        // Estimate tokens for rate limiting (rough approximation)
        let estimated_tokens = messages.iter().map(|m| m.content.len() / 4).sum::<usize>();

        self.rate_limiter.wait(estimated_tokens).await;

        let mut request = OpenRouterRequest {
            model: self.model.clone(),
            messages: self.convert_messages(messages),
            temperature,
            tools: None,
            tool_choice: None,
        };

        if !tools.is_empty() {
            request.tools = Some(self.convert_tools(tools));
            request.tool_choice = Some(serde_json::json!("auto"));
        }

        debug!(
            "OpenRouter request: model={}, messages={}",
            self.model,
            request.messages.len()
        );

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", "https://rust-ai-agents.dev")
            .header("X-Title", "Rust AI Agents")
            .json(&request)
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return Err(match status.as_u16() {
                429 => LLMError::RateLimitExceeded,
                401 | 403 => LLMError::AuthenticationFailed(error_text),
                _ => LLMError::ApiError(format!("Status {}: {}", status, error_text)),
            });
        }

        let api_response: OpenRouterResponse = response
            .json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        let choice = api_response
            .choices
            .first()
            .ok_or_else(|| LLMError::InvalidResponse("No choices in response".to_string()))?;

        let tool_calls = choice.message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .filter_map(|call| {
                    serde_json::from_str(&call.function.arguments)
                        .ok()
                        .map(|args| ToolCall {
                            id: call.id.clone(),
                            name: call.function.name.clone(),
                            arguments: args,
                        })
                })
                .collect()
        });

        let token_usage = TokenUsage::new(
            api_response.usage.prompt_tokens,
            api_response.usage.completion_tokens,
        );

        Ok(InferenceOutput {
            content: choice.message.content.clone().unwrap_or_default(),
            tool_calls,
            reasoning: None,
            confidence: 1.0,
            token_usage,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, LLMError> {
        // OpenRouter doesn't provide embeddings directly
        // Use a specific embedding model
        warn!("OpenRouter embedding not implemented, returning zeros");
        Ok(vec![0.0; 1536])
    }

    fn model_info(&self) -> ModelInfo {
        // Default info - should be fetched from OpenRouter's models endpoint
        ModelInfo {
            model: self.model.clone(),
            provider: "openrouter".to_string(),
            max_tokens: 128_000,
            input_cost_per_1m: 0.15,
            output_cost_per_1m: 0.60,
            supports_functions: true,
            supports_vision: false,
        }
    }
}

// OpenRouter API types

#[derive(Debug, Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenRouterTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenRouterToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterToolCall {
    id: String,
    r#type: String,
    function: OpenRouterFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenRouterTool {
    r#type: String,
    function: OpenRouterToolFunction,
}

#[derive(Debug, Serialize)]
struct OpenRouterToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
    usage: OpenRouterUsage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterMessage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}
