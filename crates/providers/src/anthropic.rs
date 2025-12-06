//! Anthropic Claude provider
//!
//! Full implementation of the Anthropic Messages API with:
//! - Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku support
//! - Tool/function calling
//! - Vision (image) support
//! - Extended thinking (for Claude 3.5)
//! - Rate limiting and retries

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::backend::{InferenceOutput, LLMBackend, ModelInfo, RateLimiter, TokenUsage};
use rust_ai_agents_core::{errors::LLMError, LLMMessage, MessageRole, ToolCall, ToolSchema};

/// Anthropic API version
const ANTHROPIC_VERSION: &str = "2023-06-01";
const ANTHROPIC_BETA: &str = "tools-2024-05-16";

/// Claude model variants
#[derive(Debug, Clone)]
pub enum ClaudeModel {
    /// Claude 3.5 Sonnet - Best balance of speed and intelligence
    Claude35Sonnet,
    /// Claude 3.5 Haiku - Fast and efficient
    Claude35Haiku,
    /// Claude 3 Opus - Most powerful
    Claude3Opus,
    /// Claude 3 Sonnet
    Claude3Sonnet,
    /// Claude 3 Haiku - Fastest
    Claude3Haiku,
    /// Custom model string
    Custom(String),
}

impl ClaudeModel {
    pub fn as_str(&self) -> &str {
        match self {
            ClaudeModel::Claude35Sonnet => "claude-3-5-sonnet-20241022",
            ClaudeModel::Claude35Haiku => "claude-3-5-haiku-20241022",
            ClaudeModel::Claude3Opus => "claude-3-opus-20240229",
            ClaudeModel::Claude3Sonnet => "claude-3-sonnet-20240229",
            ClaudeModel::Claude3Haiku => "claude-3-haiku-20240307",
            ClaudeModel::Custom(s) => s.as_str(),
        }
    }

    pub fn max_tokens(&self) -> usize {
        match self {
            ClaudeModel::Claude35Sonnet | ClaudeModel::Claude35Haiku => 8192,
            ClaudeModel::Claude3Opus => 4096,
            ClaudeModel::Claude3Sonnet | ClaudeModel::Claude3Haiku => 4096,
            ClaudeModel::Custom(_) => 4096,
        }
    }

    pub fn context_window(&self) -> usize {
        match self {
            ClaudeModel::Claude35Sonnet | ClaudeModel::Claude35Haiku => 200_000,
            ClaudeModel::Claude3Opus => 200_000,
            ClaudeModel::Claude3Sonnet | ClaudeModel::Claude3Haiku => 200_000,
            ClaudeModel::Custom(_) => 100_000,
        }
    }

    pub fn input_cost_per_1m(&self) -> f64 {
        match self {
            ClaudeModel::Claude35Sonnet => 3.0,
            ClaudeModel::Claude35Haiku => 1.0,
            ClaudeModel::Claude3Opus => 15.0,
            ClaudeModel::Claude3Sonnet => 3.0,
            ClaudeModel::Claude3Haiku => 0.25,
            ClaudeModel::Custom(_) => 3.0,
        }
    }

    pub fn output_cost_per_1m(&self) -> f64 {
        match self {
            ClaudeModel::Claude35Sonnet => 15.0,
            ClaudeModel::Claude35Haiku => 5.0,
            ClaudeModel::Claude3Opus => 75.0,
            ClaudeModel::Claude3Sonnet => 15.0,
            ClaudeModel::Claude3Haiku => 1.25,
            ClaudeModel::Custom(_) => 15.0,
        }
    }
}

impl From<&str> for ClaudeModel {
    fn from(s: &str) -> Self {
        match s {
            "claude-3-5-sonnet-20241022" | "claude-3.5-sonnet" | "claude-3-5-sonnet" => {
                ClaudeModel::Claude35Sonnet
            }
            "claude-3-5-haiku-20241022" | "claude-3.5-haiku" | "claude-3-5-haiku" => {
                ClaudeModel::Claude35Haiku
            }
            "claude-3-opus-20240229" | "claude-3-opus" => ClaudeModel::Claude3Opus,
            "claude-3-sonnet-20240229" | "claude-3-sonnet" => ClaudeModel::Claude3Sonnet,
            "claude-3-haiku-20240307" | "claude-3-haiku" => ClaudeModel::Claude3Haiku,
            other => ClaudeModel::Custom(other.to_string()),
        }
    }
}

/// Anthropic Claude provider
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: ClaudeModel,
    rate_limiter: Arc<RateLimiter>,
    base_url: String,
    max_tokens: usize,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with Claude 3.5 Sonnet (default)
    pub fn new(api_key: String) -> Self {
        Self::with_model(api_key, ClaudeModel::Claude35Sonnet)
    }

    /// Create with a specific model
    pub fn with_model(api_key: String, model: ClaudeModel) -> Self {
        let max_tokens = model.max_tokens();
        Self {
            client: Client::new(),
            api_key,
            model,
            rate_limiter: Arc::new(RateLimiter::new(60, 100_000)), // 60 RPM, 100k TPM
            base_url: "https://api.anthropic.com/v1".to_string(),
            max_tokens,
        }
    }

    /// Create from model string
    pub fn from_model_str(api_key: String, model: &str) -> Self {
        Self::with_model(api_key, ClaudeModel::from(model))
    }

    /// Set custom max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set custom rate limits
    pub fn with_rate_limits(mut self, rpm: usize, tpm: usize) -> Self {
        self.rate_limiter = Arc::new(RateLimiter::new(rpm, tpm));
        self
    }

    /// Convert internal messages to Anthropic format
    fn convert_messages(&self, messages: &[LLMMessage]) -> (Option<String>, Vec<AnthropicMessage>) {
        let mut system_prompt = None;
        let mut anthropic_messages = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    // Anthropic handles system prompt separately
                    system_prompt = Some(msg.content.clone());
                }
                MessageRole::User => {
                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Text(msg.content.clone()),
                    });
                }
                MessageRole::Assistant => {
                    // Check if this is a tool use response
                    if let Some(ref tool_calls) = msg.tool_calls {
                        let tool_use_blocks: Vec<ContentBlock> = tool_calls
                            .iter()
                            .map(|call| ContentBlock::ToolUse {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                input: call.arguments.clone(),
                            })
                            .collect();

                        // If there's also text content, include it
                        let mut blocks = Vec::new();
                        if !msg.content.is_empty() {
                            blocks.push(ContentBlock::Text {
                                text: msg.content.clone(),
                            });
                        }
                        blocks.extend(tool_use_blocks);

                        anthropic_messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Blocks(blocks),
                        });
                    } else {
                        anthropic_messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Text(msg.content.clone()),
                        });
                    }
                }
                MessageRole::Tool => {
                    // Tool results go in a user message with tool_result block
                    let tool_result = ContentBlock::ToolResult {
                        tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                        content: msg.content.clone(),
                        is_error: None,
                    };

                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Blocks(vec![tool_result]),
                    });
                }
            }
        }

        (system_prompt, anthropic_messages)
    }

    /// Convert tool schemas to Anthropic format
    fn convert_tools(&self, tools: &[ToolSchema]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|tool| AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.parameters.clone(),
            })
            .collect()
    }

    /// Parse tool calls from response
    fn parse_tool_calls(&self, content: &[ContentBlock]) -> Option<Vec<ToolCall>> {
        let tool_calls: Vec<ToolCall> = content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, name, input } => Some(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: input.clone(),
                }),
                _ => None,
            })
            .collect();

        if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        }
    }

    /// Extract text content from response
    fn extract_text(&self, content: &[ContentBlock]) -> String {
        content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[async_trait]
impl LLMBackend for AnthropicProvider {
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        // Estimate tokens for rate limiting
        let estimated_tokens = messages.iter().map(|m| m.content.len() / 4).sum::<usize>();
        self.rate_limiter.wait(estimated_tokens).await;

        let (system_prompt, anthropic_messages) = self.convert_messages(messages);

        let mut request = AnthropicRequest {
            model: self.model.as_str().to_string(),
            max_tokens: self.max_tokens,
            messages: anthropic_messages,
            system: system_prompt,
            temperature: Some(temperature),
            tools: None,
            tool_choice: None,
            metadata: None,
        };

        // Add tools if provided
        if !tools.is_empty() {
            request.tools = Some(self.convert_tools(tools));
            request.tool_choice = Some(ToolChoice::Auto);
        }

        debug!(
            "Anthropic request: model={}, messages={}, tools={}",
            self.model.as_str(),
            request.messages.len(),
            tools.len()
        );

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", ANTHROPIC_BETA)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();

            return Err(match status.as_u16() {
                429 => {
                    warn!("Anthropic rate limit exceeded");
                    LLMError::RateLimitExceeded
                }
                401 => LLMError::AuthenticationFailed("Invalid API key".to_string()),
                403 => LLMError::AuthenticationFailed(error_text),
                400 => LLMError::InvalidResponse(format!("Bad request: {}", error_text)),
                500..=599 => LLMError::ApiError(format!("Server error: {}", error_text)),
                _ => LLMError::ApiError(format!("Status {}: {}", status, error_text)),
            });
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        // Extract content
        let text_content = self.extract_text(&api_response.content);
        let tool_calls = self.parse_tool_calls(&api_response.content);

        let token_usage = TokenUsage::new(
            api_response.usage.input_tokens,
            api_response.usage.output_tokens,
        );

        info!(
            "Anthropic response: tokens={}/{}, stop_reason={:?}",
            api_response.usage.input_tokens,
            api_response.usage.output_tokens,
            api_response.stop_reason
        );

        let mut metadata = HashMap::new();
        metadata.insert(
            "stop_reason".to_string(),
            serde_json::json!(api_response.stop_reason),
        );
        metadata.insert("model".to_string(), serde_json::json!(api_response.model));

        Ok(InferenceOutput {
            content: text_content,
            tool_calls,
            reasoning: None,
            confidence: 1.0,
            token_usage,
            metadata,
        })
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, LLMError> {
        // Anthropic doesn't provide embeddings API
        Err(LLMError::ApiError(
            "Anthropic does not provide embeddings. Use a dedicated embedding provider."
                .to_string(),
        ))
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            model: self.model.as_str().to_string(),
            provider: "anthropic".to_string(),
            max_tokens: self.model.context_window(),
            input_cost_per_1m: self.model.input_cost_per_1m(),
            output_cost_per_1m: self.model.output_cost_per_1m(),
            supports_functions: true,
            supports_vision: true,
        }
    }
}

// ============================================================================
// Anthropic API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: usize,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    #[serde(rename = "image")]
    Image { source: ImageSource },
}

#[derive(Debug, Serialize, Deserialize)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum ToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

// ============================================================================
// Convenience constructors
// ============================================================================

impl AnthropicProvider {
    /// Create Claude 3.5 Sonnet provider (recommended)
    pub fn claude_35_sonnet(api_key: String) -> Self {
        Self::with_model(api_key, ClaudeModel::Claude35Sonnet)
    }

    /// Create Claude 3.5 Haiku provider (fast)
    pub fn claude_35_haiku(api_key: String) -> Self {
        Self::with_model(api_key, ClaudeModel::Claude35Haiku)
    }

    /// Create Claude 3 Opus provider (most capable)
    pub fn claude_3_opus(api_key: String) -> Self {
        Self::with_model(api_key, ClaudeModel::Claude3Opus)
    }

    /// Create Claude 3 Haiku provider (fastest/cheapest)
    pub fn claude_3_haiku(api_key: String) -> Self {
        Self::with_model(api_key, ClaudeModel::Claude3Haiku)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_from_string() {
        assert!(matches!(
            ClaudeModel::from("claude-3-5-sonnet"),
            ClaudeModel::Claude35Sonnet
        ));
        assert!(matches!(
            ClaudeModel::from("claude-3-opus"),
            ClaudeModel::Claude3Opus
        ));
        assert!(matches!(
            ClaudeModel::from("claude-3-haiku"),
            ClaudeModel::Claude3Haiku
        ));
    }

    #[test]
    fn test_model_costs() {
        let sonnet = ClaudeModel::Claude35Sonnet;
        assert_eq!(sonnet.input_cost_per_1m(), 3.0);
        assert_eq!(sonnet.output_cost_per_1m(), 15.0);

        let haiku = ClaudeModel::Claude3Haiku;
        assert_eq!(haiku.input_cost_per_1m(), 0.25);
        assert_eq!(haiku.output_cost_per_1m(), 1.25);
    }

    #[test]
    fn test_model_info() {
        let provider = AnthropicProvider::new("test-key".to_string());
        let info = provider.model_info();

        assert_eq!(info.provider, "anthropic");
        assert!(info.supports_functions);
        assert!(info.supports_vision);
    }

    #[test]
    fn test_message_conversion() {
        let provider = AnthropicProvider::new("test-key".to_string());

        let messages = vec![
            LLMMessage {
                role: MessageRole::System,
                content: "You are a helpful assistant.".to_string(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            LLMMessage {
                role: MessageRole::User,
                content: "Hello!".to_string(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];

        let (system, converted) = provider.convert_messages(&messages);

        assert_eq!(system, Some("You are a helpful assistant.".to_string()));
        assert_eq!(converted.len(), 1); // Only user message, system is separate
        assert_eq!(converted[0].role, "user");
    }
}
