//! OpenAI provider (GPT-4, GPT-3.5, etc.)
//!
//! Full implementation of the OpenAI Chat Completions API with:
//! - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo support
//! - Tool/function calling
//! - Rate limiting and retries
//! - Streaming support with Server-Sent Events (SSE)

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use std::sync::Arc;
use tracing::debug;

use crate::backend::{
    InferenceOutput, LLMBackend, ModelInfo, RateLimiter, StreamEvent, StreamResponse, TokenUsage,
};
use rust_ai_agents_core::{errors::LLMError, LLMMessage, MessageRole, ToolCall, ToolSchema};

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
    rate_limiter: Arc<RateLimiter>,
}

impl OpenAIProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            rate_limiter: Arc::new(RateLimiter::new(500, 150_000)), // GPT-4 Turbo limits
        }
    }

    pub fn with_rate_limits(mut self, rpm: usize, tpm: usize) -> Self {
        self.rate_limiter = Arc::new(RateLimiter::new(rpm, tpm));
        self
    }

    fn convert_messages(&self, messages: &[LLMMessage]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };

                let mut json = serde_json::json!({
                    "role": role,
                    "content": msg.content.clone(),
                });

                if let Some(tool_calls) = &msg.tool_calls {
                    json["tool_calls"] = serde_json::json!(tool_calls.iter().map(|call| {
                    serde_json::json!({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": serde_json::to_string(&call.arguments).unwrap_or_default(),
                        }
                    })
                }).collect::<Vec<_>>());
                }

                if let Some(tool_call_id) = &msg.tool_call_id {
                    json["tool_call_id"] = serde_json::json!(tool_call_id);
                }

                if let Some(name) = &msg.name {
                    json["name"] = serde_json::json!(name);
                }

                json
            })
            .collect()
    }

    fn convert_tools(&self, tools: &[ToolSchema]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect()
    }
}

#[async_trait]
impl LLMBackend for OpenAIProvider {
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        let estimated_tokens = messages.iter().map(|m| m.content.len() / 4).sum::<usize>();

        self.rate_limiter.wait(estimated_tokens).await;

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": self.convert_messages(messages),
            "temperature": temperature,
        });

        if !tools.is_empty() {
            body["tools"] = serde_json::json!(self.convert_tools(tools));
            body["tool_choice"] = serde_json::json!("auto");
        }

        debug!(
            "OpenAI request: model={}, messages={}",
            self.model,
            messages.len()
        );

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return Err(match status.as_u16() {
                429 => LLMError::RateLimitExceeded,
                401 => LLMError::AuthenticationFailed(error_text),
                _ => LLMError::ApiError(format!("Status {}: {}", status, error_text)),
            });
        }

        let api_response: OpenAIResponse = response
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

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError> {
        let body = serde_json::json!({
            "model": "text-embedding-3-small",
            "input": text,
        });

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(LLMError::ApiError(
                response.text().await.unwrap_or_default(),
            ));
        }

        let api_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        Ok(api_response
            .data
            .first()
            .ok_or_else(|| LLMError::InvalidResponse("No embeddings in response".to_string()))?
            .embedding
            .clone())
    }

    fn model_info(&self) -> ModelInfo {
        match self.model.as_str() {
            "gpt-4-turbo-preview" | "gpt-4-turbo" => ModelInfo {
                model: self.model.clone(),
                provider: "openai".to_string(),
                max_tokens: 128_000,
                input_cost_per_1m: 10.0,
                output_cost_per_1m: 30.0,
                supports_functions: true,
                supports_vision: false,
            },
            "gpt-4" => ModelInfo {
                model: self.model.clone(),
                provider: "openai".to_string(),
                max_tokens: 8_192,
                input_cost_per_1m: 30.0,
                output_cost_per_1m: 60.0,
                supports_functions: true,
                supports_vision: false,
            },
            "gpt-3.5-turbo" => ModelInfo {
                model: self.model.clone(),
                provider: "openai".to_string(),
                max_tokens: 16_385,
                input_cost_per_1m: 0.50,
                output_cost_per_1m: 1.50,
                supports_functions: true,
                supports_vision: false,
            },
            _ => ModelInfo {
                model: self.model.clone(),
                provider: "openai".to_string(),
                max_tokens: 4_096,
                input_cost_per_1m: 1.0,
                output_cost_per_1m: 2.0,
                supports_functions: false,
                supports_vision: false,
            },
        }
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn infer_stream(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<StreamResponse, LLMError> {
        let estimated_tokens = messages.iter().map(|m| m.content.len() / 4).sum::<usize>();
        self.rate_limiter.wait(estimated_tokens).await;

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": self.convert_messages(messages),
            "temperature": temperature,
            "stream": true,
        });

        if !tools.is_empty() {
            body["tools"] = serde_json::json!(self.convert_tools(tools));
            body["tool_choice"] = serde_json::json!("auto");
        }

        debug!(
            "OpenAI streaming request: model={}, messages={}",
            self.model,
            messages.len()
        );

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return Err(match status.as_u16() {
                429 => LLMError::RateLimitExceeded,
                401 => LLMError::AuthenticationFailed(error_text),
                _ => LLMError::ApiError(format!("Status {}: {}", status, error_text)),
            });
        }

        // Parse SSE stream
        let byte_stream = response.bytes_stream();

        let stream = async_stream::stream! {
            let mut buffer = String::new();
            let mut tool_calls: std::collections::HashMap<usize, (String, String, String)> = std::collections::HashMap::new();

            tokio::pin!(byte_stream);

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(LLMError::NetworkError(e.to_string()));
                        break;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE lines
                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            // Emit any pending tool calls
                            for (_, (id, _, _)) in tool_calls.drain() {
                                yield Ok(StreamEvent::ToolCallEnd { id });
                            }
                            yield Ok(StreamEvent::Done { token_usage: None });
                            break;
                        }

                        match serde_json::from_str::<OpenAIStreamResponse>(data) {
                            Ok(response) => {
                                if let Some(choice) = response.choices.first() {
                                    let delta = &choice.delta;

                                    // Text content
                                    if let Some(ref content) = delta.content {
                                        yield Ok(StreamEvent::TextDelta(content.clone()));
                                    }

                                    // Tool calls
                                    if let Some(ref tool_call_deltas) = delta.tool_calls {
                                        for tc in tool_call_deltas {
                                            let index = tc.index;

                                            // New tool call
                                            if let Some(ref func) = tc.function {
                                                if let Some(ref name) = func.name {
                                                    let id = tc.id.clone().unwrap_or_else(|| format!("call_{}", index));
                                                    tool_calls.insert(index, (id.clone(), name.clone(), String::new()));
                                                    yield Ok(StreamEvent::ToolCallStart {
                                                        id,
                                                        name: name.clone(),
                                                    });
                                                }

                                                // Arguments delta
                                                if let Some(ref args) = func.arguments {
                                                    if let Some((id, _, ref mut accumulated_args)) = tool_calls.get_mut(&index) {
                                                        accumulated_args.push_str(args);
                                                        yield Ok(StreamEvent::ToolCallDelta {
                                                            id: id.clone(),
                                                            arguments_delta: args.clone(),
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Check for finish reason
                                    if let Some(ref finish_reason) = choice.finish_reason {
                                        if finish_reason == "tool_calls" {
                                            for (_, (id, _, _)) in tool_calls.drain() {
                                                yield Ok(StreamEvent::ToolCallEnd { id });
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                debug!("Failed to parse OpenAI SSE event: {} - data: {}", e, data);
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

// OpenAI API types

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolCall {
    id: String,
    function: OpenAIFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

// ============================================================================
// Streaming API Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolCall {
    index: usize,
    id: Option<String>,
    function: Option<OpenAIStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}
