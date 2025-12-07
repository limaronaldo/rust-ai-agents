//! Response parsing for LLM API responses
//!
//! This module provides runtime-agnostic parsing of responses from
//! OpenAI, Anthropic, and OpenRouter APIs.

use serde::{Deserialize, Serialize};

use crate::error::{LlmClientError, Result};
use crate::provider::Provider;

/// Parsed LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// The generated text content
    pub content: String,
    /// Token usage information (if available)
    pub usage: Option<Usage>,
    /// Tool calls requested by the model (if any)
    pub tool_calls: Vec<ToolCall>,
    /// The finish reason
    pub finish_reason: Option<String>,
    /// Model used for generation
    pub model: Option<String>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total tokens used
    pub total_tokens: u32,
}

/// Tool call requested by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: String,
    /// Name of the function to call
    pub function_name: String,
    /// JSON arguments for the function
    pub arguments: String,
}

/// Streaming chunk from SSE response
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Content delta (partial text)
    pub content: Option<String>,
    /// Whether this is the final chunk
    pub done: bool,
    /// Finish reason (on final chunk)
    pub finish_reason: Option<String>,
    /// Tool call chunks (for streaming tool calls)
    pub tool_call_chunk: Option<ToolCallChunk>,
}

/// Partial tool call information from streaming
#[derive(Debug, Clone)]
pub struct ToolCallChunk {
    /// Index of the tool call (for ordering)
    pub index: usize,
    /// Tool call ID (first chunk only)
    pub id: Option<String>,
    /// Function name (first chunk only)
    pub function_name: Option<String>,
    /// Arguments delta
    pub arguments_delta: Option<String>,
}

/// Response parser for different providers
pub struct ResponseParser;

impl ResponseParser {
    /// Parse a complete (non-streaming) response
    pub fn parse(provider: Provider, body: &str) -> Result<LlmResponse> {
        match provider {
            Provider::OpenAI | Provider::OpenRouter => Self::parse_openai(body),
            Provider::Anthropic => Self::parse_anthropic(body),
        }
    }

    /// Parse a streaming SSE line
    pub fn parse_stream_line(provider: Provider, line: &str) -> Result<Option<StreamChunk>> {
        // Skip empty lines and comments
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            return Ok(None);
        }

        // Extract data from SSE format
        let data = if let Some(stripped) = line.strip_prefix("data: ") {
            stripped.trim()
        } else {
            return Ok(None);
        };

        // Check for stream end
        if data == "[DONE]" {
            return Ok(Some(StreamChunk {
                content: None,
                done: true,
                finish_reason: None,
                tool_call_chunk: None,
            }));
        }

        match provider {
            Provider::OpenAI | Provider::OpenRouter => Self::parse_openai_stream_chunk(data),
            Provider::Anthropic => Self::parse_anthropic_stream_chunk(data),
        }
    }

    fn parse_openai(body: &str) -> Result<LlmResponse> {
        let json: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| LlmClientError::UnexpectedFormat(e.to_string()))?;

        // Check for error response
        if json.get("error").is_some() {
            return Err(LlmClientError::from_api_response(&json));
        }

        let choices = json
            .get("choices")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                LlmClientError::UnexpectedFormat("Missing 'choices' field".to_string())
            })?;

        let first_choice = choices
            .first()
            .ok_or_else(|| LlmClientError::UnexpectedFormat("Empty choices array".to_string()))?;

        let message = first_choice.get("message").ok_or_else(|| {
            LlmClientError::UnexpectedFormat("Missing 'message' field".to_string())
        })?;

        let content = message
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let finish_reason = first_choice
            .get("finish_reason")
            .and_then(|f| f.as_str())
            .map(|s| s.to_string());

        let model = json
            .get("model")
            .and_then(|m| m.as_str())
            .map(|s| s.to_string());

        // Parse tool calls if present
        let tool_calls = Self::parse_openai_tool_calls(message);

        // Parse usage
        let usage = json.get("usage").and_then(|u| {
            Some(Usage {
                prompt_tokens: u.get("prompt_tokens")?.as_u64()? as u32,
                completion_tokens: u.get("completion_tokens")?.as_u64()? as u32,
                total_tokens: u.get("total_tokens")?.as_u64()? as u32,
            })
        });

        Ok(LlmResponse {
            content,
            usage,
            tool_calls,
            finish_reason,
            model,
        })
    }

    fn parse_openai_tool_calls(message: &serde_json::Value) -> Vec<ToolCall> {
        let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) else {
            return Vec::new();
        };

        tool_calls
            .iter()
            .filter_map(|tc| {
                let id = tc.get("id")?.as_str()?.to_string();
                let function = tc.get("function")?;
                let function_name = function.get("name")?.as_str()?.to_string();
                let arguments = function.get("arguments")?.as_str()?.to_string();
                Some(ToolCall {
                    id,
                    function_name,
                    arguments,
                })
            })
            .collect()
    }

    fn parse_anthropic(body: &str) -> Result<LlmResponse> {
        let json: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| LlmClientError::UnexpectedFormat(e.to_string()))?;

        // Check for error response
        if json.get("error").is_some() {
            return Err(LlmClientError::from_api_response(&json));
        }

        // Anthropic returns content as an array of blocks
        let content_blocks = json
            .get("content")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                LlmClientError::UnexpectedFormat("Missing 'content' field".to_string())
            })?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for block in content_blocks {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match block_type {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        if !content.is_empty() {
                            content.push('\n');
                        }
                        content.push_str(text);
                    }
                }
                "tool_use" => {
                    if let (Some(id), Some(name), Some(input)) = (
                        block.get("id").and_then(|i| i.as_str()),
                        block.get("name").and_then(|n| n.as_str()),
                        block.get("input"),
                    ) {
                        tool_calls.push(ToolCall {
                            id: id.to_string(),
                            function_name: name.to_string(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        });
                    }
                }
                _ => {}
            }
        }

        let finish_reason = json
            .get("stop_reason")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());

        let model = json
            .get("model")
            .and_then(|m| m.as_str())
            .map(|s| s.to_string());

        let usage = json.get("usage").and_then(|u| {
            Some(Usage {
                prompt_tokens: u.get("input_tokens")?.as_u64()? as u32,
                completion_tokens: u.get("output_tokens")?.as_u64()? as u32,
                total_tokens: (u.get("input_tokens")?.as_u64()?
                    + u.get("output_tokens")?.as_u64()?) as u32,
            })
        });

        Ok(LlmResponse {
            content,
            usage,
            tool_calls,
            finish_reason,
            model,
        })
    }

    fn parse_openai_stream_chunk(data: &str) -> Result<Option<StreamChunk>> {
        let json: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| LlmClientError::UnexpectedFormat(e.to_string()))?;

        let choices = json.get("choices").and_then(|c| c.as_array());
        let Some(choices) = choices else {
            return Ok(None);
        };

        let Some(first_choice) = choices.first() else {
            return Ok(None);
        };

        let finish_reason = first_choice.get("finish_reason").and_then(|f| {
            if f.is_null() {
                None
            } else {
                f.as_str().map(|s| s.to_string())
            }
        });

        let delta = first_choice.get("delta");

        let content = delta
            .and_then(|d| d.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());

        // Parse streaming tool calls
        let tool_call_chunk = delta
            .and_then(|d| d.get("tool_calls"))
            .and_then(|t| t.as_array())
            .and_then(|arr| arr.first())
            .and_then(|tc| {
                let index = tc.get("index")?.as_u64()? as usize;
                let id = tc.get("id").and_then(|i| i.as_str()).map(|s| s.to_string());
                let function = tc.get("function");
                let function_name = function
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string());
                let arguments_delta = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .map(|s| s.to_string());

                Some(ToolCallChunk {
                    index,
                    id,
                    function_name,
                    arguments_delta,
                })
            });

        let done = finish_reason.is_some();

        Ok(Some(StreamChunk {
            content,
            done,
            finish_reason,
            tool_call_chunk,
        }))
    }

    fn parse_anthropic_stream_chunk(data: &str) -> Result<Option<StreamChunk>> {
        let json: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| LlmClientError::UnexpectedFormat(e.to_string()))?;

        let event_type = json.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match event_type {
            "content_block_delta" => {
                let delta = json.get("delta");
                let delta_type = delta
                    .and_then(|d| d.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");

                match delta_type {
                    "text_delta" => {
                        let content = delta
                            .and_then(|d| d.get("text"))
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string());

                        Ok(Some(StreamChunk {
                            content,
                            done: false,
                            finish_reason: None,
                            tool_call_chunk: None,
                        }))
                    }
                    "input_json_delta" => {
                        let partial_json = delta
                            .and_then(|d| d.get("partial_json"))
                            .and_then(|p| p.as_str())
                            .map(|s| s.to_string());

                        let index =
                            json.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;

                        Ok(Some(StreamChunk {
                            content: None,
                            done: false,
                            finish_reason: None,
                            tool_call_chunk: Some(ToolCallChunk {
                                index,
                                id: None,
                                function_name: None,
                                arguments_delta: partial_json,
                            }),
                        }))
                    }
                    _ => Ok(None),
                }
            }
            "content_block_start" => {
                let content_block = json.get("content_block");
                let block_type = content_block
                    .and_then(|b| b.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");

                if block_type == "tool_use" {
                    let id = content_block
                        .and_then(|b| b.get("id"))
                        .and_then(|i| i.as_str())
                        .map(|s| s.to_string());
                    let name = content_block
                        .and_then(|b| b.get("name"))
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string());
                    let index = json.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;

                    Ok(Some(StreamChunk {
                        content: None,
                        done: false,
                        finish_reason: None,
                        tool_call_chunk: Some(ToolCallChunk {
                            index,
                            id,
                            function_name: name,
                            arguments_delta: None,
                        }),
                    }))
                } else {
                    Ok(None)
                }
            }
            "message_delta" => {
                let stop_reason = json
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string());

                Ok(Some(StreamChunk {
                    content: None,
                    done: stop_reason.is_some(),
                    finish_reason: stop_reason,
                    tool_call_chunk: None,
                }))
            }
            "message_stop" => Ok(Some(StreamChunk {
                content: None,
                done: true,
                finish_reason: Some("end_turn".to_string()),
                tool_call_chunk: None,
            })),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_openai_response() {
        let body = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response = ResponseParser::parse(Provider::OpenAI, body).unwrap();
        assert_eq!(response.content, "Hello, world!");
        assert_eq!(response.finish_reason, Some("stop".to_string()));
        assert_eq!(response.model, Some("gpt-4".to_string()));
        assert!(response.tool_calls.is_empty());

        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_openai_with_tool_calls() {
        let body = r#"{
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"Paris\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;

        let response = ResponseParser::parse(Provider::OpenAI, body).unwrap();
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_123");
        assert_eq!(response.tool_calls[0].function_name, "get_weather");
        assert_eq!(
            response.tool_calls[0].arguments,
            "{\"location\": \"Paris\"}"
        );
    }

    #[test]
    fn test_parse_anthropic_response() {
        let body = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [{
                "type": "text",
                "text": "Hello from Claude!"
            }],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        }"#;

        let response = ResponseParser::parse(Provider::Anthropic, body).unwrap();
        assert_eq!(response.content, "Hello from Claude!");
        assert_eq!(response.finish_reason, Some("end_turn".to_string()));
        assert_eq!(response.model, Some("claude-3-opus-20240229".to_string()));

        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_anthropic_with_tool_use() {
        let body = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [{
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "Paris"}
            }],
            "stop_reason": "tool_use"
        }"#;

        let response = ResponseParser::parse(Provider::Anthropic, body).unwrap();
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "toolu_123");
        assert_eq!(response.tool_calls[0].function_name, "get_weather");
    }

    #[test]
    fn test_parse_openai_stream_chunk() {
        let data = r#"{"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

        let chunk = ResponseParser::parse_stream_line(Provider::OpenAI, &format!("data: {}", data))
            .unwrap()
            .unwrap();

        assert_eq!(chunk.content, Some("Hello".to_string()));
        assert!(!chunk.done);
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn test_parse_stream_done() {
        let chunk = ResponseParser::parse_stream_line(Provider::OpenAI, "data: [DONE]")
            .unwrap()
            .unwrap();

        assert!(chunk.done);
        assert!(chunk.content.is_none());
    }

    #[test]
    fn test_parse_anthropic_stream_text_delta() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;

        let chunk =
            ResponseParser::parse_stream_line(Provider::Anthropic, &format!("data: {}", data))
                .unwrap()
                .unwrap();

        assert_eq!(chunk.content, Some("Hello".to_string()));
        assert!(!chunk.done);
    }

    #[test]
    fn test_parse_error_response() {
        let body = r#"{"error": {"message": "Invalid API key", "type": "invalid_request_error"}}"#;

        let result = ResponseParser::parse(Provider::OpenAI, body);
        assert!(result.is_err());
        assert!(matches!(result, Err(LlmClientError::ApiError { .. })));
    }
}
