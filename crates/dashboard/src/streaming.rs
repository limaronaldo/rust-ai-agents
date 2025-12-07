//! Server-Sent Events (SSE) streaming for LLM responses.
//!
//! This module provides SSE endpoints for streaming LLM inference results
//! to clients in real-time.
//!
//! # Feature Flag
//!
//! This module requires the `streaming` feature:
//!
//! ```toml
//! rust-ai-agents-dashboard = { version = "0.1", features = ["streaming"] }
//! ```

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Json,
};
use rust_ai_agents_core::{tool::ToolSchema, LLMMessage, MessageRole};
use rust_ai_agents_providers::{LLMBackend, StreamEvent};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio_stream::StreamExt;

use crate::state::DashboardState;

/// Request body for streaming inference
#[derive(Debug, Clone, Deserialize)]
pub struct StreamInferenceRequest {
    /// Messages to send to the LLM
    pub messages: Vec<MessageInput>,
    /// Optional tools available to the model
    #[serde(default)]
    pub tools: Vec<ToolSchema>,
    /// Temperature for sampling (0.0 - 1.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Model to use (optional, uses default if not specified)
    pub model: Option<String>,
}

fn default_temperature() -> f32 {
    0.7
}

/// Input message format
#[derive(Debug, Clone, Deserialize)]
pub struct MessageInput {
    pub role: String,
    pub content: String,
}

impl From<MessageInput> for LLMMessage {
    fn from(msg: MessageInput) -> Self {
        let role = match msg.role.to_lowercase().as_str() {
            "system" => MessageRole::System,
            "assistant" => MessageRole::Assistant,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User,
        };
        LLMMessage {
            role,
            content: msg.content,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
}

/// SSE event types sent to clients
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum StreamEventOutput {
    /// Incremental text chunk
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    /// Tool call started
    #[serde(rename = "tool_call_start")]
    ToolCallStart { id: String, name: String },
    /// Tool call arguments chunk
    #[serde(rename = "tool_call_delta")]
    ToolCallDelta { id: String, arguments_delta: String },
    /// Tool call completed
    #[serde(rename = "tool_call_end")]
    ToolCallEnd { id: String },
    /// Stream completed
    #[serde(rename = "done")]
    Done {
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    },
    /// Error occurred
    #[serde(rename = "error")]
    Error { message: String },
}

impl From<StreamEvent> for StreamEventOutput {
    fn from(event: StreamEvent) -> Self {
        match event {
            StreamEvent::TextDelta(text) => StreamEventOutput::TextDelta { text },
            StreamEvent::ToolCallStart { id, name } => {
                StreamEventOutput::ToolCallStart { id, name }
            }
            StreamEvent::ToolCallDelta {
                id,
                arguments_delta,
            } => StreamEventOutput::ToolCallDelta {
                id,
                arguments_delta,
            },
            StreamEvent::ToolCallEnd { id } => StreamEventOutput::ToolCallEnd { id },
            StreamEvent::Done { token_usage } => StreamEventOutput::Done {
                input_tokens: token_usage.as_ref().map(|t| t.prompt_tokens as u32),
                output_tokens: token_usage.as_ref().map(|t| t.completion_tokens as u32),
            },
            StreamEvent::Error(message) => StreamEventOutput::Error { message },
        }
    }
}

/// Streaming state that holds the LLM backend
pub struct StreamingState<B: LLMBackend> {
    pub backend: Arc<B>,
    pub dashboard: Arc<DashboardState>,
}

impl<B: LLMBackend> StreamingState<B> {
    pub fn new(backend: Arc<B>, dashboard: Arc<DashboardState>) -> Self {
        Self { backend, dashboard }
    }
}

/// SSE streaming inference handler
///
/// Returns a Server-Sent Events stream with LLM response chunks.
///
/// # Example
///
/// ```javascript
/// const eventSource = new EventSource('/api/inference/stream');
/// eventSource.onmessage = (event) => {
///     const data = JSON.parse(event.data);
///     if (data.type === 'text_delta') {
///         appendText(data.data.text);
///     }
/// };
/// ```
pub async fn stream_inference_handler<B: LLMBackend + 'static>(
    State(state): State<Arc<StreamingState<B>>>,
    Json(request): Json<StreamInferenceRequest>,
) -> impl IntoResponse {
    let messages: Vec<LLMMessage> = request.messages.into_iter().map(Into::into).collect();
    let tools = request.tools;
    let temperature = request.temperature;
    let backend = state.backend.clone();

    // Create the SSE stream
    let stream = async_stream::stream! {
        match backend.infer_stream(&messages, &tools, temperature).await {
            Ok(mut response_stream) => {
                while let Some(result) = response_stream.next().await {
                    match result {
                        Ok(event) => {
                            let output: StreamEventOutput = event.into();
                            let json = serde_json::to_string(&output).unwrap_or_default();
                            yield Ok::<_, Infallible>(Event::default().data(json));
                        }
                        Err(e) => {
                            let error = StreamEventOutput::Error {
                                message: e.to_string(),
                            };
                            let json = serde_json::to_string(&error).unwrap_or_default();
                            yield Ok(Event::default().data(json));
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                let error = StreamEventOutput::Error {
                    message: e.to_string(),
                };
                let json = serde_json::to_string(&error).unwrap_or_default();
                yield Ok(Event::default().data(json));
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

/// POST handler for streaming inference (accepts JSON body)
pub async fn post_stream_inference_handler<B: LLMBackend + 'static>(
    State(state): State<Arc<StreamingState<B>>>,
    Json(request): Json<StreamInferenceRequest>,
) -> impl IntoResponse {
    stream_inference_handler(State(state), Json(request)).await
}

/// Simple chat completion endpoint (non-streaming, for comparison)
pub async fn chat_completion_handler<B: LLMBackend + 'static>(
    State(state): State<Arc<StreamingState<B>>>,
    Json(request): Json<StreamInferenceRequest>,
) -> impl IntoResponse {
    let messages: Vec<LLMMessage> = request.messages.into_iter().map(Into::into).collect();
    let tools = request.tools;
    let temperature = request.temperature;

    match state.backend.infer(&messages, &tools, temperature).await {
        Ok(output) => Json(serde_json::json!({
            "content": output.content,
            "tool_calls": output.tool_calls,
            "token_usage": output.token_usage,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// Create SSE routes for streaming
///
/// # Example
///
/// ```rust,ignore
/// use rust_ai_agents_dashboard::streaming::{streaming_routes, StreamingState};
/// use rust_ai_agents_providers::AnthropicProvider;
///
/// let backend = Arc::new(AnthropicProvider::new(api_key));
/// let streaming_state = Arc::new(StreamingState::new(backend, dashboard_state));
///
/// let app = Router::new()
///     .merge(streaming_routes())
///     .with_state(streaming_state);
/// ```
pub fn streaming_routes<B: LLMBackend + 'static>() -> axum::Router<Arc<StreamingState<B>>> {
    use axum::routing::post;

    axum::Router::new()
        .route(
            "/api/inference/stream",
            post(post_stream_inference_handler::<B>),
        )
        .route("/api/inference/chat", post(chat_completion_handler::<B>))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_input_conversion() {
        let input = MessageInput {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        };

        let llm_msg: LLMMessage = input.into();
        assert!(matches!(llm_msg.role, MessageRole::User));
        assert_eq!(llm_msg.content, "Hello!");
    }

    #[test]
    fn test_stream_event_output_serialization() {
        let event = StreamEventOutput::TextDelta {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("text_delta"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_stream_event_conversion() {
        let event = StreamEvent::TextDelta("test".to_string());
        let output: StreamEventOutput = event.into();
        match output {
            StreamEventOutput::TextDelta { text } => assert_eq!(text, "test"),
            _ => panic!("Wrong variant"),
        }
    }
}
