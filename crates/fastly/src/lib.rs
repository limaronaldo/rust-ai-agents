//! # Rust AI Agents - Fastly Compute
//!
//! Run AI agents on Fastly Compute@Edge.
//!
//! This crate provides a Fastly-native implementation using:
//! - `rust-ai-agents-llm-client` for request/response logic
//! - Fastly SDK for HTTP requests
//!
//! ## Usage
//!
//! ```rust,ignore
//! use rust_ai_agents_fastly::{FastlyAgent, FastlyAgentConfig};
//! use fastly::{Request, Response};
//!
//! #[fastly::main]
//! fn main(req: Request) -> Result<Response, fastly::Error> {
//!     let config = FastlyAgentConfig::new(
//!         "openai",
//!         std::env::var("OPENAI_API_KEY").unwrap(),
//!         "gpt-4o-mini",
//!     );
//!
//!     let mut agent = FastlyAgent::new(config, "llm-backend");
//!     let response = agent.chat("Hello!")?;
//!
//!     Ok(Response::from_body(response.content))
//! }
//! ```

use std::io::Read;

use fastly::http::{header, Method, StatusCode};
use fastly::{Backend, Body, Request, Response};
use rust_ai_agents_llm_client::{
    LlmResponse, Message, Provider, RequestBuilder, ResponseParser, StreamChunk,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur in the Fastly agent
#[derive(Debug, Error)]
pub enum FastlyAgentError {
    #[error("LLM client error: {0}")]
    LlmClient(#[from] rust_ai_agents_llm_client::LlmClientError),

    #[error("Fastly error: {0}")]
    Fastly(String),

    #[error("HTTP error: {status} - {message}")]
    Http { status: u16, message: String },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid provider: {0}")]
    InvalidProvider(String),
}

impl From<fastly::Error> for FastlyAgentError {
    fn from(e: fastly::Error) -> Self {
        FastlyAgentError::Fastly(e.to_string())
    }
}

impl From<fastly::http::request::SendError> for FastlyAgentError {
    fn from(e: fastly::http::request::SendError) -> Self {
        FastlyAgentError::Fastly(format!("Send error: {:?}", e))
    }
}

pub type Result<T> = std::result::Result<T, FastlyAgentError>;

/// Configuration for the Fastly agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastlyAgentConfig {
    /// LLM provider
    pub provider: String,
    /// API key
    pub api_key: String,
    /// Model identifier
    pub model: String,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Temperature (0.0 - 2.0)
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
}

impl FastlyAgentConfig {
    /// Create a new configuration
    pub fn new(
        provider: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            api_key: api_key.into(),
            model: model.into(),
            system_prompt: None,
            temperature: 0.7,
            max_tokens: 4096,
        }
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }
}

/// AI Agent for Fastly Compute
pub struct FastlyAgent {
    config: FastlyAgentConfig,
    backend_name: String,
    messages: Vec<Message>,
    provider: Provider,
}

impl FastlyAgent {
    /// Create a new Fastly agent
    ///
    /// # Arguments
    /// * `config` - Agent configuration
    /// * `backend_name` - Name of the Fastly backend configured for LLM API
    pub fn new(config: FastlyAgentConfig, backend_name: impl Into<String>) -> Result<Self> {
        let provider = config
            .provider
            .parse::<Provider>()
            .map_err(|_| FastlyAgentError::InvalidProvider(config.provider.clone()))?;

        Ok(Self {
            config,
            backend_name: backend_name.into(),
            messages: Vec::new(),
            provider,
        })
    }

    /// Send a message and get a response
    pub fn chat(&mut self, message: &str) -> Result<LlmResponse> {
        // Add user message
        self.messages.push(Message::user(message));

        // Build request using llm-client
        let mut builder = RequestBuilder::new(self.provider)
            .model(&self.config.model)
            .api_key(&self.config.api_key)
            .temperature(self.config.temperature)
            .max_tokens(self.config.max_tokens)
            .stream(false);

        // Add system prompt if present
        if let Some(ref system) = self.config.system_prompt {
            builder = builder.add_message(Message::system(system));
        }

        // Add conversation messages
        builder = builder.messages(&self.messages);

        let http_request = builder.build()?;

        // Create Fastly request
        let mut req = Request::new(Method::POST, &http_request.url);

        for (key, value) in &http_request.headers {
            req.set_header(key, value);
        }

        req.set_body(http_request.body);

        // Send request via Fastly backend
        let backend = Backend::from_name(&self.backend_name)
            .map_err(|e| FastlyAgentError::Fastly(format!("Backend not found: {}", e)))?;

        let resp = req.send(backend)?;

        // Check status
        if resp.get_status() != StatusCode::OK {
            let status = resp.get_status().as_u16();
            let body = resp.into_body_str();
            return Err(FastlyAgentError::Http {
                status,
                message: body,
            });
        }

        // Parse response using llm-client
        let body = resp.into_body_str();
        let response = ResponseParser::parse(self.provider, &body)?;

        // Add assistant message to history
        self.messages.push(Message::assistant(&response.content));

        Ok(response)
    }

    /// Send a message and get a streaming response
    ///
    /// Returns an iterator over stream chunks
    pub fn chat_stream(&mut self, message: &str) -> Result<StreamingResponse> {
        // Add user message
        self.messages.push(Message::user(message));

        // Build request using llm-client
        let mut builder = RequestBuilder::new(self.provider)
            .model(&self.config.model)
            .api_key(&self.config.api_key)
            .temperature(self.config.temperature)
            .max_tokens(self.config.max_tokens)
            .stream(true);

        // Add system prompt if present
        if let Some(ref system) = self.config.system_prompt {
            builder = builder.add_message(Message::system(system));
        }

        // Add conversation messages
        builder = builder.messages(&self.messages);

        let http_request = builder.build()?;

        // Create Fastly request
        let mut req = Request::new(Method::POST, &http_request.url);

        for (key, value) in &http_request.headers {
            req.set_header(key, value);
        }

        req.set_body(http_request.body);

        // Send request via Fastly backend
        let backend = Backend::from_name(&self.backend_name)
            .map_err(|e| FastlyAgentError::Fastly(format!("Backend not found: {}", e)))?;

        let resp = req.send(backend)?;

        // Check status
        if resp.get_status() != StatusCode::OK {
            let status = resp.get_status().as_u16();
            let body = resp.into_body_str();
            return Err(FastlyAgentError::Http {
                status,
                message: body,
            });
        }

        Ok(StreamingResponse {
            body: resp.into_body(),
            provider: self.provider,
            buffer: String::new(),
            accumulated_content: String::new(),
        })
    }

    /// Clear conversation history
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get conversation history
    pub fn history(&self) -> &[Message] {
        &self.messages
    }

    /// Add assistant response to history (for streaming)
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(Message::assistant(content));
    }
}

/// Streaming response iterator
pub struct StreamingResponse {
    body: Body,
    provider: Provider,
    buffer: String,
    accumulated_content: String,
}

impl StreamingResponse {
    /// Get the full accumulated content after streaming completes
    pub fn content(&self) -> &str {
        &self.accumulated_content
    }

    /// Process the next chunk from the stream
    pub fn next_chunk(&mut self) -> Result<Option<StreamChunk>> {
        // Read more data into buffer
        let mut chunk_buf = [0u8; 1024];
        let bytes_read = self
            .body
            .read(&mut chunk_buf)
            .map_err(|e| FastlyAgentError::Fastly(e.to_string()))?;

        if bytes_read == 0 {
            return Ok(None);
        }

        // Add to buffer
        let chunk_str = String::from_utf8_lossy(&chunk_buf[..bytes_read]);
        self.buffer.push_str(&chunk_str);

        // Process complete lines
        while let Some(line_end) = self.buffer.find('\n') {
            let line = self.buffer[..line_end].to_string();
            self.buffer = self.buffer[line_end + 1..].to_string();

            if let Ok(Some(chunk)) = ResponseParser::parse_stream_line(self.provider, &line) {
                if let Some(ref content) = chunk.content {
                    self.accumulated_content.push_str(content);
                }
                return Ok(Some(chunk));
            }
        }

        // No complete chunk yet, keep reading
        self.next_chunk()
    }
}

impl std::io::Read for StreamingResponse {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.body.read(buf)
    }
}

/// Create a simple chat completion handler for Fastly
///
/// This is a convenience function for simple use cases.
pub fn handle_chat_request(
    incoming_req: Request,
    config: FastlyAgentConfig,
    backend_name: &str,
) -> Result<Response> {
    // Parse incoming request
    let body: serde_json::Value = serde_json::from_str(&incoming_req.into_body_str())?;

    let message = body["message"]
        .as_str()
        .ok_or_else(|| FastlyAgentError::Fastly("Missing 'message' field".to_string()))?;

    // Create agent and chat
    let mut agent = FastlyAgent::new(config, backend_name)?;
    let response = agent.chat(message)?;

    // Build response
    let response_body = serde_json::json!({
        "content": response.content,
        "usage": response.usage,
    });

    let mut resp = Response::from_body(response_body.to_string());
    resp.set_header(header::CONTENT_TYPE, "application/json");

    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = FastlyAgentConfig::new("openai", "sk-test", "gpt-4o-mini")
            .with_system_prompt("You are helpful")
            .with_temperature(0.5)
            .with_max_tokens(2048);

        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "gpt-4o-mini");
        assert_eq!(config.system_prompt, Some("You are helpful".to_string()));
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, 2048);
    }

    #[test]
    fn test_provider_parsing() {
        assert!("openai".parse::<Provider>().is_ok());
        assert!("anthropic".parse::<Provider>().is_ok());
        assert!("openrouter".parse::<Provider>().is_ok());
        assert!("invalid".parse::<Provider>().is_err());
    }
}
