//! # Rust AI Agents - WebAssembly
//!
//! WebAssembly bindings for running AI agents in the browser.
//!
//! This crate provides a JavaScript-friendly API for:
//! - Making LLM API calls (OpenAI, Anthropic, OpenRouter)
//! - Streaming responses
//! - Tool/function calling
//!
//! ## Usage in JavaScript
//!
//! ```javascript
//! import init, { WasmAgent, WasmMessage } from 'rust-ai-agents-wasm';
//!
//! await init();
//!
//! const agent = new WasmAgent({
//!     provider: 'openai',
//!     apiKey: 'sk-...',
//!     model: 'gpt-4',
//!     systemPrompt: 'You are a helpful assistant.',
//! });
//!
//! // Non-streaming
//! const response = await agent.chat('Hello!');
//! console.log(response.content);
//!
//! // Streaming
//! for await (const chunk of agent.chatStream('Tell me a story')) {
//!     process.stdout.write(chunk.text);
//! }
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

mod error;
mod provider;
mod streaming;
mod types;

pub use error::*;
pub use provider::*;
pub use streaming::*;
pub use types::*;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Set up panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    // Initialize tracing for WASM
    tracing_wasm::set_as_global_default();
}

/// Agent configuration for WASM
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmAgentConfig {
    /// LLM provider: "openai", "anthropic", "openrouter"
    #[wasm_bindgen(skip)]
    pub provider: String,

    /// API key
    #[wasm_bindgen(skip)]
    pub api_key: String,

    /// Model identifier
    #[wasm_bindgen(skip)]
    pub model: String,

    /// System prompt
    #[wasm_bindgen(skip)]
    pub system_prompt: Option<String>,

    /// Temperature (0.0 - 2.0)
    #[wasm_bindgen(skip)]
    pub temperature: f32,

    /// Maximum tokens to generate
    #[wasm_bindgen(skip)]
    pub max_tokens: u32,
}

#[wasm_bindgen]
impl WasmAgentConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(provider: String, api_key: String, model: String) -> Self {
        Self {
            provider,
            api_key,
            model,
            system_prompt: None,
            temperature: 0.7,
            max_tokens: 4096,
        }
    }

    #[wasm_bindgen(setter)]
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = Some(prompt);
    }

    #[wasm_bindgen(setter)]
    pub fn set_temperature(&mut self, temp: f32) {
        self.temperature = temp;
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_tokens(&mut self, tokens: u32) {
        self.max_tokens = tokens;
    }
}

/// WASM-compatible AI Agent
#[wasm_bindgen]
pub struct WasmAgent {
    config: WasmAgentConfig,
    messages: Vec<WasmMessage>,
}

#[wasm_bindgen]
impl WasmAgent {
    /// Create a new WASM agent
    #[wasm_bindgen(constructor)]
    pub fn new(config: WasmAgentConfig) -> Self {
        Self {
            config,
            messages: Vec::new(),
        }
    }

    /// Create agent from JavaScript object
    #[wasm_bindgen(js_name = fromObject)]
    pub fn from_object(obj: JsValue) -> Result<WasmAgent, JsValue> {
        let config: WasmAgentConfig = serde_wasm_bindgen::from_value(obj)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
        Ok(Self::new(config))
    }

    /// Send a message and get a response (non-streaming)
    #[wasm_bindgen]
    pub async fn chat(&mut self, message: &str) -> Result<JsValue, JsValue> {
        // Add user message
        self.messages.push(WasmMessage::user(message.to_string()));

        // Call LLM
        let response = self.call_llm(false).await?;

        // Parse response
        let content = self.extract_content(&response)?;

        // Add assistant message
        self.messages.push(WasmMessage::assistant(content.clone()));

        Ok(serde_wasm_bindgen::to_value(&WasmResponse {
            content,
            tool_calls: None,
            usage: None,
        })?)
    }

    /// Send a message and get a streaming response
    /// Returns a ReadableStream that yields text chunks
    #[wasm_bindgen(js_name = chatStream)]
    pub async fn chat_stream(&mut self, message: &str) -> Result<JsValue, JsValue> {
        // Add user message
        self.messages.push(WasmMessage::user(message.to_string()));

        // Call LLM with streaming
        let response = self.call_llm(true).await?;

        // Return the readable stream from the response body
        let body = Response::from(response)
            .body()
            .ok_or_else(|| JsValue::from_str("No response body"))?;

        Ok(body.into())
    }

    /// Clear conversation history
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get conversation history as JSON
    #[wasm_bindgen(js_name = getHistory)]
    pub fn get_history(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.messages)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Internal: Call the LLM API
    async fn call_llm(&self, stream: bool) -> Result<JsValue, JsValue> {
        let (url, headers, body) = self.build_request(stream)?;

        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_mode(RequestMode::Cors);

        // Set headers
        let js_headers = web_sys::Headers::new()?;
        for (key, value) in headers {
            js_headers.set(&key, &value)?;
        }
        opts.set_headers(&js_headers);

        // Set body
        opts.set_body(&JsValue::from_str(&body));

        let request = Request::new_with_str_and_init(&url, &opts)?;

        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

        let resp: Response = resp_value.dyn_into()?;

        if !resp.ok() {
            let status = resp.status();
            let text = JsFuture::from(resp.text()?).await?;
            return Err(JsValue::from_str(&format!(
                "HTTP {}: {}",
                status,
                text.as_string().unwrap_or_default()
            )));
        }

        if stream {
            Ok(resp.into())
        } else {
            let json = JsFuture::from(resp.json()?).await?;
            Ok(json)
        }
    }

    /// Build request based on provider
    fn build_request(
        &self,
        stream: bool,
    ) -> Result<(String, Vec<(String, String)>, String), JsValue> {
        match self.config.provider.as_str() {
            "openai" => self.build_openai_request(stream),
            "anthropic" => self.build_anthropic_request(stream),
            "openrouter" => self.build_openrouter_request(stream),
            _ => Err(JsValue::from_str(&format!(
                "Unknown provider: {}",
                self.config.provider
            ))),
        }
    }

    fn build_openai_request(
        &self,
        stream: bool,
    ) -> Result<(String, Vec<(String, String)>, String), JsValue> {
        let url = "https://api.openai.com/v1/chat/completions".to_string();

        let headers = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            (
                "Authorization".to_string(),
                format!("Bearer {}", self.config.api_key),
            ),
        ];

        let messages: Vec<serde_json::Value> = self.build_messages();

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream
        });

        Ok((url, headers, body.to_string()))
    }

    fn build_anthropic_request(
        &self,
        stream: bool,
    ) -> Result<(String, Vec<(String, String)>, String), JsValue> {
        let url = "https://api.anthropic.com/v1/messages".to_string();

        let headers = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            ("x-api-key".to_string(), self.config.api_key.clone()),
            ("anthropic-version".to_string(), "2023-06-01".to_string()),
            (
                "anthropic-dangerous-direct-browser-access".to_string(),
                "true".to_string(),
            ),
        ];

        // Anthropic uses separate system prompt
        let (system, messages) = self.build_anthropic_messages();

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": stream
        });

        if let Some(sys) = system {
            body["system"] = serde_json::json!(sys);
        }

        Ok((url, headers, body.to_string()))
    }

    fn build_openrouter_request(
        &self,
        stream: bool,
    ) -> Result<(String, Vec<(String, String)>, String), JsValue> {
        let url = "https://openrouter.ai/api/v1/chat/completions".to_string();

        let headers = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            (
                "Authorization".to_string(),
                format!("Bearer {}", self.config.api_key),
            ),
            (
                "HTTP-Referer".to_string(),
                "https://rust-ai-agents.dev".to_string(),
            ),
        ];

        let messages: Vec<serde_json::Value> = self.build_messages();

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream
        });

        Ok((url, headers, body.to_string()))
    }

    fn build_messages(&self) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();

        // Add system prompt if present
        if let Some(ref system) = self.config.system_prompt {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }

        // Add conversation history
        for msg in &self.messages {
            messages.push(serde_json::json!({
                "role": msg.role,
                "content": msg.content
            }));
        }

        messages
    }

    fn build_anthropic_messages(&self) -> (Option<String>, Vec<serde_json::Value>) {
        let system = self.config.system_prompt.clone();

        let messages: Vec<serde_json::Value> = self
            .messages
            .iter()
            .map(|msg| {
                serde_json::json!({
                    "role": if msg.role == "user" { "user" } else { "assistant" },
                    "content": msg.content
                })
            })
            .collect();

        (system, messages)
    }

    fn extract_content(&self, response: &JsValue) -> Result<String, JsValue> {
        let obj: serde_json::Value = serde_wasm_bindgen::from_value(response.clone())
            .map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;

        match self.config.provider.as_str() {
            "openai" | "openrouter" => obj["choices"][0]["message"]["content"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| JsValue::from_str("No content in response")),
            "anthropic" => obj["content"][0]["text"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| JsValue::from_str("No content in response")),
            _ => Err(JsValue::from_str("Unknown provider")),
        }
    }
}

/// Message in conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmMessage {
    #[wasm_bindgen(skip)]
    pub role: String,
    #[wasm_bindgen(skip)]
    pub content: String,
}

#[wasm_bindgen]
impl WasmMessage {
    #[wasm_bindgen(constructor)]
    pub fn new(role: String, content: String) -> Self {
        Self { role, content }
    }

    #[wasm_bindgen(getter)]
    pub fn role(&self) -> String {
        self.role.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn content(&self) -> String {
        self.content.clone()
    }

    pub fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
        }
    }
}

/// Response from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmResponse {
    pub content: String,
    pub tool_calls: Option<Vec<WasmToolCall>>,
    pub usage: Option<WasmUsage>,
}

/// Tool call in response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Token usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_config_creation() {
        let config = WasmAgentConfig::new(
            "openai".to_string(),
            "test-key".to_string(),
            "gpt-4".to_string(),
        );

        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.temperature, 0.7);
    }

    #[wasm_bindgen_test]
    fn test_agent_creation() {
        let config = WasmAgentConfig::new(
            "openai".to_string(),
            "test-key".to_string(),
            "gpt-4".to_string(),
        );

        let agent = WasmAgent::new(config);
        assert!(agent.messages.is_empty());
    }

    #[wasm_bindgen_test]
    fn test_message_creation() {
        let msg = WasmMessage::user("Hello".to_string());
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");

        let msg = WasmMessage::assistant("Hi there!".to_string());
        assert_eq!(msg.role, "assistant");
    }
}
