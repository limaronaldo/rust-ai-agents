//! Request building for LLM APIs.

use crate::{LlmClientError, Message, Provider, Result};
use serde::Serialize;

/// An HTTP request ready to be sent.
///
/// This struct contains all the information needed to make an HTTP request,
/// but does NOT include any HTTP client implementation.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP method (always POST for LLM APIs)
    pub method: &'static str,
    /// Full URL
    pub url: String,
    /// Headers as key-value pairs
    pub headers: Vec<(String, String)>,
    /// JSON body as string
    pub body: String,
}

/// Builder for constructing LLM API requests.
#[derive(Debug, Clone)]
pub struct RequestBuilder {
    provider: Provider,
    model: Option<String>,
    messages: Vec<Message>,
    api_key: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stream: bool,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
}

impl RequestBuilder {
    /// Create a new request builder for the given provider.
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            model: None,
            messages: Vec::new(),
            api_key: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            top_p: None,
            stop: None,
        }
    }

    /// Set the model to use.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the messages for the conversation.
    pub fn messages(mut self, messages: &[Message]) -> Self {
        self.messages = messages.to_vec();
        self
    }

    /// Add a single message to the conversation.
    pub fn add_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Set the API key.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the temperature (0.0 - 2.0).
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp.clamp(0.0, 2.0));
        self
    }

    /// Set the maximum tokens to generate.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Enable or disable streaming.
    pub fn stream(mut self, enable: bool) -> Self {
        self.stream = enable;
        self
    }

    /// Set top_p (nucleus sampling).
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p.clamp(0.0, 1.0));
        self
    }

    /// Set stop sequences.
    pub fn stop(mut self, sequences: Vec<String>) -> Self {
        self.stop = Some(sequences);
        self
    }

    /// Build the HTTP request.
    pub fn build(&self) -> Result<HttpRequest> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| LlmClientError::missing("model"))?;
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| LlmClientError::missing("api_key"))?;

        if self.messages.is_empty() {
            return Err(LlmClientError::missing("messages"));
        }

        let url = self.provider.endpoint().to_string();
        let headers = self.build_headers(api_key);
        let body = self.build_body(model)?;

        Ok(HttpRequest {
            method: "POST",
            url,
            headers,
            body,
        })
    }

    fn build_headers(&self, api_key: &str) -> Vec<(String, String)> {
        let mut headers = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            (
                self.provider.auth_header().to_string(),
                self.provider.format_auth(api_key),
            ),
        ];

        for (key, value) in self.provider.extra_headers() {
            headers.push((key.to_string(), value.to_string()));
        }

        headers
    }

    fn build_body(&self, model: &str) -> Result<String> {
        match self.provider {
            Provider::OpenAI | Provider::OpenRouter => self.build_openai_body(model),
            Provider::Anthropic => self.build_anthropic_body(model),
        }
    }

    fn build_openai_body(&self, model: &str) -> Result<String> {
        #[derive(Serialize)]
        struct OpenAIRequest<'a> {
            model: &'a str,
            messages: &'a [OpenAIMessage<'a>],
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            max_tokens: Option<u32>,
            stream: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            top_p: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            stop: Option<&'a [String]>,
        }

        #[derive(Serialize)]
        struct OpenAIMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        let messages: Vec<OpenAIMessage> = self
            .messages
            .iter()
            .map(|m| OpenAIMessage {
                role: m.role.as_str(),
                content: &m.content,
            })
            .collect();

        let request = OpenAIRequest {
            model,
            messages: &messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: self.stream,
            top_p: self.top_p,
            stop: self.stop.as_deref(),
        };

        Ok(serde_json::to_string(&request)?)
    }

    fn build_anthropic_body(&self, model: &str) -> Result<String> {
        #[derive(Serialize)]
        struct AnthropicRequest<'a> {
            model: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            system: Option<&'a str>,
            messages: Vec<AnthropicMessage<'a>>,
            max_tokens: u32,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
            stream: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            top_p: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            stop_sequences: Option<&'a [String]>,
        }

        #[derive(Serialize)]
        struct AnthropicMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        // Extract system message (Anthropic handles it separately)
        let system = self
            .messages
            .iter()
            .find(|m| m.role == crate::Role::System)
            .map(|m| m.content.as_str());

        // Filter out system messages for the messages array
        let messages: Vec<AnthropicMessage> = self
            .messages
            .iter()
            .filter(|m| m.role != crate::Role::System)
            .map(|m| AnthropicMessage {
                role: if m.role == crate::Role::User {
                    "user"
                } else {
                    "assistant"
                },
                content: &m.content,
            })
            .collect();

        let request = AnthropicRequest {
            model,
            system,
            messages,
            max_tokens: self.max_tokens.unwrap_or(4096),
            temperature: self.temperature,
            stream: self.stream,
            top_p: self.top_p,
            stop_sequences: self.stop.as_deref(),
        };

        Ok(serde_json::to_string(&request)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_request() {
        let request = RequestBuilder::new(Provider::OpenAI)
            .model("gpt-4o-mini")
            .api_key("sk-test")
            .add_message(Message::system("You are helpful"))
            .add_message(Message::user("Hello"))
            .temperature(0.7)
            .max_tokens(1024)
            .build()
            .unwrap();

        assert_eq!(request.method, "POST");
        assert!(request.url.contains("openai.com"));
        assert!(request.body.contains("gpt-4o-mini"));
        assert!(request.body.contains("Hello"));

        // Check headers
        let auth_header = request.headers.iter().find(|(k, _)| k == "Authorization");
        assert!(auth_header.is_some());
        assert!(auth_header.unwrap().1.starts_with("Bearer "));
    }

    #[test]
    fn test_anthropic_request() {
        let request = RequestBuilder::new(Provider::Anthropic)
            .model("claude-3-sonnet-20240229")
            .api_key("sk-ant-test")
            .add_message(Message::system("You are helpful"))
            .add_message(Message::user("Hello"))
            .max_tokens(1024)
            .build()
            .unwrap();

        assert!(request.url.contains("anthropic.com"));
        assert!(request.body.contains("claude-3"));
        // Anthropic puts system message separately
        assert!(request.body.contains(r#""system":"You are helpful"#));

        // Check anthropic-version header
        let version_header = request
            .headers
            .iter()
            .find(|(k, _)| k == "anthropic-version");
        assert!(version_header.is_some());
    }

    #[test]
    fn test_missing_model() {
        let result = RequestBuilder::new(Provider::OpenAI)
            .api_key("sk-test")
            .add_message(Message::user("Hello"))
            .build();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model"));
    }

    #[test]
    fn test_missing_messages() {
        let result = RequestBuilder::new(Provider::OpenAI)
            .model("gpt-4")
            .api_key("sk-test")
            .build();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("messages"));
    }

    #[test]
    fn test_streaming() {
        let request = RequestBuilder::new(Provider::OpenAI)
            .model("gpt-4")
            .api_key("sk-test")
            .add_message(Message::user("Hello"))
            .stream(true)
            .build()
            .unwrap();

        assert!(request.body.contains(r#""stream":true"#));
    }
}
