//! Cloudflare Workers AI Agent

use crate::config::CloudflareConfig;
use crate::error::{CloudflareError, Result};
use crate::http::CloudflareHttpClient;
use crate::kv::{KvStore, SessionMetadata};
use rust_ai_agents_llm_client::{Message, RequestBuilder, Role};

/// AI Agent for Cloudflare Workers
///
/// Provides chat functionality with optional KV-based conversation persistence.
pub struct CloudflareAgent {
    config: CloudflareConfig,
    messages: Vec<Message>,
    kv: Option<KvStore>,
}

impl CloudflareAgent {
    /// Create a new Cloudflare agent with the given configuration
    pub fn new(config: CloudflareConfig) -> Self {
        let mut messages = Vec::new();

        // Add system prompt if configured
        if let Some(ref prompt) = config.system_prompt {
            messages.push(Message::system(prompt));
        }

        Self {
            config,
            messages,
            kv: None,
        }
    }

    /// Attach a KV store for conversation persistence
    pub fn with_kv(mut self, kv: KvStore) -> Self {
        self.kv = Some(kv);
        self
    }

    /// Send a message and get a response (non-streaming)
    pub async fn chat(&mut self, message: &str) -> Result<AgentResponse> {
        // Add user message
        self.messages.push(Message::user(message));

        // Build and send request
        let request = self.build_request(false)?;
        let response = CloudflareHttpClient::execute(request).await?;

        // Add assistant response to history
        self.messages.push(Message::assistant(&response.content));

        Ok(AgentResponse {
            content: response.content,
            tool_calls: if response.tool_calls.is_empty() {
                None
            } else {
                Some(response.tool_calls)
            },
            usage: response.usage.map(|u| AgentUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        })
    }

    /// Send a message with session persistence
    pub async fn chat_with_session(
        &mut self,
        session_id: &str,
        message: &str,
    ) -> Result<AgentResponse> {
        // Check KV is configured
        if self.kv.is_none() {
            return Err(CloudflareError::ConfigError(
                "KV store not configured".to_string(),
            ));
        }

        // Load existing conversation
        let history = self
            .kv
            .as_ref()
            .unwrap()
            .get_conversation(session_id)
            .await?;

        if !history.is_empty() {
            // Keep system prompt if exists, then add history
            let system_prompt = self.messages.first().cloned();
            self.messages.clear();
            if let Some(prompt) = system_prompt {
                if matches!(prompt.role, Role::System) {
                    self.messages.push(prompt);
                }
            }
            self.messages.extend(history);
        }

        // Chat normally
        let response = self.chat(message).await?;

        // Save updated conversation
        let kv = self.kv.as_ref().unwrap();
        kv.save_conversation(session_id, &self.messages).await?;

        // Update metadata
        let mut metadata = kv
            .get_metadata(session_id)
            .await?
            .unwrap_or_else(|| SessionMetadata::new(session_id));

        if let Some(usage) = &response.usage {
            metadata.add_tokens(usage.total_tokens);
        }
        metadata.increment_messages();
        kv.save_metadata(session_id, &metadata).await?;

        Ok(response)
    }

    /// Send a message and get a streaming response
    pub async fn chat_stream(&mut self, message: &str) -> Result<worker::Response> {
        // Add user message
        self.messages.push(Message::user(message));

        // Build and send streaming request
        let request = self.build_request(true)?;
        let response = CloudflareHttpClient::execute_stream(request).await?;

        Ok(response)
    }

    /// Get current conversation history
    pub fn history(&self) -> &[Message] {
        &self.messages
    }

    /// Clear conversation history (keeps system prompt)
    pub fn clear_history(&mut self) {
        let system_prompt = self.messages.first().cloned();
        self.messages.clear();
        if let Some(prompt) = system_prompt {
            if matches!(prompt.role, Role::System) {
                self.messages.push(prompt);
            }
        }
    }

    /// Build an HTTP request for the LLM API
    fn build_request(&self, stream: bool) -> Result<rust_ai_agents_llm_client::HttpRequest> {
        let mut builder = RequestBuilder::new(self.config.provider)
            .api_key(&self.config.api_key)
            .model(&self.config.model)
            .temperature(self.config.temperature)
            .stream(stream)
            .messages(&self.messages);

        if let Some(max_tokens) = self.config.max_tokens {
            builder = builder.max_tokens(max_tokens);
        }

        builder
            .build()
            .map_err(|e| CloudflareError::ConfigError(e.to_string()))
    }
}

/// Response from the agent
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// Generated content
    pub content: String,
    /// Tool calls (if any)
    pub tool_calls: Option<Vec<rust_ai_agents_llm_client::ToolCall>>,
    /// Token usage
    pub usage: Option<AgentUsage>,
}

/// Token usage information
#[derive(Debug, Clone)]
pub struct AgentUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_ai_agents_llm_client::Provider;

    #[test]
    fn test_agent_creation() {
        let config = CloudflareConfig::builder()
            .provider(Provider::Anthropic)
            .api_key("test-key")
            .model("claude-3")
            .system_prompt("You are helpful")
            .build();

        let agent = CloudflareAgent::new(config);

        assert_eq!(agent.messages.len(), 1);
        assert!(matches!(agent.messages[0].role, Role::System));
        assert_eq!(agent.messages[0].content, "You are helpful");
    }

    #[test]
    fn test_clear_history() {
        let config = CloudflareConfig::builder()
            .provider(Provider::OpenAI)
            .api_key("test")
            .system_prompt("System")
            .build();

        let mut agent = CloudflareAgent::new(config);

        // Simulate adding messages
        agent.messages.push(Message::user("Hello"));
        agent.messages.push(Message::assistant("Hi"));

        assert_eq!(agent.messages.len(), 3);

        agent.clear_history();

        // Should only have system prompt
        assert_eq!(agent.messages.len(), 1);
        assert!(matches!(agent.messages[0].role, Role::System));
    }

    #[test]
    fn test_agent_without_system_prompt() {
        let config = CloudflareConfig::builder()
            .provider(Provider::OpenAI)
            .api_key("test")
            .build();

        let agent = CloudflareAgent::new(config);
        assert!(agent.messages.is_empty());
    }
}
