//! Configuration for Cloudflare Workers agent

use rust_ai_agents_llm_client::Provider;

/// Configuration for the Cloudflare agent
#[derive(Debug, Clone)]
pub struct CloudflareConfig {
    /// LLM provider (OpenAI, Anthropic, OpenRouter)
    pub provider: Provider,

    /// API key for the provider
    pub api_key: String,

    /// Model identifier (e.g., "gpt-4", "claude-3-5-sonnet-20241022")
    pub model: String,

    /// System prompt for the agent
    pub system_prompt: Option<String>,

    /// Temperature for response generation (0.0 - 2.0)
    pub temperature: f32,

    /// Maximum tokens in response
    pub max_tokens: Option<u32>,

    /// Whether to use streaming responses
    pub stream: bool,
}

impl Default for CloudflareConfig {
    fn default() -> Self {
        Self {
            provider: Provider::OpenAI,
            api_key: String::new(),
            model: "gpt-4".to_string(),
            system_prompt: None,
            temperature: 0.7,
            max_tokens: None,
            stream: false,
        }
    }
}

impl CloudflareConfig {
    /// Create a new configuration builder
    pub fn builder() -> CloudflareConfigBuilder {
        CloudflareConfigBuilder::default()
    }
}

/// Builder for CloudflareConfig
#[derive(Debug, Default)]
pub struct CloudflareConfigBuilder {
    provider: Option<Provider>,
    api_key: Option<String>,
    model: Option<String>,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
}

impl CloudflareConfigBuilder {
    /// Set the LLM provider
    pub fn provider(mut self, provider: Provider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the API key
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the model identifier
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set maximum tokens
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Enable streaming responses
    pub fn stream(mut self, enable: bool) -> Self {
        self.stream = Some(enable);
        self
    }

    /// Build the configuration
    pub fn build(self) -> CloudflareConfig {
        CloudflareConfig {
            provider: self.provider.unwrap_or(Provider::OpenAI),
            api_key: self.api_key.unwrap_or_default(),
            model: self.model.unwrap_or_else(|| "gpt-4".to_string()),
            system_prompt: self.system_prompt,
            temperature: self.temperature.unwrap_or(0.7),
            max_tokens: self.max_tokens,
            stream: self.stream.unwrap_or(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = CloudflareConfig::builder()
            .provider(Provider::Anthropic)
            .api_key("test-key")
            .model("claude-3")
            .temperature(0.5)
            .build();

        assert!(matches!(config.provider, Provider::Anthropic));
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model, "claude-3");
        assert_eq!(config.temperature, 0.5);
    }

    #[test]
    fn test_config_defaults() {
        let config = CloudflareConfig::default();

        assert!(matches!(config.provider, Provider::OpenAI));
        assert_eq!(config.temperature, 0.7);
        assert!(!config.stream);
    }
}
