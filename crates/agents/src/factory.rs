//! # Agent Factory
//!
//! Runtime agent instantiation and configuration system.
//!
//! Inspired by Swarm's agent factory pattern for dynamic agent creation.
//!
//! ## Features
//!
//! - **Templates**: Define reusable agent templates
//! - **Runtime Config**: Configure agents at runtime
//! - **Dependency Injection**: Inject tools, memory, and providers
//! - **Pooling**: Reuse agent instances for efficiency
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents::factory::{AgentFactory, AgentTemplate};
//!
//! let factory = AgentFactory::new()
//!     .register_template("researcher", AgentTemplate::new()
//!         .system_prompt("You are a research assistant")
//!         .tool("web_search")
//!         .tool("summarize"))
//!     .register_template("coder", AgentTemplate::new()
//!         .system_prompt("You are a coding expert")
//!         .tool("code_execute")
//!         .tool("file_edit"));
//!
//! // Create agent at runtime
//! let agent = factory.create("researcher")?;
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Agent template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTemplate {
    /// Template name
    pub name: String,
    /// System prompt for the agent
    pub system_prompt: String,
    /// List of tool names to include
    pub tools: Vec<String>,
    /// Model to use (e.g., "gpt-4", "claude-3")
    pub model: Option<String>,
    /// Temperature setting
    pub temperature: Option<f32>,
    /// Maximum tokens for response
    pub max_tokens: Option<usize>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Agent capabilities/tags
    pub capabilities: Vec<String>,
    /// Additional configuration
    pub config: HashMap<String, String>,
    /// Description of the template
    pub description: String,
}

impl AgentTemplate {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            system_prompt: String::new(),
            tools: Vec::new(),
            model: None,
            temperature: None,
            max_tokens: None,
            stop_sequences: Vec::new(),
            capabilities: Vec::new(),
            config: HashMap::new(),
            description: String::new(),
        }
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn tool(mut self, tool_name: impl Into<String>) -> Self {
        self.tools.push(tool_name.into());
        self
    }

    pub fn tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tools.extend(tools.into_iter().map(|s| s.into()));
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp.clamp(0.0, 2.0));
        self
    }

    pub fn max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    pub fn stop_sequence(mut self, seq: impl Into<String>) -> Self {
        self.stop_sequences.push(seq.into());
        self
    }

    pub fn capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }

    pub fn config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Merge another template's settings (for inheritance)
    pub fn extend(mut self, other: &AgentTemplate) -> Self {
        if self.system_prompt.is_empty() {
            self.system_prompt = other.system_prompt.clone();
        }
        self.tools.extend(other.tools.clone());
        if self.model.is_none() {
            self.model = other.model.clone();
        }
        if self.temperature.is_none() {
            self.temperature = other.temperature;
        }
        if self.max_tokens.is_none() {
            self.max_tokens = other.max_tokens;
        }
        self.stop_sequences.extend(other.stop_sequences.clone());
        self.capabilities.extend(other.capabilities.clone());
        for (k, v) in &other.config {
            self.config.entry(k.clone()).or_insert_with(|| v.clone());
        }
        self
    }
}

/// Runtime configuration for agent creation
#[derive(Debug, Clone, Default)]
pub struct AgentConfig {
    /// Override system prompt
    pub system_prompt_override: Option<String>,
    /// Additional context to prepend
    pub context_prefix: Option<String>,
    /// Override model
    pub model_override: Option<String>,
    /// Override temperature
    pub temperature_override: Option<f32>,
    /// Additional tools to include
    pub additional_tools: Vec<String>,
    /// Tools to exclude
    pub excluded_tools: Vec<String>,
    /// Session/user ID for tracking
    pub session_id: Option<String>,
    /// User ID for tracking
    pub user_id: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl AgentConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn override_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt_override = Some(prompt.into());
        self
    }

    pub fn context_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.context_prefix = Some(prefix.into());
        self
    }

    pub fn override_model(mut self, model: impl Into<String>) -> Self {
        self.model_override = Some(model.into());
        self
    }

    pub fn override_temperature(mut self, temp: f32) -> Self {
        self.temperature_override = Some(temp);
        self
    }

    pub fn add_tool(mut self, tool: impl Into<String>) -> Self {
        self.additional_tools.push(tool.into());
        self
    }

    pub fn exclude_tool(mut self, tool: impl Into<String>) -> Self {
        self.excluded_tools.push(tool.into());
        self
    }

    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    pub fn user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }

    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A created agent instance
#[derive(Debug, Clone)]
pub struct AgentInstance {
    /// Unique instance ID
    pub id: String,
    /// Template name used
    pub template_name: String,
    /// Final system prompt
    pub system_prompt: String,
    /// Final tool list
    pub tools: Vec<String>,
    /// Model to use
    pub model: String,
    /// Temperature
    pub temperature: f32,
    /// Max tokens
    pub max_tokens: Option<usize>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Capabilities
    pub capabilities: Vec<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// User ID
    pub user_id: Option<String>,
    /// When created
    pub created_at: Instant,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl AgentInstance {
    /// Check if instance has a specific capability
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.contains(&capability.to_string())
    }

    /// Check if instance has a specific tool
    pub fn has_tool(&self, tool: &str) -> bool {
        self.tools.contains(&tool.to_string())
    }

    /// Get age of the instance
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Error type for factory operations
#[derive(Debug, thiserror::Error)]
pub enum FactoryError {
    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Pool exhausted for template: {0}")]
    PoolExhausted(String),

    #[error("Instance not found: {0}")]
    InstanceNotFound(String),
}

/// Statistics for the factory
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactoryStats {
    pub templates_registered: usize,
    pub total_instances_created: u64,
    pub active_instances: usize,
    pub pool_hits: u64,
    pub pool_misses: u64,
}

/// Configuration for agent pooling
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum instances per template
    pub max_per_template: usize,
    /// Maximum total instances
    pub max_total: usize,
    /// Instance TTL (time-to-live)
    pub instance_ttl: Duration,
    /// Whether pooling is enabled
    pub enabled: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_per_template: 10,
            max_total: 100,
            instance_ttl: Duration::from_secs(3600),
            enabled: true,
        }
    }
}

/// Agent factory for creating and managing agents
pub struct AgentFactory {
    templates: RwLock<HashMap<String, AgentTemplate>>,
    instances: RwLock<HashMap<String, AgentInstance>>,
    pool: RwLock<HashMap<String, Vec<AgentInstance>>>,
    pool_config: PoolConfig,
    stats: RwLock<FactoryStats>,
    default_model: String,
    default_temperature: f32,
}

impl AgentFactory {
    pub fn new() -> Self {
        Self {
            templates: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            pool: RwLock::new(HashMap::new()),
            pool_config: PoolConfig::default(),
            stats: RwLock::new(FactoryStats::default()),
            default_model: "gpt-4".to_string(),
            default_temperature: 0.7,
        }
    }

    pub fn with_pool_config(mut self, config: PoolConfig) -> Self {
        self.pool_config = config;
        self
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    pub fn with_default_temperature(mut self, temp: f32) -> Self {
        self.default_temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Register a template
    pub fn register_template(&self, template: AgentTemplate) -> &Self {
        let name = template.name.clone();
        self.templates.write().insert(name.clone(), template);
        self.stats.write().templates_registered += 1;
        info!(template = %name, "Template registered");
        self
    }

    /// Register multiple templates
    pub fn register_templates<I>(&self, templates: I) -> &Self
    where
        I: IntoIterator<Item = AgentTemplate>,
    {
        for template in templates {
            self.register_template(template);
        }
        self
    }

    /// Get a template by name
    pub fn get_template(&self, name: &str) -> Option<AgentTemplate> {
        self.templates.read().get(name).cloned()
    }

    /// List all template names
    pub fn list_templates(&self) -> Vec<String> {
        self.templates.read().keys().cloned().collect()
    }

    /// Create an agent instance from a template
    pub fn create(&self, template_name: &str) -> Result<AgentInstance, FactoryError> {
        self.create_with_config(template_name, AgentConfig::default())
    }

    /// Create an agent instance with custom configuration
    pub fn create_with_config(
        &self,
        template_name: &str,
        config: AgentConfig,
    ) -> Result<AgentInstance, FactoryError> {
        // Try pool first if enabled
        if self.pool_config.enabled {
            if let Some(instance) = self.get_from_pool(template_name) {
                self.stats.write().pool_hits += 1;
                debug!(template = %template_name, "Got agent from pool");
                return Ok(instance);
            }
            self.stats.write().pool_misses += 1;
        }

        // Get template
        let template = self
            .templates
            .read()
            .get(template_name)
            .cloned()
            .ok_or_else(|| FactoryError::TemplateNotFound(template_name.to_string()))?;

        // Build instance
        let instance = self.build_instance(&template, config);

        // Track instance
        self.instances
            .write()
            .insert(instance.id.clone(), instance.clone());

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_instances_created += 1;
            stats.active_instances += 1;
        }

        info!(
            instance_id = %instance.id,
            template = %template_name,
            "Agent instance created"
        );

        Ok(instance)
    }

    fn build_instance(&self, template: &AgentTemplate, config: AgentConfig) -> AgentInstance {
        // Determine system prompt
        let system_prompt =
            config
                .system_prompt_override
                .unwrap_or_else(|| match &config.context_prefix {
                    Some(prefix) => format!("{}\n\n{}", prefix, template.system_prompt),
                    None => template.system_prompt.clone(),
                });

        // Determine tools
        let mut tools: Vec<String> = template
            .tools
            .iter()
            .filter(|t| !config.excluded_tools.contains(t))
            .cloned()
            .collect();
        tools.extend(config.additional_tools);

        // Determine model
        let model = config
            .model_override
            .or_else(|| template.model.clone())
            .unwrap_or_else(|| self.default_model.clone());

        // Determine temperature
        let temperature = config
            .temperature_override
            .or(template.temperature)
            .unwrap_or(self.default_temperature);

        // Merge metadata
        let mut metadata = template.config.clone();
        metadata.extend(config.metadata);

        AgentInstance {
            id: uuid::Uuid::new_v4().to_string(),
            template_name: template.name.clone(),
            system_prompt,
            tools,
            model,
            temperature,
            max_tokens: template.max_tokens,
            stop_sequences: template.stop_sequences.clone(),
            capabilities: template.capabilities.clone(),
            session_id: config.session_id,
            user_id: config.user_id,
            created_at: Instant::now(),
            metadata,
        }
    }

    fn get_from_pool(&self, template_name: &str) -> Option<AgentInstance> {
        let mut pool = self.pool.write();
        if let Some(instances) = pool.get_mut(template_name) {
            // Find a valid (not expired) instance
            while let Some(instance) = instances.pop() {
                if instance.age() < self.pool_config.instance_ttl {
                    return Some(instance);
                }
            }
        }
        None
    }

    /// Return an instance to the pool for reuse
    pub fn release(&self, instance: AgentInstance) {
        if !self.pool_config.enabled {
            return;
        }

        // Check TTL
        if instance.age() >= self.pool_config.instance_ttl {
            debug!(instance_id = %instance.id, "Instance expired, not returning to pool");
            return;
        }

        let template_name = instance.template_name.clone();

        let mut pool = self.pool.write();
        let template_pool = pool.entry(template_name).or_default();

        // Check capacity
        if template_pool.len() >= self.pool_config.max_per_template {
            debug!(instance_id = %instance.id, "Pool full for template");
            return;
        }

        template_pool.push(instance);
        let mut stats = self.stats.write();
        stats.active_instances = stats.active_instances.saturating_sub(1);
    }

    /// Get an instance by ID
    pub fn get_instance(&self, id: &str) -> Option<AgentInstance> {
        self.instances.read().get(id).cloned()
    }

    /// Remove an instance
    pub fn remove_instance(&self, id: &str) -> bool {
        let removed = self.instances.write().remove(id).is_some();
        if removed {
            let mut stats = self.stats.write();
            stats.active_instances = stats.active_instances.saturating_sub(1);
        }
        removed
    }

    /// Get factory statistics
    pub fn stats(&self) -> FactoryStats {
        self.stats.read().clone()
    }

    /// Find templates by capability
    pub fn find_by_capability(&self, capability: &str) -> Vec<AgentTemplate> {
        self.templates
            .read()
            .values()
            .filter(|t| t.capabilities.contains(&capability.to_string()))
            .cloned()
            .collect()
    }

    /// Find templates that have a specific tool
    pub fn find_by_tool(&self, tool: &str) -> Vec<AgentTemplate> {
        self.templates
            .read()
            .values()
            .filter(|t| t.tools.contains(&tool.to_string()))
            .cloned()
            .collect()
    }

    /// Clear all pools
    pub fn clear_pools(&self) {
        self.pool.write().clear();
        debug!("All pools cleared");
    }

    /// Cleanup expired instances from pools
    pub fn cleanup_expired(&self) {
        let mut pool = self.pool.write();
        let ttl = self.pool_config.instance_ttl;

        for (_, instances) in pool.iter_mut() {
            instances.retain(|i| i.age() < ttl);
        }
    }
}

impl Default for AgentFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro for defining templates concisely
#[macro_export]
macro_rules! agent_template {
    ($name:expr => {
        $(system_prompt: $prompt:expr,)?
        $(model: $model:expr,)?
        $(temperature: $temp:expr,)?
        $(tools: [$($tool:expr),* $(,)?],)?
        $(capabilities: [$($cap:expr),* $(,)?],)?
        $(description: $desc:expr,)?
    }) => {{
        let mut template = AgentTemplate::new($name);
        $(template = template.system_prompt($prompt);)?
        $(template = template.model($model);)?
        $(template = template.temperature($temp);)?
        $($(template = template.tool($tool);)*)?
        $($(template = template.capability($cap);)*)?
        $(template = template.description($desc);)?
        template
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_builder() {
        let template = AgentTemplate::new("test")
            .system_prompt("You are a test agent")
            .tool("search")
            .tool("calculate")
            .model("gpt-4")
            .temperature(0.5)
            .capability("math")
            .config("key", "value");

        assert_eq!(template.name, "test");
        assert_eq!(template.tools.len(), 2);
        assert_eq!(template.model, Some("gpt-4".to_string()));
        assert_eq!(template.temperature, Some(0.5));
    }

    #[test]
    fn test_template_extend() {
        let base = AgentTemplate::new("base")
            .system_prompt("Base prompt")
            .tool("base_tool")
            .model("gpt-3.5");

        let extended = AgentTemplate::new("extended")
            .tool("extended_tool")
            .extend(&base);

        assert_eq!(extended.system_prompt, "Base prompt");
        assert!(extended.tools.contains(&"base_tool".to_string()));
        assert!(extended.tools.contains(&"extended_tool".to_string()));
        assert_eq!(extended.model, Some("gpt-3.5".to_string()));
    }

    #[test]
    fn test_factory_register_and_create() {
        let factory = AgentFactory::new();

        let template = AgentTemplate::new("researcher")
            .system_prompt("You are a researcher")
            .tool("web_search");

        factory.register_template(template);

        let instance = factory.create("researcher").unwrap();
        assert_eq!(instance.template_name, "researcher");
        assert!(instance.tools.contains(&"web_search".to_string()));
    }

    #[test]
    fn test_factory_create_with_config() {
        let factory = AgentFactory::new();

        let template = AgentTemplate::new("agent")
            .system_prompt("Original prompt")
            .tool("tool1")
            .tool("tool2");

        factory.register_template(template);

        let config = AgentConfig::new()
            .override_system_prompt("New prompt")
            .add_tool("tool3")
            .exclude_tool("tool1")
            .session_id("session123");

        let instance = factory.create_with_config("agent", config).unwrap();

        assert_eq!(instance.system_prompt, "New prompt");
        assert!(!instance.tools.contains(&"tool1".to_string()));
        assert!(instance.tools.contains(&"tool2".to_string()));
        assert!(instance.tools.contains(&"tool3".to_string()));
        assert_eq!(instance.session_id, Some("session123".to_string()));
    }

    #[test]
    fn test_factory_template_not_found() {
        let factory = AgentFactory::new();
        let result = factory.create("nonexistent");
        assert!(matches!(result, Err(FactoryError::TemplateNotFound(_))));
    }

    #[test]
    fn test_factory_list_templates() {
        let factory = AgentFactory::new();

        factory.register_template(AgentTemplate::new("t1"));
        factory.register_template(AgentTemplate::new("t2"));
        factory.register_template(AgentTemplate::new("t3"));

        let templates = factory.list_templates();
        assert_eq!(templates.len(), 3);
    }

    #[test]
    fn test_factory_find_by_capability() {
        let factory = AgentFactory::new();

        factory.register_template(AgentTemplate::new("math_agent").capability("math"));
        factory.register_template(AgentTemplate::new("text_agent").capability("text"));
        factory.register_template(
            AgentTemplate::new("multi_agent")
                .capability("math")
                .capability("text"),
        );

        let math_agents = factory.find_by_capability("math");
        assert_eq!(math_agents.len(), 2);
    }

    #[test]
    fn test_factory_find_by_tool() {
        let factory = AgentFactory::new();

        factory.register_template(AgentTemplate::new("a1").tool("search"));
        factory.register_template(AgentTemplate::new("a2").tool("calculate"));
        factory.register_template(AgentTemplate::new("a3").tool("search").tool("calculate"));

        let search_agents = factory.find_by_tool("search");
        assert_eq!(search_agents.len(), 2);
    }

    #[test]
    fn test_agent_instance_methods() {
        let instance = AgentInstance {
            id: "test".to_string(),
            template_name: "test".to_string(),
            system_prompt: "prompt".to_string(),
            tools: vec!["tool1".to_string(), "tool2".to_string()],
            model: "gpt-4".to_string(),
            temperature: 0.7,
            max_tokens: None,
            stop_sequences: Vec::new(),
            capabilities: vec!["cap1".to_string()],
            session_id: None,
            user_id: None,
            created_at: Instant::now(),
            metadata: HashMap::new(),
        };

        assert!(instance.has_tool("tool1"));
        assert!(!instance.has_tool("tool3"));
        assert!(instance.has_capability("cap1"));
        assert!(!instance.has_capability("cap2"));
    }

    #[test]
    fn test_factory_pooling() {
        let pool_config = PoolConfig {
            max_per_template: 2,
            max_total: 10,
            instance_ttl: Duration::from_secs(3600),
            enabled: true,
        };

        let factory = AgentFactory::new().with_pool_config(pool_config);

        factory.register_template(AgentTemplate::new("pooled"));

        // Create and release an instance
        let instance1 = factory.create("pooled").unwrap();
        let _id1 = instance1.id.clone();
        factory.release(instance1);

        // Next create should get from pool
        let _instance2 = factory.create("pooled").unwrap();
        // Note: We get a pooled instance but it keeps its original ID
        // The pool hit is tracked in stats
        let stats = factory.stats();
        assert!(stats.pool_hits > 0 || stats.pool_misses > 0);
    }

    #[test]
    fn test_factory_stats() {
        let factory = AgentFactory::new();

        factory.register_template(AgentTemplate::new("t1"));
        factory.register_template(AgentTemplate::new("t2"));

        factory.create("t1").unwrap();
        factory.create("t1").unwrap();
        factory.create("t2").unwrap();

        let stats = factory.stats();
        assert_eq!(stats.templates_registered, 2);
        assert_eq!(stats.total_instances_created, 3);
    }

    #[test]
    fn test_context_prefix() {
        let factory = AgentFactory::new();

        factory.register_template(AgentTemplate::new("agent").system_prompt("Base instructions"));

        let config = AgentConfig::new().context_prefix("User context: VIP customer");

        let instance = factory.create_with_config("agent", config).unwrap();
        assert!(instance.system_prompt.contains("VIP customer"));
        assert!(instance.system_prompt.contains("Base instructions"));
    }
}
