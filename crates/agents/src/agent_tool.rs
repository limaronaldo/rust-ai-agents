//! # Agent-as-Tool
//!
//! Compose agents as tools, allowing agents to invoke other agents.
//!
//! Inspired by AutoGen's agent composition pattern.
//!
//! ## Features
//!
//! - **Agent Wrapping**: Wrap any agent as a callable tool
//! - **Typed I/O**: Strong typing for agent input/output
//! - **Delegation**: Hierarchical agent organization
//! - **Context Passing**: Share context between agents
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents::agent_tool::{AgentTool, AgentToolBuilder};
//!
//! // Create a specialized agent
//! let math_agent = AgentTool::builder("math_expert")
//!     .description("Expert at solving mathematical problems")
//!     .handler(|input| async {
//!         // Process math query
//!         Ok(format!("Result: {}", solve_math(&input)))
//!     })
//!     .build();
//!
//! // Use it as a tool in another agent
//! let main_agent = Agent::new()
//!     .add_tool(math_agent.as_tool());
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Input to an agent tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolInput {
    /// The query or task for the agent
    pub query: String,
    /// Additional context
    pub context: HashMap<String, String>,
    /// Conversation history (if applicable)
    pub history: Vec<String>,
    /// Maximum tokens for response (hint to agent)
    pub max_tokens: Option<usize>,
}

impl AgentToolInput {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            context: HashMap::new(),
            history: Vec::new(),
            max_tokens: None,
        }
    }

    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    pub fn with_history(mut self, history: Vec<String>) -> Self {
        self.history = history;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl From<&str> for AgentToolInput {
    fn from(query: &str) -> Self {
        Self::new(query)
    }
}

impl From<String> for AgentToolInput {
    fn from(query: String) -> Self {
        Self::new(query)
    }
}

/// Output from an agent tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolOutput {
    /// The response content
    pub content: String,
    /// Whether the agent successfully completed the task
    pub success: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Additional metadata from the agent
    pub metadata: HashMap<String, String>,
    /// Time taken to process
    pub duration_ms: u64,
    /// Tools used by the agent
    pub tools_used: Vec<String>,
}

impl AgentToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            success: true,
            confidence: 1.0,
            metadata: HashMap::new(),
            duration_ms: 0,
            tools_used: Vec::new(),
        }
    }

    pub fn failure(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            success: false,
            confidence: 0.0,
            metadata: HashMap::new(),
            duration_ms: 0,
            tools_used: Vec::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn with_tools_used(mut self, tools: Vec<String>) -> Self {
        self.tools_used = tools;
        self
    }
}

/// Error from agent tool execution
#[derive(Debug, thiserror::Error)]
pub enum AgentToolError {
    #[error("Agent execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Agent timeout after {0:?}")]
    Timeout(Duration),

    #[error("Agent not available: {0}")]
    NotAvailable(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Rate limited: retry after {0:?}")]
    RateLimited(Duration),
}

/// Trait for agent tool handlers
#[async_trait]
pub trait AgentToolHandler: Send + Sync {
    /// Execute the agent with the given input
    async fn execute(&self, input: AgentToolInput) -> Result<AgentToolOutput, AgentToolError>;

    /// Get the agent's name
    fn name(&self) -> &str;

    /// Get the agent's description
    fn description(&self) -> &str;

    /// Check if the agent is available
    fn is_available(&self) -> bool {
        true
    }

    /// Get the agent's capabilities/tags
    fn capabilities(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Type alias for boxed handler
pub type BoxedHandler = Arc<dyn AgentToolHandler>;

/// Configuration for an agent tool
#[derive(Debug, Clone)]
pub struct AgentToolConfig {
    /// Timeout for agent execution
    pub timeout: Duration,
    /// Maximum retries on failure
    pub max_retries: u32,
    /// Whether to cache results
    pub cache_enabled: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
}

impl Default for AgentToolConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            max_retries: 2,
            cache_enabled: false,
            cache_ttl: Duration::from_secs(300),
        }
    }
}

/// Statistics for an agent tool
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentToolStats {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub timeouts: u64,
    pub total_duration_ms: u64,
    pub cache_hits: u64,
}

impl AgentToolStats {
    pub fn average_duration_ms(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.total_duration_ms as f64 / self.total_calls as f64
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.successful_calls as f64 / self.total_calls as f64
        }
    }
}

/// An agent wrapped as a tool
pub struct AgentTool {
    name: String,
    description: String,
    handler: BoxedHandler,
    config: AgentToolConfig,
    stats: Arc<RwLock<AgentToolStats>>,
    cache: Arc<RwLock<HashMap<String, (AgentToolOutput, Instant)>>>,
}

impl AgentTool {
    /// Create a new agent tool with a handler
    pub fn new<H: AgentToolHandler + 'static>(handler: H) -> Self {
        Self {
            name: handler.name().to_string(),
            description: handler.description().to_string(),
            handler: Arc::new(handler),
            config: AgentToolConfig::default(),
            stats: Arc::new(RwLock::new(AgentToolStats::default())),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a builder for fluent construction
    pub fn builder(name: impl Into<String>) -> AgentToolBuilder {
        AgentToolBuilder::new(name)
    }

    pub fn with_config(mut self, config: AgentToolConfig) -> Self {
        self.config = config;
        self
    }

    /// Execute the agent tool
    pub async fn call(
        &self,
        input: impl Into<AgentToolInput>,
    ) -> Result<AgentToolOutput, AgentToolError> {
        let input = input.into();
        let start = Instant::now();

        // Check cache
        if self.config.cache_enabled {
            let cache_key = self.cache_key(&input);
            if let Some(cached) = self.get_cached(&cache_key) {
                self.stats.write().cache_hits += 1;
                return Ok(cached);
            }
        }

        // Execute with timeout and retries
        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                debug!(name = %self.name, attempt, "Retrying agent tool");
            }

            match tokio::time::timeout(self.config.timeout, self.handler.execute(input.clone()))
                .await
            {
                Ok(Ok(mut output)) => {
                    let duration = start.elapsed();
                    output.duration_ms = duration.as_millis() as u64;

                    // Update stats
                    {
                        let mut stats = self.stats.write();
                        stats.total_calls += 1;
                        stats.successful_calls += 1;
                        stats.total_duration_ms += output.duration_ms;
                    }

                    // Cache result
                    if self.config.cache_enabled {
                        let cache_key = self.cache_key(&input);
                        self.set_cached(cache_key, output.clone());
                    }

                    return Ok(output);
                }
                Ok(Err(e)) => {
                    warn!(name = %self.name, error = %e, attempt, "Agent tool execution failed");
                    last_error = Some(e);
                }
                Err(_) => {
                    self.stats.write().timeouts += 1;
                    last_error = Some(AgentToolError::Timeout(self.config.timeout));
                }
            }
        }

        // All retries exhausted
        {
            let mut stats = self.stats.write();
            stats.total_calls += 1;
            stats.failed_calls += 1;
            stats.total_duration_ms += start.elapsed().as_millis() as u64;
        }

        Err(last_error
            .unwrap_or_else(|| AgentToolError::ExecutionFailed("Unknown error".to_string())))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn stats(&self) -> AgentToolStats {
        self.stats.read().clone()
    }

    pub fn is_available(&self) -> bool {
        self.handler.is_available()
    }

    pub fn capabilities(&self) -> Vec<String> {
        self.handler.capabilities()
    }

    fn cache_key(&self, input: &AgentToolInput) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        input.query.hash(&mut hasher);
        for (k, v) in &input.context {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        format!("{}:{}", self.name, hasher.finish())
    }

    fn get_cached(&self, key: &str) -> Option<AgentToolOutput> {
        let cache = self.cache.read();
        if let Some((output, cached_at)) = cache.get(key) {
            if cached_at.elapsed() < self.config.cache_ttl {
                return Some(output.clone());
            }
        }
        None
    }

    fn set_cached(&self, key: String, output: AgentToolOutput) {
        let mut cache = self.cache.write();
        cache.insert(key, (output, Instant::now()));

        // Simple cache cleanup - remove old entries
        let ttl = self.config.cache_ttl;
        cache.retain(|_, (_, cached_at)| cached_at.elapsed() < ttl);
    }
}

/// Builder for AgentTool
pub struct AgentToolBuilder {
    name: String,
    description: String,
    config: AgentToolConfig,
    capabilities: Vec<String>,
}

impl AgentToolBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            config: AgentToolConfig::default(),
            capabilities: Vec::new(),
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.config.max_retries = max_retries;
        self
    }

    pub fn cache(mut self, enabled: bool, ttl: Duration) -> Self {
        self.config.cache_enabled = enabled;
        self.config.cache_ttl = ttl;
        self
    }

    pub fn capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    /// Build with a closure handler
    pub fn handler<F, Fut>(self, handler: F) -> AgentTool
    where
        F: Fn(AgentToolInput) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<AgentToolOutput, AgentToolError>> + Send + 'static,
    {
        let fn_handler = FnAgentHandler {
            name: self.name.clone(),
            description: self.description.clone(),
            capabilities: self.capabilities.clone(),
            handler: Arc::new(move |input| Box::pin(handler(input))),
        };

        AgentTool::new(fn_handler).with_config(self.config)
    }
}

/// Handler implemented with a closure
struct FnAgentHandler {
    name: String,
    description: String,
    capabilities: Vec<String>,
    handler: Arc<
        dyn Fn(
                AgentToolInput,
            )
                -> Pin<Box<dyn Future<Output = Result<AgentToolOutput, AgentToolError>> + Send>>
            + Send
            + Sync,
    >,
}

#[async_trait]
impl AgentToolHandler for FnAgentHandler {
    async fn execute(&self, input: AgentToolInput) -> Result<AgentToolOutput, AgentToolError> {
        (self.handler)(input).await
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn capabilities(&self) -> Vec<String> {
        self.capabilities.clone()
    }
}

/// A registry for agent tools
pub struct AgentToolRegistry {
    tools: RwLock<HashMap<String, Arc<AgentTool>>>,
}

impl AgentToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Register an agent tool
    pub fn register(&self, tool: AgentTool) {
        let name = tool.name().to_string();
        self.tools.write().insert(name.clone(), Arc::new(tool));
        info!(name = %name, "Agent tool registered");
    }

    /// Get an agent tool by name
    pub fn get(&self, name: &str) -> Option<Arc<AgentTool>> {
        self.tools.read().get(name).cloned()
    }

    /// List all registered tools
    pub fn list(&self) -> Vec<String> {
        self.tools.read().keys().cloned().collect()
    }

    /// List available tools
    pub fn list_available(&self) -> Vec<String> {
        self.tools
            .read()
            .iter()
            .filter(|(_, tool)| tool.is_available())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Find tools by capability
    pub fn find_by_capability(&self, capability: &str) -> Vec<Arc<AgentTool>> {
        self.tools
            .read()
            .values()
            .filter(|tool| tool.capabilities().contains(&capability.to_string()))
            .cloned()
            .collect()
    }

    /// Remove a tool
    pub fn remove(&self, name: &str) -> bool {
        self.tools.write().remove(name).is_some()
    }

    /// Get stats for all tools
    pub fn all_stats(&self) -> HashMap<String, AgentToolStats> {
        self.tools
            .read()
            .iter()
            .map(|(name, tool)| (name.clone(), tool.stats()))
            .collect()
    }
}

impl Default for AgentToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create a delegation chain
pub struct DelegationChain {
    tools: Vec<Arc<AgentTool>>,
    strategy: DelegationStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum DelegationStrategy {
    /// First tool that succeeds
    FirstSuccess,
    /// All tools must succeed, combine results
    All,
    /// Fallback chain - try next on failure
    Fallback,
    /// Best result by confidence
    BestConfidence,
}

impl DelegationChain {
    pub fn new(strategy: DelegationStrategy) -> Self {
        Self {
            tools: Vec::new(),
            strategy,
        }
    }

    pub fn add(mut self, tool: Arc<AgentTool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub async fn execute(&self, input: AgentToolInput) -> Result<AgentToolOutput, AgentToolError> {
        match self.strategy {
            DelegationStrategy::FirstSuccess => self.execute_first_success(input).await,
            DelegationStrategy::Fallback => self.execute_fallback(input).await,
            DelegationStrategy::BestConfidence => self.execute_best_confidence(input).await,
            DelegationStrategy::All => self.execute_all(input).await,
        }
    }

    async fn execute_first_success(
        &self,
        input: AgentToolInput,
    ) -> Result<AgentToolOutput, AgentToolError> {
        for tool in &self.tools {
            if let Ok(output) = tool.call(input.clone()).await {
                if output.success {
                    return Ok(output);
                }
            }
        }
        Err(AgentToolError::ExecutionFailed(
            "No tool succeeded".to_string(),
        ))
    }

    async fn execute_fallback(
        &self,
        input: AgentToolInput,
    ) -> Result<AgentToolOutput, AgentToolError> {
        let mut last_error = None;
        for tool in &self.tools {
            match tool.call(input.clone()).await {
                Ok(output) => return Ok(output),
                Err(e) => last_error = Some(e),
            }
        }
        Err(last_error
            .unwrap_or_else(|| AgentToolError::ExecutionFailed("No tools available".to_string())))
    }

    async fn execute_best_confidence(
        &self,
        input: AgentToolInput,
    ) -> Result<AgentToolOutput, AgentToolError> {
        let mut best: Option<AgentToolOutput> = None;

        for tool in &self.tools {
            if let Ok(output) = tool.call(input.clone()).await {
                if output.success {
                    match &best {
                        None => best = Some(output),
                        Some(current) if output.confidence > current.confidence => {
                            best = Some(output)
                        }
                        _ => {}
                    }
                }
            }
        }

        best.ok_or_else(|| AgentToolError::ExecutionFailed("No tool succeeded".to_string()))
    }

    async fn execute_all(&self, input: AgentToolInput) -> Result<AgentToolOutput, AgentToolError> {
        let mut results = Vec::new();
        let mut all_success = true;
        let mut total_confidence = 0.0;
        let mut all_tools_used = Vec::new();

        for tool in &self.tools {
            match tool.call(input.clone()).await {
                Ok(output) => {
                    all_success = all_success && output.success;
                    total_confidence += output.confidence;
                    all_tools_used.extend(output.tools_used.clone());
                    results.push(output.content);
                }
                Err(_) => {
                    all_success = false;
                }
            }
        }

        if results.is_empty() {
            return Err(AgentToolError::ExecutionFailed(
                "All tools failed".to_string(),
            ));
        }

        let avg_confidence = total_confidence / self.tools.len() as f32;
        let combined_content = results.join("\n---\n");

        Ok(AgentToolOutput {
            content: combined_content,
            success: all_success,
            confidence: avg_confidence,
            metadata: HashMap::new(),
            duration_ms: 0,
            tools_used: all_tools_used,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_tool_basic() {
        let tool = AgentTool::builder("test_agent")
            .description("A test agent")
            .handler(|input: AgentToolInput| async move {
                Ok(AgentToolOutput::success(format!(
                    "Processed: {}",
                    input.query
                )))
            });

        let result = tool.call("Hello").await.unwrap();
        assert!(result.success);
        assert!(result.content.contains("Processed: Hello"));
    }

    #[tokio::test]
    async fn test_agent_tool_with_context() {
        let tool = AgentTool::builder("context_agent")
            .description("Agent that uses context")
            .handler(|input: AgentToolInput| async move {
                let name = input.context.get("name").cloned().unwrap_or_default();
                Ok(AgentToolOutput::success(format!("Hello, {}!", name)))
            });

        let input = AgentToolInput::new("greet").with_context("name", "World");
        let result = tool.call(input).await.unwrap();
        assert!(result.content.contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_agent_tool_failure() {
        let tool = AgentTool::builder("failing_agent")
            .description("Agent that fails")
            .max_retries(0)
            .handler(|_: AgentToolInput| async move {
                Err(AgentToolError::ExecutionFailed(
                    "Intentional failure".to_string(),
                ))
            });

        let result = tool.call("test").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_tool_stats() {
        let tool = AgentTool::builder("stats_agent")
            .description("Agent for testing stats")
            .handler(|_: AgentToolInput| async move { Ok(AgentToolOutput::success("OK")) });

        tool.call("test1").await.unwrap();
        tool.call("test2").await.unwrap();
        tool.call("test3").await.unwrap();

        let stats = tool.stats();
        assert_eq!(stats.total_calls, 3);
        assert_eq!(stats.successful_calls, 3);
        assert_eq!(stats.failed_calls, 0);
    }

    #[tokio::test]
    async fn test_agent_tool_registry() {
        let registry = AgentToolRegistry::new();

        let tool1 = AgentTool::builder("agent1")
            .description("First agent")
            .capability("math")
            .handler(|_: AgentToolInput| async move { Ok(AgentToolOutput::success("1")) });

        let tool2 = AgentTool::builder("agent2")
            .description("Second agent")
            .capability("text")
            .handler(|_: AgentToolInput| async move { Ok(AgentToolOutput::success("2")) });

        registry.register(tool1);
        registry.register(tool2);

        assert_eq!(registry.list().len(), 2);
        assert!(registry.get("agent1").is_some());
        assert!(registry.get("nonexistent").is_none());

        let math_tools = registry.find_by_capability("math");
        assert_eq!(math_tools.len(), 1);
    }

    #[tokio::test]
    async fn test_delegation_chain_fallback() {
        let failing_tool = Arc::new(
            AgentTool::builder("failing")
                .description("Fails")
                .max_retries(0)
                .handler(|_: AgentToolInput| async move {
                    Err(AgentToolError::ExecutionFailed("fail".to_string()))
                }),
        );

        let success_tool = Arc::new(
            AgentTool::builder("success")
                .description("Succeeds")
                .handler(|_: AgentToolInput| async move { Ok(AgentToolOutput::success("OK")) }),
        );

        let chain = DelegationChain::new(DelegationStrategy::Fallback)
            .add(failing_tool)
            .add(success_tool);

        let result = chain.execute(AgentToolInput::new("test")).await.unwrap();
        assert!(result.success);
        assert_eq!(result.content, "OK");
    }

    #[tokio::test]
    async fn test_delegation_chain_best_confidence() {
        let low_conf = Arc::new(
            AgentTool::builder("low")
                .description("Low confidence")
                .handler(|_: AgentToolInput| async move {
                    Ok(AgentToolOutput::success("low").with_confidence(0.3))
                }),
        );

        let high_conf = Arc::new(
            AgentTool::builder("high")
                .description("High confidence")
                .handler(|_: AgentToolInput| async move {
                    Ok(AgentToolOutput::success("high").with_confidence(0.9))
                }),
        );

        let chain = DelegationChain::new(DelegationStrategy::BestConfidence)
            .add(low_conf)
            .add(high_conf);

        let result = chain.execute(AgentToolInput::new("test")).await.unwrap();
        assert_eq!(result.content, "high");
        assert_eq!(result.confidence, 0.9);
    }

    #[tokio::test]
    async fn test_agent_tool_output_builder() {
        let output = AgentToolOutput::success("Result")
            .with_confidence(0.85)
            .with_metadata("key", "value")
            .with_tools_used(vec!["tool1".to_string()]);

        assert!(output.success);
        assert_eq!(output.confidence, 0.85);
        assert_eq!(output.metadata.get("key").unwrap(), "value");
        assert_eq!(output.tools_used.len(), 1);
    }

    #[tokio::test]
    async fn test_agent_tool_input_builder() {
        let input = AgentToolInput::new("query")
            .with_context("key", "value")
            .with_history(vec!["previous".to_string()])
            .with_max_tokens(100);

        assert_eq!(input.query, "query");
        assert_eq!(input.context.get("key").unwrap(), "value");
        assert_eq!(input.history.len(), 1);
        assert_eq!(input.max_tokens, Some(100));
    }

    #[tokio::test]
    async fn test_agent_tool_capabilities() {
        let tool = AgentTool::builder("capable")
            .description("Agent with capabilities")
            .capability("math")
            .capability("science")
            .handler(|_: AgentToolInput| async move { Ok(AgentToolOutput::success("OK")) });

        let caps = tool.capabilities();
        assert!(caps.contains(&"math".to_string()));
        assert!(caps.contains(&"science".to_string()));
    }
}
