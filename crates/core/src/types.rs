//! Core type definitions for the agent framework

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for an agent
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Agent role in a crew
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentRole {
    /// Agent performs research and analysis
    Researcher,
    /// Agent writes content
    Writer,
    /// Agent reviews and validates
    Reviewer,
    /// Agent coordinates other agents
    Coordinator,
    /// Agent executes specific tasks
    Executor,
    /// Custom role
    Custom(String),
}

/// Agent capabilities
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Can perform analysis
    Analysis,
    /// Can assess risk
    RiskAssessment,
    /// Can make predictions
    Prediction,
    /// Can write content
    ContentGeneration,
    /// Can execute code
    CodeExecution,
    /// Can search the web
    WebSearch,
    /// Can access databases
    DatabaseAccess,
    /// Custom capability
    Custom(String),
}

/// Message routing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Direct message to specific agent
    Direct,
    /// Broadcast to all agents
    Broadcast,
    /// Round-robin distribution
    RoundRobin,
    /// Load-balanced distribution
    LoadBalanced,
    /// Priority-based routing
    Priority(u8),
}

/// Memory configuration for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory size in bytes
    pub max_size: usize,
    /// Enable long-term memory persistence
    pub persist: bool,
    /// Memory retention policy
    pub retention_policy: RetentionPolicy,
}

impl MemoryConfig {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            persist: false,
            retention_policy: RetentionPolicy::KeepRecent(100),
        }
    }

    pub fn with_persistence(mut self, persist: bool) -> Self {
        self.persist = persist;
        self
    }

    pub fn with_retention(mut self, policy: RetentionPolicy) -> Self {
        self.retention_policy = policy;
        self
    }
}

/// Memory retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionPolicy {
    /// Keep all messages
    KeepAll,
    /// Keep only the N most recent messages
    KeepRecent(usize),
    /// Keep messages based on importance score
    KeepImportant(f64),
    /// Custom retention logic
    Custom,
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Unique agent identifier
    pub id: AgentId,
    /// Human-readable agent name
    pub name: String,
    /// Agent role
    pub role: AgentRole,
    /// Agent capabilities
    pub capabilities: Vec<Capability>,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// Routing strategy
    pub routing_strategy: RoutingStrategy,
    /// System prompt/instructions
    pub system_prompt: Option<String>,
    /// Maximum iterations for reasoning loops
    pub max_iterations: usize,
    /// Temperature for LLM sampling
    pub temperature: f32,
}

impl AgentConfig {
    pub fn new(name: impl Into<String>, role: AgentRole) -> Self {
        Self {
            id: AgentId::generate(),
            name: name.into(),
            role,
            capabilities: Vec::new(),
            memory_config: MemoryConfig::new(2 * 1024 * 1024), // 2MB default
            routing_strategy: RoutingStrategy::Direct,
            system_prompt: None,
            max_iterations: 10,
            temperature: 0.7,
        }
    }

    pub fn with_id(mut self, id: AgentId) -> Self {
        self.id = id;
        self
    }

    pub fn with_capabilities(mut self, capabilities: Vec<Capability>) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
}

/// Agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Current agent status
    pub status: AgentStatus,
    /// Message history
    pub history: Vec<crate::message::Message>,
    /// Agent-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Current iteration count
    pub iteration: usize,
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            status: AgentStatus::Idle,
            history: Vec::new(),
            metadata: HashMap::new(),
            iteration: 0,
        }
    }

    pub fn is_busy(&self) -> bool {
        matches!(self.status, AgentStatus::Processing | AgentStatus::Thinking)
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
}

/// Agent status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is idle and ready
    Idle,
    /// Agent is processing a message
    Processing,
    /// Agent is performing reasoning
    Thinking,
    /// Agent is executing a tool
    ExecutingTool,
    /// Agent encountered an error
    Error,
    /// Agent is stopped
    Stopped,
}

/// Task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Task ID
    pub id: String,
    /// Task description
    pub description: String,
    /// Expected output description
    pub expected_output: String,
    /// Agent assigned to this task
    pub agent_id: Option<AgentId>,
    /// Task dependencies
    pub dependencies: Vec<String>,
    /// Task context data
    pub context: HashMap<String, serde_json::Value>,
}

impl Task {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description: description.into(),
            expected_output: String::new(),
            agent_id: None,
            dependencies: Vec::new(),
            context: HashMap::new(),
        }
    }

    pub fn with_expected_output(mut self, output: impl Into<String>) -> Self {
        self.expected_output = output.into();
        self
    }

    pub fn with_agent(mut self, agent_id: AgentId) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    pub fn add_context(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.context.insert(key.into(), value);
        self
    }
}

/// Task result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task ID
    pub task_id: String,
    /// Success status
    pub success: bool,
    /// Output data
    pub output: serde_json::Value,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl TaskResult {
    pub fn success(task_id: String, output: serde_json::Value) -> Self {
        Self {
            task_id,
            success: true,
            output,
            error: None,
            metadata: HashMap::new(),
        }
    }

    pub fn failure(task_id: String, error: impl Into<String>) -> Self {
        Self {
            task_id,
            success: false,
            output: serde_json::Value::Null,
            error: Some(error.into()),
            metadata: HashMap::new(),
        }
    }
}
