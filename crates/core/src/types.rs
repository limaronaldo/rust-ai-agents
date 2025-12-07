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

// Default value functions for serde
fn default_max_iterations() -> usize {
    10
}
fn default_timeout_secs() -> u64 {
    60
}
fn default_temperature() -> f32 {
    0.7
}
fn default_planning_mode() -> PlanningMode {
    PlanningMode::Disabled
}

/// Planning mode configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum PlanningMode {
    /// No planning - direct execution (default)
    #[default]
    Disabled,
    /// Plan before each task execution
    BeforeTask,
    /// Plan once at the beginning, then execute all steps
    FullPlan,
    /// Adaptive planning - re-plan after each step based on results
    Adaptive,
}

impl PlanningMode {
    pub fn is_enabled(&self) -> bool {
        !matches!(self, PlanningMode::Disabled)
    }
}

/// A step in an execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step number (1-indexed)
    pub step_number: usize,
    /// Description of what this step does
    pub description: String,
    /// Expected output/result from this step
    pub expected_result: String,
    /// Tools that might be used in this step
    pub tools: Vec<String>,
    /// Whether this step has been completed
    pub completed: bool,
    /// Actual result after execution
    pub actual_result: Option<String>,
}

impl PlanStep {
    pub fn new(step_number: usize, description: impl Into<String>) -> Self {
        Self {
            step_number,
            description: description.into(),
            expected_result: String::new(),
            tools: Vec::new(),
            completed: false,
            actual_result: None,
        }
    }

    pub fn with_expected_result(mut self, result: impl Into<String>) -> Self {
        self.expected_result = result.into();
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    pub fn mark_completed(&mut self, result: impl Into<String>) {
        self.completed = true;
        self.actual_result = Some(result.into());
    }
}

/// An execution plan generated by the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Plan identifier
    pub id: String,
    /// Original task/goal
    pub goal: String,
    /// Ordered list of steps to execute
    pub steps: Vec<PlanStep>,
    /// Current step being executed (0 = not started)
    pub current_step: usize,
    /// Whether the plan has been fully executed
    pub completed: bool,
    /// Overall success status
    pub success: bool,
    /// Any notes or reasoning about the plan
    pub reasoning: String,
}

impl ExecutionPlan {
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            goal: goal.into(),
            steps: Vec::new(),
            current_step: 0,
            completed: false,
            success: false,
            reasoning: String::new(),
        }
    }

    pub fn add_step(&mut self, step: PlanStep) {
        self.steps.push(step);
    }

    pub fn with_steps(mut self, steps: Vec<PlanStep>) -> Self {
        self.steps = steps;
        self
    }

    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = reasoning.into();
        self
    }

    pub fn current_step(&self) -> Option<&PlanStep> {
        if self.current_step > 0 && self.current_step <= self.steps.len() {
            Some(&self.steps[self.current_step - 1])
        } else {
            None
        }
    }

    pub fn advance(&mut self) -> bool {
        if self.current_step < self.steps.len() {
            self.current_step += 1;
            true
        } else {
            self.completed = true;
            false
        }
    }

    pub fn mark_current_completed(&mut self, result: impl Into<String>) {
        if self.current_step > 0 && self.current_step <= self.steps.len() {
            self.steps[self.current_step - 1].mark_completed(result);
        }
    }

    pub fn progress(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let completed = self.steps.iter().filter(|s| s.completed).count();
        completed as f64 / self.steps.len() as f64
    }

    pub fn is_complete(&self) -> bool {
        self.completed || self.steps.iter().all(|s| s.completed)
    }
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
    /// Maximum iterations for reasoning loops (prevents infinite loops)
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    /// Timeout in seconds for message processing
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Temperature for LLM sampling
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Planning mode for task execution
    #[serde(default = "default_planning_mode")]
    pub planning_mode: PlanningMode,
    /// Stop words that terminate agent execution
    #[serde(default)]
    pub stop_words: Vec<String>,
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
            max_iterations: default_max_iterations(),
            timeout_secs: default_timeout_secs(),
            temperature: default_temperature(),
            planning_mode: default_planning_mode(),
            stop_words: Vec::new(),
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

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    pub fn with_planning_mode(mut self, mode: PlanningMode) -> Self {
        self.planning_mode = mode;
        self
    }

    pub fn with_stop_words(mut self, words: Vec<String>) -> Self {
        self.stop_words = words;
        self
    }

    pub fn add_stop_word(mut self, word: impl Into<String>) -> Self {
        self.stop_words.push(word.into());
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
    /// Current execution plan (if planning mode is enabled)
    pub current_plan: Option<ExecutionPlan>,
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            status: AgentStatus::Idle,
            history: Vec::new(),
            metadata: HashMap::new(),
            iteration: 0,
            current_plan: None,
        }
    }

    pub fn is_busy(&self) -> bool {
        matches!(
            self.status,
            AgentStatus::Processing | AgentStatus::Thinking | AgentStatus::Planning
        )
    }

    pub fn set_plan(&mut self, plan: ExecutionPlan) {
        self.current_plan = Some(plan);
    }

    pub fn clear_plan(&mut self) {
        self.current_plan = None;
    }

    pub fn plan_progress(&self) -> Option<f64> {
        self.current_plan.as_ref().map(|p| p.progress())
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
    /// Agent is creating an execution plan
    Planning,
    /// Agent is executing a planned step
    ExecutingPlan,
    /// Agent is executing a tool
    ExecutingTool,
    /// Agent encountered an error
    Error,
    /// Agent is stopped
    Stopped,
    /// Agent terminated due to stop word
    StoppedByStopWord,
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
