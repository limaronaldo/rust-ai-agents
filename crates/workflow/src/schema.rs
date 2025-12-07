//! YAML schema definitions for workflows
//!
//! Defines the structure of workflow YAML files.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Root workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    /// Workflow name
    pub name: String,

    /// Workflow description
    #[serde(default)]
    pub description: String,

    /// Version of the workflow
    #[serde(default = "default_version")]
    pub version: String,

    /// Agent definitions
    #[serde(default)]
    pub agents: Vec<AgentDefinition>,

    /// Task definitions
    #[serde(default)]
    pub tasks: Vec<TaskDefinition>,

    /// Execution configuration
    #[serde(default)]
    pub execution: ExecutionConfig,

    /// Global variables
    #[serde(default)]
    pub variables: HashMap<String, serde_json::Value>,

    /// Workflow metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

fn default_version() -> String {
    "1.0".to_string()
}

/// Agent definition in workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Unique agent ID within workflow
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Agent role
    #[serde(default = "default_role")]
    pub role: AgentRoleType,

    /// System prompt/instructions
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Tools available to this agent
    #[serde(default)]
    pub tools: Vec<String>,

    /// LLM model to use
    #[serde(default)]
    pub model: Option<String>,

    /// Temperature for LLM
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Maximum iterations
    #[serde(default)]
    pub max_iterations: Option<usize>,

    /// Timeout in seconds
    #[serde(default)]
    pub timeout_secs: Option<u64>,

    /// Planning mode
    #[serde(default)]
    pub planning_mode: PlanningModeType,

    /// Stop words for termination
    #[serde(default)]
    pub stop_words: Vec<String>,

    /// Agent-specific metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

fn default_role() -> AgentRoleType {
    AgentRoleType::Executor
}

/// Agent role types
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AgentRoleType {
    Researcher,
    Writer,
    Reviewer,
    Coordinator,
    #[default]
    Executor,
    Custom(String),
}

/// Planning mode types
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PlanningModeType {
    #[default]
    Disabled,
    BeforeTask,
    FullPlan,
    Adaptive,
}

/// Task definition in workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    /// Unique task ID
    pub id: String,

    /// Agent ID to execute this task
    pub agent: String,

    /// Task description/prompt
    pub description: String,

    /// Expected output description
    #[serde(default)]
    pub expected_output: String,

    /// Task dependencies (other task IDs)
    #[serde(default)]
    pub depends_on: Vec<String>,

    /// Context data for the task
    #[serde(default)]
    pub context: HashMap<String, serde_json::Value>,

    /// Whether this task can be skipped on failure
    #[serde(default)]
    pub optional: bool,

    /// Retry configuration
    #[serde(default)]
    pub retry: Option<RetryConfig>,

    /// Task-specific timeout override
    #[serde(default)]
    pub timeout_secs: Option<u64>,

    /// Task metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Retry configuration for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    #[serde(default = "default_max_retries")]
    pub max_attempts: usize,

    /// Delay between retries in seconds
    #[serde(default = "default_retry_delay")]
    pub delay_secs: u64,

    /// Whether to use exponential backoff
    #[serde(default)]
    pub exponential_backoff: bool,
}

fn default_max_retries() -> usize {
    3
}

fn default_retry_delay() -> u64 {
    1
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: default_max_retries(),
            delay_secs: default_retry_delay(),
            exponential_backoff: false,
        }
    }
}

/// Execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Execution mode
    #[serde(default)]
    pub mode: ExecutionMode,

    /// Global maximum iterations
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// Global timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Whether to stop on first error
    #[serde(default)]
    pub fail_fast: bool,

    /// Maximum parallel tasks (for parallel mode)
    #[serde(default = "default_max_parallel")]
    pub max_parallel: usize,

    /// LLM provider configuration
    #[serde(default)]
    pub provider: ProviderConfig,
}

fn default_max_iterations() -> usize {
    10
}

fn default_timeout() -> u64 {
    300
}

fn default_max_parallel() -> usize {
    4
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Sequential,
            max_iterations: default_max_iterations(),
            timeout_secs: default_timeout(),
            fail_fast: false,
            max_parallel: default_max_parallel(),
            provider: ProviderConfig::default(),
        }
    }
}

/// Execution mode
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Execute tasks one by one in order
    #[default]
    Sequential,
    /// Execute independent tasks in parallel
    Parallel,
    /// Use a manager agent to coordinate
    Hierarchical,
}

/// LLM Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type
    #[serde(default)]
    pub provider_type: ProviderType,

    /// Default model
    #[serde(default)]
    pub model: Option<String>,

    /// API key environment variable name
    #[serde(default)]
    pub api_key_env: Option<String>,

    /// Base URL for API
    #[serde(default)]
    pub base_url: Option<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider_type: ProviderType::OpenRouter,
            model: None,
            api_key_env: None,
            base_url: None,
        }
    }
}

/// Provider type
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    #[default]
    OpenRouter,
    OpenAI,
    Anthropic,
    Local,
    Custom(String),
}

/// Result of a task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task ID
    pub task_id: String,

    /// Whether task succeeded
    pub success: bool,

    /// Output from the task
    pub output: String,

    /// Error message if failed
    pub error: Option<String>,

    /// Execution time in milliseconds
    pub duration_ms: u64,

    /// Number of LLM calls made
    pub llm_calls: usize,

    /// Number of tool calls made
    pub tool_calls: usize,
}

/// Result of a workflow execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    /// Workflow name
    pub workflow_name: String,

    /// Whether workflow succeeded
    pub success: bool,

    /// Task results in execution order
    pub task_results: Vec<TaskResult>,

    /// Total execution time in milliseconds
    pub total_duration_ms: u64,

    /// Final output (from last task)
    pub final_output: Option<String>,

    /// Any errors encountered
    pub errors: Vec<String>,
}

impl WorkflowResult {
    pub fn new(workflow_name: String) -> Self {
        Self {
            workflow_name,
            success: false,
            task_results: Vec::new(),
            total_duration_ms: 0,
            final_output: None,
            errors: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_workflow() {
        let yaml = r#"
name: test_workflow
agents:
  - id: agent1
    name: Test Agent
tasks:
  - id: task1
    agent: agent1
    description: Do something
"#;
        let workflow: WorkflowDefinition = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(workflow.name, "test_workflow");
        assert_eq!(workflow.agents.len(), 1);
        assert_eq!(workflow.tasks.len(), 1);
    }

    #[test]
    fn test_parse_full_workflow() {
        let yaml = r#"
name: research_workflow
description: Research and summarize a topic
version: "1.0"

agents:
  - id: researcher
    name: Research Agent
    role: researcher
    system_prompt: You are a research specialist.
    tools: [web_search]
    planning_mode: before_task
    stop_words: [DONE, FINISHED]

  - id: writer
    name: Writer Agent
    role: writer
    temperature: 0.7

tasks:
  - id: research
    agent: researcher
    description: Research the topic
    expected_output: Detailed notes

  - id: summarize
    agent: writer
    description: Write summary
    depends_on: [research]
    retry:
      max_attempts: 2

execution:
  mode: sequential
  timeout_secs: 600
  fail_fast: true
"#;
        let workflow: WorkflowDefinition = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(workflow.name, "research_workflow");
        assert_eq!(workflow.agents.len(), 2);
        assert_eq!(workflow.tasks.len(), 2);
        assert_eq!(workflow.agents[0].role, AgentRoleType::Researcher);
        assert_eq!(
            workflow.agents[0].planning_mode,
            PlanningModeType::BeforeTask
        );
        assert_eq!(workflow.tasks[1].depends_on, vec!["research"]);
        assert!(workflow.execution.fail_fast);
    }
}
