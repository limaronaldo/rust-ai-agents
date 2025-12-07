//! Workflow error types

use thiserror::Error;

/// Errors that can occur during workflow operations
#[derive(Debug, Error)]
pub enum WorkflowError {
    #[error("Failed to parse workflow: {0}")]
    ParseError(String),

    #[error("Invalid workflow: {0}")]
    ValidationError(String),

    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Circular dependency detected: {0}")]
    CircularDependency(String),

    #[error("Variable not found: {0}")]
    VariableNotFound(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Timeout after {0} seconds")]
    Timeout(u64),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("Agent error: {0}")]
    Agent(#[from] rust_ai_agents_core::AgentError),
}
