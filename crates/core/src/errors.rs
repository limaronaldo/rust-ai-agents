//! Error types for the agent framework

use thiserror::Error;

/// Agent-related errors
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Agent not found: {0}")]
    AgentNotFound(crate::types::AgentId),

    #[error("Failed to send message: {0}")]
    SendError(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("State error: {0}")]
    StateError(String),

    #[error("Maximum iterations exceeded")]
    MaxIterationsExceeded,

    #[error("Tool error: {0}")]
    ToolError(#[from] ToolError),

    #[error("LLM error: {0}")]
    LLMError(#[from] LLMError),

    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
}

/// Tool/function execution errors
#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Timeout: operation took longer than {0}ms")]
    TimeoutMs(u64),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Circuit breaker open for tool: {0}")]
    CircuitOpen(String),
}

/// LLM backend errors
#[derive(Error, Debug)]
pub enum LLMError {
    #[error("API error: {0}")]
    ApiError(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Timeout: request took longer than {0}ms")]
    Timeout(u64),

    #[error("Token limit exceeded: {0}/{1}")]
    TokenLimitExceeded(usize, usize),

    #[error("Model not available: {0}")]
    ModelNotAvailable(String),
}

/// Crew/orchestration errors
#[derive(Error, Debug)]
pub enum CrewError {
    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Circular dependency detected in tasks")]
    CircularDependency,

    #[error("No agent available for task: {0}")]
    NoAgentAvailable(String),

    #[error("Task execution failed: {0}")]
    TaskExecutionFailed(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Agent error: {0}")]
    AgentError(#[from] AgentError),
}

/// Memory errors
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Memory limit exceeded: {0}/{1} bytes")]
    LimitExceeded(usize, usize),

    #[error("Capacity exceeded: current={current}, required={required}, max={max}")]
    CapacityExceeded {
        current: usize,
        required: usize,
        max: usize,
    },

    #[error("Persistence error: {0}")]
    PersistenceError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Not found: {0}")]
    NotFound(String),
}
