//! # Durable Execution
//!
//! Fault-tolerant execution with automatic recovery from failures.
//!
//! Inspired by LangGraph's durable execution patterns.
//!
//! ## Features
//!
//! - **Step Persistence**: Persist each step for recovery
//! - **Resume**: Resume execution from any point after failure
//! - **Idempotency**: Ensure steps are not re-executed
//! - **Timeouts**: Handle long-running operations
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents::durable::{DurableExecution, ExecutionStep};
//!
//! let execution = DurableExecution::new("workflow_1")
//!     .add_step("fetch_data", || async { fetch_data().await })
//!     .add_step("process", || async { process_data().await })
//!     .add_step("save", || async { save_results().await });
//!
//! // Run (will resume from last completed step if previously interrupted)
//! let result = execution.run().await?;
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

/// Unique identifier for an execution
pub type ExecutionId = String;

/// Unique identifier for a step
pub type StepId = String;

/// Status of an execution step
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepStatus {
    /// Not yet started
    Pending,
    /// Currently running
    Running,
    /// Successfully completed
    Completed,
    /// Failed with error
    Failed,
    /// Skipped (e.g., conditional step)
    Skipped,
    /// Timed out
    TimedOut,
}

/// Result of a step execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Step ID
    pub step_id: StepId,
    /// Status
    pub status: StepStatus,
    /// Output value (JSON serialized)
    pub output: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// When the step started
    pub started_at: Option<u64>,
    /// When the step completed
    pub completed_at: Option<u64>,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Retry count
    pub retry_count: u32,
}

impl StepResult {
    pub fn pending(step_id: impl Into<String>) -> Self {
        Self {
            step_id: step_id.into(),
            status: StepStatus::Pending,
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
            duration_ms: None,
            retry_count: 0,
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.status, StepStatus::Completed | StepStatus::Skipped)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.status, StepStatus::Failed | StepStatus::TimedOut)
    }
}

/// State of the entire execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    /// Execution ID
    pub execution_id: ExecutionId,
    /// Current step index
    pub current_step: usize,
    /// Total steps
    pub total_steps: usize,
    /// Results for each step
    pub step_results: Vec<StepResult>,
    /// When execution started
    pub started_at: u64,
    /// When execution last updated
    pub updated_at: u64,
    /// When execution completed (if done)
    pub completed_at: Option<u64>,
    /// Overall status
    pub status: ExecutionStatus,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ExecutionState {
    pub fn new(execution_id: impl Into<String>, step_ids: &[String]) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            execution_id: execution_id.into(),
            current_step: 0,
            total_steps: step_ids.len(),
            step_results: step_ids.iter().map(StepResult::pending).collect(),
            started_at: now,
            updated_at: now,
            completed_at: None,
            status: ExecutionStatus::Pending,
            metadata: HashMap::new(),
        }
    }

    pub fn is_complete(&self) -> bool {
        // Only truly completed, not failed (failed can be resumed)
        matches!(self.status, ExecutionStatus::Completed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.status, ExecutionStatus::Failed)
    }

    pub fn progress(&self) -> f32 {
        if self.total_steps == 0 {
            return 1.0;
        }
        let completed = self.step_results.iter().filter(|r| r.is_complete()).count();
        completed as f32 / self.total_steps as f32
    }

    fn touch(&mut self) {
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

/// Overall execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Not yet started
    Pending,
    /// Currently running
    Running,
    /// Successfully completed all steps
    Completed,
    /// Failed (one or more steps failed)
    Failed,
    /// Paused (can be resumed)
    Paused,
}

/// Error type for durable execution
#[derive(Debug, thiserror::Error)]
pub enum DurableError {
    #[error("Step '{0}' failed: {1}")]
    StepFailed(StepId, String),

    #[error("Step '{0}' timed out after {1:?}")]
    StepTimeout(StepId, Duration),

    #[error("Execution '{0}' not found")]
    ExecutionNotFound(ExecutionId),

    #[error("Step '{0}' not found")]
    StepNotFound(StepId),

    #[error("Execution already completed")]
    AlreadyCompleted,

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Max retries ({0}) exceeded for step '{1}'")]
    MaxRetriesExceeded(u32, StepId),
}

/// Configuration for durable execution
#[derive(Debug, Clone)]
pub struct DurableConfig {
    /// Default timeout per step
    pub default_timeout: Duration,
    /// Maximum retries per step
    pub max_retries: u32,
    /// Delay between retries
    pub retry_delay: Duration,
    /// Whether to continue on step failure
    pub continue_on_failure: bool,
    /// Whether to persist state after each step
    pub persist_per_step: bool,
}

impl Default for DurableConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(300), // 5 minutes
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            continue_on_failure: false,
            persist_per_step: true,
        }
    }
}

/// Trait for persisting execution state
#[async_trait]
pub trait ExecutionStore: Send + Sync {
    /// Save execution state
    async fn save(&self, state: &ExecutionState) -> Result<(), DurableError>;

    /// Load execution state
    async fn load(&self, execution_id: &str) -> Result<Option<ExecutionState>, DurableError>;

    /// Delete execution state
    async fn delete(&self, execution_id: &str) -> Result<(), DurableError>;

    /// List all execution IDs
    async fn list(&self) -> Result<Vec<ExecutionId>, DurableError>;
}

/// In-memory execution store
#[derive(Default)]
pub struct MemoryExecutionStore {
    states: RwLock<HashMap<ExecutionId, ExecutionState>>,
}

impl MemoryExecutionStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ExecutionStore for MemoryExecutionStore {
    async fn save(&self, state: &ExecutionState) -> Result<(), DurableError> {
        self.states
            .write()
            .insert(state.execution_id.clone(), state.clone());
        Ok(())
    }

    async fn load(&self, execution_id: &str) -> Result<Option<ExecutionState>, DurableError> {
        Ok(self.states.read().get(execution_id).cloned())
    }

    async fn delete(&self, execution_id: &str) -> Result<(), DurableError> {
        self.states.write().remove(execution_id);
        Ok(())
    }

    async fn list(&self) -> Result<Vec<ExecutionId>, DurableError> {
        Ok(self.states.read().keys().cloned().collect())
    }
}

/// Type alias for step functions
pub type StepFn =
    Box<dyn Fn() -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> + Send + Sync>;

/// A step in the execution
pub struct ExecutionStep {
    pub id: StepId,
    pub name: String,
    pub timeout: Option<Duration>,
    pub max_retries: Option<u32>,
    pub handler: StepFn,
}

impl ExecutionStep {
    pub fn new<F, Fut>(id: impl Into<String>, handler: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, String>> + Send + 'static,
    {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            timeout: None,
            max_retries: None,
            handler: Box::new(move || Box::pin(handler())),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }
}

/// Result of a durable execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Final state
    pub state: ExecutionState,
    /// Whether all steps completed successfully
    pub success: bool,
    /// Final outputs from each step
    pub outputs: HashMap<StepId, String>,
    /// Total duration
    pub total_duration_ms: u64,
}

impl ExecutionResult {
    pub fn get_output(&self, step_id: &str) -> Option<&String> {
        self.outputs.get(step_id)
    }
}

/// A durable execution workflow
pub struct DurableExecution<S: ExecutionStore> {
    execution_id: ExecutionId,
    steps: Vec<ExecutionStep>,
    config: DurableConfig,
    store: Arc<S>,
}

impl DurableExecution<MemoryExecutionStore> {
    /// Create with in-memory store
    pub fn in_memory(execution_id: impl Into<String>) -> Self {
        Self::new(execution_id, Arc::new(MemoryExecutionStore::new()))
    }
}

impl<S: ExecutionStore> DurableExecution<S> {
    pub fn new(execution_id: impl Into<String>, store: Arc<S>) -> Self {
        Self {
            execution_id: execution_id.into(),
            steps: Vec::new(),
            config: DurableConfig::default(),
            store,
        }
    }

    pub fn with_config(mut self, config: DurableConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a step to the execution
    pub fn add_step<F, Fut>(mut self, id: impl Into<String>, handler: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, String>> + Send + 'static,
    {
        self.steps.push(ExecutionStep::new(id, handler));
        self
    }

    /// Add a pre-configured step
    pub fn add_step_config(mut self, step: ExecutionStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Run the execution (resumes from last completed step if interrupted)
    pub async fn run(&self) -> Result<ExecutionResult, DurableError> {
        // Load or create state
        let mut state = match self.store.load(&self.execution_id).await? {
            Some(existing) => {
                if existing.is_complete() {
                    return Err(DurableError::AlreadyCompleted);
                }
                info!(
                    execution_id = %self.execution_id,
                    current_step = existing.current_step,
                    "Resuming execution"
                );
                existing
            }
            None => {
                let step_ids: Vec<String> = self.steps.iter().map(|s| s.id.clone()).collect();
                let state = ExecutionState::new(&self.execution_id, &step_ids);
                self.store.save(&state).await?;
                info!(execution_id = %self.execution_id, "Starting new execution");
                state
            }
        };

        state.status = ExecutionStatus::Running;
        state.touch();
        self.store.save(&state).await?;

        let start = std::time::Instant::now();
        let mut outputs = HashMap::new();

        // Execute steps starting from current position
        for step_idx in state.current_step..self.steps.len() {
            let step = &self.steps[step_idx];

            // Skip already completed steps
            if state.step_results[step_idx].is_complete() {
                if let Some(output) = &state.step_results[step_idx].output {
                    outputs.insert(step.id.clone(), output.clone());
                }
                continue;
            }

            debug!(
                execution_id = %self.execution_id,
                step_id = %step.id,
                step_idx,
                "Executing step"
            );

            // Execute step with retries
            let result = self
                .execute_step(step, &mut state.step_results[step_idx])
                .await;

            state.current_step = step_idx;
            state.touch();

            match result {
                Ok(output) => {
                    outputs.insert(step.id.clone(), output);
                    if self.config.persist_per_step {
                        self.store.save(&state).await?;
                    }
                }
                Err(e) => {
                    error!(
                        execution_id = %self.execution_id,
                        step_id = %step.id,
                        error = %e,
                        "Step failed"
                    );

                    if !self.config.continue_on_failure {
                        state.status = ExecutionStatus::Failed;
                        state.touch();
                        self.store.save(&state).await?;
                        return Err(e);
                    }
                }
            }
        }

        // All steps completed
        state.status = ExecutionStatus::Completed;
        state.completed_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
        state.touch();
        self.store.save(&state).await?;

        info!(
            execution_id = %self.execution_id,
            duration_ms = start.elapsed().as_millis(),
            "Execution completed"
        );

        Ok(ExecutionResult {
            success: state.step_results.iter().all(|r| r.is_complete()),
            state,
            outputs,
            total_duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn execute_step(
        &self,
        step: &ExecutionStep,
        result: &mut StepResult,
    ) -> Result<String, DurableError> {
        let timeout = step.timeout.unwrap_or(self.config.default_timeout);
        let max_retries = step.max_retries.unwrap_or(self.config.max_retries);

        let start = std::time::Instant::now();
        result.started_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
        result.status = StepStatus::Running;

        for attempt in 0..=max_retries {
            result.retry_count = attempt;

            match tokio::time::timeout(timeout, (step.handler)()).await {
                Ok(Ok(output)) => {
                    result.status = StepStatus::Completed;
                    result.output = Some(output.clone());
                    result.completed_at = Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    );
                    result.duration_ms = Some(start.elapsed().as_millis() as u64);

                    debug!(
                        step_id = %step.id,
                        attempt,
                        duration_ms = result.duration_ms,
                        "Step completed"
                    );

                    return Ok(output);
                }
                Ok(Err(e)) => {
                    warn!(
                        step_id = %step.id,
                        attempt,
                        max_retries,
                        error = %e,
                        "Step attempt failed"
                    );

                    if attempt < max_retries {
                        tokio::time::sleep(self.config.retry_delay).await;
                        continue;
                    }

                    result.status = StepStatus::Failed;
                    result.error = Some(e.clone());
                    result.completed_at = Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    );
                    result.duration_ms = Some(start.elapsed().as_millis() as u64);

                    return Err(DurableError::StepFailed(step.id.clone(), e));
                }
                Err(_) => {
                    warn!(
                        step_id = %step.id,
                        attempt,
                        timeout_secs = timeout.as_secs(),
                        "Step timed out"
                    );

                    if attempt < max_retries {
                        tokio::time::sleep(self.config.retry_delay).await;
                        continue;
                    }

                    result.status = StepStatus::TimedOut;
                    result.error = Some(format!("Timed out after {:?}", timeout));
                    result.completed_at = Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    );
                    result.duration_ms = Some(start.elapsed().as_millis() as u64);

                    return Err(DurableError::StepTimeout(step.id.clone(), timeout));
                }
            }
        }

        Err(DurableError::MaxRetriesExceeded(
            max_retries,
            step.id.clone(),
        ))
    }

    /// Pause the execution (for manual resume later)
    pub async fn pause(&self) -> Result<(), DurableError> {
        if let Some(mut state) = self.store.load(&self.execution_id).await? {
            state.status = ExecutionStatus::Paused;
            state.touch();
            self.store.save(&state).await?;
            info!(execution_id = %self.execution_id, "Execution paused");
        }
        Ok(())
    }

    /// Get current state
    pub async fn state(&self) -> Result<Option<ExecutionState>, DurableError> {
        self.store.load(&self.execution_id).await
    }

    /// Reset execution to start from beginning
    pub async fn reset(&self) -> Result<(), DurableError> {
        self.store.delete(&self.execution_id).await
    }
}

/// Statistics for durable executions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DurableStats {
    pub total_executions: u64,
    pub completed_executions: u64,
    pub failed_executions: u64,
    pub total_steps_executed: u64,
    pub total_retries: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_simple_execution() {
        let execution = DurableExecution::in_memory("test_1")
            .add_step("step1", || async { Ok("result1".to_string()) })
            .add_step("step2", || async { Ok("result2".to_string()) });

        let result = execution.run().await.unwrap();

        assert!(result.success);
        assert_eq!(result.outputs.get("step1").unwrap(), "result1");
        assert_eq!(result.outputs.get("step2").unwrap(), "result2");
        assert_eq!(result.state.status, ExecutionStatus::Completed);
    }

    #[tokio::test]
    async fn test_execution_with_failure() {
        let config = DurableConfig {
            max_retries: 0,
            ..Default::default()
        };

        let execution = DurableExecution::in_memory("test_fail")
            .with_config(config)
            .add_step("step1", || async { Ok("ok".to_string()) })
            .add_step("step2", || async { Err("failed".to_string()) });

        let result = execution.run().await;

        assert!(result.is_err());
        match result {
            Err(DurableError::StepFailed(id, _)) => assert_eq!(id, "step2"),
            _ => panic!("Expected StepFailed error"),
        }
    }

    #[tokio::test]
    async fn test_execution_resume() {
        let store = Arc::new(MemoryExecutionStore::new());
        let attempt = Arc::new(AtomicU32::new(0));

        // First run - step2 will fail
        {
            let attempt_clone = attempt.clone();
            let config = DurableConfig {
                max_retries: 0,
                ..Default::default()
            };

            let execution = DurableExecution::new("test_resume", store.clone())
                .with_config(config)
                .add_step("step1", || async { Ok("done".to_string()) })
                .add_step("step2", move || {
                    let current = attempt_clone.fetch_add(1, Ordering::SeqCst);
                    async move {
                        if current == 0 {
                            Err("first attempt fails".to_string())
                        } else {
                            Ok("success".to_string())
                        }
                    }
                });

            let _ = execution.run().await; // This will fail
        }

        // Second run - should resume and succeed
        {
            let attempt_clone = attempt.clone();
            let config = DurableConfig {
                max_retries: 0,
                ..Default::default()
            };

            // Need to recreate execution with same store
            let execution = DurableExecution::new("test_resume", store.clone())
                .with_config(config)
                .add_step("step1", || async { Ok("done".to_string()) })
                .add_step("step2", move || {
                    let current = attempt_clone.fetch_add(1, Ordering::SeqCst);
                    async move {
                        if current == 0 {
                            Err("first attempt fails".to_string())
                        } else {
                            Ok("success".to_string())
                        }
                    }
                });

            let result = execution.run().await.unwrap();
            assert!(result.success);
        }
    }

    #[tokio::test]
    async fn test_step_with_retries() {
        let attempt = AtomicU32::new(0);

        let config = DurableConfig {
            max_retries: 2,
            retry_delay: Duration::from_millis(10),
            ..Default::default()
        };

        let execution = DurableExecution::in_memory("test_retry")
            .with_config(config)
            .add_step("flaky", move || {
                let current = attempt.fetch_add(1, Ordering::SeqCst);
                async move {
                    if current < 2 {
                        Err("temporary failure".to_string())
                    } else {
                        Ok("finally worked".to_string())
                    }
                }
            });

        let result = execution.run().await.unwrap();
        assert!(result.success);
        assert_eq!(result.outputs.get("flaky").unwrap(), "finally worked");
    }

    #[tokio::test]
    async fn test_execution_state() {
        let store = Arc::new(MemoryExecutionStore::new());

        let execution = DurableExecution::new("test_state", store.clone())
            .add_step("step1", || async { Ok("done".to_string()) });

        execution.run().await.unwrap();

        let state = execution.state().await.unwrap().unwrap();
        assert_eq!(state.status, ExecutionStatus::Completed);
        assert!(state.completed_at.is_some());
        assert_eq!(state.step_results[0].status, StepStatus::Completed);
    }

    #[tokio::test]
    async fn test_execution_reset() {
        let store = Arc::new(MemoryExecutionStore::new());

        let execution = DurableExecution::new("test_reset", store.clone())
            .add_step("step1", || async { Ok("done".to_string()) });

        execution.run().await.unwrap();
        assert!(execution.state().await.unwrap().is_some());

        execution.reset().await.unwrap();
        assert!(execution.state().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_progress() {
        let state = ExecutionState::new(
            "test",
            &["s1".to_string(), "s2".to_string(), "s3".to_string()],
        );
        assert_eq!(state.progress(), 0.0);

        let mut state = state;
        state.step_results[0].status = StepStatus::Completed;
        assert!((state.progress() - 0.333).abs() < 0.01);

        state.step_results[1].status = StepStatus::Completed;
        state.step_results[2].status = StepStatus::Completed;
        assert_eq!(state.progress(), 1.0);
    }

    #[tokio::test]
    async fn test_continue_on_failure() {
        let config = DurableConfig {
            max_retries: 0,
            continue_on_failure: true,
            ..Default::default()
        };

        let execution = DurableExecution::in_memory("test_continue")
            .with_config(config)
            .add_step("step1", || async { Ok("ok".to_string()) })
            .add_step("step2", || async { Err("fails".to_string()) })
            .add_step("step3", || async { Ok("also ok".to_string()) });

        let result = execution.run().await.unwrap();

        // Should complete but not be fully successful
        assert_eq!(result.state.status, ExecutionStatus::Completed);
        assert!(!result.success); // step2 failed
        assert!(result.outputs.contains_key("step1"));
        assert!(result.outputs.contains_key("step3"));
        assert!(!result.outputs.contains_key("step2"));
    }

    #[tokio::test]
    async fn test_step_result_states() {
        let result = StepResult::pending("test");
        assert!(!result.is_complete());
        assert!(!result.is_failed());

        let mut result = result;
        result.status = StepStatus::Completed;
        assert!(result.is_complete());

        result.status = StepStatus::Failed;
        assert!(result.is_failed());

        result.status = StepStatus::Skipped;
        assert!(result.is_complete());
    }

    #[tokio::test]
    async fn test_memory_store() {
        let store = MemoryExecutionStore::new();

        let state = ExecutionState::new("exec1", &["s1".to_string()]);
        store.save(&state).await.unwrap();

        let loaded = store.load("exec1").await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().execution_id, "exec1");

        let ids = store.list().await.unwrap();
        assert_eq!(ids.len(), 1);

        store.delete("exec1").await.unwrap();
        assert!(store.load("exec1").await.unwrap().is_none());
    }
}
