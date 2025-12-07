//! Agent Trajectory Recording
//!
//! Structured logging for agent operations to enable debugging,
//! metrics collection, and execution replay.
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents::trajectory::{TrajectoryRecorder, StepType};
//!
//! let mut recorder = TrajectoryRecorder::new("my-agent", "task-123");
//!
//! // Record LLM call
//! recorder.start_step();
//! // ... LLM call happens ...
//! recorder.record_llm_call("gpt-4", Some(100), Some(50));
//!
//! // Record tool call
//! recorder.start_step();
//! // ... tool call happens ...
//! recorder.record_tool_call("search", r#"{"query": "rust"}"#, true, None);
//!
//! // Complete the trajectory
//! let trajectory = recorder.complete("Task completed successfully");
//! println!("Total duration: {}ms", trajectory.total_duration_ms);
//! ```

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Type of step in a trajectory
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepType {
    /// Task started
    TaskStart,
    /// LLM/Model call
    LlmCall,
    /// Tool invocation
    ToolCall,
    /// Tool result received
    ToolResult,
    /// Handoff to another agent
    Handoff,
    /// Planning step
    Planning,
    /// Memory operation
    Memory,
    /// Task completed successfully
    TaskComplete,
    /// Task failed
    TaskFailed,
}

/// A single step in an agent's execution trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    /// Type of this step
    pub step_type: StepType,
    /// Agent that performed this step
    pub agent_name: String,
    /// Duration of this step in milliseconds
    pub duration_ms: u64,
    /// Additional details (JSON)
    pub details: serde_json::Value,
    /// Whether this step succeeded
    pub success: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Timestamp of this step
    pub timestamp: i64,
}

impl TrajectoryStep {
    /// Create a new trajectory step
    pub fn new(
        step_type: StepType,
        agent_name: impl Into<String>,
        duration_ms: u64,
        details: serde_json::Value,
        success: bool,
    ) -> Self {
        Self {
            step_type,
            agent_name: agent_name.into(),
            duration_ms,
            details,
            success,
            error: None,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Add error message
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }
}

/// Recorder for tracking agent execution trajectory
#[derive(Debug)]
pub struct TrajectoryRecorder {
    agent_name: String,
    task_id: String,
    steps: Vec<TrajectoryStep>,
    start_time: Instant,
    current_step_start: Option<Instant>,
    metadata: serde_json::Value,
}

impl TrajectoryRecorder {
    /// Create a new trajectory recorder
    pub fn new(agent_name: impl Into<String>, task_id: impl Into<String>) -> Self {
        let agent = agent_name.into();
        let task = task_id.into();

        tracing::info!(
            target: "trajectory",
            agent = %agent,
            task_id = %task,
            "Task started"
        );

        Self {
            agent_name: agent,
            task_id: task,
            steps: Vec::new(),
            start_time: Instant::now(),
            current_step_start: None,
            metadata: serde_json::json!({}),
        }
    }

    /// Set metadata for the trajectory
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Start timing a new step
    pub fn start_step(&mut self) {
        self.current_step_start = Some(Instant::now());
    }

    /// Get elapsed time since step started
    fn step_duration(&self) -> u64 {
        self.current_step_start
            .map(|s| s.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    /// Record an LLM call
    pub fn record_llm_call(
        &mut self,
        model: &str,
        tokens_in: Option<u32>,
        tokens_out: Option<u32>,
    ) {
        let duration = self.step_duration();

        let step = TrajectoryStep::new(
            StepType::LlmCall,
            &self.agent_name,
            duration,
            serde_json::json!({
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }),
            true,
        );

        tracing::debug!(
            target: "trajectory",
            agent = %self.agent_name,
            model = %model,
            duration_ms = %duration,
            tokens_in = ?tokens_in,
            tokens_out = ?tokens_out,
            "LLM call completed"
        );

        self.steps.push(step);
        self.current_step_start = None;
    }

    /// Record an LLM call failure
    pub fn record_llm_failure(&mut self, model: &str, error: &str) {
        let duration = self.step_duration();

        let step = TrajectoryStep::new(
            StepType::LlmCall,
            &self.agent_name,
            duration,
            serde_json::json!({
                "model": model,
            }),
            false,
        )
        .with_error(error);

        tracing::warn!(
            target: "trajectory",
            agent = %self.agent_name,
            model = %model,
            error = %error,
            "LLM call failed"
        );

        self.steps.push(step);
        self.current_step_start = None;
    }

    /// Record a tool call
    pub fn record_tool_call(
        &mut self,
        tool_name: &str,
        arguments: &str,
        success: bool,
        error: Option<&str>,
    ) {
        let duration = self.step_duration();

        // Truncate arguments for logging
        let args_preview = if arguments.len() > 200 {
            format!("{}...", &arguments[..200])
        } else {
            arguments.to_string()
        };

        let mut step = TrajectoryStep::new(
            StepType::ToolCall,
            &self.agent_name,
            duration,
            serde_json::json!({
                "tool": tool_name,
                "arguments_preview": args_preview,
            }),
            success,
        );

        if let Some(err) = error {
            step = step.with_error(err);
        }

        if success {
            tracing::debug!(
                target: "trajectory",
                agent = %self.agent_name,
                tool = %tool_name,
                duration_ms = %duration,
                "Tool call succeeded"
            );
        } else {
            tracing::warn!(
                target: "trajectory",
                agent = %self.agent_name,
                tool = %tool_name,
                error = ?error,
                "Tool call failed"
            );
        }

        self.steps.push(step);
        self.current_step_start = None;
    }

    /// Record a handoff to another agent
    pub fn record_handoff(&mut self, target_agent: &str, reason: &str) {
        let step = TrajectoryStep::new(
            StepType::Handoff,
            &self.agent_name,
            0,
            serde_json::json!({
                "target_agent": target_agent,
                "reason": reason,
            }),
            true,
        );

        tracing::info!(
            target: "trajectory",
            agent = %self.agent_name,
            target = %target_agent,
            reason = %reason,
            "Agent handoff"
        );

        self.steps.push(step);
    }

    /// Record a planning step
    pub fn record_planning(&mut self, plan_steps: usize, goal: &str) {
        let duration = self.step_duration();

        let step = TrajectoryStep::new(
            StepType::Planning,
            &self.agent_name,
            duration,
            serde_json::json!({
                "plan_steps": plan_steps,
                "goal": goal,
            }),
            true,
        );

        tracing::debug!(
            target: "trajectory",
            agent = %self.agent_name,
            plan_steps = %plan_steps,
            "Plan created"
        );

        self.steps.push(step);
        self.current_step_start = None;
    }

    /// Record a memory operation
    pub fn record_memory(&mut self, operation: &str, key: &str, success: bool) {
        let duration = self.step_duration();

        let step = TrajectoryStep::new(
            StepType::Memory,
            &self.agent_name,
            duration,
            serde_json::json!({
                "operation": operation,
                "key": key,
            }),
            success,
        );

        self.steps.push(step);
        self.current_step_start = None;
    }

    /// Record a custom step
    pub fn record_custom(
        &mut self,
        step_type: StepType,
        details: serde_json::Value,
        success: bool,
        error: Option<&str>,
    ) {
        let duration = self.step_duration();

        let mut step = TrajectoryStep::new(step_type, &self.agent_name, duration, details, success);

        if let Some(err) = error {
            step = step.with_error(err);
        }

        self.steps.push(step);
        self.current_step_start = None;
    }

    /// Complete the task successfully
    pub fn complete(mut self, result_preview: &str) -> Trajectory {
        let total_duration = self.start_time.elapsed();

        // Truncate result for storage
        let preview = if result_preview.len() > 500 {
            format!("{}...", &result_preview[..500])
        } else {
            result_preview.to_string()
        };

        self.steps.push(TrajectoryStep::new(
            StepType::TaskComplete,
            &self.agent_name,
            total_duration.as_millis() as u64,
            serde_json::json!({
                "result_preview": preview,
            }),
            true,
        ));

        let trajectory = Trajectory {
            agent_name: self.agent_name.clone(),
            task_id: self.task_id.clone(),
            total_duration_ms: total_duration.as_millis() as u64,
            steps: self.steps,
            success: true,
            metadata: self.metadata,
        };

        tracing::info!(
            target: "trajectory",
            agent = %trajectory.agent_name,
            task_id = %trajectory.task_id,
            duration_ms = %trajectory.total_duration_ms,
            steps = %trajectory.steps.len(),
            "Task completed successfully"
        );

        trajectory
    }

    /// Fail the task
    pub fn fail(mut self, error: &str) -> Trajectory {
        let total_duration = self.start_time.elapsed();

        self.steps.push(
            TrajectoryStep::new(
                StepType::TaskFailed,
                &self.agent_name,
                total_duration.as_millis() as u64,
                serde_json::json!({}),
                false,
            )
            .with_error(error),
        );

        let trajectory = Trajectory {
            agent_name: self.agent_name.clone(),
            task_id: self.task_id.clone(),
            total_duration_ms: total_duration.as_millis() as u64,
            steps: self.steps,
            success: false,
            metadata: self.metadata,
        };

        tracing::error!(
            target: "trajectory",
            agent = %trajectory.agent_name,
            task_id = %trajectory.task_id,
            error = %error,
            "Task failed"
        );

        trajectory
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Get elapsed time since start
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}

/// Complete trajectory for a task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Agent that executed the task
    pub agent_name: String,
    /// Task identifier
    pub task_id: String,
    /// Total execution time in milliseconds
    pub total_duration_ms: u64,
    /// All steps in the execution
    pub steps: Vec<TrajectoryStep>,
    /// Whether the task succeeded
    pub success: bool,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl Trajectory {
    /// Get count of tool calls in trajectory
    pub fn tool_call_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.step_type == StepType::ToolCall)
            .count()
    }

    /// Get count of LLM calls in trajectory
    pub fn llm_call_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.step_type == StepType::LlmCall)
            .count()
    }

    /// Get count of failed steps
    pub fn failed_step_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.success).count()
    }

    /// Get total tokens used (if tracked)
    pub fn total_tokens(&self) -> (u32, u32) {
        let mut tokens_in = 0u32;
        let mut tokens_out = 0u32;

        for step in &self.steps {
            if step.step_type == StepType::LlmCall {
                if let Some(t) = step.details.get("tokens_in").and_then(|v| v.as_u64()) {
                    tokens_in += t as u32;
                }
                if let Some(t) = step.details.get("tokens_out").and_then(|v| v.as_u64()) {
                    tokens_out += t as u32;
                }
            }
        }

        (tokens_in, tokens_out)
    }

    /// Get all tool names used
    pub fn tools_used(&self) -> Vec<String> {
        self.steps
            .iter()
            .filter(|s| s.step_type == StepType::ToolCall)
            .filter_map(|s| s.details.get("tool").and_then(|v| v.as_str()))
            .map(|s| s.to_string())
            .collect()
    }

    /// Get steps by type
    pub fn steps_by_type(&self, step_type: StepType) -> Vec<&TrajectoryStep> {
        self.steps
            .iter()
            .filter(|s| s.step_type == step_type)
            .collect()
    }

    /// Log the full trajectory as structured JSON
    pub fn log_json(&self) {
        if let Ok(json) = serde_json::to_string(self) {
            tracing::info!(target: "trajectory_json", "{}", json);
        }
    }

    /// Convert to pretty JSON
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Get summary statistics
    pub fn summary(&self) -> TrajectorySummary {
        let (tokens_in, tokens_out) = self.total_tokens();

        TrajectorySummary {
            agent_name: self.agent_name.clone(),
            task_id: self.task_id.clone(),
            success: self.success,
            total_duration_ms: self.total_duration_ms,
            step_count: self.steps.len(),
            llm_calls: self.llm_call_count(),
            tool_calls: self.tool_call_count(),
            failed_steps: self.failed_step_count(),
            tokens_in,
            tokens_out,
        }
    }
}

/// Summary of a trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectorySummary {
    pub agent_name: String,
    pub task_id: String,
    pub success: bool,
    pub total_duration_ms: u64,
    pub step_count: usize,
    pub llm_calls: usize,
    pub tool_calls: usize,
    pub failed_steps: usize,
    pub tokens_in: u32,
    pub tokens_out: u32,
}

/// Storage for trajectories
#[derive(Default)]
pub struct TrajectoryStore {
    trajectories: std::sync::RwLock<Vec<Trajectory>>,
    max_size: usize,
}

impl TrajectoryStore {
    /// Create a new trajectory store
    pub fn new(max_size: usize) -> Self {
        Self {
            trajectories: std::sync::RwLock::new(Vec::new()),
            max_size,
        }
    }

    /// Store a trajectory
    pub fn store(&self, trajectory: Trajectory) {
        let mut trajectories = self.trajectories.write().unwrap();

        if trajectories.len() >= self.max_size {
            trajectories.remove(0);
        }

        trajectories.push(trajectory);
    }

    /// Get all trajectories
    pub fn all(&self) -> Vec<Trajectory> {
        self.trajectories.read().unwrap().clone()
    }

    /// Get trajectories by agent
    pub fn by_agent(&self, agent_name: &str) -> Vec<Trajectory> {
        self.trajectories
            .read()
            .unwrap()
            .iter()
            .filter(|t| t.agent_name == agent_name)
            .cloned()
            .collect()
    }

    /// Get trajectory by task ID
    pub fn by_task_id(&self, task_id: &str) -> Option<Trajectory> {
        self.trajectories
            .read()
            .unwrap()
            .iter()
            .find(|t| t.task_id == task_id)
            .cloned()
    }

    /// Get failed trajectories
    pub fn failed(&self) -> Vec<Trajectory> {
        self.trajectories
            .read()
            .unwrap()
            .iter()
            .filter(|t| !t.success)
            .cloned()
            .collect()
    }

    /// Get count
    pub fn len(&self) -> usize {
        self.trajectories.read().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.trajectories.read().unwrap().is_empty()
    }

    /// Clear all trajectories
    pub fn clear(&self) {
        self.trajectories.write().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_trajectory_recorder_basic() {
        let mut recorder = TrajectoryRecorder::new("test-agent", "task-1");

        recorder.start_step();
        std::thread::sleep(Duration::from_millis(10));
        recorder.record_llm_call("gpt-4", Some(100), Some(50));

        recorder.start_step();
        recorder.record_tool_call("search", r#"{"query": "test"}"#, true, None);

        let trajectory = recorder.complete("Done!");

        assert!(trajectory.success);
        assert_eq!(trajectory.agent_name, "test-agent");
        assert_eq!(trajectory.task_id, "task-1");
        assert_eq!(trajectory.llm_call_count(), 1);
        assert_eq!(trajectory.tool_call_count(), 1);
        assert!(trajectory.total_duration_ms >= 10);
    }

    #[test]
    fn test_trajectory_failure() {
        let recorder = TrajectoryRecorder::new("test-agent", "task-2");
        let trajectory = recorder.fail("Something went wrong");

        assert!(!trajectory.success);
        assert_eq!(trajectory.failed_step_count(), 1);
    }

    #[test]
    fn test_trajectory_tokens() {
        let mut recorder = TrajectoryRecorder::new("test-agent", "task-3");

        recorder.start_step();
        recorder.record_llm_call("gpt-4", Some(100), Some(50));

        recorder.start_step();
        recorder.record_llm_call("gpt-4", Some(200), Some(100));

        let trajectory = recorder.complete("Done");

        let (tokens_in, tokens_out) = trajectory.total_tokens();
        assert_eq!(tokens_in, 300);
        assert_eq!(tokens_out, 150);
    }

    #[test]
    fn test_trajectory_tools_used() {
        let mut recorder = TrajectoryRecorder::new("test-agent", "task-4");

        recorder.start_step();
        recorder.record_tool_call("search", "{}", true, None);

        recorder.start_step();
        recorder.record_tool_call("calculator", "{}", true, None);

        recorder.start_step();
        recorder.record_tool_call("search", "{}", true, None);

        let trajectory = recorder.complete("Done");

        let tools = trajectory.tools_used();
        assert_eq!(tools.len(), 3);
        assert!(tools.contains(&"search".to_string()));
        assert!(tools.contains(&"calculator".to_string()));
    }

    #[test]
    fn test_trajectory_handoff() {
        let mut recorder = TrajectoryRecorder::new("agent-a", "task-5");

        recorder.record_handoff("agent-b", "Better suited for this task");

        let trajectory = recorder.complete("Handed off");

        let handoffs = trajectory.steps_by_type(StepType::Handoff);
        assert_eq!(handoffs.len(), 1);
    }

    #[test]
    fn test_trajectory_summary() {
        let mut recorder = TrajectoryRecorder::new("test-agent", "task-6");

        recorder.start_step();
        recorder.record_llm_call("gpt-4", Some(100), Some(50));

        recorder.start_step();
        recorder.record_tool_call("search", "{}", true, None);

        recorder.start_step();
        recorder.record_tool_call("failed_tool", "{}", false, Some("Error"));

        let trajectory = recorder.complete("Done");
        let summary = trajectory.summary();

        assert_eq!(summary.llm_calls, 1);
        assert_eq!(summary.tool_calls, 2);
        assert_eq!(summary.failed_steps, 1);
        assert_eq!(summary.tokens_in, 100);
        assert_eq!(summary.tokens_out, 50);
    }

    #[test]
    fn test_trajectory_store() {
        let store = TrajectoryStore::new(10);

        let recorder1 = TrajectoryRecorder::new("agent-1", "task-1");
        store.store(recorder1.complete("Done 1"));

        let recorder2 = TrajectoryRecorder::new("agent-2", "task-2");
        store.store(recorder2.complete("Done 2"));

        assert_eq!(store.len(), 2);
        assert_eq!(store.by_agent("agent-1").len(), 1);
        assert!(store.by_task_id("task-1").is_some());
    }

    #[test]
    fn test_trajectory_store_max_size() {
        let store = TrajectoryStore::new(2);

        for i in 0..5 {
            let recorder = TrajectoryRecorder::new("agent", &format!("task-{}", i));
            store.store(recorder.complete("Done"));
        }

        assert_eq!(store.len(), 2);
        // First 3 should have been evicted
        assert!(store.by_task_id("task-0").is_none());
        assert!(store.by_task_id("task-1").is_none());
        assert!(store.by_task_id("task-2").is_none());
        // Last 2 should exist
        assert!(store.by_task_id("task-3").is_some());
        assert!(store.by_task_id("task-4").is_some());
    }
}
