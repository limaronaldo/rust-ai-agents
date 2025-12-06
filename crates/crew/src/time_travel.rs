//! Time Travel / State History
//!
//! Provides the ability to replay, fork, and debug workflow executions:
//! - Full execution history with state snapshots at each step
//! - Replay from any point in history
//! - Fork execution to explore alternative paths
//! - Compare states across different branches
//! - Debug mode with step-by-step execution
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_crew::time_travel::{TimeTravelGraph, HistoryBrowser};
//!
//! // Wrap a graph with time travel capabilities
//! let tt_graph = TimeTravelGraph::new(graph);
//!
//! // Execute and record full history
//! let execution = tt_graph.execute(initial_state).await?;
//!
//! // Browse history
//! let browser = execution.browser();
//! println!("Steps: {}", browser.len());
//!
//! // Replay from step 3
//! let forked = tt_graph.replay_from(&execution, 3).await?;
//!
//! // Fork and modify state at step 5
//! let mut state_at_5 = browser.state_at(5)?;
//! state_at_5.set("override", true);
//! let branched = tt_graph.fork_from(&execution, 5, state_at_5).await?;
//!
//! // Compare branches
//! let diff = browser.diff(5, branched.browser(), 5)?;
//! ```

use crate::graph::{Graph, GraphResult, GraphState, GraphStatus, END};
use chrono::{DateTime, Utc};
use rust_ai_agents_core::errors::CrewError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A single step in execution history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryStep {
    /// Step number (0-indexed)
    pub step: usize,
    /// Node that was executed
    pub node_id: String,
    /// State before node execution
    pub state_before: GraphState,
    /// State after node execution
    pub state_after: GraphState,
    /// Execution duration in microseconds
    pub duration_us: u64,
    /// Timestamp when this step executed
    pub timestamp: DateTime<Utc>,
    /// Any error that occurred
    pub error: Option<String>,
}

impl HistoryStep {
    /// Get the state changes made by this step
    pub fn changes(&self) -> StateChanges {
        diff_states(&self.state_before, &self.state_after)
    }
}

/// Represents changes between two states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateChanges {
    /// Keys that were added
    pub added: Vec<String>,
    /// Keys that were removed
    pub removed: Vec<String>,
    /// Keys that were modified (old_value, new_value)
    pub modified: HashMap<String, (serde_json::Value, serde_json::Value)>,
}

impl StateChanges {
    /// Check if there are any changes
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }

    /// Get a summary of changes
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.added.is_empty() {
            parts.push(format!("+{} added", self.added.len()));
        }
        if !self.removed.is_empty() {
            parts.push(format!("-{} removed", self.removed.len()));
        }
        if !self.modified.is_empty() {
            parts.push(format!("~{} modified", self.modified.len()));
        }
        if parts.is_empty() {
            "no changes".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Compare two states and return their differences
pub fn diff_states(before: &GraphState, after: &GraphState) -> StateChanges {
    let mut changes = StateChanges {
        added: Vec::new(),
        removed: Vec::new(),
        modified: HashMap::new(),
    };

    let before_obj = before.data.as_object();
    let after_obj = after.data.as_object();

    if let (Some(b), Some(a)) = (before_obj, after_obj) {
        // Find added and modified
        for (key, new_val) in a {
            match b.get(key) {
                None => changes.added.push(key.clone()),
                Some(old_val) if old_val != new_val => {
                    changes
                        .modified
                        .insert(key.clone(), (old_val.clone(), new_val.clone()));
                }
                _ => {}
            }
        }

        // Find removed
        for key in b.keys() {
            if !a.contains_key(key) {
                changes.removed.push(key.clone());
            }
        }
    }

    changes
}

/// Full execution history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionHistory {
    /// Unique execution ID
    pub id: String,
    /// Graph ID
    pub graph_id: String,
    /// All steps in order
    pub steps: Vec<HistoryStep>,
    /// Final result
    pub result: GraphResult,
    /// Branch ID (for forked executions)
    pub branch_id: Option<String>,
    /// Parent execution ID (if forked)
    pub parent_id: Option<String>,
    /// Step in parent where fork occurred
    pub fork_step: Option<usize>,
    /// Execution start time
    pub started_at: DateTime<Utc>,
    /// Execution end time
    pub ended_at: DateTime<Utc>,
}

impl ExecutionHistory {
    /// Create a browser for this history
    pub fn browser(&self) -> HistoryBrowser<'_> {
        HistoryBrowser { history: self }
    }

    /// Get total execution time in milliseconds
    pub fn duration_ms(&self) -> i64 {
        self.ended_at
            .signed_duration_since(self.started_at)
            .num_milliseconds()
    }

    /// Check if this is a forked execution
    pub fn is_fork(&self) -> bool {
        self.parent_id.is_some()
    }
}

/// Browser for navigating execution history
#[derive(Debug)]
pub struct HistoryBrowser<'a> {
    history: &'a ExecutionHistory,
}

impl<'a> HistoryBrowser<'a> {
    /// Get number of steps
    pub fn len(&self) -> usize {
        self.history.steps.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.history.steps.is_empty()
    }

    /// Get step at index
    pub fn step_at(&self, index: usize) -> Option<&HistoryStep> {
        self.history.steps.get(index)
    }

    /// Get state at a specific step (after execution)
    pub fn state_at(&self, index: usize) -> Result<GraphState, CrewError> {
        self.history
            .steps
            .get(index)
            .map(|s| s.state_after.clone())
            .ok_or_else(|| CrewError::TaskNotFound(format!("Step {} not found", index)))
    }

    /// Get state before a specific step
    pub fn state_before(&self, index: usize) -> Result<GraphState, CrewError> {
        self.history
            .steps
            .get(index)
            .map(|s| s.state_before.clone())
            .ok_or_else(|| CrewError::TaskNotFound(format!("Step {} not found", index)))
    }

    /// Get all nodes visited in order
    pub fn visited_nodes(&self) -> Vec<&str> {
        self.history
            .steps
            .iter()
            .map(|s| s.node_id.as_str())
            .collect()
    }

    /// Find steps where a specific key changed
    pub fn find_key_changes(&self, key: &str) -> Vec<usize> {
        self.history
            .steps
            .iter()
            .enumerate()
            .filter(|(_, step)| {
                let changes = step.changes();
                changes.added.contains(&key.to_string())
                    || changes.removed.contains(&key.to_string())
                    || changes.modified.contains_key(key)
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Find steps that executed a specific node
    pub fn find_node_executions(&self, node_id: &str) -> Vec<usize> {
        self.history
            .steps
            .iter()
            .enumerate()
            .filter(|(_, step)| step.node_id == node_id)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get a summary of the execution
    pub fn summary(&self) -> ExecutionSummary {
        let mut node_counts: HashMap<String, usize> = HashMap::new();
        let mut total_duration_us = 0u64;

        for step in &self.history.steps {
            *node_counts.entry(step.node_id.clone()).or_insert(0) += 1;
            total_duration_us += step.duration_us;
        }

        ExecutionSummary {
            total_steps: self.history.steps.len(),
            unique_nodes: node_counts.len(),
            node_execution_counts: node_counts,
            total_duration_us,
            status: self.history.result.status,
            is_fork: self.history.is_fork(),
        }
    }

    /// Compare this execution with another at specific steps
    pub fn diff(
        &self,
        self_step: usize,
        other: &HistoryBrowser,
        other_step: usize,
    ) -> Result<StateChanges, CrewError> {
        let self_state = self.state_at(self_step)?;
        let other_state = other.state_at(other_step)?;
        Ok(diff_states(&self_state, &other_state))
    }

    /// Get execution timeline as a formatted string
    pub fn timeline(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Execution {} Timeline:", self.history.id));
        lines.push("─".repeat(50));

        for step in &self.history.steps {
            let changes = step.changes();
            let status = if step.error.is_some() { "✗" } else { "✓" };
            lines.push(format!(
                "{} Step {}: {} ({:.2}ms) - {}",
                status,
                step.step,
                step.node_id,
                step.duration_us as f64 / 1000.0,
                changes.summary()
            ));
        }

        lines.push("─".repeat(50));
        lines.push(format!("Status: {:?}", self.history.result.status));
        lines.join("\n")
    }
}

/// Summary of an execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSummary {
    /// Total number of steps
    pub total_steps: usize,
    /// Number of unique nodes executed
    pub unique_nodes: usize,
    /// How many times each node was executed
    pub node_execution_counts: HashMap<String, usize>,
    /// Total execution time in microseconds
    pub total_duration_us: u64,
    /// Final status
    pub status: GraphStatus,
    /// Whether this was a forked execution
    pub is_fork: bool,
}

/// Storage for execution histories
#[async_trait::async_trait]
pub trait HistoryStore: Send + Sync {
    /// Save an execution history
    async fn save(&self, history: ExecutionHistory) -> Result<(), CrewError>;
    /// Load an execution history by ID
    async fn load(&self, id: &str) -> Result<Option<ExecutionHistory>, CrewError>;
    /// List all executions for a graph
    async fn list_for_graph(&self, graph_id: &str) -> Result<Vec<String>, CrewError>;
    /// List all forks of an execution
    async fn list_forks(&self, parent_id: &str) -> Result<Vec<String>, CrewError>;
    /// Delete an execution history
    async fn delete(&self, id: &str) -> Result<(), CrewError>;
}

/// In-memory history store
#[derive(Default)]
pub struct InMemoryHistoryStore {
    histories: RwLock<HashMap<String, ExecutionHistory>>,
}

impl InMemoryHistoryStore {
    /// Create a new in-memory store
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl HistoryStore for InMemoryHistoryStore {
    async fn save(&self, history: ExecutionHistory) -> Result<(), CrewError> {
        self.histories
            .write()
            .await
            .insert(history.id.clone(), history);
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<ExecutionHistory>, CrewError> {
        Ok(self.histories.read().await.get(id).cloned())
    }

    async fn list_for_graph(&self, graph_id: &str) -> Result<Vec<String>, CrewError> {
        Ok(self
            .histories
            .read()
            .await
            .values()
            .filter(|h| h.graph_id == graph_id)
            .map(|h| h.id.clone())
            .collect())
    }

    async fn list_forks(&self, parent_id: &str) -> Result<Vec<String>, CrewError> {
        Ok(self
            .histories
            .read()
            .await
            .values()
            .filter(|h| h.parent_id.as_deref() == Some(parent_id))
            .map(|h| h.id.clone())
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<(), CrewError> {
        self.histories.write().await.remove(id);
        Ok(())
    }
}

/// Graph wrapper with time travel capabilities
pub struct TimeTravelGraph {
    graph: Arc<Graph>,
    store: Arc<dyn HistoryStore>,
}

impl TimeTravelGraph {
    /// Create a new time travel graph with in-memory storage
    pub fn new(graph: Graph) -> Self {
        Self {
            graph: Arc::new(graph),
            store: Arc::new(InMemoryHistoryStore::new()),
        }
    }

    /// Create with custom history store
    pub fn with_store(graph: Graph, store: Arc<dyn HistoryStore>) -> Self {
        Self {
            graph: Arc::new(graph),
            store,
        }
    }

    /// Get the underlying graph
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Execute the graph and record full history
    pub async fn execute(&self, initial_state: GraphState) -> Result<ExecutionHistory, CrewError> {
        let execution_id = uuid::Uuid::new_v4().to_string();
        let started_at = Utc::now();
        let mut steps = Vec::new();
        let mut state = initial_state;
        state.metadata.started_at = Some(started_at);
        state.metadata.iterations = 0;

        let mut current_node = self.graph.entry_node.clone();
        let mut step_num = 0;

        let result = loop {
            // Check max iterations
            if state.metadata.iterations >= self.graph.config.max_iterations {
                break GraphResult {
                    state: state.clone(),
                    status: GraphStatus::MaxIterations,
                    error: Some(format!(
                        "Hit maximum iterations: {}",
                        self.graph.config.max_iterations
                    )),
                };
            }

            // Check for END node
            if current_node == END {
                state.metadata.execution_time_ms = started_at
                    .signed_duration_since(Utc::now())
                    .num_milliseconds()
                    .unsigned_abs();
                break GraphResult {
                    state: state.clone(),
                    status: GraphStatus::Success,
                    error: None,
                };
            }

            // Get node
            let node = match self.graph.nodes.get(&current_node) {
                Some(n) => n,
                None => {
                    break GraphResult {
                        state: state.clone(),
                        status: GraphStatus::Failed,
                        error: Some(format!("Node not found: {}", current_node)),
                    };
                }
            };

            // Record state before
            let state_before = state.clone();
            let step_start = std::time::Instant::now();

            // Execute node
            state.metadata.visited_nodes.push(current_node.clone());
            state.metadata.iterations += 1;

            let (new_state, error) = match node.executor.call(state).await {
                Ok(s) => (s, None),
                Err(e) => {
                    let err_msg = e.to_string();
                    // Record the failed step
                    steps.push(HistoryStep {
                        step: step_num,
                        node_id: current_node.clone(),
                        state_before: state_before.clone(),
                        state_after: state_before.clone(),
                        duration_us: step_start.elapsed().as_micros() as u64,
                        timestamp: Utc::now(),
                        error: Some(err_msg.clone()),
                    });

                    break GraphResult {
                        state: state_before,
                        status: GraphStatus::Failed,
                        error: Some(err_msg),
                    };
                }
            };

            let duration_us = step_start.elapsed().as_micros() as u64;

            // Record step
            steps.push(HistoryStep {
                step: step_num,
                node_id: current_node.clone(),
                state_before,
                state_after: new_state.clone(),
                duration_us,
                timestamp: Utc::now(),
                error,
            });

            state = new_state;
            step_num += 1;

            // Find next node
            current_node = self.graph.find_next_node(&current_node, &state)?;
        };

        let history = ExecutionHistory {
            id: execution_id,
            graph_id: self.graph.id.clone(),
            steps,
            result,
            branch_id: None,
            parent_id: None,
            fork_step: None,
            started_at,
            ended_at: Utc::now(),
        };

        // Store history
        self.store.save(history.clone()).await?;

        Ok(history)
    }

    /// Replay execution from a specific step
    pub async fn replay_from(
        &self,
        history: &ExecutionHistory,
        from_step: usize,
    ) -> Result<ExecutionHistory, CrewError> {
        // Get state at the specified step
        let state = history
            .steps
            .get(from_step)
            .map(|s| s.state_after.clone())
            .ok_or_else(|| CrewError::TaskNotFound(format!("Step {} not found", from_step)))?;

        // Get the next node after that step
        let next_node = if from_step + 1 < history.steps.len() {
            history.steps[from_step + 1].node_id.clone()
        } else {
            // Re-evaluate the conditional edge
            let current_node = &history.steps[from_step].node_id;
            self.graph.find_next_node(current_node, &state)?
        };

        self.execute_from_state(state, &next_node, Some(history.id.clone()), Some(from_step))
            .await
    }

    /// Fork execution from a specific step with modified state
    pub async fn fork_from(
        &self,
        history: &ExecutionHistory,
        from_step: usize,
        modified_state: GraphState,
    ) -> Result<ExecutionHistory, CrewError> {
        // Get the next node after that step
        let next_node = if from_step + 1 < history.steps.len() {
            history.steps[from_step + 1].node_id.clone()
        } else {
            let current_node = &history.steps[from_step].node_id;
            self.graph.find_next_node(current_node, &modified_state)?
        };

        self.execute_from_state(
            modified_state,
            &next_node,
            Some(history.id.clone()),
            Some(from_step),
        )
        .await
    }

    /// Execute from a specific state and node (internal helper)
    async fn execute_from_state(
        &self,
        mut state: GraphState,
        start_node: &str,
        parent_id: Option<String>,
        fork_step: Option<usize>,
    ) -> Result<ExecutionHistory, CrewError> {
        let execution_id = uuid::Uuid::new_v4().to_string();
        let branch_id = if parent_id.is_some() {
            Some(uuid::Uuid::new_v4().to_string())
        } else {
            None
        };
        let started_at = Utc::now();
        let mut steps = Vec::new();
        state.metadata.started_at = Some(started_at);

        let mut current_node = start_node.to_string();
        let mut step_num = 0;

        let result = loop {
            if state.metadata.iterations >= self.graph.config.max_iterations {
                break GraphResult {
                    state: state.clone(),
                    status: GraphStatus::MaxIterations,
                    error: Some(format!(
                        "Hit maximum iterations: {}",
                        self.graph.config.max_iterations
                    )),
                };
            }

            if current_node == END {
                break GraphResult {
                    state: state.clone(),
                    status: GraphStatus::Success,
                    error: None,
                };
            }

            let node = match self.graph.nodes.get(&current_node) {
                Some(n) => n,
                None => {
                    break GraphResult {
                        state: state.clone(),
                        status: GraphStatus::Failed,
                        error: Some(format!("Node not found: {}", current_node)),
                    };
                }
            };

            let state_before = state.clone();
            let step_start = std::time::Instant::now();

            state.metadata.visited_nodes.push(current_node.clone());
            state.metadata.iterations += 1;

            let (new_state, error) = match node.executor.call(state).await {
                Ok(s) => (s, None),
                Err(e) => {
                    let err_msg = e.to_string();
                    steps.push(HistoryStep {
                        step: step_num,
                        node_id: current_node.clone(),
                        state_before: state_before.clone(),
                        state_after: state_before.clone(),
                        duration_us: step_start.elapsed().as_micros() as u64,
                        timestamp: Utc::now(),
                        error: Some(err_msg.clone()),
                    });

                    break GraphResult {
                        state: state_before,
                        status: GraphStatus::Failed,
                        error: Some(err_msg),
                    };
                }
            };

            let duration_us = step_start.elapsed().as_micros() as u64;

            steps.push(HistoryStep {
                step: step_num,
                node_id: current_node.clone(),
                state_before,
                state_after: new_state.clone(),
                duration_us,
                timestamp: Utc::now(),
                error,
            });

            state = new_state;
            step_num += 1;
            current_node = self.graph.find_next_node(&current_node, &state)?;
        };

        let history = ExecutionHistory {
            id: execution_id,
            graph_id: self.graph.id.clone(),
            steps,
            result,
            branch_id,
            parent_id,
            fork_step,
            started_at,
            ended_at: Utc::now(),
        };

        self.store.save(history.clone()).await?;

        Ok(history)
    }

    /// Load an execution history by ID
    pub async fn load_history(&self, id: &str) -> Result<Option<ExecutionHistory>, CrewError> {
        self.store.load(id).await
    }

    /// List all executions for this graph
    pub async fn list_executions(&self) -> Result<Vec<String>, CrewError> {
        self.store.list_for_graph(&self.graph.id).await
    }

    /// List all forks of an execution
    pub async fn list_forks(&self, parent_id: &str) -> Result<Vec<String>, CrewError> {
        self.store.list_forks(parent_id).await
    }

    /// Compare two executions
    pub async fn compare_executions(
        &self,
        id1: &str,
        id2: &str,
    ) -> Result<ExecutionComparison, CrewError> {
        let h1 = self
            .store
            .load(id1)
            .await?
            .ok_or_else(|| CrewError::TaskNotFound(format!("Execution {} not found", id1)))?;

        let h2 = self
            .store
            .load(id2)
            .await?
            .ok_or_else(|| CrewError::TaskNotFound(format!("Execution {} not found", id2)))?;

        Ok(ExecutionComparison::new(&h1, &h2))
    }
}

/// Comparison between two executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionComparison {
    /// ID of first execution
    pub execution1_id: String,
    /// ID of second execution
    pub execution2_id: String,
    /// Steps that diverged (first differing step)
    pub divergence_step: Option<usize>,
    /// Number of steps in execution 1
    pub steps1: usize,
    /// Number of steps in execution 2
    pub steps2: usize,
    /// Status of execution 1
    pub status1: GraphStatus,
    /// Status of execution 2
    pub status2: GraphStatus,
    /// Final state differences
    pub final_state_diff: StateChanges,
    /// Whether they share a common ancestor (fork relationship)
    pub is_fork_comparison: bool,
}

impl ExecutionComparison {
    /// Create a comparison between two executions
    pub fn new(h1: &ExecutionHistory, h2: &ExecutionHistory) -> Self {
        let divergence_step = Self::find_divergence(h1, h2);
        let final_state_diff = diff_states(&h1.result.state, &h2.result.state);
        let is_fork_comparison =
            h1.parent_id == Some(h2.id.clone()) || h2.parent_id == Some(h1.id.clone());

        Self {
            execution1_id: h1.id.clone(),
            execution2_id: h2.id.clone(),
            divergence_step,
            steps1: h1.steps.len(),
            steps2: h2.steps.len(),
            status1: h1.result.status,
            status2: h2.result.status,
            final_state_diff,
            is_fork_comparison,
        }
    }

    fn find_divergence(h1: &ExecutionHistory, h2: &ExecutionHistory) -> Option<usize> {
        let min_len = h1.steps.len().min(h2.steps.len());
        for i in 0..min_len {
            if h1.steps[i].node_id != h2.steps[i].node_id {
                return Some(i);
            }
            // Also check if state diverged even with same node
            let diff = diff_states(&h1.steps[i].state_after, &h2.steps[i].state_after);
            if !diff.is_empty() {
                return Some(i);
            }
        }
        // If one is longer, divergence is at the end of the shorter
        if h1.steps.len() != h2.steps.len() {
            Some(min_len)
        } else {
            None // Identical executions
        }
    }

    /// Get a summary of the comparison
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Comparing {} vs {}",
            &self.execution1_id[..8],
            &self.execution2_id[..8]
        ));

        if self.is_fork_comparison {
            lines.push("  (Fork relationship detected)".to_string());
        }

        lines.push(format!("  Steps: {} vs {}", self.steps1, self.steps2));
        lines.push(format!(
            "  Status: {:?} vs {:?}",
            self.status1, self.status2
        ));

        if let Some(step) = self.divergence_step {
            lines.push(format!("  Diverged at step: {}", step));
        } else {
            lines.push("  No divergence (identical executions)".to_string());
        }

        lines.push(format!(
            "  Final state diff: {}",
            self.final_state_diff.summary()
        ));

        lines.join("\n")
    }
}

/// Debug mode for step-by-step execution
pub struct DebugSession {
    graph: Arc<Graph>,
    state: GraphState,
    current_node: String,
    step_count: usize,
    history: Vec<HistoryStep>,
    breakpoints: Vec<String>,
    started_at: DateTime<Utc>,
}

impl DebugSession {
    /// Create a new debug session
    pub fn new(graph: Arc<Graph>, initial_state: GraphState) -> Self {
        let entry_node = graph.entry_node.clone();
        Self {
            graph,
            state: initial_state,
            current_node: entry_node,
            step_count: 0,
            history: Vec::new(),
            breakpoints: Vec::new(),
            started_at: Utc::now(),
        }
    }

    /// Add a breakpoint at a node
    pub fn add_breakpoint(&mut self, node_id: impl Into<String>) {
        self.breakpoints.push(node_id.into());
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&mut self, node_id: &str) {
        self.breakpoints.retain(|n| n != node_id);
    }

    /// Get current state
    pub fn current_state(&self) -> &GraphState {
        &self.state
    }

    /// Get current node
    pub fn current_node(&self) -> &str {
        &self.current_node
    }

    /// Check if execution is finished
    pub fn is_finished(&self) -> bool {
        self.current_node == END
            || self.state.metadata.iterations >= self.graph.config.max_iterations
    }

    /// Execute a single step
    pub async fn step(&mut self) -> Result<StepResult, CrewError> {
        if self.is_finished() {
            return Ok(StepResult::Finished);
        }

        let node = self
            .graph
            .nodes
            .get(&self.current_node)
            .ok_or_else(|| CrewError::TaskNotFound(self.current_node.clone()))?;

        let state_before = self.state.clone();
        let step_start = std::time::Instant::now();

        self.state
            .metadata
            .visited_nodes
            .push(self.current_node.clone());
        self.state.metadata.iterations += 1;

        let new_state = node.executor.call(self.state.clone()).await?;
        let duration_us = step_start.elapsed().as_micros() as u64;

        let step = HistoryStep {
            step: self.step_count,
            node_id: self.current_node.clone(),
            state_before,
            state_after: new_state.clone(),
            duration_us,
            timestamp: Utc::now(),
            error: None,
        };

        self.history.push(step.clone());
        self.state = new_state;
        self.step_count += 1;

        let next_node = self.graph.find_next_node(&self.current_node, &self.state)?;
        self.current_node = next_node.clone();

        if next_node == END {
            Ok(StepResult::Finished)
        } else if self.breakpoints.contains(&next_node) {
            Ok(StepResult::Breakpoint(next_node))
        } else {
            Ok(StepResult::Continue(step))
        }
    }

    /// Run until a breakpoint or completion
    pub async fn run(&mut self) -> Result<StepResult, CrewError> {
        loop {
            let result = self.step().await?;
            match &result {
                StepResult::Continue(_) => continue,
                _ => return Ok(result),
            }
        }
    }

    /// Modify current state
    pub fn modify_state<F>(&mut self, f: F)
    where
        F: FnOnce(&mut GraphState),
    {
        f(&mut self.state);
    }

    /// Get execution history so far
    pub fn history(&self) -> &[HistoryStep] {
        &self.history
    }

    /// Convert to final ExecutionHistory
    pub fn into_history(self, status: GraphStatus, error: Option<String>) -> ExecutionHistory {
        ExecutionHistory {
            id: uuid::Uuid::new_v4().to_string(),
            graph_id: self.graph.id.clone(),
            steps: self.history,
            result: GraphResult {
                state: self.state,
                status,
                error,
            },
            branch_id: None,
            parent_id: None,
            fork_step: None,
            started_at: self.started_at,
            ended_at: Utc::now(),
        }
    }
}

/// Result of a single debug step
#[derive(Debug)]
pub enum StepResult {
    /// Continue to next step (includes the completed step)
    Continue(HistoryStep),
    /// Hit a breakpoint (includes the node ID)
    Breakpoint(String),
    /// Execution finished
    Finished,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn create_test_graph() -> Graph {
        GraphBuilder::new("test_graph")
            .add_node("step1", |mut state: GraphState| async move {
                state.set("step1_done", true);
                state.set("counter", 1);
                Ok(state)
            })
            .add_node("step2", |mut state: GraphState| async move {
                let counter: i32 = state.get("counter").unwrap_or(0);
                state.set("counter", counter + 1);
                state.set("step2_done", true);
                Ok(state)
            })
            .add_node("step3", |mut state: GraphState| async move {
                let counter: i32 = state.get("counter").unwrap_or(0);
                state.set("counter", counter + 1);
                state.set("step3_done", true);
                Ok(state)
            })
            .add_edge("step1", "step2")
            .add_edge("step2", "step3")
            .add_edge("step3", END)
            .set_entry("step1")
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_execute_with_history() {
        let graph = create_test_graph();
        let tt = TimeTravelGraph::new(graph);

        let history = tt.execute(GraphState::new()).await.unwrap();

        assert_eq!(history.steps.len(), 3);
        assert_eq!(history.result.status, GraphStatus::Success);
        assert_eq!(history.steps[0].node_id, "step1");
        assert_eq!(history.steps[1].node_id, "step2");
        assert_eq!(history.steps[2].node_id, "step3");
    }

    #[tokio::test]
    async fn test_history_browser() {
        let graph = create_test_graph();
        let tt = TimeTravelGraph::new(graph);

        let history = tt.execute(GraphState::new()).await.unwrap();
        let browser = history.browser();

        assert_eq!(browser.len(), 3);
        assert_eq!(browser.visited_nodes(), vec!["step1", "step2", "step3"]);

        let state_at_1 = browser.state_at(1).unwrap();
        let counter: i32 = state_at_1.get("counter").unwrap();
        assert_eq!(counter, 2);
    }

    #[tokio::test]
    async fn test_find_key_changes() {
        let graph = create_test_graph();
        let tt = TimeTravelGraph::new(graph);

        let history = tt.execute(GraphState::new()).await.unwrap();
        let browser = history.browser();

        // counter changes in all 3 steps
        let counter_changes = browser.find_key_changes("counter");
        assert_eq!(counter_changes.len(), 3);

        // step1_done only changes in step 0
        let step1_changes = browser.find_key_changes("step1_done");
        assert_eq!(step1_changes.len(), 1);
        assert_eq!(step1_changes[0], 0);
    }

    #[tokio::test]
    async fn test_replay_from_step() {
        let graph = create_test_graph();
        let tt = TimeTravelGraph::new(graph);

        let original = tt.execute(GraphState::new()).await.unwrap();

        // Replay from step 1 (after step2 executed)
        let replayed = tt.replay_from(&original, 1).await.unwrap();

        // Should only have step3
        assert_eq!(replayed.steps.len(), 1);
        assert_eq!(replayed.steps[0].node_id, "step3");
        assert!(replayed.parent_id.is_some());
        assert_eq!(replayed.fork_step, Some(1));
    }

    #[tokio::test]
    async fn test_fork_with_modified_state() {
        let graph = create_test_graph();
        let tt = TimeTravelGraph::new(graph);

        let original = tt.execute(GraphState::new()).await.unwrap();

        // Fork from step 0 with modified state
        let mut modified_state = original.steps[0].state_after.clone();
        modified_state.set("counter", 100); // Override counter

        let forked = tt.fork_from(&original, 0, modified_state).await.unwrap();

        // Forked execution should continue with modified counter
        let final_counter: i32 = forked.result.state.get("counter").unwrap();
        assert_eq!(final_counter, 102); // 100 + 1 + 1
    }

    #[tokio::test]
    async fn test_state_diff() {
        let mut before = GraphState::new();
        before.set("a", 1);
        before.set("b", 2);

        let mut after = GraphState::new();
        after.set("a", 1); // unchanged
        after.set("b", 3); // modified
        after.set("c", 4); // added

        let diff = diff_states(&before, &after);

        assert!(diff.added.contains(&"c".to_string()));
        assert!(diff.modified.contains_key("b"));
        assert!(!diff.removed.contains(&"a".to_string()));
    }

    #[tokio::test]
    async fn test_compare_executions() {
        let graph = create_test_graph();
        let tt = TimeTravelGraph::new(graph);

        let exec1 = tt.execute(GraphState::new()).await.unwrap();

        // Fork with different state
        let mut modified = exec1.steps[0].state_after.clone();
        modified.set("custom", "value");
        let exec2 = tt.fork_from(&exec1, 0, modified).await.unwrap();

        let comparison = tt.compare_executions(&exec1.id, &exec2.id).await.unwrap();

        assert!(comparison.is_fork_comparison);
        assert!(!comparison.final_state_diff.is_empty());
    }

    #[tokio::test]
    async fn test_debug_session() {
        let graph = Arc::new(create_test_graph());
        let mut session = DebugSession::new(graph, GraphState::new());

        // Step through manually
        let result1 = session.step().await.unwrap();
        assert!(matches!(result1, StepResult::Continue(_)));
        assert_eq!(session.current_node(), "step2");

        let result2 = session.step().await.unwrap();
        assert!(matches!(result2, StepResult::Continue(_)));
        assert_eq!(session.current_node(), "step3");

        let result3 = session.step().await.unwrap();
        assert!(matches!(result3, StepResult::Finished));
    }

    #[tokio::test]
    async fn test_debug_breakpoints() {
        let graph = Arc::new(create_test_graph());
        let mut session = DebugSession::new(graph, GraphState::new());

        session.add_breakpoint("step3");

        // Run until breakpoint
        let result = session.run().await.unwrap();
        assert!(matches!(result, StepResult::Breakpoint(ref n) if n == "step3"));
        assert_eq!(session.current_node(), "step3");

        // Continue to end
        let result = session.run().await.unwrap();
        assert!(matches!(result, StepResult::Finished));
    }

    #[tokio::test]
    async fn test_debug_modify_state() {
        let graph = Arc::new(create_test_graph());
        let mut session = DebugSession::new(graph, GraphState::new());

        // Execute step1
        session.step().await.unwrap();

        // Modify state before step2
        session.modify_state(|state| {
            state.set("counter", 50);
        });

        // Continue execution
        session.run().await.unwrap();

        // Final counter should be 52 (50 + 1 + 1)
        let final_counter: i32 = session.current_state().get("counter").unwrap();
        assert_eq!(final_counter, 52);
    }

    #[tokio::test]
    async fn test_execution_summary() {
        let graph = create_test_graph();
        let tt = TimeTravelGraph::new(graph);

        let history = tt.execute(GraphState::new()).await.unwrap();
        let summary = history.browser().summary();

        assert_eq!(summary.total_steps, 3);
        assert_eq!(summary.unique_nodes, 3);
        assert_eq!(summary.status, GraphStatus::Success);
        assert!(!summary.is_fork);
    }

    #[tokio::test]
    async fn test_timeline_output() {
        let graph = create_test_graph();
        let tt = TimeTravelGraph::new(graph);

        let history = tt.execute(GraphState::new()).await.unwrap();
        let timeline = history.browser().timeline();

        assert!(timeline.contains("step1"));
        assert!(timeline.contains("step2"));
        assert!(timeline.contains("step3"));
        assert!(timeline.contains("Success"));
    }
}
