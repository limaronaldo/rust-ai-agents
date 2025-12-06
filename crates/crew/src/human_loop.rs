//! Human-in-the-Loop Support
//!
//! Provides pause/resume, approval gates, and interactive breakpoints
//! for graph-based workflows.
//!
//! ## Features
//!
//! - **Approval Gates**: Pause execution for human approval before continuing
//! - **Breakpoints**: Define nodes where execution pauses for inspection/input
//! - **Input Collection**: Request structured input from humans during execution
//! - **Interrupt/Resume**: Pause and resume execution at any point
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_crew::human_loop::{HumanLoop, ApprovalGate, InteractiveGraph};
//!
//! // Create a graph with approval gates
//! let graph = GraphBuilder::new("approval_workflow")
//!     .add_node("generate", generate_content)
//!     .add_node("review", review_node)
//!     .add_node("publish", publish_content)
//!     .add_edge("generate", "review")
//!     .add_edge("review", "publish")
//!     .set_entry("generate")
//!     .build()?;
//!
//! // Wrap with human-in-the-loop
//! let interactive = InteractiveGraph::new(graph)
//!     .with_approval_gate("review", ApprovalGate::new("Review the generated content"))
//!     .with_breakpoint("publish");
//!
//! // Run interactively
//! let mut session = interactive.start(initial_state).await?;
//!
//! loop {
//!     match session.next().await? {
//!         HumanLoopAction::Continue(state) => { /* auto-continues */ }
//!         HumanLoopAction::AwaitApproval { gate, state } => {
//!             // Show content to user, get approval
//!             if user_approves() {
//!                 session.approve().await?;
//!             } else {
//!                 session.reject("Needs more work").await?;
//!             }
//!         }
//!         HumanLoopAction::AwaitInput { request, state } => {
//!             let input = get_user_input(&request);
//!             session.provide_input(input).await?;
//!         }
//!         HumanLoopAction::Breakpoint { node_id, state } => {
//!             // Inspect state, optionally modify
//!             session.resume().await?;
//!         }
//!         HumanLoopAction::Complete(result) => break,
//!     }
//! }
//! ```

use crate::graph::{
    Checkpoint, CheckpointStore, Graph, GraphBuilder, GraphResult, GraphState, GraphStatus,
    InMemoryCheckpointStore, END,
};
use chrono::{DateTime, Utc};
use rust_ai_agents_core::errors::CrewError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::oneshot;

/// Action required from human
#[derive(Debug, Clone)]
pub enum HumanLoopAction {
    /// Execution continues automatically
    Continue(GraphState),

    /// Awaiting human approval to proceed
    AwaitApproval {
        /// The approval gate configuration
        gate: ApprovalGate,
        /// Current state
        state: GraphState,
        /// Node that triggered the gate
        node_id: String,
    },

    /// Awaiting human input
    AwaitInput {
        /// Input request details
        request: HumanInputRequest,
        /// Current state
        state: GraphState,
    },

    /// Hit a breakpoint - execution paused
    Breakpoint {
        /// Node where breakpoint was hit
        node_id: String,
        /// Current state
        state: GraphState,
    },

    /// Execution interrupted by user
    Interrupted {
        /// State at interruption
        state: GraphState,
        /// Reason for interruption
        reason: String,
    },

    /// Execution completed
    Complete(GraphResult),

    /// Execution failed
    Failed(String),
}

/// Approval gate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalGate {
    /// Human-readable description of what needs approval
    pub description: String,
    /// Fields from state to show for review
    pub show_fields: Vec<String>,
    /// Whether to allow modification of state
    pub allow_edit: bool,
    /// Timeout in seconds (None = wait indefinitely)
    pub timeout_secs: Option<u64>,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ApprovalGate {
    /// Create a new approval gate
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            show_fields: Vec::new(),
            allow_edit: false,
            timeout_secs: None,
            metadata: HashMap::new(),
        }
    }

    /// Specify fields to show for review
    pub fn show_fields(mut self, fields: Vec<String>) -> Self {
        self.show_fields = fields;
        self
    }

    /// Allow editing state during approval
    pub fn allow_edit(mut self) -> Self {
        self.allow_edit = true;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Input request for human interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanInputRequest {
    /// Unique request ID
    pub id: String,
    /// Human-readable prompt
    pub prompt: String,
    /// Type of input expected
    pub input_type: HumanInputType,
    /// Field name to store result
    pub field_name: String,
    /// Whether input is required
    pub required: bool,
    /// Default value
    pub default: Option<serde_json::Value>,
    /// Validation schema (JSON Schema format)
    pub validation: Option<serde_json::Value>,
    /// Timeout in seconds
    pub timeout_secs: Option<u64>,
}

impl HumanInputRequest {
    /// Create a new input request
    pub fn new(prompt: impl Into<String>, field_name: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            prompt: prompt.into(),
            input_type: HumanInputType::Text,
            field_name: field_name.into(),
            required: true,
            default: None,
            validation: None,
            timeout_secs: None,
        }
    }

    /// Set input type
    pub fn input_type(mut self, t: HumanInputType) -> Self {
        self.input_type = t;
        self
    }

    /// Set as optional
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Set default value
    pub fn with_default(mut self, value: serde_json::Value) -> Self {
        self.default = Some(value);
        self.required = false;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }
}

/// Type of input expected for human interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HumanInputType {
    /// Free-form text
    Text,
    /// Multi-line text
    TextArea,
    /// Yes/No confirmation
    Boolean,
    /// Number input
    Number,
    /// Selection from options
    Select(Vec<SelectOption>),
    /// Multiple selection
    MultiSelect(Vec<SelectOption>),
    /// Date/time input
    DateTime,
    /// File upload (returns base64 or path)
    File,
    /// Structured JSON
    Json,
}

/// Option for select inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectOption {
    /// Display label
    pub label: String,
    /// Value
    pub value: serde_json::Value,
    /// Description
    pub description: Option<String>,
}

impl SelectOption {
    /// Create a new select option
    pub fn new(label: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            description: None,
        }
    }

    /// Add description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// Human response to an approval gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalResponse {
    /// Approved to continue
    Approved,
    /// Approved with modified state
    ApprovedWithChanges(serde_json::Value),
    /// Rejected with reason
    Rejected(String),
    /// Request to retry the previous node
    Retry,
}

/// Interactive graph wrapper with human-in-the-loop support
pub struct InteractiveGraph {
    /// Underlying graph
    graph: Graph,
    /// Approval gates by node ID
    approval_gates: HashMap<String, ApprovalGate>,
    /// Breakpoint nodes
    breakpoints: HashSet<String>,
    /// Input requests by node ID
    input_requests: HashMap<String, HumanInputRequest>,
    /// Checkpoint store for persistence
    checkpoint_store: Arc<dyn CheckpointStore>,
}

impl InteractiveGraph {
    /// Create an interactive graph wrapper
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            approval_gates: HashMap::new(),
            breakpoints: HashSet::new(),
            input_requests: HashMap::new(),
            checkpoint_store: Arc::new(InMemoryCheckpointStore::default()),
        }
    }

    /// Add an approval gate after a node
    pub fn with_approval_gate(mut self, node_id: impl Into<String>, gate: ApprovalGate) -> Self {
        self.approval_gates.insert(node_id.into(), gate);
        self
    }

    /// Add a breakpoint at a node
    pub fn with_breakpoint(mut self, node_id: impl Into<String>) -> Self {
        self.breakpoints.insert(node_id.into());
        self
    }

    /// Add an input request before a node
    pub fn with_input_request(
        mut self,
        node_id: impl Into<String>,
        request: HumanInputRequest,
    ) -> Self {
        self.input_requests.insert(node_id.into(), request);
        self
    }

    /// Set custom checkpoint store
    pub fn with_checkpoint_store(mut self, store: Arc<dyn CheckpointStore>) -> Self {
        self.checkpoint_store = store;
        self
    }

    /// Start an interactive session
    pub async fn start(
        &self,
        initial_state: GraphState,
    ) -> Result<InteractiveSession<'_>, CrewError> {
        Ok(InteractiveSession::new(
            &self.graph,
            initial_state,
            self.approval_gates.clone(),
            self.breakpoints.clone(),
            self.input_requests.clone(),
            self.checkpoint_store.clone(),
        ))
    }

    /// Resume from a checkpoint
    pub async fn resume(&self, checkpoint_id: &str) -> Result<InteractiveSession<'_>, CrewError> {
        let checkpoint = self
            .checkpoint_store
            .load(checkpoint_id)
            .await?
            .ok_or_else(|| {
                CrewError::TaskNotFound(format!("Checkpoint not found: {}", checkpoint_id))
            })?;

        Ok(InteractiveSession::from_checkpoint(
            &self.graph,
            checkpoint,
            self.approval_gates.clone(),
            self.breakpoints.clone(),
            self.input_requests.clone(),
            self.checkpoint_store.clone(),
        ))
    }
}

/// Interactive execution session
pub struct InteractiveSession<'a> {
    /// Reference to graph
    graph: &'a Graph,
    /// Current state
    state: GraphState,
    /// Current node
    current_node: String,
    /// Session status
    status: SessionStatus,
    /// Approval gates
    approval_gates: HashMap<String, ApprovalGate>,
    /// Breakpoints
    breakpoints: HashSet<String>,
    /// Input requests
    input_requests: HashMap<String, HumanInputRequest>,
    /// Checkpoint store
    checkpoint_store: Arc<dyn CheckpointStore>,
    /// Pending approval response (reserved for future async approval handling)
    #[allow(dead_code)]
    pending_approval: Option<oneshot::Sender<ApprovalResponse>>,
    /// Pending input response (reserved for future async input handling)
    #[allow(dead_code)]
    pending_input: Option<oneshot::Sender<serde_json::Value>>,
    /// Session ID
    session_id: String,
    /// Started at
    started_at: DateTime<Utc>,
}

/// Session status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    /// Running normally
    Running,
    /// Waiting for approval
    AwaitingApproval,
    /// Waiting for input
    AwaitingInput,
    /// Paused at breakpoint
    Paused,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Interrupted
    Interrupted,
}

impl<'a> InteractiveSession<'a> {
    /// Create a new session
    fn new(
        graph: &'a Graph,
        initial_state: GraphState,
        approval_gates: HashMap<String, ApprovalGate>,
        breakpoints: HashSet<String>,
        input_requests: HashMap<String, HumanInputRequest>,
        checkpoint_store: Arc<dyn CheckpointStore>,
    ) -> Self {
        Self {
            graph,
            state: initial_state,
            current_node: graph.entry_node.clone(),
            status: SessionStatus::Running,
            approval_gates,
            breakpoints,
            input_requests,
            checkpoint_store,
            pending_approval: None,
            pending_input: None,
            session_id: uuid::Uuid::new_v4().to_string(),
            started_at: Utc::now(),
        }
    }

    /// Create from checkpoint
    fn from_checkpoint(
        graph: &'a Graph,
        checkpoint: Checkpoint,
        approval_gates: HashMap<String, ApprovalGate>,
        breakpoints: HashSet<String>,
        input_requests: HashMap<String, HumanInputRequest>,
        checkpoint_store: Arc<dyn CheckpointStore>,
    ) -> Self {
        Self {
            graph,
            state: checkpoint.state,
            current_node: checkpoint.next_node,
            status: SessionStatus::Running,
            approval_gates,
            breakpoints,
            input_requests,
            checkpoint_store,
            pending_approval: None,
            pending_input: None,
            session_id: uuid::Uuid::new_v4().to_string(),
            started_at: Utc::now(),
        }
    }

    /// Get current state
    pub fn state(&self) -> &GraphState {
        &self.state
    }

    /// Get current node
    pub fn current_node(&self) -> &str {
        &self.current_node
    }

    /// Get session status
    pub fn status(&self) -> SessionStatus {
        self.status
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Execute next step
    pub async fn next(&mut self) -> Result<HumanLoopAction, CrewError> {
        // Check if completed
        if self.current_node == END {
            self.status = SessionStatus::Completed;
            let duration = Utc::now()
                .signed_duration_since(self.started_at)
                .num_milliseconds() as u64;
            self.state.metadata.execution_time_ms = duration;

            return Ok(HumanLoopAction::Complete(GraphResult {
                state: self.state.clone(),
                status: GraphStatus::Success,
                error: None,
            }));
        }

        // Check for max iterations
        if self.state.metadata.iterations >= 100 {
            self.status = SessionStatus::Failed;
            return Ok(HumanLoopAction::Failed(
                "Max iterations reached".to_string(),
            ));
        }

        // Check for input request before node
        if let Some(request) = self.input_requests.get(&self.current_node).cloned() {
            self.status = SessionStatus::AwaitingInput;
            return Ok(HumanLoopAction::AwaitInput {
                request,
                state: self.state.clone(),
            });
        }

        // Check for breakpoint
        if self.breakpoints.contains(&self.current_node) {
            self.status = SessionStatus::Paused;
            // Save checkpoint
            self.save_checkpoint().await?;
            return Ok(HumanLoopAction::Breakpoint {
                node_id: self.current_node.clone(),
                state: self.state.clone(),
            });
        }

        // Execute the current node
        let node = self
            .graph
            .nodes
            .get(&self.current_node)
            .ok_or_else(|| CrewError::TaskNotFound(self.current_node.clone()))?;

        self.state
            .metadata
            .visited_nodes
            .push(self.current_node.clone());
        self.state.metadata.iterations += 1;

        self.state = node.executor.call(self.state.clone()).await?;

        // Check for approval gate after node
        if let Some(gate) = self.approval_gates.get(&self.current_node).cloned() {
            self.status = SessionStatus::AwaitingApproval;
            // Save checkpoint before waiting
            self.save_checkpoint().await?;
            return Ok(HumanLoopAction::AwaitApproval {
                gate,
                state: self.state.clone(),
                node_id: self.current_node.clone(),
            });
        }

        // Find next node
        self.current_node = self.find_next_node()?;
        self.status = SessionStatus::Running;

        Ok(HumanLoopAction::Continue(self.state.clone()))
    }

    /// Approve and continue
    pub async fn approve(&mut self) -> Result<(), CrewError> {
        if self.status != SessionStatus::AwaitingApproval {
            return Err(CrewError::ExecutionFailed(
                "Not awaiting approval".to_string(),
            ));
        }

        // Move to next node
        self.current_node = self.find_next_node()?;
        self.status = SessionStatus::Running;
        Ok(())
    }

    /// Approve with modified state
    pub async fn approve_with_changes(
        &mut self,
        changes: serde_json::Value,
    ) -> Result<(), CrewError> {
        if self.status != SessionStatus::AwaitingApproval {
            return Err(CrewError::ExecutionFailed(
                "Not awaiting approval".to_string(),
            ));
        }

        // Apply changes to state
        if let Some(obj) = changes.as_object() {
            for (k, v) in obj {
                self.state.set(k, v.clone());
            }
        }

        // Move to next node
        self.current_node = self.find_next_node()?;
        self.status = SessionStatus::Running;
        Ok(())
    }

    /// Reject and stop or go back
    pub async fn reject(&mut self, reason: impl Into<String>) -> Result<(), CrewError> {
        if self.status != SessionStatus::AwaitingApproval {
            return Err(CrewError::ExecutionFailed(
                "Not awaiting approval".to_string(),
            ));
        }

        self.status = SessionStatus::Interrupted;
        self.state.set("_rejection_reason", reason.into());
        Ok(())
    }

    /// Provide input and continue
    pub async fn provide_input(&mut self, value: serde_json::Value) -> Result<(), CrewError> {
        if self.status != SessionStatus::AwaitingInput {
            return Err(CrewError::ExecutionFailed("Not awaiting input".to_string()));
        }

        // Get the field name for this input and remove it so we don't ask again
        if let Some(request) = self.input_requests.remove(&self.current_node) {
            self.state.set(&request.field_name, value);
        }

        self.status = SessionStatus::Running;
        Ok(())
    }

    /// Resume from breakpoint
    pub async fn resume(&mut self) -> Result<(), CrewError> {
        if self.status != SessionStatus::Paused {
            return Err(CrewError::ExecutionFailed("Not paused".to_string()));
        }

        // Remove the breakpoint that was hit (so we don't hit it again)
        self.breakpoints.remove(&self.current_node);
        self.status = SessionStatus::Running;
        Ok(())
    }

    /// Resume with modified state
    pub async fn resume_with_state(&mut self, new_state: GraphState) -> Result<(), CrewError> {
        if self.status != SessionStatus::Paused {
            return Err(CrewError::ExecutionFailed("Not paused".to_string()));
        }

        self.state = new_state;
        self.breakpoints.remove(&self.current_node);
        self.status = SessionStatus::Running;
        Ok(())
    }

    /// Interrupt execution
    pub async fn interrupt(&mut self, reason: impl Into<String>) -> Result<(), CrewError> {
        self.status = SessionStatus::Interrupted;
        self.state.set("_interrupt_reason", reason.into());
        // Save checkpoint for potential resume
        self.save_checkpoint().await?;
        Ok(())
    }

    /// Get checkpoint ID for this session
    pub fn checkpoint_id(&self) -> String {
        format!("{}_{}", self.session_id, self.state.metadata.iterations)
    }

    /// Save current state as checkpoint
    async fn save_checkpoint(&self) -> Result<(), CrewError> {
        let checkpoint = Checkpoint {
            id: self.checkpoint_id(),
            state: self.state.clone(),
            next_node: self.current_node.clone(),
            created_at: Utc::now(),
        };
        self.checkpoint_store.save(checkpoint).await
    }

    /// Find next node based on edges
    fn find_next_node(&self) -> Result<String, CrewError> {
        for edge in &self.graph.edges {
            match edge {
                crate::graph::GraphEdge::Direct { from, to } if *from == self.current_node => {
                    return Ok(to.clone());
                }
                crate::graph::GraphEdge::Conditional { from, router }
                    if *from == self.current_node =>
                {
                    return Ok(router.route(&self.state));
                }
                _ => continue,
            }
        }
        Ok(END.to_string())
    }

    /// Run to completion or next human action
    pub async fn run_until_human_action(&mut self) -> Result<HumanLoopAction, CrewError> {
        loop {
            let action = self.next().await?;
            match &action {
                HumanLoopAction::Continue(_) => continue,
                _ => return Ok(action),
            }
        }
    }
}

/// Builder for creating interactive workflows
pub struct InteractiveWorkflowBuilder {
    graph_builder: GraphBuilder,
    approval_gates: HashMap<String, ApprovalGate>,
    breakpoints: HashSet<String>,
    input_requests: HashMap<String, HumanInputRequest>,
}

impl InteractiveWorkflowBuilder {
    /// Create a new builder
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            graph_builder: GraphBuilder::new(id),
            approval_gates: HashMap::new(),
            breakpoints: HashSet::new(),
            input_requests: HashMap::new(),
        }
    }

    /// Add a node
    pub fn add_node<F, Fut>(mut self, id: impl Into<String>, func: F) -> Self
    where
        F: Fn(GraphState) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<GraphState, CrewError>> + Send + 'static,
    {
        self.graph_builder = self.graph_builder.add_node(id, func);
        self
    }

    /// Add a node with approval gate
    pub fn add_node_with_approval<F, Fut>(
        mut self,
        id: impl Into<String>,
        func: F,
        gate: ApprovalGate,
    ) -> Self
    where
        F: Fn(GraphState) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<GraphState, CrewError>> + Send + 'static,
    {
        let id = id.into();
        self.graph_builder = self.graph_builder.add_node(id.clone(), func);
        self.approval_gates.insert(id, gate);
        self
    }

    /// Add a node with input request
    pub fn add_node_with_input<F, Fut>(
        mut self,
        id: impl Into<String>,
        func: F,
        request: HumanInputRequest,
    ) -> Self
    where
        F: Fn(GraphState) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<GraphState, CrewError>> + Send + 'static,
    {
        let id = id.into();
        self.graph_builder = self.graph_builder.add_node(id.clone(), func);
        self.input_requests.insert(id, request);
        self
    }

    /// Add a breakpoint node
    pub fn add_breakpoint_node<F, Fut>(mut self, id: impl Into<String>, func: F) -> Self
    where
        F: Fn(GraphState) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<GraphState, CrewError>> + Send + 'static,
    {
        let id = id.into();
        self.graph_builder = self.graph_builder.add_node(id.clone(), func);
        self.breakpoints.insert(id);
        self
    }

    /// Add edge
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.graph_builder = self.graph_builder.add_edge(from, to);
        self
    }

    /// Add conditional edge
    pub fn add_conditional_edge<F>(mut self, from: impl Into<String>, router: F) -> Self
    where
        F: Fn(&GraphState) -> String + Send + Sync + 'static,
    {
        self.graph_builder = self.graph_builder.add_conditional_edge(from, router);
        self
    }

    /// Set entry node
    pub fn set_entry(mut self, node_id: impl Into<String>) -> Self {
        self.graph_builder = self.graph_builder.set_entry(node_id);
        self
    }

    /// Build the interactive graph
    pub fn build(self) -> Result<InteractiveGraph, CrewError> {
        let graph = self.graph_builder.build()?;

        let mut interactive = InteractiveGraph::new(graph);
        interactive.approval_gates = self.approval_gates;
        interactive.breakpoints = self.breakpoints;
        interactive.input_requests = self.input_requests;

        Ok(interactive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_approval_gate() {
        let graph = GraphBuilder::new("approval_test")
            .add_node("generate", |mut state: GraphState| async move {
                state.set("content", "Generated content");
                Ok(state)
            })
            .add_node("publish", |mut state: GraphState| async move {
                state.set("published", true);
                Ok(state)
            })
            .add_edge("generate", "publish")
            .add_edge("publish", END)
            .set_entry("generate")
            .build()
            .unwrap();

        let interactive = InteractiveGraph::new(graph)
            .with_approval_gate("generate", ApprovalGate::new("Review content"));

        let mut session = interactive.start(GraphState::new()).await.unwrap();

        // First step executes generate
        let action = session.next().await.unwrap();
        match action {
            HumanLoopAction::AwaitApproval { gate, state, .. } => {
                assert_eq!(gate.description, "Review content");
                assert_eq!(
                    state.get::<String>("content"),
                    Some("Generated content".to_string())
                );
            }
            _ => panic!("Expected AwaitApproval"),
        }

        // Approve
        session.approve().await.unwrap();

        // Continue to publish
        let action = session.next().await.unwrap();
        assert!(matches!(action, HumanLoopAction::Continue(_)));

        // Complete
        let action = session.next().await.unwrap();
        match action {
            HumanLoopAction::Complete(result) => {
                assert_eq!(result.status, GraphStatus::Success);
                assert_eq!(result.state.get::<bool>("published"), Some(true));
            }
            _ => panic!("Expected Complete"),
        }
    }

    #[tokio::test]
    async fn test_breakpoint() {
        let graph = GraphBuilder::new("breakpoint_test")
            .add_node("step1", |mut state: GraphState| async move {
                state.set("step", 1);
                Ok(state)
            })
            .add_node("step2", |mut state: GraphState| async move {
                state.set("step", 2);
                Ok(state)
            })
            .add_edge("step1", "step2")
            .add_edge("step2", END)
            .set_entry("step1")
            .build()
            .unwrap();

        let interactive = InteractiveGraph::new(graph).with_breakpoint("step2");

        let mut session = interactive.start(GraphState::new()).await.unwrap();

        // Execute step1
        let action = session.next().await.unwrap();
        assert!(matches!(action, HumanLoopAction::Continue(_)));

        // Hit breakpoint at step2
        let action = session.next().await.unwrap();
        match action {
            HumanLoopAction::Breakpoint { node_id, state } => {
                assert_eq!(node_id, "step2");
                assert_eq!(state.get::<i32>("step"), Some(1));
            }
            _ => panic!("Expected Breakpoint"),
        }

        // Resume
        session.resume().await.unwrap();

        // Execute step2
        let action = session.next().await.unwrap();
        assert!(matches!(action, HumanLoopAction::Continue(_)));

        // Complete
        let action = session.next().await.unwrap();
        assert!(matches!(action, HumanLoopAction::Complete(_)));
    }

    #[tokio::test]
    async fn test_input_request() {
        let graph = GraphBuilder::new("input_test")
            .add_node("process", |state: GraphState| async move {
                // Just pass through - input should already be set
                Ok(state)
            })
            .add_edge("process", END)
            .set_entry("process")
            .build()
            .unwrap();

        let interactive = InteractiveGraph::new(graph).with_input_request(
            "process",
            HumanInputRequest::new("Enter your name", "user_name"),
        );

        let mut session = interactive.start(GraphState::new()).await.unwrap();

        // Should request input before process
        let action = session.next().await.unwrap();
        match action {
            HumanLoopAction::AwaitInput { request, .. } => {
                assert_eq!(request.prompt, "Enter your name");
                assert_eq!(request.field_name, "user_name");
            }
            _ => panic!("Expected AwaitInput"),
        }

        // Provide input
        session
            .provide_input(serde_json::json!("Alice"))
            .await
            .unwrap();

        // Execute process
        let action = session.next().await.unwrap();
        assert!(matches!(action, HumanLoopAction::Continue(_)));

        // Complete
        let action = session.next().await.unwrap();
        match action {
            HumanLoopAction::Complete(result) => {
                assert_eq!(
                    result.state.get::<String>("user_name"),
                    Some("Alice".to_string())
                );
            }
            _ => panic!("Expected Complete"),
        }
    }

    #[tokio::test]
    async fn test_rejection() {
        let graph = GraphBuilder::new("reject_test")
            .add_node("generate", |mut state: GraphState| async move {
                state.set("content", "Bad content");
                Ok(state)
            })
            .add_edge("generate", END)
            .set_entry("generate")
            .build()
            .unwrap();

        let interactive = InteractiveGraph::new(graph)
            .with_approval_gate("generate", ApprovalGate::new("Review content"));

        let mut session = interactive.start(GraphState::new()).await.unwrap();

        // Execute generate
        let action = session.next().await.unwrap();
        assert!(matches!(action, HumanLoopAction::AwaitApproval { .. }));

        // Reject
        session.reject("Content not good enough").await.unwrap();

        assert_eq!(session.status(), SessionStatus::Interrupted);
    }

    #[tokio::test]
    async fn test_run_until_human_action() {
        let graph = GraphBuilder::new("run_test")
            .add_node("auto1", |mut state: GraphState| async move {
                state.set("auto1", true);
                Ok(state)
            })
            .add_node("auto2", |mut state: GraphState| async move {
                state.set("auto2", true);
                Ok(state)
            })
            .add_node("manual", |mut state: GraphState| async move {
                state.set("manual", true);
                Ok(state)
            })
            .add_edge("auto1", "auto2")
            .add_edge("auto2", "manual")
            .add_edge("manual", END)
            .set_entry("auto1")
            .build()
            .unwrap();

        let interactive =
            InteractiveGraph::new(graph).with_approval_gate("manual", ApprovalGate::new("Review"));

        let mut session = interactive.start(GraphState::new()).await.unwrap();

        // Run until human action (should skip auto1, auto2 and stop at manual approval)
        let action = session.run_until_human_action().await.unwrap();

        match action {
            HumanLoopAction::AwaitApproval { state, .. } => {
                // Both auto nodes should have run
                assert_eq!(state.get::<bool>("auto1"), Some(true));
                assert_eq!(state.get::<bool>("auto2"), Some(true));
                assert_eq!(state.get::<bool>("manual"), Some(true));
            }
            _ => panic!("Expected AwaitApproval"),
        }
    }

    #[test]
    fn test_approval_gate_builder() {
        let gate = ApprovalGate::new("Review document")
            .show_fields(vec!["title".to_string(), "content".to_string()])
            .allow_edit()
            .with_timeout(300)
            .with_metadata("priority", serde_json::json!("high"));

        assert_eq!(gate.description, "Review document");
        assert_eq!(gate.show_fields.len(), 2);
        assert!(gate.allow_edit);
        assert_eq!(gate.timeout_secs, Some(300));
        assert_eq!(
            gate.metadata.get("priority"),
            Some(&serde_json::json!("high"))
        );
    }

    #[test]
    fn test_input_request_builder() {
        let request = HumanInputRequest::new("Select priority", "priority")
            .input_type(HumanInputType::Select(vec![
                SelectOption::new("Low", "low"),
                SelectOption::new("Medium", "medium").with_description("Default"),
                SelectOption::new("High", "high"),
            ]))
            .with_default(serde_json::json!("medium"))
            .with_timeout(60);

        assert_eq!(request.prompt, "Select priority");
        assert_eq!(request.field_name, "priority");
        assert!(!request.required); // optional because has default
        assert_eq!(request.timeout_secs, Some(60));
    }
}
