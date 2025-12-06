//! DAG-based Workflow System
//!
//! Provides a directed acyclic graph (DAG) based workflow execution system
//! with support for conditional branching, human-in-the-loop patterns,
//! and complex task orchestration.
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_crew::workflow::{Workflow, WorkflowBuilder, NextAction};
//!
//! let workflow = WorkflowBuilder::new("approval_flow")
//!     .add_node("validate", validate_task)
//!     .add_node("approve", approve_task)
//!     .add_node("execute", execute_task)
//!     .add_node("reject", reject_task)
//!     .connect("validate", "approve", Some(Condition::when("valid", true)))
//!     .connect("validate", "reject", Some(Condition::when("valid", false)))
//!     .connect("approve", "execute", None)
//!     .set_entry("validate")
//!     .build();
//!
//! let mut runner = WorkflowRunner::new(workflow);
//! loop {
//!     match runner.step(context).await? {
//!         NextAction::Continue => continue,
//!         NextAction::WaitForInput(prompt) => {
//!             let input = get_user_input(&prompt);
//!             runner.provide_input(input);
//!         }
//!         NextAction::Complete(result) => break result,
//!     }
//! }
//! ```

use rust_ai_agents_core::{errors::CrewError, Task};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Action to take after a workflow step
#[derive(Debug, Clone)]
pub enum NextAction {
    /// Continue to next step automatically
    Continue,

    /// Continue and execute until completion or pause
    ContinueAndExecute,

    /// Wait for external input before continuing
    WaitForInput(InputRequest),

    /// Branch to multiple paths
    Branch(Vec<String>),

    /// Workflow completed successfully
    Complete(WorkflowResult),

    /// Workflow failed
    Failed(String),
}

/// Request for external input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRequest {
    /// Unique identifier for this input request
    pub id: String,

    /// Human-readable prompt
    pub prompt: String,

    /// Type of input expected
    pub input_type: InputType,

    /// Node waiting for input
    pub waiting_node: String,

    /// Optional default value
    pub default: Option<serde_json::Value>,

    /// Timeout in seconds (None = no timeout)
    pub timeout_secs: Option<u64>,
}

/// Type of input expected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputType {
    /// Free-form text
    Text,

    /// Yes/No confirmation
    Confirmation,

    /// Selection from options
    Selection(Vec<String>),

    /// Approval (approve/reject)
    Approval,

    /// Structured JSON data
    Json(serde_json::Value), // JSON Schema
}

/// Result of a completed workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    /// Final output data
    pub output: serde_json::Value,

    /// Execution trace
    pub trace: Vec<NodeExecution>,

    /// Total execution time in milliseconds
    pub duration_ms: u64,

    /// Final status
    pub status: WorkflowStatus,
}

/// Workflow execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    Success,
    Failed,
    Cancelled,
    TimedOut,
}

/// Record of a node execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeExecution {
    /// Node ID
    pub node_id: String,

    /// Start timestamp
    pub started_at: chrono::DateTime<chrono::Utc>,

    /// End timestamp
    pub ended_at: chrono::DateTime<chrono::Utc>,

    /// Execution result
    pub result: NodeResult,

    /// Output data
    pub output: serde_json::Value,
}

/// Result of a single node execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeResult {
    Success,
    Failed(String),
    Skipped,
    WaitingForInput,
}

/// Condition for edge traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    /// Field to check in context
    pub field: String,

    /// Comparison operator
    pub operator: ConditionOperator,

    /// Value to compare against
    pub value: serde_json::Value,
}

impl Condition {
    /// Create a simple equality condition
    pub fn when(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: ConditionOperator::Equals,
            value: value.into(),
        }
    }

    /// Create a not-equals condition
    pub fn when_not(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: ConditionOperator::NotEquals,
            value: value.into(),
        }
    }

    /// Create a greater-than condition
    pub fn when_gt(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: ConditionOperator::GreaterThan,
            value: value.into(),
        }
    }

    /// Evaluate the condition against a context
    pub fn evaluate(&self, context: &serde_json::Value) -> bool {
        let field_value = context.get(&self.field);

        match (&self.operator, field_value) {
            (ConditionOperator::Equals, Some(v)) => v == &self.value,
            (ConditionOperator::NotEquals, Some(v)) => v != &self.value,
            (ConditionOperator::GreaterThan, Some(v)) => match (v.as_f64(), self.value.as_f64()) {
                (Some(a), Some(b)) => a > b,
                _ => false,
            },
            (ConditionOperator::LessThan, Some(v)) => match (v.as_f64(), self.value.as_f64()) {
                (Some(a), Some(b)) => a < b,
                _ => false,
            },
            (ConditionOperator::Contains, Some(v)) => {
                if let (Some(arr), Some(needle)) = (v.as_array(), self.value.as_str()) {
                    arr.iter().any(|x| x.as_str() == Some(needle))
                } else if let (Some(s), Some(needle)) = (v.as_str(), self.value.as_str()) {
                    s.contains(needle)
                } else {
                    false
                }
            }
            (ConditionOperator::Exists, _) => field_value.is_some(),
            _ => false,
        }
    }
}

/// Comparison operators for conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    Exists,
}

/// A node in the workflow graph
#[derive(Clone)]
pub struct WorkflowNode {
    /// Unique node identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Node type
    pub node_type: NodeType,

    /// Associated task (if any)
    pub task: Option<Task>,

    /// Custom executor function
    pub executor: Option<Arc<dyn NodeExecutor>>,

    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl std::fmt::Debug for WorkflowNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkflowNode")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("node_type", &self.node_type)
            .finish()
    }
}

/// Type of workflow node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    /// Task execution node
    Task,

    /// Human input required
    HumanInput,

    /// Conditional branching
    Decision,

    /// Join multiple branches
    Join,

    /// Start node
    Start,

    /// End node
    End,

    /// Parallel execution of children
    Parallel,
}

/// Trait for custom node executors
#[async_trait::async_trait]
pub trait NodeExecutor: Send + Sync {
    /// Execute the node
    async fn execute(
        &self,
        node: &WorkflowNode,
        context: &mut WorkflowContext,
    ) -> Result<NextAction, CrewError>;
}

/// An edge connecting two nodes
#[derive(Debug, Clone)]
pub struct WorkflowEdge {
    /// Source node ID
    pub from: String,

    /// Target node ID
    pub to: String,

    /// Optional condition for traversal
    pub condition: Option<Condition>,

    /// Edge priority (higher = checked first)
    pub priority: i32,
}

/// Workflow definition
#[derive(Clone)]
pub struct Workflow {
    /// Workflow identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Workflow nodes
    pub nodes: HashMap<String, WorkflowNode>,

    /// Workflow edges
    pub edges: Vec<WorkflowEdge>,

    /// Entry node ID
    pub entry_node: String,

    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl std::fmt::Debug for Workflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Workflow")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("entry_node", &self.entry_node)
            .finish()
    }
}

/// Builder for creating workflows
pub struct WorkflowBuilder {
    id: String,
    name: String,
    nodes: HashMap<String, WorkflowNode>,
    edges: Vec<WorkflowEdge>,
    entry_node: Option<String>,
    metadata: HashMap<String, serde_json::Value>,
}

impl WorkflowBuilder {
    /// Create a new workflow builder
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_node: None,
            metadata: HashMap::new(),
        }
    }

    /// Set workflow name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a task node
    pub fn add_task_node(mut self, id: impl Into<String>, task: Task) -> Self {
        let id = id.into();
        self.nodes.insert(
            id.clone(),
            WorkflowNode {
                id: id.clone(),
                name: task.description.clone(),
                node_type: NodeType::Task,
                task: Some(task),
                executor: None,
                metadata: HashMap::new(),
            },
        );
        self
    }

    /// Add a human input node
    pub fn add_input_node(
        mut self,
        id: impl Into<String>,
        prompt: impl Into<String>,
        input_type: InputType,
    ) -> Self {
        let id = id.into();
        let prompt = prompt.into();
        self.nodes.insert(
            id.clone(),
            WorkflowNode {
                id: id.clone(),
                name: prompt.clone(),
                node_type: NodeType::HumanInput,
                task: None,
                executor: None,
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("prompt".to_string(), serde_json::json!(prompt));
                    m.insert(
                        "input_type".to_string(),
                        serde_json::to_value(&input_type).unwrap(),
                    );
                    m
                },
            },
        );
        self
    }

    /// Add a decision node
    pub fn add_decision_node(mut self, id: impl Into<String>, name: impl Into<String>) -> Self {
        let id = id.into();
        self.nodes.insert(
            id.clone(),
            WorkflowNode {
                id: id.clone(),
                name: name.into(),
                node_type: NodeType::Decision,
                task: None,
                executor: None,
                metadata: HashMap::new(),
            },
        );
        self
    }

    /// Add a custom executor node
    pub fn add_custom_node(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        executor: Arc<dyn NodeExecutor>,
    ) -> Self {
        let id = id.into();
        self.nodes.insert(
            id.clone(),
            WorkflowNode {
                id: id.clone(),
                name: name.into(),
                node_type: NodeType::Task,
                task: None,
                executor: Some(executor),
                metadata: HashMap::new(),
            },
        );
        self
    }

    /// Connect two nodes
    pub fn connect(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: Option<Condition>,
    ) -> Self {
        self.edges.push(WorkflowEdge {
            from: from.into(),
            to: to.into(),
            condition,
            priority: 0,
        });
        self
    }

    /// Connect with priority
    pub fn connect_with_priority(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: Option<Condition>,
        priority: i32,
    ) -> Self {
        self.edges.push(WorkflowEdge {
            from: from.into(),
            to: to.into(),
            condition,
            priority,
        });
        self
    }

    /// Set entry node
    pub fn set_entry(mut self, node_id: impl Into<String>) -> Self {
        self.entry_node = Some(node_id.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Build the workflow
    pub fn build(self) -> Result<Workflow, CrewError> {
        let entry_node = self.entry_node.ok_or_else(|| {
            CrewError::InvalidConfiguration("No entry node specified".to_string())
        })?;

        if !self.nodes.contains_key(&entry_node) {
            return Err(CrewError::InvalidConfiguration(format!(
                "Entry node '{}' not found",
                entry_node
            )));
        }

        // Validate edges
        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.from) {
                return Err(CrewError::InvalidConfiguration(format!(
                    "Edge source '{}' not found",
                    edge.from
                )));
            }
            if !self.nodes.contains_key(&edge.to) {
                return Err(CrewError::InvalidConfiguration(format!(
                    "Edge target '{}' not found",
                    edge.to
                )));
            }
        }

        Ok(Workflow {
            id: self.id,
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            entry_node,
            metadata: self.metadata,
        })
    }
}

/// Execution context for workflows
#[derive(Debug, Clone, Default)]
pub struct WorkflowContext {
    /// Shared data between nodes
    pub data: serde_json::Value,

    /// Execution trace
    pub trace: Vec<NodeExecution>,

    /// Current node ID
    pub current_node: Option<String>,

    /// Pending input requests
    pub pending_input: Option<InputRequest>,

    /// User-provided input
    pub provided_input: Option<serde_json::Value>,
}

impl WorkflowContext {
    /// Create a new context
    pub fn new() -> Self {
        Self {
            data: serde_json::json!({}),
            trace: Vec::new(),
            current_node: None,
            pending_input: None,
            provided_input: None,
        }
    }

    /// Create with initial data
    pub fn with_data(data: serde_json::Value) -> Self {
        Self {
            data,
            ..Self::new()
        }
    }

    /// Set a value in context
    pub fn set(&mut self, key: &str, value: serde_json::Value) {
        if let Some(obj) = self.data.as_object_mut() {
            obj.insert(key.to_string(), value);
        }
    }

    /// Get a value from context
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }
}

/// Workflow execution runner
pub struct WorkflowRunner {
    workflow: Workflow,
    context: Arc<RwLock<WorkflowContext>>,
    started_at: Option<std::time::Instant>,
}

impl WorkflowRunner {
    /// Create a new runner
    pub fn new(workflow: Workflow) -> Self {
        Self {
            workflow,
            context: Arc::new(RwLock::new(WorkflowContext::new())),
            started_at: None,
        }
    }

    /// Create with initial context
    pub fn with_context(workflow: Workflow, context: WorkflowContext) -> Self {
        Self {
            workflow,
            context: Arc::new(RwLock::new(context)),
            started_at: None,
        }
    }

    /// Execute one step of the workflow
    pub async fn step(&mut self) -> Result<NextAction, CrewError> {
        if self.started_at.is_none() {
            self.started_at = Some(std::time::Instant::now());
        }

        let mut ctx = self.context.write().await;

        // Check for pending input
        if ctx.pending_input.is_some() && ctx.provided_input.is_none() {
            return Ok(NextAction::WaitForInput(ctx.pending_input.clone().unwrap()));
        }

        // Get current node
        let current_id = ctx
            .current_node
            .clone()
            .unwrap_or_else(|| self.workflow.entry_node.clone());

        let node = self
            .workflow
            .nodes
            .get(&current_id)
            .ok_or_else(|| CrewError::TaskNotFound(current_id.clone()))?
            .clone();

        let start_time = chrono::Utc::now();

        // Execute node based on type
        let (result, output) = match &node.node_type {
            NodeType::Start => (NodeResult::Success, serde_json::json!({})),

            NodeType::End => {
                let duration = self
                    .started_at
                    .map(|s| s.elapsed().as_millis() as u64)
                    .unwrap_or(0);
                return Ok(NextAction::Complete(WorkflowResult {
                    output: ctx.data.clone(),
                    trace: ctx.trace.clone(),
                    duration_ms: duration,
                    status: WorkflowStatus::Success,
                }));
            }

            NodeType::HumanInput => {
                // Check if we have input
                if let Some(input) = ctx.provided_input.take() {
                    // Store input in context
                    ctx.set(&current_id, input.clone());
                    ctx.pending_input = None;
                    (NodeResult::Success, input)
                } else {
                    // Request input
                    let prompt = node
                        .metadata
                        .get("prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Please provide input")
                        .to_string();

                    let input_type: InputType = node
                        .metadata
                        .get("input_type")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or(InputType::Text);

                    let request = InputRequest {
                        id: format!("{}_{}", current_id, chrono::Utc::now().timestamp()),
                        prompt,
                        input_type,
                        waiting_node: current_id.clone(),
                        default: None,
                        timeout_secs: None,
                    };

                    ctx.pending_input = Some(request.clone());
                    return Ok(NextAction::WaitForInput(request));
                }
            }

            NodeType::Decision => {
                // Decision nodes just pass through, edges determine path
                (NodeResult::Success, serde_json::json!({}))
            }

            NodeType::Task => {
                if let Some(executor) = &node.executor {
                    match executor.execute(&node, &mut ctx).await {
                        Ok(action) => return Ok(action),
                        Err(e) => (NodeResult::Failed(e.to_string()), serde_json::json!({})),
                    }
                } else {
                    // Default task execution - just mark as success
                    (NodeResult::Success, serde_json::json!({"task": node.name}))
                }
            }

            NodeType::Join => {
                // Join node waits for all incoming branches
                (NodeResult::Success, serde_json::json!({}))
            }

            NodeType::Parallel => {
                // Parallel execution of children - simplified for now
                (NodeResult::Success, serde_json::json!({}))
            }
        };

        // Record execution
        ctx.trace.push(NodeExecution {
            node_id: current_id.clone(),
            started_at: start_time,
            ended_at: chrono::Utc::now(),
            result,
            output,
        });

        // Find next node
        let next_node = self.find_next_node(&current_id, &ctx.data)?;

        if let Some(next_id) = next_node {
            ctx.current_node = Some(next_id);
            Ok(NextAction::Continue)
        } else {
            // No next node - workflow complete
            let duration = self
                .started_at
                .map(|s| s.elapsed().as_millis() as u64)
                .unwrap_or(0);
            Ok(NextAction::Complete(WorkflowResult {
                output: ctx.data.clone(),
                trace: ctx.trace.clone(),
                duration_ms: duration,
                status: WorkflowStatus::Success,
            }))
        }
    }

    /// Provide input for a pending request
    pub async fn provide_input(&mut self, input: serde_json::Value) {
        let mut ctx = self.context.write().await;
        ctx.provided_input = Some(input);
    }

    /// Run workflow to completion
    pub async fn run(&mut self) -> Result<WorkflowResult, CrewError> {
        loop {
            match self.step().await? {
                NextAction::Continue | NextAction::ContinueAndExecute => continue,
                NextAction::Complete(result) => return Ok(result),
                NextAction::Failed(err) => return Err(CrewError::ExecutionFailed(err)),
                NextAction::WaitForInput(_) => {
                    return Err(CrewError::ExecutionFailed(
                        "Workflow requires input but running in non-interactive mode".to_string(),
                    ))
                }
                NextAction::Branch(_) => {
                    // Handle branching - for now, take first branch
                    continue;
                }
            }
        }
    }

    /// Find next node based on edges and conditions
    fn find_next_node(
        &self,
        current: &str,
        context: &serde_json::Value,
    ) -> Result<Option<String>, CrewError> {
        // Get outgoing edges sorted by priority
        let mut edges: Vec<_> = self
            .workflow
            .edges
            .iter()
            .filter(|e| e.from == current)
            .collect();

        edges.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Find first edge with satisfied condition
        for edge in edges {
            match &edge.condition {
                None => return Ok(Some(edge.to.clone())),
                Some(cond) if cond.evaluate(context) => return Ok(Some(edge.to.clone())),
                _ => continue,
            }
        }

        Ok(None)
    }

    /// Get current context
    pub async fn context(&self) -> WorkflowContext {
        self.context.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_builder() {
        let workflow = WorkflowBuilder::new("test")
            .add_task_node("step1", Task::new("First step"))
            .add_task_node("step2", Task::new("Second step"))
            .connect("step1", "step2", None)
            .set_entry("step1")
            .build()
            .unwrap();

        assert_eq!(workflow.nodes.len(), 2);
        assert_eq!(workflow.edges.len(), 1);
        assert_eq!(workflow.entry_node, "step1");
    }

    #[test]
    fn test_condition_evaluation() {
        let ctx = serde_json::json!({
            "approved": true,
            "amount": 100,
            "name": "test"
        });

        assert!(Condition::when("approved", true).evaluate(&ctx));
        assert!(!Condition::when("approved", false).evaluate(&ctx));
        assert!(Condition::when_gt("amount", 50).evaluate(&ctx));
        assert!(!Condition::when_gt("amount", 150).evaluate(&ctx));
    }

    #[tokio::test]
    async fn test_simple_workflow_execution() {
        let workflow = WorkflowBuilder::new("simple")
            .add_task_node("start", Task::new("Start task"))
            .add_task_node("end", Task::new("End task"))
            .connect("start", "end", None)
            .set_entry("start")
            .build()
            .unwrap();

        let mut runner = WorkflowRunner::new(workflow);
        let result = runner.run().await.unwrap();

        assert_eq!(result.status, WorkflowStatus::Success);
        assert_eq!(result.trace.len(), 2);
    }

    #[tokio::test]
    async fn test_conditional_workflow() {
        let workflow = WorkflowBuilder::new("conditional")
            .add_task_node("check", Task::new("Check condition"))
            .add_task_node("yes_path", Task::new("Yes path"))
            .add_task_node("no_path", Task::new("No path"))
            .connect("check", "yes_path", Some(Condition::when("approved", true)))
            .connect("check", "no_path", Some(Condition::when("approved", false)))
            .set_entry("check")
            .build()
            .unwrap();

        // Test with approved = true
        let ctx = WorkflowContext::with_data(serde_json::json!({"approved": true}));
        let mut runner = WorkflowRunner::with_context(workflow.clone(), ctx);
        let result = runner.run().await.unwrap();

        assert!(result.trace.iter().any(|t| t.node_id == "yes_path"));
        assert!(!result.trace.iter().any(|t| t.node_id == "no_path"));
    }

    #[test]
    fn test_workflow_validation() {
        // Missing entry node
        let result = WorkflowBuilder::new("test")
            .add_task_node("step1", Task::new("Step"))
            .build();

        assert!(result.is_err());

        // Invalid entry node
        let result = WorkflowBuilder::new("test")
            .add_task_node("step1", Task::new("Step"))
            .set_entry("nonexistent")
            .build();

        assert!(result.is_err());

        // Invalid edge
        let result = WorkflowBuilder::new("test")
            .add_task_node("step1", Task::new("Step"))
            .connect("step1", "nonexistent", None)
            .set_entry("step1")
            .build();

        assert!(result.is_err());
    }
}
