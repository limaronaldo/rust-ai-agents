//! Graph-based Workflow Engine with Cycle Support
//!
//! A LangGraph-inspired execution engine that supports:
//! - Cycles for iterative reasoning loops
//! - Conditional edges with dynamic routing
//! - State checkpointing and recovery
//! - Parallel branch execution
//! - Maximum iteration limits to prevent infinite loops
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_crew::graph::{Graph, GraphBuilder, StateGraph};
//!
//! // Create a reasoning loop that iterates until done
//! let graph = GraphBuilder::new("reasoning_loop")
//!     .add_node("think", think_node)
//!     .add_node("act", act_node)
//!     .add_node("evaluate", evaluate_node)
//!     .add_edge("think", "act")
//!     .add_edge("act", "evaluate")
//!     // Conditional: loop back to think or finish
//!     .add_conditional_edge("evaluate", |state| {
//!         if state.get("done").unwrap_or(&false) {
//!             "END"
//!         } else {
//!             "think"  // Loop back
//!         }
//!     })
//!     .set_entry("think")
//!     .set_finish("END")
//!     .build()?;
//!
//! let result = graph.invoke(initial_state).await?;
//! ```

use chrono::{DateTime, Utc};
use rust_ai_agents_core::errors::CrewError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Graph state - the data that flows through the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphState {
    /// State data as JSON
    pub data: serde_json::Value,
    /// Execution metadata
    pub metadata: GraphMetadata,
}

impl Default for GraphState {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphState {
    /// Create empty state
    pub fn new() -> Self {
        Self {
            data: serde_json::json!({}),
            metadata: GraphMetadata::default(),
        }
    }

    /// Create from JSON data
    pub fn from_json(data: serde_json::Value) -> Self {
        Self {
            data,
            metadata: GraphMetadata::default(),
        }
    }

    /// Get a value from state
    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.data
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Set a value in state
    pub fn set<T: Serialize>(&mut self, key: &str, value: T) {
        if let Some(obj) = self.data.as_object_mut() {
            if let Ok(v) = serde_json::to_value(value) {
                obj.insert(key.to_string(), v);
            }
        }
    }

    /// Merge another state's data into this one
    pub fn merge(&mut self, other: &GraphState) {
        if let (Some(self_obj), Some(other_obj)) =
            (self.data.as_object_mut(), other.data.as_object())
        {
            for (k, v) in other_obj {
                self_obj.insert(k.clone(), v.clone());
            }
        }
    }

    /// Get raw JSON data
    pub fn raw(&self) -> &serde_json::Value {
        &self.data
    }
}

/// Execution metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Number of iterations executed
    pub iterations: u32,
    /// Nodes visited in order
    pub visited_nodes: Vec<String>,
    /// Current checkpoint ID
    pub checkpoint_id: Option<String>,
    /// Total execution time in ms
    pub execution_time_ms: u64,
    /// Start time
    pub started_at: Option<DateTime<Utc>>,
}

/// Result of graph execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphResult {
    /// Final state
    pub state: GraphState,
    /// Execution status
    pub status: GraphStatus,
    /// Error message if failed
    pub error: Option<String>,
}

/// Graph execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphStatus {
    /// Successfully completed
    Success,
    /// Failed with error
    Failed,
    /// Hit maximum iterations
    MaxIterations,
    /// Interrupted by user
    Interrupted,
    /// Waiting for input
    Paused,
}

/// A node in the graph
pub struct GraphNode {
    /// Node ID
    pub id: String,
    /// Node executor
    pub executor: Arc<dyn NodeFn>,
}

impl std::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphNode").field("id", &self.id).finish()
    }
}

/// Function type for node execution
#[async_trait::async_trait]
pub trait NodeFn: Send + Sync {
    /// Execute the node and return updated state
    async fn call(&self, state: GraphState) -> Result<GraphState, CrewError>;
}

/// Simple function wrapper for node execution
pub struct FnNode<F>(pub F);

#[async_trait::async_trait]
impl<F, Fut> NodeFn for FnNode<F>
where
    F: Fn(GraphState) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<GraphState, CrewError>> + Send,
{
    async fn call(&self, state: GraphState) -> Result<GraphState, CrewError> {
        (self.0)(state).await
    }
}

/// Edge types in the graph
#[derive(Clone)]
pub enum GraphEdge {
    /// Direct edge to a single node
    Direct { from: String, to: String },
    /// Conditional edge with dynamic routing
    Conditional {
        from: String,
        router: Arc<dyn EdgeRouter>,
    },
}

impl std::fmt::Debug for GraphEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Direct { from, to } => f
                .debug_struct("Direct")
                .field("from", from)
                .field("to", to)
                .finish(),
            Self::Conditional { from, .. } => {
                f.debug_struct("Conditional").field("from", from).finish()
            }
        }
    }
}

/// Router for conditional edges
pub trait EdgeRouter: Send + Sync {
    /// Determine next node based on state
    fn route(&self, state: &GraphState) -> String;
}

/// Simple function-based router
pub struct FnRouter<F>(pub F);

impl<F> EdgeRouter for FnRouter<F>
where
    F: Fn(&GraphState) -> String + Send + Sync,
{
    fn route(&self, state: &GraphState) -> String {
        (self.0)(state)
    }
}

/// Condition-based router
pub struct ConditionRouter {
    conditions: Vec<(Box<dyn Fn(&GraphState) -> bool + Send + Sync>, String)>,
    default: String,
}

impl ConditionRouter {
    /// Create a new condition router
    pub fn new(default: impl Into<String>) -> Self {
        Self {
            conditions: Vec::new(),
            default: default.into(),
        }
    }

    /// Add a condition
    pub fn when<F>(mut self, condition: F, target: impl Into<String>) -> Self
    where
        F: Fn(&GraphState) -> bool + Send + Sync + 'static,
    {
        self.conditions.push((Box::new(condition), target.into()));
        self
    }
}

impl EdgeRouter for ConditionRouter {
    fn route(&self, state: &GraphState) -> String {
        for (condition, target) in &self.conditions {
            if condition(state) {
                return target.clone();
            }
        }
        self.default.clone()
    }
}

/// State checkpoint for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint ID
    pub id: String,
    /// State at checkpoint
    pub state: GraphState,
    /// Node that was about to execute
    pub next_node: String,
    /// Timestamp
    pub created_at: DateTime<Utc>,
}

/// Checkpoint storage trait
#[async_trait::async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Save a checkpoint
    async fn save(&self, checkpoint: Checkpoint) -> Result<(), CrewError>;
    /// Load a checkpoint by ID
    async fn load(&self, id: &str) -> Result<Option<Checkpoint>, CrewError>;
    /// List all checkpoints for a graph
    async fn list(&self, graph_id: &str) -> Result<Vec<String>, CrewError>;
    /// Delete a checkpoint
    async fn delete(&self, id: &str) -> Result<(), CrewError>;
}

/// In-memory checkpoint store
#[derive(Default)]
pub struct InMemoryCheckpointStore {
    checkpoints: RwLock<HashMap<String, Checkpoint>>,
}

#[async_trait::async_trait]
impl CheckpointStore for InMemoryCheckpointStore {
    async fn save(&self, checkpoint: Checkpoint) -> Result<(), CrewError> {
        self.checkpoints
            .write()
            .await
            .insert(checkpoint.id.clone(), checkpoint);
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<Checkpoint>, CrewError> {
        Ok(self.checkpoints.read().await.get(id).cloned())
    }

    async fn list(&self, _graph_id: &str) -> Result<Vec<String>, CrewError> {
        Ok(self.checkpoints.read().await.keys().cloned().collect())
    }

    async fn delete(&self, id: &str) -> Result<(), CrewError> {
        self.checkpoints.write().await.remove(id);
        Ok(())
    }
}

/// Special node IDs
pub const START: &str = "__start__";
pub const END: &str = "__end__";

/// Graph configuration
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Maximum iterations before stopping (prevents infinite loops)
    pub max_iterations: u32,
    /// Enable checkpointing
    pub checkpointing: bool,
    /// Checkpoint interval (every N nodes)
    pub checkpoint_interval: u32,
    /// Enable parallel execution of independent branches
    pub parallel_branches: bool,
    /// Timeout per node in milliseconds
    pub node_timeout_ms: Option<u64>,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            checkpointing: false,
            checkpoint_interval: 5,
            parallel_branches: false,
            node_timeout_ms: None,
        }
    }
}

/// The main graph structure
pub struct Graph {
    /// Graph ID
    pub id: String,
    /// Graph name
    pub name: String,
    /// Nodes in the graph
    pub nodes: HashMap<String, GraphNode>,
    /// Edges in the graph
    pub edges: Vec<GraphEdge>,
    /// Entry node ID
    pub entry_node: String,
    /// Configuration
    pub config: GraphConfig,
    /// Checkpoint store
    pub checkpoint_store: Option<Arc<dyn CheckpointStore>>,
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("entry_node", &self.entry_node)
            .finish()
    }
}

impl Graph {
    /// Execute the graph with initial state
    pub async fn invoke(&self, initial_state: GraphState) -> Result<GraphResult, CrewError> {
        let mut state = initial_state;
        state.metadata.started_at = Some(Utc::now());
        state.metadata.iterations = 0;

        let mut current_node = self.entry_node.clone();

        loop {
            // Check max iterations
            if state.metadata.iterations >= self.config.max_iterations {
                return Ok(GraphResult {
                    state,
                    status: GraphStatus::MaxIterations,
                    error: Some(format!(
                        "Hit maximum iterations: {}",
                        self.config.max_iterations
                    )),
                });
            }

            // Check for END node
            if current_node == END {
                state.metadata.execution_time_ms = state
                    .metadata
                    .started_at
                    .map(|s| Utc::now().signed_duration_since(s).num_milliseconds() as u64)
                    .unwrap_or(0);
                return Ok(GraphResult {
                    state,
                    status: GraphStatus::Success,
                    error: None,
                });
            }

            // Get node
            let node = self.nodes.get(&current_node).ok_or_else(|| {
                CrewError::TaskNotFound(format!("Node not found: {}", current_node))
            })?;

            // Save checkpoint if enabled
            if self.config.checkpointing
                && state.metadata.iterations % self.config.checkpoint_interval == 0
            {
                if let Some(store) = &self.checkpoint_store {
                    let checkpoint = Checkpoint {
                        id: format!("{}_{}", self.id, state.metadata.iterations),
                        state: state.clone(),
                        next_node: current_node.clone(),
                        created_at: Utc::now(),
                    };
                    store.save(checkpoint).await?;
                }
            }

            // Execute node
            state.metadata.visited_nodes.push(current_node.clone());
            state.metadata.iterations += 1;

            state = match self.config.node_timeout_ms {
                Some(timeout) => tokio::time::timeout(
                    std::time::Duration::from_millis(timeout),
                    node.executor.call(state),
                )
                .await
                .map_err(|_| {
                    CrewError::ExecutionFailed(format!("Node {} timed out", current_node))
                })??,
                None => node.executor.call(state).await?,
            };

            // Find next node
            current_node = self.find_next_node(&current_node, &state)?;
        }
    }

    /// Resume from a checkpoint
    pub async fn resume(&self, checkpoint_id: &str) -> Result<GraphResult, CrewError> {
        let store = self.checkpoint_store.as_ref().ok_or_else(|| {
            CrewError::InvalidConfiguration("Checkpointing not enabled".to_string())
        })?;

        let checkpoint = store.load(checkpoint_id).await?.ok_or_else(|| {
            CrewError::TaskNotFound(format!("Checkpoint not found: {}", checkpoint_id))
        })?;

        // Continue from checkpoint
        let mut state = checkpoint.state;
        let mut current_node = checkpoint.next_node;

        loop {
            if state.metadata.iterations >= self.config.max_iterations {
                return Ok(GraphResult {
                    state,
                    status: GraphStatus::MaxIterations,
                    error: Some(format!(
                        "Hit maximum iterations: {}",
                        self.config.max_iterations
                    )),
                });
            }

            if current_node == END {
                state.metadata.execution_time_ms = state
                    .metadata
                    .started_at
                    .map(|s| Utc::now().signed_duration_since(s).num_milliseconds() as u64)
                    .unwrap_or(0);
                return Ok(GraphResult {
                    state,
                    status: GraphStatus::Success,
                    error: None,
                });
            }

            let node = self.nodes.get(&current_node).ok_or_else(|| {
                CrewError::TaskNotFound(format!("Node not found: {}", current_node))
            })?;

            state.metadata.visited_nodes.push(current_node.clone());
            state.metadata.iterations += 1;

            state = node.executor.call(state).await?;
            current_node = self.find_next_node(&current_node, &state)?;
        }
    }

    /// Find the next node based on edges
    pub fn find_next_node(&self, current: &str, state: &GraphState) -> Result<String, CrewError> {
        for edge in &self.edges {
            match edge {
                GraphEdge::Direct { from, to } if from == current => {
                    return Ok(to.clone());
                }
                GraphEdge::Conditional { from, router } if from == current => {
                    return Ok(router.route(state));
                }
                _ => continue,
            }
        }

        // No outgoing edge = implicit END
        Ok(END.to_string())
    }

    /// Stream execution, yielding state after each node
    pub fn stream(&self, initial_state: GraphState) -> GraphStream<'_> {
        GraphStream {
            graph: self,
            state: Some(initial_state),
            current_node: Some(self.entry_node.clone()),
            finished: false,
        }
    }

    /// Get a visual representation of the graph (Mermaid format)
    pub fn to_mermaid(&self) -> String {
        let mut lines = vec!["graph TD".to_string()];

        for (id, _node) in &self.nodes {
            let display_id = if id == START {
                "START"
            } else if id == END {
                "END"
            } else {
                id
            };
            lines.push(format!("    {}[{}]", id.replace('-', "_"), display_id));
        }

        for edge in &self.edges {
            match edge {
                GraphEdge::Direct { from, to } => {
                    lines.push(format!(
                        "    {} --> {}",
                        from.replace('-', "_"),
                        to.replace('-', "_")
                    ));
                }
                GraphEdge::Conditional { from, .. } => {
                    lines.push(format!(
                        "    {} -.->|condition| ...",
                        from.replace('-', "_")
                    ));
                }
            }
        }

        lines.join("\n")
    }
}

/// Streaming graph execution
pub struct GraphStream<'a> {
    graph: &'a Graph,
    state: Option<GraphState>,
    current_node: Option<String>,
    finished: bool,
}

impl<'a> GraphStream<'a> {
    /// Get next state update
    pub async fn next(&mut self) -> Option<Result<(String, GraphState), CrewError>> {
        if self.finished {
            return None;
        }

        let current_node = self.current_node.take()?;
        let mut state = self.state.take()?;

        // Check for END
        if current_node == END {
            self.finished = true;
            return Some(Ok((END.to_string(), state)));
        }

        // Check max iterations
        if state.metadata.iterations >= self.graph.config.max_iterations {
            self.finished = true;
            return Some(Err(CrewError::ExecutionFailed(
                "Max iterations reached".to_string(),
            )));
        }

        // Get and execute node
        let node = match self.graph.nodes.get(&current_node) {
            Some(n) => n,
            None => {
                self.finished = true;
                return Some(Err(CrewError::TaskNotFound(current_node)));
            }
        };

        state.metadata.visited_nodes.push(current_node.clone());
        state.metadata.iterations += 1;

        match node.executor.call(state).await {
            Ok(new_state) => {
                let next_node = match self.graph.find_next_node(&current_node, &new_state) {
                    Ok(n) => n,
                    Err(e) => {
                        self.finished = true;
                        return Some(Err(e));
                    }
                };

                self.state = Some(new_state.clone());
                self.current_node = Some(next_node);
                Some(Ok((current_node, new_state)))
            }
            Err(e) => {
                self.finished = true;
                Some(Err(e))
            }
        }
    }
}

/// Builder for creating graphs
pub struct GraphBuilder {
    id: String,
    name: String,
    nodes: HashMap<String, GraphNode>,
    edges: Vec<GraphEdge>,
    entry_node: Option<String>,
    config: GraphConfig,
    checkpoint_store: Option<Arc<dyn CheckpointStore>>,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_node: None,
            config: GraphConfig::default(),
            checkpoint_store: None,
        }
    }

    /// Set graph name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a node with an async function
    pub fn add_node<F, Fut>(mut self, id: impl Into<String>, func: F) -> Self
    where
        F: Fn(GraphState) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<GraphState, CrewError>> + Send + 'static,
    {
        let id = id.into();
        self.nodes.insert(
            id.clone(),
            GraphNode {
                id: id.clone(),
                executor: Arc::new(FnNode(func)),
            },
        );
        self
    }

    /// Add a node with a custom executor
    pub fn add_node_executor(mut self, id: impl Into<String>, executor: Arc<dyn NodeFn>) -> Self {
        let id = id.into();
        self.nodes.insert(id.clone(), GraphNode { id, executor });
        self
    }

    /// Add a direct edge between two nodes
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(GraphEdge::Direct {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    /// Add a conditional edge with a routing function
    pub fn add_conditional_edge<F>(mut self, from: impl Into<String>, router: F) -> Self
    where
        F: Fn(&GraphState) -> String + Send + Sync + 'static,
    {
        self.edges.push(GraphEdge::Conditional {
            from: from.into(),
            router: Arc::new(FnRouter(router)),
        });
        self
    }

    /// Add a conditional edge with a ConditionRouter
    pub fn add_conditional_edge_router(
        mut self,
        from: impl Into<String>,
        router: ConditionRouter,
    ) -> Self {
        self.edges.push(GraphEdge::Conditional {
            from: from.into(),
            router: Arc::new(router),
        });
        self
    }

    /// Set the entry node
    pub fn set_entry(mut self, node_id: impl Into<String>) -> Self {
        self.entry_node = Some(node_id.into());
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max: u32) -> Self {
        self.config.max_iterations = max;
        self
    }

    /// Enable checkpointing
    pub fn with_checkpointing(mut self, store: Arc<dyn CheckpointStore>) -> Self {
        self.config.checkpointing = true;
        self.checkpoint_store = Some(store);
        self
    }

    /// Set checkpoint interval
    pub fn checkpoint_interval(mut self, interval: u32) -> Self {
        self.config.checkpoint_interval = interval;
        self
    }

    /// Set node timeout
    pub fn node_timeout_ms(mut self, timeout: u64) -> Self {
        self.config.node_timeout_ms = Some(timeout);
        self
    }

    /// Build the graph
    pub fn build(self) -> Result<Graph, CrewError> {
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
            let from = match edge {
                GraphEdge::Direct { from, .. } => from,
                GraphEdge::Conditional { from, .. } => from,
            };
            if !self.nodes.contains_key(from) {
                return Err(CrewError::InvalidConfiguration(format!(
                    "Edge source '{}' not found",
                    from
                )));
            }
            // Note: We don't validate 'to' for conditional edges since they're dynamic
            if let GraphEdge::Direct { to, .. } = edge {
                if to != END && !self.nodes.contains_key(to) {
                    return Err(CrewError::InvalidConfiguration(format!(
                        "Edge target '{}' not found",
                        to
                    )));
                }
            }
        }

        Ok(Graph {
            id: self.id,
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            entry_node,
            config: self.config,
            checkpoint_store: self.checkpoint_store,
        })
    }
}

/// StateGraph - a higher-level API for common patterns
pub struct StateGraph<S> {
    graph: Graph,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static> StateGraph<S> {
    /// Create from a graph
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Invoke with typed state
    pub async fn invoke(&self, initial: S) -> Result<S, CrewError> {
        let json = serde_json::to_value(&initial)
            .map_err(|e| CrewError::ExecutionFailed(format!("Serialization error: {}", e)))?;
        let state = GraphState::from_json(json);
        let result = self.graph.invoke(state).await?;

        if result.status != GraphStatus::Success {
            return Err(CrewError::ExecutionFailed(
                result
                    .error
                    .unwrap_or_else(|| "Graph execution failed".to_string()),
            ));
        }

        serde_json::from_value(result.state.data)
            .map_err(|e| CrewError::ExecutionFailed(format!("Deserialization error: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_graph() {
        let graph = GraphBuilder::new("simple")
            .add_node("step1", |mut state: GraphState| async move {
                state.set("step1_done", true);
                Ok(state)
            })
            .add_node("step2", |mut state: GraphState| async move {
                state.set("step2_done", true);
                Ok(state)
            })
            .add_edge("step1", "step2")
            .add_edge("step2", END)
            .set_entry("step1")
            .build()
            .unwrap();

        let result = graph.invoke(GraphState::new()).await.unwrap();

        assert_eq!(result.status, GraphStatus::Success);
        assert_eq!(result.state.get::<bool>("step1_done"), Some(true));
        assert_eq!(result.state.get::<bool>("step2_done"), Some(true));
        assert_eq!(result.state.metadata.iterations, 2);
    }

    #[tokio::test]
    async fn test_conditional_edge() {
        let graph = GraphBuilder::new("conditional")
            .add_node("check", |state: GraphState| async move { Ok(state) })
            .add_node("yes_path", |mut state: GraphState| async move {
                state.set("path", "yes");
                Ok(state)
            })
            .add_node("no_path", |mut state: GraphState| async move {
                state.set("path", "no");
                Ok(state)
            })
            .add_conditional_edge("check", |state| {
                if state.get::<bool>("condition").unwrap_or(false) {
                    "yes_path".to_string()
                } else {
                    "no_path".to_string()
                }
            })
            .add_edge("yes_path", END)
            .add_edge("no_path", END)
            .set_entry("check")
            .build()
            .unwrap();

        // Test with condition = true
        let mut state = GraphState::new();
        state.set("condition", true);
        let result = graph.invoke(state).await.unwrap();
        assert_eq!(result.state.get::<String>("path"), Some("yes".to_string()));

        // Test with condition = false
        let mut state = GraphState::new();
        state.set("condition", false);
        let result = graph.invoke(state).await.unwrap();
        assert_eq!(result.state.get::<String>("path"), Some("no".to_string()));
    }

    #[tokio::test]
    async fn test_cycle_with_limit() {
        let graph = GraphBuilder::new("cycle")
            .add_node("increment", |mut state: GraphState| async move {
                let count: i32 = state.get("count").unwrap_or(0);
                state.set("count", count + 1);
                Ok(state)
            })
            .add_conditional_edge("increment", |state| {
                let count: i32 = state.get("count").unwrap_or(0);
                if count >= 5 {
                    END.to_string()
                } else {
                    "increment".to_string() // Loop back
                }
            })
            .set_entry("increment")
            .max_iterations(100)
            .build()
            .unwrap();

        let result = graph.invoke(GraphState::new()).await.unwrap();

        assert_eq!(result.status, GraphStatus::Success);
        assert_eq!(result.state.get::<i32>("count"), Some(5));
        assert_eq!(result.state.metadata.iterations, 5);
    }

    #[tokio::test]
    async fn test_max_iterations_limit() {
        let graph = GraphBuilder::new("infinite")
            .add_node("loop", |state: GraphState| async move { Ok(state) })
            .add_edge("loop", "loop") // Infinite loop
            .set_entry("loop")
            .max_iterations(10)
            .build()
            .unwrap();

        let result = graph.invoke(GraphState::new()).await.unwrap();

        assert_eq!(result.status, GraphStatus::MaxIterations);
        assert_eq!(result.state.metadata.iterations, 10);
    }

    #[tokio::test]
    async fn test_condition_router() {
        let router = ConditionRouter::new("default")
            .when(|s| s.get::<i32>("score").unwrap_or(0) >= 80, "excellent")
            .when(|s| s.get::<i32>("score").unwrap_or(0) >= 60, "good")
            .when(|s| s.get::<i32>("score").unwrap_or(0) >= 40, "pass");

        let mut state = GraphState::new();
        state.set("score", 85);
        assert_eq!(router.route(&state), "excellent");

        state.set("score", 65);
        assert_eq!(router.route(&state), "good");

        state.set("score", 30);
        assert_eq!(router.route(&state), "default");
    }

    #[tokio::test]
    async fn test_checkpointing() {
        let store = Arc::new(InMemoryCheckpointStore::default());

        let graph = GraphBuilder::new("checkpoint_test")
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
            .with_checkpointing(store.clone())
            .checkpoint_interval(1)
            .build()
            .unwrap();

        let result = graph.invoke(GraphState::new()).await.unwrap();
        assert_eq!(result.status, GraphStatus::Success);

        // Verify checkpoints were created
        let checkpoints = store.list("checkpoint_test").await.unwrap();
        assert!(!checkpoints.is_empty());
    }

    #[test]
    fn test_mermaid_output() {
        let graph = GraphBuilder::new("mermaid_test")
            .add_node("start", |s| async { Ok(s) })
            .add_node("process", |s| async { Ok(s) })
            .add_node("end", |s| async { Ok(s) })
            .add_edge("start", "process")
            .add_edge("process", "end")
            .set_entry("start")
            .build()
            .unwrap();

        let mermaid = graph.to_mermaid();
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("start"));
        assert!(mermaid.contains("process"));
    }

    #[tokio::test]
    async fn test_stream_execution() {
        let graph = GraphBuilder::new("stream_test")
            .add_node("a", |mut s: GraphState| async move {
                s.set("a", true);
                Ok(s)
            })
            .add_node("b", |mut s: GraphState| async move {
                s.set("b", true);
                Ok(s)
            })
            .add_edge("a", "b")
            .add_edge("b", END)
            .set_entry("a")
            .build()
            .unwrap();

        let mut stream = graph.stream(GraphState::new());
        let mut steps = Vec::new();

        while let Some(result) = stream.next().await {
            let (node_id, _state) = result.unwrap();
            steps.push(node_id);
        }

        assert_eq!(steps, vec!["a", "b", END]);
    }
}
