//! Subgraphs / Nested Workflows
//!
//! Enables composing graphs within graphs for modular, reusable workflow components.
//!
//! ## Features
//!
//! - Nest graphs as nodes within parent graphs
//! - State mapping between parent and child graphs
//! - Parallel subgraph execution
//! - Isolated or shared state modes
//! - Subgraph libraries for reusable components
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_crew::subgraph::{SubgraphNode, StateMapping};
//!
//! // Create a reusable research subgraph
//! let research_graph = GraphBuilder::new("research")
//!     .add_node("search", search_node)
//!     .add_node("analyze", analyze_node)
//!     .add_edge("search", "analyze")
//!     .set_entry("search")
//!     .build()?;
//!
//! // Create a reusable writing subgraph
//! let writing_graph = GraphBuilder::new("writing")
//!     .add_node("draft", draft_node)
//!     .add_node("edit", edit_node)
//!     .add_edge("draft", "edit")
//!     .set_entry("draft")
//!     .build()?;
//!
//! // Compose into a parent graph
//! let main_graph = GraphBuilder::new("main")
//!     .add_node("init", init_node)
//!     .add_subgraph("research", research_graph, StateMapping::default())
//!     .add_subgraph("writing", writing_graph, StateMapping::default())
//!     .add_edge("init", "research")
//!     .add_edge("research", "writing")
//!     .set_entry("init")
//!     .build()?;
//! ```

use crate::graph::{Graph, GraphState, GraphStatus, NodeFn};
use rust_ai_agents_core::errors::CrewError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// State mapping configuration between parent and child graphs
#[derive(Debug, Clone, Default)]
pub struct StateMapping {
    /// Map parent keys to child keys (parent_key -> child_key)
    pub input_mapping: HashMap<String, String>,
    /// Map child keys back to parent keys (child_key -> parent_key)
    pub output_mapping: HashMap<String, String>,
    /// Keys to pass through unchanged (same key in both)
    pub passthrough: Vec<String>,
    /// Whether to merge all child state back (vs only mapped keys)
    pub merge_all: bool,
    /// Prefix for child keys when merging all
    pub output_prefix: Option<String>,
}

impl StateMapping {
    /// Create a new state mapping
    pub fn new() -> Self {
        Self::default()
    }

    /// Map a parent key to a child key on input
    pub fn map_input(
        mut self,
        parent_key: impl Into<String>,
        child_key: impl Into<String>,
    ) -> Self {
        self.input_mapping
            .insert(parent_key.into(), child_key.into());
        self
    }

    /// Map a child key back to a parent key on output
    pub fn map_output(
        mut self,
        child_key: impl Into<String>,
        parent_key: impl Into<String>,
    ) -> Self {
        self.output_mapping
            .insert(child_key.into(), parent_key.into());
        self
    }

    /// Add a passthrough key (same name in parent and child)
    pub fn passthrough(mut self, key: impl Into<String>) -> Self {
        self.passthrough.push(key.into());
        self
    }

    /// Add multiple passthrough keys
    pub fn passthrough_keys(mut self, keys: Vec<String>) -> Self {
        self.passthrough.extend(keys);
        self
    }

    /// Merge all child state back to parent
    pub fn merge_all(mut self) -> Self {
        self.merge_all = true;
        self
    }

    /// Set prefix for merged keys
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.output_prefix = Some(prefix.into());
        self.merge_all = true;
        self
    }

    /// Apply input mapping: parent state -> child state
    pub fn apply_input(&self, parent_state: &GraphState) -> GraphState {
        let mut child_state = GraphState::new();

        // Apply explicit mappings
        if let Some(parent_obj) = parent_state.data.as_object() {
            // Input mappings
            for (parent_key, child_key) in &self.input_mapping {
                if let Some(value) = parent_obj.get(parent_key) {
                    child_state.set(child_key, value.clone());
                }
            }

            // Passthrough keys
            for key in &self.passthrough {
                if let Some(value) = parent_obj.get(key) {
                    child_state.set(key, value.clone());
                }
            }

            // If merge_all and no specific input mappings, pass all parent state
            if self.merge_all && self.input_mapping.is_empty() {
                for (key, value) in parent_obj {
                    child_state.set(key, value.clone());
                }
            }
        }

        child_state
    }

    /// Apply output mapping: child state -> updated parent state
    pub fn apply_output(&self, parent_state: &GraphState, child_state: &GraphState) -> GraphState {
        let mut result = parent_state.clone();

        if let Some(child_obj) = child_state.data.as_object() {
            // Output mappings
            for (child_key, parent_key) in &self.output_mapping {
                if let Some(value) = child_obj.get(child_key) {
                    result.set(parent_key, value.clone());
                }
            }

            // Passthrough keys
            for key in &self.passthrough {
                if let Some(value) = child_obj.get(key) {
                    result.set(key, value.clone());
                }
            }

            // Merge all with optional prefix
            if self.merge_all {
                for (key, value) in child_obj {
                    let target_key = match &self.output_prefix {
                        Some(prefix) => format!("{}.{}", prefix, key),
                        None => key.clone(),
                    };
                    result.set(&target_key, value.clone());
                }
            }
        }

        result
    }
}

/// A node that executes a subgraph
pub struct SubgraphNode {
    /// The subgraph to execute
    graph: Arc<Graph>,
    /// State mapping configuration
    mapping: StateMapping,
    /// Optional name for this subgraph instance
    name: Option<String>,
}

impl SubgraphNode {
    /// Create a new subgraph node
    pub fn new(graph: Graph) -> Self {
        Self {
            graph: Arc::new(graph),
            mapping: StateMapping::default(),
            name: None,
        }
    }

    /// Create with state mapping
    pub fn with_mapping(graph: Graph, mapping: StateMapping) -> Self {
        Self {
            graph: Arc::new(graph),
            mapping,
            name: None,
        }
    }

    /// Set a name for this subgraph instance
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get the subgraph
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
}

#[async_trait::async_trait]
impl NodeFn for SubgraphNode {
    async fn call(&self, state: GraphState) -> Result<GraphState, CrewError> {
        // Apply input mapping
        let child_state = self.mapping.apply_input(&state);

        // Execute subgraph
        let result = self.graph.invoke(child_state).await?;

        // Check for errors
        if result.status != GraphStatus::Success {
            return Err(CrewError::ExecutionFailed(format!(
                "Subgraph '{}' failed: {}",
                self.name.as_deref().unwrap_or(&self.graph.name),
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            )));
        }

        // Apply output mapping
        let final_state = self.mapping.apply_output(&state, &result.state);

        Ok(final_state)
    }
}

/// Parallel subgraph executor - runs multiple subgraphs concurrently
pub struct ParallelSubgraphs {
    /// Subgraphs to execute in parallel
    subgraphs: Vec<(String, Arc<Graph>, StateMapping)>,
    /// How to merge results
    merge_strategy: MergeStrategy,
}

/// Strategy for merging parallel subgraph results
#[derive(Debug, Clone, Default)]
pub enum MergeStrategy {
    /// Merge all results, later overwrites earlier on conflicts
    #[default]
    MergeAll,
    /// Prefix each subgraph's output with its name
    Prefixed,
    /// Collect results into a map
    Collect,
    /// Custom merge function (not serializable)
    Custom,
}

impl ParallelSubgraphs {
    /// Create a new parallel subgraph executor
    pub fn new() -> Self {
        Self {
            subgraphs: Vec::new(),
            merge_strategy: MergeStrategy::default(),
        }
    }

    /// Add a subgraph
    pub fn add(mut self, name: impl Into<String>, graph: Graph, mapping: StateMapping) -> Self {
        self.subgraphs.push((name.into(), Arc::new(graph), mapping));
        self
    }

    /// Set merge strategy
    pub fn with_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Execute all subgraphs in parallel
    pub async fn execute(&self, state: &GraphState) -> Result<GraphState, CrewError> {
        use futures::future::join_all;

        // Prepare child states
        let tasks: Vec<_> = self
            .subgraphs
            .iter()
            .map(|(name, graph, mapping)| {
                let child_state = mapping.apply_input(state);
                let graph = Arc::clone(graph);
                let name = name.clone();
                let mapping = mapping.clone();

                async move {
                    let result = graph.invoke(child_state).await?;
                    if result.status != GraphStatus::Success {
                        return Err(CrewError::ExecutionFailed(format!(
                            "Parallel subgraph '{}' failed: {}",
                            name,
                            result.error.unwrap_or_else(|| "Unknown error".to_string())
                        )));
                    }
                    Ok((name, result.state, mapping))
                }
            })
            .collect();

        // Execute in parallel
        let results = join_all(tasks).await;

        // Check for errors
        let mut successful: Vec<(String, GraphState, StateMapping)> = Vec::new();
        for result in results {
            successful.push(result?);
        }

        // Merge results based on strategy
        let mut final_state = state.clone();
        match self.merge_strategy {
            MergeStrategy::MergeAll => {
                for (_, child_state, mapping) in successful {
                    final_state = mapping.apply_output(&final_state, &child_state);
                }
            }
            MergeStrategy::Prefixed => {
                for (name, child_state, _) in successful {
                    if let Some(obj) = child_state.data.as_object() {
                        for (key, value) in obj {
                            final_state.set(&format!("{}.{}", name, key), value.clone());
                        }
                    }
                }
            }
            MergeStrategy::Collect => {
                let mut collected = serde_json::Map::new();
                for (name, child_state, _) in successful {
                    collected.insert(name, child_state.data);
                }
                final_state.set("parallel_results", serde_json::Value::Object(collected));
            }
            MergeStrategy::Custom => {
                // Custom strategy would need a closure, which we can't easily support
                // For now, fall back to MergeAll
                for (_, child_state, mapping) in successful {
                    final_state = mapping.apply_output(&final_state, &child_state);
                }
            }
        }

        Ok(final_state)
    }
}

impl Default for ParallelSubgraphs {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl NodeFn for ParallelSubgraphs {
    async fn call(&self, state: GraphState) -> Result<GraphState, CrewError> {
        self.execute(&state).await
    }
}

/// Subgraph library for reusable components
#[derive(Default)]
pub struct SubgraphLibrary {
    graphs: HashMap<String, Arc<Graph>>,
}

impl SubgraphLibrary {
    /// Create a new library
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a graph in the library
    pub fn register(&mut self, name: impl Into<String>, graph: Graph) {
        self.graphs.insert(name.into(), Arc::new(graph));
    }

    /// Get a graph from the library
    pub fn get(&self, name: &str) -> Option<Arc<Graph>> {
        self.graphs.get(name).cloned()
    }

    /// Create a subgraph node from a library graph
    pub fn create_node(&self, name: &str, mapping: StateMapping) -> Option<SubgraphNode> {
        self.graphs.get(name).map(|graph| SubgraphNode {
            graph: Arc::clone(graph),
            mapping,
            name: Some(name.to_string()),
        })
    }

    /// List all registered graphs
    pub fn list(&self) -> Vec<&str> {
        self.graphs.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a graph is registered
    pub fn contains(&self, name: &str) -> bool {
        self.graphs.contains_key(name)
    }

    /// Remove a graph from the library
    pub fn remove(&mut self, name: &str) -> Option<Arc<Graph>> {
        self.graphs.remove(name)
    }
}

/// Conditional subgraph - executes one of several subgraphs based on state
pub struct ConditionalSubgraph {
    /// Branches with their conditions
    branches: Vec<(Box<dyn Fn(&GraphState) -> bool + Send + Sync>, SubgraphNode)>,
    /// Default branch if no condition matches
    default: Option<SubgraphNode>,
}

impl ConditionalSubgraph {
    /// Create a new conditional subgraph
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
            default: None,
        }
    }

    /// Add a conditional branch
    pub fn when<F>(mut self, condition: F, graph: Graph, mapping: StateMapping) -> Self
    where
        F: Fn(&GraphState) -> bool + Send + Sync + 'static,
    {
        self.branches.push((
            Box::new(condition),
            SubgraphNode::with_mapping(graph, mapping),
        ));
        self
    }

    /// Set the default branch
    pub fn otherwise(mut self, graph: Graph, mapping: StateMapping) -> Self {
        self.default = Some(SubgraphNode::with_mapping(graph, mapping));
        self
    }
}

impl Default for ConditionalSubgraph {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl NodeFn for ConditionalSubgraph {
    async fn call(&self, state: GraphState) -> Result<GraphState, CrewError> {
        // Find first matching condition
        for (condition, subgraph) in &self.branches {
            if condition(&state) {
                return subgraph.call(state).await;
            }
        }

        // Use default if no condition matched
        if let Some(default) = &self.default {
            return default.call(state).await;
        }

        // No match and no default - pass through unchanged
        Ok(state)
    }
}

/// Loop subgraph - executes a subgraph repeatedly until a condition is met
pub struct LoopSubgraph {
    /// The subgraph to loop
    subgraph: SubgraphNode,
    /// Continue condition (returns true to continue looping)
    continue_condition: Box<dyn Fn(&GraphState) -> bool + Send + Sync>,
    /// Maximum iterations
    max_iterations: u32,
}

impl LoopSubgraph {
    /// Create a new loop subgraph
    pub fn new<F>(graph: Graph, mapping: StateMapping, continue_while: F) -> Self
    where
        F: Fn(&GraphState) -> bool + Send + Sync + 'static,
    {
        Self {
            subgraph: SubgraphNode::with_mapping(graph, mapping),
            continue_condition: Box::new(continue_while),
            max_iterations: 100,
        }
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }
}

#[async_trait::async_trait]
impl NodeFn for LoopSubgraph {
    async fn call(&self, mut state: GraphState) -> Result<GraphState, CrewError> {
        let mut iterations = 0;

        while (self.continue_condition)(&state) && iterations < self.max_iterations {
            state = self.subgraph.call(state).await?;
            iterations += 1;
        }

        if iterations >= self.max_iterations {
            return Err(CrewError::ExecutionFailed(format!(
                "Loop subgraph exceeded max iterations: {}",
                self.max_iterations
            )));
        }

        Ok(state)
    }
}

/// Retry subgraph - retries a subgraph on failure
pub struct RetrySubgraph {
    /// The subgraph to retry
    subgraph: SubgraphNode,
    /// Maximum retry attempts
    max_retries: u32,
    /// Delay between retries in milliseconds
    retry_delay_ms: u64,
    /// Exponential backoff multiplier
    backoff_multiplier: f64,
}

impl RetrySubgraph {
    /// Create a new retry subgraph
    pub fn new(graph: Graph, mapping: StateMapping) -> Self {
        Self {
            subgraph: SubgraphNode::with_mapping(graph, mapping),
            max_retries: 3,
            retry_delay_ms: 100,
            backoff_multiplier: 2.0,
        }
    }

    /// Set maximum retries
    pub fn max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set retry delay
    pub fn retry_delay_ms(mut self, delay: u64) -> Self {
        self.retry_delay_ms = delay;
        self
    }

    /// Set backoff multiplier
    pub fn backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }
}

#[async_trait::async_trait]
impl NodeFn for RetrySubgraph {
    async fn call(&self, state: GraphState) -> Result<GraphState, CrewError> {
        let mut last_error = None;
        let mut delay = self.retry_delay_ms;

        for attempt in 0..=self.max_retries {
            match self.subgraph.call(state.clone()).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.max_retries {
                        tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                        delay = (delay as f64 * self.backoff_multiplier) as u64;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            CrewError::ExecutionFailed("Retry subgraph failed with unknown error".to_string())
        }))
    }
}

/// Extension trait for GraphBuilder to add subgraph nodes
pub trait GraphBuilderSubgraphExt {
    /// Add a subgraph as a node
    fn add_subgraph(self, id: impl Into<String>, graph: Graph, mapping: StateMapping) -> Self;

    /// Add a parallel subgraphs node
    fn add_parallel_subgraphs(self, id: impl Into<String>, parallel: ParallelSubgraphs) -> Self;

    /// Add a conditional subgraph node
    fn add_conditional_subgraph(
        self,
        id: impl Into<String>,
        conditional: ConditionalSubgraph,
    ) -> Self;

    /// Add a loop subgraph node
    fn add_loop_subgraph(self, id: impl Into<String>, loop_sg: LoopSubgraph) -> Self;

    /// Add a retry subgraph node
    fn add_retry_subgraph(self, id: impl Into<String>, retry: RetrySubgraph) -> Self;
}

impl GraphBuilderSubgraphExt for crate::graph::GraphBuilder {
    fn add_subgraph(self, id: impl Into<String>, graph: Graph, mapping: StateMapping) -> Self {
        self.add_node_executor(id, Arc::new(SubgraphNode::with_mapping(graph, mapping)))
    }

    fn add_parallel_subgraphs(self, id: impl Into<String>, parallel: ParallelSubgraphs) -> Self {
        self.add_node_executor(id, Arc::new(parallel))
    }

    fn add_conditional_subgraph(
        self,
        id: impl Into<String>,
        conditional: ConditionalSubgraph,
    ) -> Self {
        self.add_node_executor(id, Arc::new(conditional))
    }

    fn add_loop_subgraph(self, id: impl Into<String>, loop_sg: LoopSubgraph) -> Self {
        self.add_node_executor(id, Arc::new(loop_sg))
    }

    fn add_retry_subgraph(self, id: impl Into<String>, retry: RetrySubgraph) -> Self {
        self.add_node_executor(id, Arc::new(retry))
    }
}

/// Subgraph execution result with detailed info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphExecutionInfo {
    /// Name of the subgraph
    pub name: String,
    /// Execution status
    pub status: GraphStatus,
    /// Input state (after mapping)
    pub input_state: serde_json::Value,
    /// Output state (before mapping back)
    pub output_state: serde_json::Value,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Number of nodes executed
    pub nodes_executed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphBuilder, END};

    fn create_adder_graph(amount: i32) -> Graph {
        GraphBuilder::new(format!("adder_{}", amount))
            .add_node("add", move |mut state: GraphState| async move {
                let value: i32 = state.get("value").unwrap_or(0);
                state.set("value", value + amount);
                Ok(state)
            })
            .add_edge("add", END)
            .set_entry("add")
            .build()
            .unwrap()
    }

    fn create_multiplier_graph(factor: i32) -> Graph {
        GraphBuilder::new(format!("multiplier_{}", factor))
            .add_node("multiply", move |mut state: GraphState| async move {
                let value: i32 = state.get("value").unwrap_or(0);
                state.set("value", value * factor);
                Ok(state)
            })
            .add_edge("multiply", END)
            .set_entry("multiply")
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_simple_subgraph() {
        let adder = create_adder_graph(5);
        // Use merge_all to pass through all state
        let subgraph_node = SubgraphNode::with_mapping(adder, StateMapping::new().merge_all());

        let mut state = GraphState::new();
        state.set("value", 10);

        let result = subgraph_node.call(state).await.unwrap();
        let value: i32 = result.get("value").unwrap();
        assert_eq!(value, 15);
    }

    #[tokio::test]
    async fn test_subgraph_with_mapping() {
        let adder = create_adder_graph(5);
        let mapping = StateMapping::new()
            .map_input("input_value", "value")
            .map_output("value", "output_value");

        let subgraph_node = SubgraphNode::with_mapping(adder, mapping);

        let mut state = GraphState::new();
        state.set("input_value", 10);

        let result = subgraph_node.call(state).await.unwrap();
        let value: i32 = result.get("output_value").unwrap();
        assert_eq!(value, 15);
    }

    #[tokio::test]
    async fn test_nested_subgraphs() {
        // Inner subgraph: add 5
        let inner = create_adder_graph(5);

        // Outer subgraph: contains inner, then multiplies by 2
        let outer = GraphBuilder::new("outer")
            .add_subgraph("add_step", inner, StateMapping::new().passthrough("value"))
            .add_node("multiply", |mut state: GraphState| async move {
                let value: i32 = state.get("value").unwrap_or(0);
                state.set("value", value * 2);
                Ok(state)
            })
            .add_edge("add_step", "multiply")
            .add_edge("multiply", END)
            .set_entry("add_step")
            .build()
            .unwrap();

        let mut state = GraphState::new();
        state.set("value", 10);

        let result = outer.invoke(state).await.unwrap();
        let value: i32 = result.state.get("value").unwrap();
        assert_eq!(value, 30); // (10 + 5) * 2 = 30
    }

    #[tokio::test]
    async fn test_parallel_subgraphs() {
        let adder = create_adder_graph(5);
        let multiplier = create_multiplier_graph(3);

        let parallel = ParallelSubgraphs::new()
            .add("adder", adder, StateMapping::new().passthrough("value"))
            .add(
                "multiplier",
                multiplier,
                StateMapping::new().passthrough("value"),
            )
            .with_strategy(MergeStrategy::Prefixed);

        let mut state = GraphState::new();
        state.set("value", 10);

        let result = parallel.call(state).await.unwrap();

        // With Prefixed strategy, results are prefixed with subgraph name
        let added: i32 = result.get("adder.value").unwrap();
        let multiplied: i32 = result.get("multiplier.value").unwrap();

        assert_eq!(added, 15);
        assert_eq!(multiplied, 30);
    }

    #[tokio::test]
    async fn test_parallel_subgraphs_collect() {
        let adder = create_adder_graph(5);
        let multiplier = create_multiplier_graph(3);

        let parallel = ParallelSubgraphs::new()
            .add("adder", adder, StateMapping::new().passthrough("value"))
            .add(
                "multiplier",
                multiplier,
                StateMapping::new().passthrough("value"),
            )
            .with_strategy(MergeStrategy::Collect);

        let mut state = GraphState::new();
        state.set("value", 10);

        let result = parallel.call(state).await.unwrap();

        // With Collect strategy, results are in parallel_results map
        let parallel_results: serde_json::Value = result.get("parallel_results").unwrap();
        let added = parallel_results["adder"]["value"].as_i64().unwrap();
        let multiplied = parallel_results["multiplier"]["value"].as_i64().unwrap();

        assert_eq!(added, 15);
        assert_eq!(multiplied, 30);
    }

    #[tokio::test]
    async fn test_conditional_subgraph() {
        let adder = create_adder_graph(10);
        let multiplier = create_multiplier_graph(2);

        let conditional = ConditionalSubgraph::new()
            .when(
                |state| state.get::<bool>("use_addition").unwrap_or(false),
                adder,
                StateMapping::new().passthrough("value"),
            )
            .otherwise(multiplier, StateMapping::new().passthrough("value"));

        // Test addition branch
        let mut state1 = GraphState::new();
        state1.set("value", 5);
        state1.set("use_addition", true);
        let result1 = conditional.call(state1).await.unwrap();
        assert_eq!(result1.get::<i32>("value").unwrap(), 15);

        // Test multiplication branch (default)
        let mut state2 = GraphState::new();
        state2.set("value", 5);
        state2.set("use_addition", false);
        let result2 = conditional.call(state2).await.unwrap();
        assert_eq!(result2.get::<i32>("value").unwrap(), 10);
    }

    #[tokio::test]
    async fn test_loop_subgraph() {
        let adder = create_adder_graph(1);

        let loop_sg = LoopSubgraph::new(adder, StateMapping::new().passthrough("value"), |state| {
            state.get::<i32>("value").unwrap_or(0) < 5
        })
        .max_iterations(10);

        let mut state = GraphState::new();
        state.set("value", 0);

        let result = loop_sg.call(state).await.unwrap();
        let value: i32 = result.get("value").unwrap();
        assert_eq!(value, 5); // Loop until value >= 5
    }

    #[tokio::test]
    async fn test_subgraph_library() {
        let mut library = SubgraphLibrary::new();
        library.register("add5", create_adder_graph(5));
        library.register("mult2", create_multiplier_graph(2));

        assert!(library.contains("add5"));
        assert!(library.contains("mult2"));
        assert!(!library.contains("unknown"));

        let node = library
            .create_node("add5", StateMapping::new().passthrough("value"))
            .unwrap();
        let mut state = GraphState::new();
        state.set("value", 10);

        let result = node.call(state).await.unwrap();
        assert_eq!(result.get::<i32>("value").unwrap(), 15);
    }

    #[tokio::test]
    async fn test_state_mapping_merge_all() {
        let graph = GraphBuilder::new("multi_output")
            .add_node("compute", |mut state: GraphState| async move {
                let value: i32 = state.get("value").unwrap_or(0);
                state.set("doubled", value * 2);
                state.set("tripled", value * 3);
                state.set("squared", value * value);
                Ok(state)
            })
            .add_edge("compute", END)
            .set_entry("compute")
            .build()
            .unwrap();

        let mapping = StateMapping::new().passthrough("value").merge_all();

        let subgraph = SubgraphNode::with_mapping(graph, mapping);

        let mut state = GraphState::new();
        state.set("value", 5);

        let result = subgraph.call(state).await.unwrap();
        assert_eq!(result.get::<i32>("doubled").unwrap(), 10);
        assert_eq!(result.get::<i32>("tripled").unwrap(), 15);
        assert_eq!(result.get::<i32>("squared").unwrap(), 25);
    }

    #[tokio::test]
    async fn test_state_mapping_with_prefix() {
        let graph = GraphBuilder::new("outputs")
            .add_node("compute", |mut state: GraphState| async move {
                state.set("result", 42);
                Ok(state)
            })
            .add_edge("compute", END)
            .set_entry("compute")
            .build()
            .unwrap();

        let mapping = StateMapping::new().with_prefix("child");
        let subgraph = SubgraphNode::with_mapping(graph, mapping);

        let state = GraphState::new();
        let result = subgraph.call(state).await.unwrap();

        assert_eq!(result.get::<i32>("child.result").unwrap(), 42);
    }

    #[tokio::test]
    async fn test_graph_builder_extension() {
        let inner = create_adder_graph(10);

        let graph = GraphBuilder::new("with_subgraph")
            .add_node("init", |mut state: GraphState| async move {
                state.set("value", 5);
                Ok(state)
            })
            .add_subgraph("add", inner, StateMapping::new().passthrough("value"))
            .add_edge("init", "add")
            .add_edge("add", END)
            .set_entry("init")
            .build()
            .unwrap();

        let result = graph.invoke(GraphState::new()).await.unwrap();
        assert_eq!(result.state.get::<i32>("value").unwrap(), 15);
    }
}
