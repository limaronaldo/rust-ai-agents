//! # Crew Orchestration
//!
//! Multi-agent orchestration system for coordinating tasks across multiple agents.

#![allow(clippy::type_complexity)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::should_implement_trait)]
//!
//! ## Features
//!
//! - **Crew**: Traditional multi-agent coordination with sequential/parallel/hierarchical processes
//! - **Workflow**: DAG-based workflow system with conditional branching and human-in-the-loop support
//! - **Orchestra**: Multi-perspective analysis with parallel execution and synthesis
//! - **Graph**: LangGraph-style graph execution with cycles, conditionals, and checkpointing
//! - **HumanLoop**: Human-in-the-loop support with approval gates, breakpoints, and input collection
//! - **TimeTravel**: State history, replay, fork, and debugging for graph executions
//! - **Subgraph**: Nested workflows, parallel subgraphs, and reusable graph components
//! - **Handoff**: Agent-to-agent handoffs with context passing and return support
//! - **Streaming**: Unified streaming events across graphs, handoffs, and workflows

pub mod crew;
pub mod graph;
pub mod handoff;
pub mod human_loop;
pub mod orchestra;
pub mod process;
pub mod streaming;
pub mod subgraph;
pub mod task_manager;
pub mod time_travel;
pub mod workflow;

pub use crew::*;
pub use graph::*;
pub use handoff::*;
pub use human_loop::*;
pub use orchestra::*;
pub use process::*;
pub use streaming::*;
pub use subgraph::*;
pub use task_manager::*;
pub use time_travel::*;
pub use workflow::*;
