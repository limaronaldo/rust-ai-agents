//! # Crew Orchestration
//!
//! Multi-agent orchestration system for coordinating tasks across multiple agents.
//!
//! ## Features
//!
//! - **Crew**: Traditional multi-agent coordination with sequential/parallel/hierarchical processes
//! - **Workflow**: DAG-based workflow system with conditional branching and human-in-the-loop support
//! - **Orchestra**: Multi-perspective analysis with parallel execution and synthesis
//! - **Graph**: LangGraph-style graph execution with cycles, conditionals, and checkpointing
//! - **HumanLoop**: Human-in-the-loop support with approval gates, breakpoints, and input collection
//! - **TimeTravel**: State history, replay, fork, and debugging for graph executions

pub mod crew;
pub mod graph;
pub mod human_loop;
pub mod orchestra;
pub mod process;
pub mod task_manager;
pub mod time_travel;
pub mod workflow;

pub use crew::*;
pub use graph::*;
pub use human_loop::*;
pub use orchestra::*;
pub use process::*;
pub use task_manager::*;
pub use time_travel::*;
pub use workflow::*;
