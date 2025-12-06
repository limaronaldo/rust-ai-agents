//! # Crew Orchestration
//!
//! Multi-agent orchestration system for coordinating tasks across multiple agents.
//!
//! ## Features
//!
//! - **Crew**: Traditional multi-agent coordination with sequential/parallel/hierarchical processes
//! - **Workflow**: DAG-based workflow system with conditional branching and human-in-the-loop support

pub mod crew;
pub mod process;
pub mod task_manager;
pub mod workflow;

pub use crew::*;
pub use process::*;
pub use task_manager::*;
pub use workflow::*;
