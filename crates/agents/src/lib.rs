//! # Agent Implementation
//!
//! High-performance agent execution with ReACT loop (Reasoning + Acting)

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::should_implement_trait)]
//!
//! ## Features
//!
//! - **ReACT Loop**: Reason -> Act -> Observe cycle for intelligent task execution
//! - **Planning Mode**: Optional planning before execution for complex tasks
//! - **Stop Words**: Configurable termination triggers
//! - **Tool Execution**: Parallel tool execution with timeout handling
//! - **Memory**: Conversation history and context management
//! - **Discovery**: Dynamic agent discovery via heartbeat/capabilities
//! - **Handoff**: Capability-based routing between agents
//! - **Guardrails**: Input/output validation with tripwire functionality
//! - **Checkpointing**: State persistence for recovery and time-travel
//! - **Multi-Memory**: Hierarchical memory (short-term, long-term, entity)
//! - **Agent-as-Tool**: Compose agents as callable tools
//! - **Self-Correcting**: LLM-as-Judge pattern for automatic quality improvement
//! - **Agent Factory**: Runtime agent instantiation and configuration
//! - **Durable Execution**: Fault-tolerant execution with recovery
//! - **Structured Sessions**: Conversation memory and session management
//! - **Approvals**: Human-in-the-loop approval for dangerous operations

pub mod agent_tool;
pub mod approvals;
#[cfg(feature = "audit")]
pub mod audited_executor;
pub mod checkpoint;
pub mod discovery;
pub mod durable;
#[cfg(feature = "encryption")]
pub mod encrypted_stores;
pub mod engine;
#[cfg(test)]
mod engine_integration_tests;
pub mod executor;
pub mod factory;
pub mod guardrails;
pub mod handoff;
pub mod memory;
pub mod multi_memory;
pub mod persistence;
pub mod planning;
pub mod self_correct;
pub mod session;
pub mod sqlite_store;
pub mod trajectory;
pub mod vector_store;

pub use agent_tool::*;
pub use approvals::*;
#[cfg(feature = "audit")]
pub use audited_executor::*;
pub use checkpoint::*;
pub use discovery::*;
pub use durable::*;
#[cfg(feature = "encryption")]
pub use encrypted_stores::*;
pub use engine::*;
pub use executor::*;
pub use factory::*;
pub use guardrails::*;
pub use handoff::*;
pub use memory::*;
pub use multi_memory::*;
pub use persistence::*;
pub use planning::*;
pub use self_correct::*;
pub use session::*;
pub use sqlite_store::*;
pub use trajectory::*;
pub use vector_store::*;
