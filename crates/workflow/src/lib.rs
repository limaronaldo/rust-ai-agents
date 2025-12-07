//! # YAML Workflow DSL
//!
//! Define agent workflows declaratively using YAML.
//!
//! ## Features
//!
//! - **Declarative Workflows**: Define multi-agent workflows in YAML
//! - **Agent Definitions**: Configure agents with roles, tools, and prompts
//! - **Task Orchestration**: Define task dependencies and execution order
//! - **Variable Interpolation**: Pass data between tasks using `{{variable}}` syntax
//!
//! ## Example YAML
//!
//! ```yaml
//! name: research_workflow
//! description: Research and summarize a topic
//!
//! agents:
//!   - id: researcher
//!     name: Research Agent
//!     role: researcher
//!     system_prompt: You are a research specialist.
//!     tools: [web_search, read_file]
//!     planning_mode: before_task
//!
//!   - id: writer
//!     name: Writer Agent
//!     role: writer
//!     system_prompt: You are a technical writer.
//!
//! tasks:
//!   - id: research
//!     agent: researcher
//!     description: Research the topic
//!     expected_output: Detailed research notes
//!
//!   - id: summarize
//!     agent: writer
//!     description: Write a summary based on {{research.output}}
//!     depends_on: [research]
//!     expected_output: A concise summary
//!
//! execution:
//!   mode: sequential  # or 'parallel', 'hierarchical'
//!   max_iterations: 10
//!   timeout_secs: 300
//! ```

pub mod error;
pub mod parser;
pub mod runner;
pub mod schema;

pub use error::WorkflowError;
pub use parser::WorkflowParser;
pub use runner::WorkflowRunner;
pub use schema::*;
