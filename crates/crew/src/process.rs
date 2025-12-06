//! Process types for crew execution

use serde::{Deserialize, Serialize};

/// Process type for crew execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Process {
    /// Execute tasks one by one in order
    Sequential,

    /// Execute tasks in parallel when possible
    Parallel,

    /// Hierarchical process with manager agent coordinating
    Hierarchical,
}

impl Process {
    pub fn as_str(&self) -> &'static str {
        match self {
            Process::Sequential => "sequential",
            Process::Parallel => "parallel",
            Process::Hierarchical => "hierarchical",
        }
    }
}

impl std::fmt::Display for Process {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for Process {
    fn default() -> Self {
        Process::Sequential
    }
}
