//! Crew management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use rust_ai_agents_agents::AgentEngine;
use rust_ai_agents_core::*;

use crate::process::Process;
use crate::task_manager::TaskManager;

/// Crew configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrewConfig {
    /// Crew name
    pub name: String,

    /// Crew description
    pub description: String,

    /// Execution process type
    pub process: Process,

    /// Maximum concurrent tasks
    pub max_concurrency: usize,

    /// Verbose logging
    pub verbose: bool,
}

impl CrewConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            process: Process::Sequential,
            max_concurrency: 4,
            verbose: false,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn with_process(mut self, process: Process) -> Self {
        self.process = process;
        self
    }

    pub fn with_max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = max;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Crew of agents working together
pub struct Crew {
    config: CrewConfig,
    engine: Arc<AgentEngine>,
    task_manager: TaskManager,
    agents: HashMap<AgentId, AgentConfig>,
}

impl Crew {
    /// Create a new crew
    pub fn new(config: CrewConfig, engine: Arc<AgentEngine>) -> Self {
        Self {
            config: config.clone(),
            engine,
            task_manager: TaskManager::new(config.max_concurrency),
            agents: HashMap::new(),
        }
    }

    /// Add an agent to the crew
    pub fn add_agent(&mut self, agent_config: AgentConfig) {
        self.agents.insert(agent_config.id.clone(), agent_config);
    }

    /// Add a task to the crew
    pub fn add_task(&mut self, task: Task) -> Result<(), CrewError> {
        self.task_manager.add_task(task)
    }

    /// Execute all tasks
    pub async fn kickoff(&mut self) -> Result<Vec<TaskResult>, CrewError> {
        tracing::info!(
            "Crew '{}' starting execution with {} tasks",
            self.config.name,
            self.task_manager.task_count()
        );

        match self.config.process {
            Process::Sequential => self.execute_sequential().await,
            Process::Parallel => self.execute_parallel().await,
            Process::Hierarchical => self.execute_hierarchical().await,
        }
    }

    /// Execute tasks sequentially
    async fn execute_sequential(&mut self) -> Result<Vec<TaskResult>, CrewError> {
        let mut results = Vec::new();
        let tasks = self.task_manager.get_all_tasks();

        for task in tasks {
            tracing::info!("Executing task: {}", task.description);
            let result = self.execute_single_task(task).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute tasks in parallel (respecting dependencies)
    async fn execute_parallel(&mut self) -> Result<Vec<TaskResult>, CrewError> {
        // For parallel execution, we use sequential for now due to borrow checker constraints
        // TODO: Implement proper parallel execution with Arc<Self>
        self.execute_sequential().await
    }

    /// Execute tasks with hierarchical coordination
    async fn execute_hierarchical(&mut self) -> Result<Vec<TaskResult>, CrewError> {
        // For hierarchical, we need a manager agent
        // For now, fall back to parallel execution
        tracing::warn!("Hierarchical process not fully implemented, using parallel");
        self.execute_parallel().await
    }

    /// Execute a single task
    async fn execute_single_task(&self, task: Task) -> Result<TaskResult, CrewError> {
        // Find agent for task
        let agent_id = task
            .agent_id
            .as_ref()
            .or_else(|| self.find_best_agent_for_task(&task))
            .ok_or_else(|| CrewError::NoAgentAvailable(task.description.clone()))?;

        // Get agent runtime to access its state and memory
        let agent_runtime = self
            .engine
            .get_agent(agent_id)
            .ok_or_else(|| CrewError::NoAgentAvailable(agent_id.to_string()))?;

        // Create message for agent
        let message = Message::new(
            AgentId::new("crew"),
            agent_id.clone(),
            Content::Text(task.description.clone()),
        );

        // Send to agent
        self.engine
            .send_message(message)
            .map_err(CrewError::AgentError)?;

        // Wait for agent to process with timeout
        // Use timeout from task context or default to 60 seconds
        let timeout_secs = task
            .context
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(60);
        let timeout_duration = std::time::Duration::from_secs(timeout_secs);
        let poll_interval = std::time::Duration::from_millis(100);
        let start = std::time::Instant::now();

        // Phase 1: Wait for agent to START processing (leave Idle state)
        loop {
            if start.elapsed() > timeout_duration {
                return Ok(TaskResult::failure(
                    task.id.clone(),
                    format!(
                        "Task timed out waiting to start after {:?}",
                        timeout_duration
                    ),
                ));
            }

            let state = agent_runtime.state.read().await;
            let is_idle = matches!(state.status, AgentStatus::Idle);
            drop(state);

            if !is_idle {
                tracing::debug!("Agent {} started processing task", agent_id);
                break;
            }

            tokio::time::sleep(poll_interval).await;
        }

        // Phase 2: Wait for agent to FINISH processing (return to Idle state)
        loop {
            if start.elapsed() > timeout_duration {
                return Ok(TaskResult::failure(
                    task.id.clone(),
                    format!("Task timed out after {:?}", timeout_duration),
                ));
            }

            let state = agent_runtime.state.read().await;
            let is_idle = matches!(state.status, AgentStatus::Idle);
            drop(state);

            if is_idle {
                tracing::debug!("Agent {} finished processing task", agent_id);

                // Agent finished processing, check memory for response
                if let Ok(history) = agent_runtime.memory.get_history().await {
                    // Find the last response from the agent after our message
                    for msg in history.iter().rev() {
                        if msg.from == *agent_id {
                            if let Content::Text(response_text) = &msg.content {
                                return Ok(TaskResult::success(
                                    task.id.clone(),
                                    serde_json::json!({
                                        "status": "completed",
                                        "response": response_text,
                                        "agent_id": agent_id.to_string(),
                                        "elapsed_ms": start.elapsed().as_millis()
                                    }),
                                ));
                            }
                        }
                    }
                }

                // Agent is idle but no response found - might have errored
                return Ok(TaskResult::failure(
                    task.id.clone(),
                    "Agent completed but no response found".to_string(),
                ));
            }

            // Agent still processing, wait before checking again
            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Find best agent for a task based on capabilities
    fn find_best_agent_for_task(&self, _task: &Task) -> Option<&AgentId> {
        // Simple strategy: return first available agent
        // TODO: Implement capability matching
        self.agents.keys().next()
    }

    /// Get crew statistics
    pub fn stats(&self) -> CrewStats {
        CrewStats {
            name: self.config.name.clone(),
            agent_count: self.agents.len(),
            task_count: self.task_manager.task_count(),
            completed_tasks: 0, // TODO: Track this
        }
    }
}

/// Crew statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrewStats {
    pub name: String,
    pub agent_count: usize,
    pub task_count: usize,
    pub completed_tasks: usize,
}
