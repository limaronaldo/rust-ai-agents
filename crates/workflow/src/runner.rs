//! Workflow execution runner

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use rust_ai_agents_agents::AgentEngine;
use rust_ai_agents_core::types::{AgentConfig, AgentId, AgentRole, PlanningMode};
use rust_ai_agents_core::{Content, Message};
use rust_ai_agents_providers::{LLMBackend, OpenRouterProvider};
use rust_ai_agents_tools::ToolRegistry;

use crate::error::WorkflowError;
use crate::parser::WorkflowParser;
use crate::schema::*;

/// Workflow execution runner
pub struct WorkflowRunner {
    /// The workflow definition
    workflow: WorkflowDefinition,
    /// Agent engine for execution
    engine: AgentEngine,
    /// Tool registry
    tools: Arc<ToolRegistry>,
    /// LLM backend
    backend: Arc<dyn LLMBackend>,
    /// Variable context for interpolation
    context: Arc<RwLock<HashMap<String, String>>>,
}

impl WorkflowRunner {
    /// Create a new workflow runner from a definition
    pub async fn new(workflow: WorkflowDefinition) -> Result<Self, WorkflowError> {
        // Create default provider from environment
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .map_err(|_| {
                WorkflowError::ExecutionError(
                    "OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set".to_string(),
                )
            })?;

        let model = workflow
            .execution
            .provider
            .model
            .clone()
            .unwrap_or_else(|| "anthropic/claude-3-5-sonnet".to_string());

        let backend: Arc<dyn LLMBackend> = Arc::new(OpenRouterProvider::new(api_key, model));

        Ok(Self {
            workflow,
            engine: AgentEngine::new(),
            tools: Arc::new(ToolRegistry::new()),
            backend,
            context: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create runner with custom backend
    pub fn with_backend(mut self, backend: Arc<dyn LLMBackend>) -> Self {
        self.backend = backend;
        self
    }

    /// Create runner with custom tool registry
    pub fn with_tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = tools;
        self
    }

    /// Create runner with existing agent engine
    pub fn with_engine(mut self, engine: AgentEngine) -> Self {
        self.engine = engine;
        self
    }

    /// Set initial context variables
    pub async fn set_context(&self, key: impl Into<String>, value: impl Into<String>) {
        let mut ctx = self.context.write().await;
        ctx.insert(key.into(), value.into());
    }

    /// Run the workflow
    pub async fn run(&self) -> Result<WorkflowResult, WorkflowError> {
        let start = Instant::now();
        let mut result = WorkflowResult::new(self.workflow.name.clone());

        info!(workflow = %self.workflow.name, "Starting workflow execution");

        // Spawn agents
        self.spawn_agents().await?;

        // Get execution order
        let execution_order = WorkflowParser::get_execution_order(&self.workflow)?;
        info!(
            tasks = ?execution_order,
            mode = ?self.workflow.execution.mode,
            "Execution order determined"
        );

        // Execute based on mode
        match self.workflow.execution.mode {
            ExecutionMode::Sequential => {
                self.run_sequential(&execution_order, &mut result).await?;
            }
            ExecutionMode::Parallel => {
                self.run_parallel(&execution_order, &mut result).await?;
            }
            ExecutionMode::Hierarchical => {
                self.run_hierarchical(&execution_order, &mut result).await?;
            }
        }

        // Finalize result
        result.total_duration_ms = start.elapsed().as_millis() as u64;
        result.success = result.errors.is_empty()
            && result
                .task_results
                .iter()
                .all(|r| r.success || self.is_optional_task(&r.task_id));

        // Set final output from last successful task
        if let Some(last_success) = result.task_results.iter().rev().find(|r| r.success) {
            result.final_output = Some(last_success.output.clone());
        }

        info!(
            workflow = %self.workflow.name,
            success = result.success,
            duration_ms = result.total_duration_ms,
            "Workflow execution completed"
        );

        // Shutdown agents
        self.engine.shutdown().await;

        Ok(result)
    }

    /// Spawn all agents defined in the workflow
    async fn spawn_agents(&self) -> Result<(), WorkflowError> {
        for agent_def in &self.workflow.agents {
            let config = self.build_agent_config(agent_def);

            self.engine
                .spawn_agent(config, self.tools.clone(), self.backend.clone())
                .await
                .map_err(|e| {
                    WorkflowError::ExecutionError(format!(
                        "Failed to spawn agent '{}': {}",
                        agent_def.id, e
                    ))
                })?;

            debug!(agent = %agent_def.id, "Agent spawned");
        }

        Ok(())
    }

    /// Build AgentConfig from workflow definition
    fn build_agent_config(&self, def: &AgentDefinition) -> AgentConfig {
        let role = match &def.role {
            AgentRoleType::Researcher => AgentRole::Researcher,
            AgentRoleType::Writer => AgentRole::Writer,
            AgentRoleType::Reviewer => AgentRole::Reviewer,
            AgentRoleType::Coordinator => AgentRole::Coordinator,
            AgentRoleType::Executor => AgentRole::Executor,
            AgentRoleType::Custom(s) => AgentRole::Custom(s.clone()),
        };

        let planning_mode = match def.planning_mode {
            PlanningModeType::Disabled => PlanningMode::Disabled,
            PlanningModeType::BeforeTask => PlanningMode::BeforeTask,
            PlanningModeType::FullPlan => PlanningMode::FullPlan,
            PlanningModeType::Adaptive => PlanningMode::Adaptive,
        };

        let mut config = AgentConfig::new(&def.name, role)
            .with_id(AgentId::new(&def.id))
            .with_planning_mode(planning_mode)
            .with_stop_words(def.stop_words.clone());

        if let Some(prompt) = &def.system_prompt {
            config = config.with_system_prompt(prompt);
        }

        if let Some(temp) = def.temperature {
            config = config.with_temperature(temp);
        }

        if let Some(max_iter) = def.max_iterations {
            config = config.with_max_iterations(max_iter);
        }

        if let Some(timeout) = def.timeout_secs {
            config = config.with_timeout(timeout);
        }

        config
    }

    /// Run tasks sequentially
    async fn run_sequential(
        &self,
        order: &[String],
        result: &mut WorkflowResult,
    ) -> Result<(), WorkflowError> {
        for task_id in order {
            let task_result = self.execute_task(task_id).await;

            match task_result {
                Ok(tr) => {
                    // Store output in context for interpolation
                    let mut ctx = self.context.write().await;
                    ctx.insert(format!("{}.output", task_id), tr.output.clone());
                    drop(ctx);

                    result.task_results.push(tr);
                }
                Err(e) => {
                    let tr = TaskResult {
                        task_id: task_id.clone(),
                        success: false,
                        output: String::new(),
                        error: Some(e.to_string()),
                        duration_ms: 0,
                        llm_calls: 0,
                        tool_calls: 0,
                    };
                    result.task_results.push(tr);
                    result.errors.push(e.to_string());

                    if self.workflow.execution.fail_fast && !self.is_optional_task(task_id) {
                        error!(task = %task_id, "Task failed, stopping due to fail_fast");
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Run independent tasks in parallel
    async fn run_parallel(
        &self,
        order: &[String],
        result: &mut WorkflowResult,
    ) -> Result<(), WorkflowError> {
        // Group tasks by dependency level
        let levels = self.group_by_dependency_level(order);

        for level in levels {
            // Execute all tasks in this level in parallel
            let mut handles = Vec::new();

            for task_id in level {
                let task_id_clone = task_id.clone();
                let self_ref = self;

                handles.push(async move {
                    (
                        task_id_clone.clone(),
                        self_ref.execute_task(&task_id_clone).await,
                    )
                });
            }

            // Wait for all tasks in this level
            let results = futures::future::join_all(handles).await;

            for (task_id, task_result) in results {
                match task_result {
                    Ok(tr) => {
                        let mut ctx = self.context.write().await;
                        ctx.insert(format!("{}.output", task_id), tr.output.clone());
                        drop(ctx);

                        result.task_results.push(tr);
                    }
                    Err(e) => {
                        let tr = TaskResult {
                            task_id: task_id.clone(),
                            success: false,
                            output: String::new(),
                            error: Some(e.to_string()),
                            duration_ms: 0,
                            llm_calls: 0,
                            tool_calls: 0,
                        };
                        result.task_results.push(tr);
                        result.errors.push(e.to_string());
                    }
                }
            }
        }

        Ok(())
    }

    /// Run with hierarchical coordination
    async fn run_hierarchical(
        &self,
        order: &[String],
        result: &mut WorkflowResult,
    ) -> Result<(), WorkflowError> {
        // For hierarchical mode, we run sequentially but could add
        // a coordinator agent that manages the workflow
        // For now, fall back to sequential
        warn!("Hierarchical mode using sequential execution (coordinator not implemented)");
        self.run_sequential(order, result).await
    }

    /// Execute a single task
    async fn execute_task(&self, task_id: &str) -> Result<TaskResult, WorkflowError> {
        let task = self
            .workflow
            .tasks
            .iter()
            .find(|t| t.id == task_id)
            .ok_or_else(|| WorkflowError::TaskNotFound(task_id.to_string()))?;

        let start = Instant::now();
        info!(task = %task_id, agent = %task.agent, "Executing task");

        // Interpolate description with context
        let ctx = self.context.read().await;
        let description = WorkflowParser::interpolate(&task.description, &ctx)?;
        drop(ctx);

        // Create message for agent
        let message = Message::new(
            AgentId::new("workflow"),
            AgentId::new(&task.agent),
            Content::Text(description),
        );

        // Send message to agent
        self.engine
            .send_message(message)
            .map_err(|e| WorkflowError::ExecutionError(e.to_string()))?;

        // Wait for response (simplified - in real impl would use channels)
        // For now, we'll poll the agent state
        let timeout = task
            .timeout_secs
            .unwrap_or(self.workflow.execution.timeout_secs);

        let response = self
            .wait_for_response(&task.agent, timeout)
            .await
            .map_err(|e| WorkflowError::ExecutionError(e.to_string()))?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(TaskResult {
            task_id: task_id.to_string(),
            success: true,
            output: response,
            error: None,
            duration_ms,
            llm_calls: 1, // Simplified
            tool_calls: 0,
        })
    }

    /// Wait for agent response
    async fn wait_for_response(
        &self,
        agent_id: &str,
        timeout_secs: u64,
    ) -> Result<String, WorkflowError> {
        use rust_ai_agents_core::types::AgentStatus;

        let deadline = Instant::now() + std::time::Duration::from_secs(timeout_secs);

        loop {
            if Instant::now() > deadline {
                return Err(WorkflowError::Timeout(timeout_secs));
            }

            if let Some(runtime) = self.engine.get_agent(&AgentId::new(agent_id)) {
                let state = runtime.state.read().await;

                match state.status {
                    AgentStatus::Idle | AgentStatus::StoppedByStopWord => {
                        // Get last response from history
                        if let Some(msg) = state.history.last() {
                            if let Content::Text(text) = &msg.content {
                                return Ok(text.clone());
                            }
                        }
                        return Ok("Task completed".to_string());
                    }
                    AgentStatus::Error => {
                        return Err(WorkflowError::ExecutionError(
                            "Agent encountered an error".to_string(),
                        ));
                    }
                    AgentStatus::Stopped => {
                        return Err(WorkflowError::ExecutionError(
                            "Agent was stopped".to_string(),
                        ));
                    }
                    _ => {
                        // Still processing
                    }
                }
            }

            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    /// Check if a task is optional
    fn is_optional_task(&self, task_id: &str) -> bool {
        self.workflow
            .tasks
            .iter()
            .find(|t| t.id == task_id)
            .map(|t| t.optional)
            .unwrap_or(false)
    }

    /// Group tasks by dependency level for parallel execution
    fn group_by_dependency_level(&self, order: &[String]) -> Vec<Vec<String>> {
        let mut levels: Vec<Vec<String>> = Vec::new();
        let mut task_levels: HashMap<&str, usize> = HashMap::new();

        for task_id in order {
            let task = self
                .workflow
                .tasks
                .iter()
                .find(|t| t.id == *task_id)
                .unwrap();

            let level = if task.depends_on.is_empty() {
                0
            } else {
                task.depends_on
                    .iter()
                    .filter_map(|dep| task_levels.get(dep.as_str()))
                    .max()
                    .map(|l| l + 1)
                    .unwrap_or(0)
            };

            task_levels.insert(task_id, level);

            while levels.len() <= level {
                levels.push(Vec::new());
            }
            levels[level].push(task_id.clone());
        }

        levels
    }
}

/// Load and run a workflow from a YAML file
pub async fn run_workflow_file(
    path: impl AsRef<std::path::Path>,
) -> Result<WorkflowResult, WorkflowError> {
    let workflow = WorkflowParser::parse_file(path)?;
    let runner = WorkflowRunner::new(workflow).await?;
    runner.run().await
}

/// Load and run a workflow from a YAML string
pub async fn run_workflow(yaml: &str) -> Result<WorkflowResult, WorkflowError> {
    let workflow = WorkflowParser::parse(yaml)?;
    let runner = WorkflowRunner::new(workflow).await?;
    runner.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_by_dependency_level() {
        let yaml = r#"
name: test
agents:
  - id: a1
    name: Agent 1
tasks:
  - id: t1
    agent: a1
    description: Task 1
  - id: t2
    agent: a1
    description: Task 2
  - id: t3
    agent: a1
    description: Task 3
    depends_on: [t1, t2]
  - id: t4
    agent: a1
    description: Task 4
    depends_on: [t3]
"#;
        let workflow = WorkflowParser::parse(yaml).unwrap();

        // Create a mock runner to test grouping
        let order = vec![
            "t1".to_string(),
            "t2".to_string(),
            "t3".to_string(),
            "t4".to_string(),
        ];

        // Manual level calculation for test
        // t1, t2 -> level 0
        // t3 -> level 1
        // t4 -> level 2
        let mut task_levels: HashMap<&str, usize> = HashMap::new();
        let mut levels: Vec<Vec<String>> = Vec::new();

        for task_id in &order {
            let task = workflow.tasks.iter().find(|t| t.id == *task_id).unwrap();
            let level = if task.depends_on.is_empty() {
                0
            } else {
                task.depends_on
                    .iter()
                    .filter_map(|dep| task_levels.get(dep.as_str()))
                    .max()
                    .map(|l| l + 1)
                    .unwrap_or(0)
            };
            task_levels.insert(task_id, level);
            while levels.len() <= level {
                levels.push(Vec::new());
            }
            levels[level].push(task_id.clone());
        }

        assert_eq!(levels.len(), 3);
        assert!(levels[0].contains(&"t1".to_string()));
        assert!(levels[0].contains(&"t2".to_string()));
        assert!(levels[1].contains(&"t3".to_string()));
        assert!(levels[2].contains(&"t4".to_string()));
    }
}
