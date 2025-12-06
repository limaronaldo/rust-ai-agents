//! Parallel tool executor

use std::sync::Arc;
use futures::future::join_all;
use tracing::{debug, error};

use rust_ai_agents_core::{AgentId, ToolCall, ToolResult};
use rust_ai_agents_tools::{ToolRegistry, ExecutionContext};

/// Parallel tool executor
#[derive(Clone)]
pub struct ToolExecutor {
    max_concurrency: usize,
}

impl ToolExecutor {
    pub fn new(max_concurrency: usize) -> Self {
        Self { max_concurrency }
    }

    /// Execute multiple tools in parallel
    pub async fn execute_tools(
        &self,
        tool_calls: &[ToolCall],
        registry: &Arc<ToolRegistry>,
        agent_id: &AgentId,
    ) -> Vec<ToolResult> {
        debug!("Executing {} tools with concurrency {}", tool_calls.len(), self.max_concurrency);

        // Create execution tasks
        let tasks: Vec<_> = tool_calls.iter().map(|call| {
            let registry = registry.clone();
            let agent_id = agent_id.clone();
            let call = call.clone();

            async move {
                Self::execute_single_tool(call, registry, agent_id).await
            }
        }).collect();

        // Execute in parallel with semaphore for concurrency control
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrency));
        let tasks_with_semaphore: Vec<_> = tasks.into_iter().map(|task| {
            let semaphore = semaphore.clone();
            async move {
                let _permit = semaphore.acquire().await.unwrap();
                task.await
            }
        }).collect();

        join_all(tasks_with_semaphore).await
    }

    async fn execute_single_tool(
        call: ToolCall,
        registry: Arc<ToolRegistry>,
        agent_id: AgentId,
    ) -> ToolResult {
        debug!("Executing tool: {} (call_id: {})", call.name, call.id);

        let tool = match registry.get(&call.name) {
            Some(tool) => tool,
            None => {
                error!("Tool not found: {}", call.name);
                return ToolResult::failure(
                    call.id,
                    format!("Tool '{}' not found", call.name),
                );
            }
        };

        // Validate arguments
        if let Err(e) = tool.validate(&call.arguments) {
            error!("Tool validation failed: {}", e);
            return ToolResult::failure(call.id, e.to_string());
        }

        // Create execution context
        let context = ExecutionContext::new(agent_id);

        // Execute with timeout
        let timeout_duration = std::time::Duration::from_secs(30);
        match tokio::time::timeout(
            timeout_duration,
            tool.execute(&context, call.arguments.clone())
        ).await {
            Ok(Ok(data)) => {
                debug!("Tool {} executed successfully", call.name);
                ToolResult::success(call.id, data)
            }
            Ok(Err(e)) => {
                error!("Tool {} execution failed: {}", call.name, e);
                ToolResult::failure(call.id, e.to_string())
            }
            Err(_) => {
                error!("Tool {} timed out after {:?}", call.name, timeout_duration);
                ToolResult::failure(
                    call.id,
                    format!("Tool execution timed out after {:?}", timeout_duration),
                )
            }
        }
    }
}
