//! Parallel tool executor with approval support
//!
//! This module provides tool execution with optional human-in-the-loop
//! approval for dangerous operations.
//!
//! # Safety Sandwich Pattern
//!
//! ```text
//! ┌─────────────────┐
//! │   Input Guard   │  ← Validate inputs
//! ├─────────────────┤
//! │    Approvals    │  ← Human approval for dangerous tools
//! ├─────────────────┤
//! │    Executor     │  ← Execute tool
//! ├─────────────────┤
//! │  Output Guard   │  ← Validate outputs
//! └─────────────────┘
//! ```

use std::sync::Arc;

use futures::future::join_all;
use tracing::{debug, error, warn};

use rust_ai_agents_core::{AgentId, ToolCall, ToolResult};
use rust_ai_agents_tools::{ExecutionContext, ToolRegistry};

use crate::approvals::{ApprovalDecision, ApprovalHandler, ApprovalRequest, AutoApproveHandler};

/// Result of a tool execution attempt
#[derive(Debug)]
pub enum ExecutionOutcome {
    /// Tool executed successfully
    Success(ToolResult),

    /// Tool execution was denied
    Denied {
        tool_call_id: String,
        reason: String,
    },

    /// Tool was skipped
    Skipped { tool_call_id: String },

    /// Approval error occurred
    ApprovalError { tool_call_id: String, error: String },
}

impl ExecutionOutcome {
    pub fn into_tool_result(self) -> ToolResult {
        match self {
            Self::Success(result) => result,
            Self::Denied {
                tool_call_id,
                reason,
            } => ToolResult::failure(tool_call_id, format!("Denied: {}", reason)),
            Self::Skipped { tool_call_id } => {
                ToolResult::failure(tool_call_id, "Tool execution skipped".to_string())
            }
            Self::ApprovalError {
                tool_call_id,
                error,
            } => ToolResult::failure(tool_call_id, format!("Approval error: {}", error)),
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }
}

/// Configuration for tool executor
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum concurrent tool executions
    pub max_concurrency: usize,

    /// Default timeout for tool execution in seconds
    pub timeout_seconds: u64,

    /// Whether to continue on tool failure
    pub continue_on_failure: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 4,
            timeout_seconds: 30,
            continue_on_failure: true,
        }
    }
}

/// Parallel tool executor with approval support
pub struct ToolExecutor<H: ApprovalHandler = AutoApproveHandler> {
    config: ExecutorConfig,
    approval_handler: Arc<H>,
}

impl ToolExecutor<AutoApproveHandler> {
    /// Create executor without approval (auto-approve everything)
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            config: ExecutorConfig {
                max_concurrency,
                ..Default::default()
            },
            approval_handler: Arc::new(AutoApproveHandler::new()),
        }
    }
}

impl<H: ApprovalHandler + 'static> ToolExecutor<H> {
    /// Create executor with custom approval handler
    pub fn with_approval_handler(config: ExecutorConfig, handler: H) -> Self {
        Self {
            config,
            approval_handler: Arc::new(handler),
        }
    }

    /// Create executor with shared approval handler
    pub fn with_shared_handler(config: ExecutorConfig, handler: Arc<H>) -> Self {
        Self {
            config,
            approval_handler: handler,
        }
    }

    /// Execute multiple tools in parallel with approval checks
    pub async fn execute_tools(
        &self,
        tool_calls: &[ToolCall],
        registry: &Arc<ToolRegistry>,
        agent_id: &AgentId,
    ) -> Vec<ToolResult> {
        let outcomes = self
            .execute_tools_with_outcomes(tool_calls, registry, agent_id)
            .await;
        outcomes.into_iter().map(|o| o.into_tool_result()).collect()
    }

    /// Execute tools and return detailed outcomes
    pub async fn execute_tools_with_outcomes(
        &self,
        tool_calls: &[ToolCall],
        registry: &Arc<ToolRegistry>,
        agent_id: &AgentId,
    ) -> Vec<ExecutionOutcome> {
        debug!(
            "Executing {} tools with concurrency {}",
            tool_calls.len(),
            self.config.max_concurrency
        );

        // Create execution tasks
        let tasks: Vec<_> = tool_calls
            .iter()
            .map(|call| {
                let registry = registry.clone();
                let agent_id = agent_id.clone();
                let call = call.clone();
                let handler = self.approval_handler.clone();
                let config = self.config.clone();

                async move {
                    Self::execute_with_approval(call, registry, agent_id, handler, config).await
                }
            })
            .collect();

        // Execute in parallel with semaphore for concurrency control
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrency));
        let tasks_with_semaphore: Vec<_> = tasks
            .into_iter()
            .map(|task| {
                let semaphore = semaphore.clone();
                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    task.await
                }
            })
            .collect();

        join_all(tasks_with_semaphore).await
    }

    async fn execute_with_approval(
        call: ToolCall,
        registry: Arc<ToolRegistry>,
        agent_id: AgentId,
        handler: Arc<H>,
        config: ExecutorConfig,
    ) -> ExecutionOutcome {
        // Get tool schema to check if dangerous
        let tool_schema = registry.get(&call.name).map(|t| t.schema());

        // Check if approval is required
        if let Some(reason) = handler.check_requires_approval(&call, tool_schema.as_ref()) {
            debug!("Tool '{}' requires approval: {}", call.name, reason);

            let request = ApprovalRequest {
                tool_call: call.clone(),
                tool_schema,
                reason: reason.clone(),
                timestamp: chrono::Utc::now(),
                context: None,
            };

            match handler.request_approval(request).await {
                Ok(ApprovalDecision::Approved) => {
                    debug!("Tool '{}' approved", call.name);
                }
                Ok(ApprovalDecision::Modify { new_arguments }) => {
                    debug!("Tool '{}' approved with modifications", call.name);
                    let modified_call = ToolCall {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: new_arguments,
                    };
                    return Self::execute_single_tool(modified_call, registry, agent_id, config)
                        .await;
                }
                Ok(ApprovalDecision::Denied { reason }) => {
                    warn!("Tool '{}' denied: {}", call.name, reason);
                    return ExecutionOutcome::Denied {
                        tool_call_id: call.id,
                        reason,
                    };
                }
                Ok(ApprovalDecision::Skip) => {
                    debug!("Tool '{}' skipped", call.name);
                    return ExecutionOutcome::Skipped {
                        tool_call_id: call.id,
                    };
                }
                Err(e) => {
                    error!("Approval error for tool '{}': {}", call.name, e);
                    return ExecutionOutcome::ApprovalError {
                        tool_call_id: call.id,
                        error: e.to_string(),
                    };
                }
            }
        }

        Self::execute_single_tool(call, registry, agent_id, config).await
    }

    async fn execute_single_tool(
        call: ToolCall,
        registry: Arc<ToolRegistry>,
        agent_id: AgentId,
        config: ExecutorConfig,
    ) -> ExecutionOutcome {
        debug!("Executing tool: {} (call_id: {})", call.name, call.id);

        let tool = match registry.get(&call.name) {
            Some(tool) => tool,
            None => {
                error!("Tool not found: {}", call.name);
                return ExecutionOutcome::Success(ToolResult::failure(
                    call.id,
                    format!("Tool '{}' not found", call.name),
                ));
            }
        };

        // Validate arguments
        if let Err(e) = tool.validate(&call.arguments) {
            error!("Tool validation failed: {}", e);
            return ExecutionOutcome::Success(ToolResult::failure(call.id, e.to_string()));
        }

        // Create execution context
        let context = ExecutionContext::new(agent_id);

        // Execute with timeout
        let timeout_duration = std::time::Duration::from_secs(config.timeout_seconds);
        let result = match tokio::time::timeout(
            timeout_duration,
            tool.execute(&context, call.arguments.clone()),
        )
        .await
        {
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
        };

        ExecutionOutcome::Success(result)
    }
}

// Implement Clone for ToolExecutor when H is Clone
impl<H: ApprovalHandler + Clone> Clone for ToolExecutor<H> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            approval_handler: self.approval_handler.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approvals::{ApprovalConfig, ApprovalReason, TestApprovalHandler};
    use rust_ai_agents_core::{Tool, ToolSchema};
    use serde_json::json;
    use std::collections::HashMap;

    // Simple test tool
    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "echo".to_string(),
                description: "Echoes input".to_string(),
                parameters: json!({"type": "object", "properties": {"message": {"type": "string"}}}),
                dangerous: false,
                metadata: HashMap::new(),
            }
        }

        async fn execute(
            &self,
            _ctx: &ExecutionContext,
            args: serde_json::Value,
        ) -> Result<serde_json::Value, rust_ai_agents_core::errors::ToolError> {
            Ok(args)
        }
    }

    // Dangerous test tool
    struct DangerousTool;

    #[async_trait::async_trait]
    impl Tool for DangerousTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "dangerous".to_string(),
                description: "A dangerous tool".to_string(),
                parameters: json!({"type": "object"}),
                dangerous: true,
                metadata: HashMap::new(),
            }
        }

        async fn execute(
            &self,
            _ctx: &ExecutionContext,
            _args: serde_json::Value,
        ) -> Result<serde_json::Value, rust_ai_agents_core::errors::ToolError> {
            Ok(json!({"result": "executed"}))
        }
    }

    #[tokio::test]
    async fn test_executor_basic() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let registry = Arc::new(registry);

        let executor = ToolExecutor::new(4);
        let agent_id = AgentId::new("test");

        let calls = vec![ToolCall {
            id: "1".to_string(),
            name: "echo".to_string(),
            arguments: json!({"message": "hello"}),
        }];

        let results = executor.execute_tools(&calls, &registry, &agent_id).await;

        assert_eq!(results.len(), 1);
        assert!(results[0].success);
    }

    #[tokio::test]
    async fn test_executor_with_approval_deny() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DangerousTool));
        let registry = Arc::new(registry);

        let config = ApprovalConfig::builder()
            .require_approval_for_dangerous(true)
            .build();

        let handler = TestApprovalHandler::with_config(config);
        handler.set_decision("dangerous", ApprovalDecision::denied("Not allowed"));

        let executor = ToolExecutor::with_approval_handler(ExecutorConfig::default(), handler);
        let agent_id = AgentId::new("test");

        let calls = vec![ToolCall {
            id: "1".to_string(),
            name: "dangerous".to_string(),
            arguments: json!({}),
        }];

        let outcomes = executor
            .execute_tools_with_outcomes(&calls, &registry, &agent_id)
            .await;

        assert_eq!(outcomes.len(), 1);
        assert!(matches!(outcomes[0], ExecutionOutcome::Denied { .. }));
    }

    #[tokio::test]
    async fn test_executor_with_approval_approve() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DangerousTool));
        let registry = Arc::new(registry);

        let config = ApprovalConfig::builder()
            .require_approval_for_dangerous(true)
            .build();

        let handler = TestApprovalHandler::with_config(config);
        // Approve by default

        let executor = ToolExecutor::with_approval_handler(ExecutorConfig::default(), handler);
        let agent_id = AgentId::new("test");

        let calls = vec![ToolCall {
            id: "1".to_string(),
            name: "dangerous".to_string(),
            arguments: json!({}),
        }];

        let outcomes = executor
            .execute_tools_with_outcomes(&calls, &registry, &agent_id)
            .await;

        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].is_success());
    }

    #[tokio::test]
    async fn test_executor_approval_request_recorded() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DangerousTool));
        let registry = Arc::new(registry);

        let config = ApprovalConfig::builder()
            .require_approval_for_dangerous(true)
            .build();

        let handler = Arc::new(TestApprovalHandler::with_config(config));

        let executor =
            ToolExecutor::with_shared_handler(ExecutorConfig::default(), handler.clone());
        let agent_id = AgentId::new("test");

        let calls = vec![ToolCall {
            id: "1".to_string(),
            name: "dangerous".to_string(),
            arguments: json!({"key": "value"}),
        }];

        executor.execute_tools(&calls, &registry, &agent_id).await;

        // Verify approval was requested
        assert!(handler.was_approval_requested("dangerous"));
        assert_eq!(handler.request_count(), 1);

        let requests = handler.get_requests();
        assert_eq!(requests[0].tool_call.name, "dangerous");
        assert!(matches!(requests[0].reason, ApprovalReason::DangerousTool));
    }

    #[tokio::test]
    async fn test_executor_skip() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DangerousTool));
        let registry = Arc::new(registry);

        let config = ApprovalConfig::builder()
            .require_approval_for_dangerous(true)
            .build();

        let handler = TestApprovalHandler::with_config(config);
        handler.set_decision("dangerous", ApprovalDecision::Skip);

        let executor = ToolExecutor::with_approval_handler(ExecutorConfig::default(), handler);
        let agent_id = AgentId::new("test");

        let calls = vec![ToolCall {
            id: "1".to_string(),
            name: "dangerous".to_string(),
            arguments: json!({}),
        }];

        let outcomes = executor
            .execute_tools_with_outcomes(&calls, &registry, &agent_id)
            .await;

        assert_eq!(outcomes.len(), 1);
        assert!(matches!(outcomes[0], ExecutionOutcome::Skipped { .. }));
    }

    #[tokio::test]
    async fn test_executor_safe_tool_no_approval() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let registry = Arc::new(registry);

        let config = ApprovalConfig::builder()
            .require_approval_for_dangerous(true)
            .build();

        let handler = Arc::new(TestApprovalHandler::with_config(config));

        let executor =
            ToolExecutor::with_shared_handler(ExecutorConfig::default(), handler.clone());
        let agent_id = AgentId::new("test");

        let calls = vec![ToolCall {
            id: "1".to_string(),
            name: "echo".to_string(),
            arguments: json!({"message": "hi"}),
        }];

        let results = executor.execute_tools(&calls, &registry, &agent_id).await;

        assert_eq!(results.len(), 1);
        assert!(results[0].success);

        // No approval should have been requested for safe tool
        assert_eq!(handler.request_count(), 0);
    }
}
