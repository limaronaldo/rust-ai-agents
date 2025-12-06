//! Agent-to-agent delegation tools
//!
//! Allows agents to delegate tasks to other specialized agents,
//! enabling hierarchical agent architectures and supervisor/worker patterns.
//!
//! # Example
//! ```ignore
//! use rust_ai_agents_tools::delegation::*;
//!
//! // Create agent registry
//! let registry = AgentRegistry::new(engine.clone());
//!
//! // Register agents with descriptions
//! registry.register_agent(
//!     researcher_id,
//!     "Research Agent",
//!     "Specialized in web research and gathering information",
//! );
//!
//! registry.register_agent(
//!     writer_id,
//!     "Writer Agent",
//!     "Specialized in writing clear, concise content",
//! );
//!
//! // Create delegation tool for supervisor
//! let delegate_tool = DelegateAgentTool::new(registry);
//! supervisor_tools.register(Arc::new(delegate_tool));
//! ```

use async_trait::async_trait;
use parking_lot::RwLock;
use rust_ai_agents_core::{
    errors::ToolError,
    tool::{ExecutionContext, Tool, ToolSchema},
    AgentId, Content, Message,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Information about a registered agent available for delegation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent ID
    pub id: AgentId,
    /// Human-readable name
    pub name: String,
    /// Description of what the agent specializes in
    pub description: String,
    /// Optional tags for categorization
    pub tags: Vec<String>,
    /// Whether agent is currently available
    pub available: bool,
}

impl AgentInfo {
    pub fn new(id: AgentId, name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: description.into(),
            tags: Vec::new(),
            available: true,
        }
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Callback type for sending messages to agents
pub type MessageSender = Arc<dyn Fn(Message) -> Result<(), String> + Send + Sync>;

/// Callback type for waiting for agent response
pub type ResponseWaiter =
    Arc<dyn Fn(AgentId, Duration) -> Option<oneshot::Receiver<Message>> + Send + Sync>;

/// Registry of agents available for delegation
pub struct AgentRegistry {
    agents: RwLock<HashMap<AgentId, AgentInfo>>,
    message_sender: MessageSender,
    response_channels: RwLock<HashMap<AgentId, Vec<oneshot::Sender<Message>>>>,
}

impl AgentRegistry {
    /// Create a new agent registry with a message sender callback
    pub fn new(message_sender: MessageSender) -> Self {
        Self {
            agents: RwLock::new(HashMap::new()),
            message_sender,
            response_channels: RwLock::new(HashMap::new()),
        }
    }

    /// Register an agent for delegation
    pub fn register_agent(&self, info: AgentInfo) {
        let id = info.id.clone();
        self.agents.write().insert(id, info);
    }

    /// Register an agent with basic info
    pub fn register(&self, id: AgentId, name: impl Into<String>, description: impl Into<String>) {
        self.register_agent(AgentInfo::new(id, name, description));
    }

    /// Unregister an agent
    pub fn unregister(&self, id: &AgentId) {
        self.agents.write().remove(id);
    }

    /// Get agent info
    pub fn get_agent(&self, id: &AgentId) -> Option<AgentInfo> {
        self.agents.read().get(id).cloned()
    }

    /// List all available agents
    pub fn list_agents(&self) -> Vec<AgentInfo> {
        self.agents
            .read()
            .values()
            .filter(|a| a.available)
            .cloned()
            .collect()
    }

    /// Find agents by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<AgentInfo> {
        self.agents
            .read()
            .values()
            .filter(|a| a.available && a.tags.iter().any(|t| t == tag))
            .cloned()
            .collect()
    }

    /// Set agent availability
    pub fn set_available(&self, id: &AgentId, available: bool) {
        if let Some(agent) = self.agents.write().get_mut(id) {
            agent.available = available;
        }
    }

    /// Send a message to an agent
    pub fn send_message(&self, message: Message) -> Result<(), String> {
        (self.message_sender)(message)
    }

    /// Register a response channel for an agent
    pub fn register_response_channel(&self, agent_id: AgentId) -> oneshot::Receiver<Message> {
        let (tx, rx) = oneshot::channel();
        self.response_channels
            .write()
            .entry(agent_id)
            .or_default()
            .push(tx);
        rx
    }

    /// Deliver a response to waiting channels
    pub fn deliver_response(&self, from_agent: &AgentId, message: Message) {
        if let Some(channels) = self.response_channels.write().remove(from_agent) {
            for tx in channels {
                let _ = tx.send(message.clone());
            }
        }
    }

    /// Get count of registered agents
    pub fn agent_count(&self) -> usize {
        self.agents.read().len()
    }
}

/// Tool for delegating tasks to other agents
///
/// This tool allows a supervisor agent to delegate specific tasks
/// to specialized worker agents and receive their responses.
pub struct DelegateAgentTool {
    registry: Arc<AgentRegistry>,
    timeout: Duration,
}

impl DelegateAgentTool {
    /// Create a new delegation tool
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self {
            registry,
            timeout: Duration::from_secs(60),
        }
    }

    /// Set timeout for waiting for agent response
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl Tool for DelegateAgentTool {
    fn schema(&self) -> ToolSchema {
        // Build available agents description
        let agents = self.registry.list_agents();
        let agents_desc = if agents.is_empty() {
            "No agents currently available for delegation.".to_string()
        } else {
            agents
                .iter()
                .map(|a| format!("- {} ({}): {}", a.name, a.id, a.description))
                .collect::<Vec<_>>()
                .join("\n")
        };

        ToolSchema::new(
            "delegate_to_agent",
            format!(
                "Delegate a task to another specialized agent. Use this when you need help \
                from an agent with specific expertise. Available agents:\n{}",
                agents_desc
            ),
        )
        .with_parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the agent to delegate to"
                },
                "task": {
                    "type": "string",
                    "description": "The task or question to send to the agent"
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context for the task"
                }
            },
            "required": ["agent_id", "task"]
        }))
    }

    async fn execute(
        &self,
        ctx: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let agent_id_str = arguments["agent_id"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("agent_id is required".to_string()))?;

        let task = arguments["task"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("task is required".to_string()))?;

        let context = arguments["context"].as_str();

        let target_agent_id = AgentId::new(agent_id_str);

        // Verify agent exists and is available
        let agent_info = self.registry.get_agent(&target_agent_id).ok_or_else(|| {
            ToolError::ExecutionFailed(format!("Agent '{}' not found", agent_id_str))
        })?;

        if !agent_info.available {
            return Err(ToolError::ExecutionFailed(format!(
                "Agent '{}' is not currently available",
                agent_id_str
            )));
        }

        info!(
            from = %ctx.agent_id,
            to = %target_agent_id,
            task = %task,
            "Delegating task to agent"
        );

        // Build message content
        let content = if let Some(ctx_str) = context {
            format!("{}\n\nContext: {}", task, ctx_str)
        } else {
            task.to_string()
        };

        // Register response channel before sending
        let response_rx = self
            .registry
            .register_response_channel(target_agent_id.clone());

        // Send message to target agent
        let message = Message::new(
            ctx.agent_id.clone(),
            target_agent_id.clone(),
            Content::Text(content),
        );

        self.registry.send_message(message).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to send message to agent: {}", e))
        })?;

        // Wait for response with timeout
        debug!(
            target = %target_agent_id,
            timeout_secs = self.timeout.as_secs(),
            "Waiting for agent response"
        );

        match timeout(self.timeout, response_rx).await {
            Ok(Ok(response)) => {
                info!(
                    from = %target_agent_id,
                    "Received response from delegated agent"
                );

                match response.content {
                    Content::Text(text) => Ok(serde_json::json!({
                        "agent": agent_info.name,
                        "agent_id": agent_id_str,
                        "response": text,
                        "success": true
                    })),
                    _ => Ok(serde_json::json!({
                        "agent": agent_info.name,
                        "agent_id": agent_id_str,
                        "response": "Agent returned non-text response",
                        "success": true
                    })),
                }
            }
            Ok(Err(_)) => {
                warn!(target = %target_agent_id, "Response channel closed");
                Err(ToolError::ExecutionFailed(
                    "Agent response channel closed unexpectedly".to_string(),
                ))
            }
            Err(_) => {
                warn!(
                    target = %target_agent_id,
                    timeout_secs = self.timeout.as_secs(),
                    "Timeout waiting for agent response"
                );
                Err(ToolError::Timeout(format!(
                    "Timeout waiting for response from agent '{}' after {} seconds",
                    target_agent_id,
                    self.timeout.as_secs()
                )))
            }
        }
    }
}

/// Tool for listing available agents
///
/// Useful for supervisor agents to discover what specialists are available.
pub struct ListAgentsTool {
    registry: Arc<AgentRegistry>,
}

impl ListAgentsTool {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for ListAgentsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "list_available_agents",
            "List all agents available for delegation, including their specializations",
        )
        .with_parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Optional tag to filter agents by specialization"
                }
            },
            "required": []
        }))
    }

    async fn execute(
        &self,
        _ctx: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let agents = if let Some(tag) = arguments["tag"].as_str() {
            self.registry.find_by_tag(tag)
        } else {
            self.registry.list_agents()
        };

        let agent_list: Vec<serde_json::Value> = agents
            .into_iter()
            .map(|a| {
                serde_json::json!({
                    "id": a.id.to_string(),
                    "name": a.name,
                    "description": a.description,
                    "tags": a.tags
                })
            })
            .collect();

        Ok(serde_json::json!({
            "agents": agent_list,
            "count": agent_list.len()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_registry() -> Arc<AgentRegistry> {
        let sender: MessageSender = Arc::new(|_msg| Ok(()));
        Arc::new(AgentRegistry::new(sender))
    }

    #[test]
    fn test_agent_registry_register() {
        let registry = create_test_registry();

        registry.register(
            AgentId::new("agent-1"),
            "Research Agent",
            "Specializes in research",
        );

        assert_eq!(registry.agent_count(), 1);

        let agent = registry.get_agent(&AgentId::new("agent-1")).unwrap();
        assert_eq!(agent.name, "Research Agent");
    }

    #[test]
    fn test_agent_registry_list() {
        let registry = create_test_registry();

        registry.register(AgentId::new("agent-1"), "Agent 1", "Description 1");
        registry.register(AgentId::new("agent-2"), "Agent 2", "Description 2");

        let agents = registry.list_agents();
        assert_eq!(agents.len(), 2);
    }

    #[test]
    fn test_agent_availability() {
        let registry = create_test_registry();

        registry.register(AgentId::new("agent-1"), "Agent 1", "Description 1");

        // Initially available
        let agents = registry.list_agents();
        assert_eq!(agents.len(), 1);

        // Set unavailable
        registry.set_available(&AgentId::new("agent-1"), false);
        let agents = registry.list_agents();
        assert_eq!(agents.len(), 0);

        // Set available again
        registry.set_available(&AgentId::new("agent-1"), true);
        let agents = registry.list_agents();
        assert_eq!(agents.len(), 1);
    }

    #[test]
    fn test_find_by_tag() {
        let registry = create_test_registry();

        registry.register_agent(
            AgentInfo::new(AgentId::new("research-1"), "Researcher", "Does research")
                .with_tags(vec!["research".to_string(), "analysis".to_string()]),
        );

        registry.register_agent(
            AgentInfo::new(AgentId::new("writer-1"), "Writer", "Writes content")
                .with_tags(vec!["writing".to_string(), "content".to_string()]),
        );

        let researchers = registry.find_by_tag("research");
        assert_eq!(researchers.len(), 1);
        assert_eq!(researchers[0].id, AgentId::new("research-1"));

        let writers = registry.find_by_tag("writing");
        assert_eq!(writers.len(), 1);
        assert_eq!(writers[0].id, AgentId::new("writer-1"));
    }

    #[tokio::test]
    async fn test_delegate_tool_schema() {
        let registry = create_test_registry();
        registry.register(AgentId::new("helper"), "Helper Agent", "Helps with tasks");

        let tool = DelegateAgentTool::new(registry);
        let schema = tool.schema();

        assert_eq!(schema.name, "delegate_to_agent");
        assert!(schema.description.contains("Helper Agent"));
    }

    #[tokio::test]
    async fn test_list_agents_tool() {
        let registry = create_test_registry();
        registry.register(AgentId::new("agent-1"), "Agent 1", "Description 1");
        registry.register(AgentId::new("agent-2"), "Agent 2", "Description 2");

        let tool = ListAgentsTool::new(registry);
        let ctx = ExecutionContext::new(AgentId::new("supervisor"));

        let result = tool.execute(&ctx, serde_json::json!({})).await.unwrap();

        assert_eq!(result["count"], 2);
        assert_eq!(result["agents"].as_array().unwrap().len(), 2);
    }
}
