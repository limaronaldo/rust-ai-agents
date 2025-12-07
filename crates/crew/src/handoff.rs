//! Agent Handoffs
//!
//! LangGraph-style agent-to-agent handoffs with full context passing,
//! supporting multi-hop chains, returns, and conditional routing.
//!
//! ## Features
//!
//! - Handoff with full conversation context
//! - Return to caller after completion
//! - Multi-agent handoff chains
//! - Conditional handoff routing
//! - Handoff history tracking
//! - Parallel handoffs to multiple agents
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_crew::handoff::{HandoffRouter, AgentNode, HandoffTrigger};
//!
//! // Create agent nodes
//! let triage = AgentNode::new("triage", triage_agent)
//!     .description("Routes requests to appropriate specialists");
//!
//! let sales = AgentNode::new("sales", sales_agent)
//!     .description("Handles sales inquiries")
//!     .can_return(); // Can return to caller
//!
//! let support = AgentNode::new("support", support_agent)
//!     .description("Handles technical support");
//!
//! // Build handoff router
//! let router = HandoffRouter::new()
//!     .add_agent(triage)
//!     .add_agent(sales)
//!     .add_agent(support)
//!     .add_handoff("triage", "sales", HandoffTrigger::keyword("buy"))
//!     .add_handoff("triage", "support", HandoffTrigger::keyword("help"))
//!     .set_entry("triage");
//!
//! // Execute conversation with automatic handoffs
//! let result = router.run(conversation).await?;
//! ```

use chrono::{DateTime, Utc};
use rust_ai_agents_core::errors::CrewError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Conversation message in a handoff context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffMessage {
    /// Message role
    pub role: MessageRole,
    /// Message content
    pub content: String,
    /// Agent that produced this message (if assistant)
    pub agent_id: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl HandoffMessage {
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            agent_id: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create an assistant message
    pub fn assistant(agent_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            agent_id: Some(agent_id.into()),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            agent_id: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Conversation context passed during handoffs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffContext {
    /// Conversation ID
    pub conversation_id: String,
    /// Full message history
    pub messages: Vec<HandoffMessage>,
    /// Current agent handling the conversation
    pub current_agent: String,
    /// Stack of agents (for return handoffs)
    pub agent_stack: Vec<String>,
    /// Handoff history
    pub handoff_history: Vec<HandoffRecord>,
    /// Custom context data
    pub data: HashMap<String, serde_json::Value>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

impl HandoffContext {
    /// Create a new context
    pub fn new(conversation_id: impl Into<String>, entry_agent: impl Into<String>) -> Self {
        let entry = entry_agent.into();
        Self {
            conversation_id: conversation_id.into(),
            messages: Vec::new(),
            current_agent: entry.clone(),
            agent_stack: vec![entry],
            handoff_history: Vec::new(),
            data: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// Add a message to the conversation
    pub fn add_message(&mut self, message: HandoffMessage) {
        self.messages.push(message);
        self.updated_at = Utc::now();
    }

    /// Add a user message
    pub fn user_message(&mut self, content: impl Into<String>) {
        self.add_message(HandoffMessage::user(content));
    }

    /// Add an assistant message from current agent
    pub fn agent_message(&mut self, content: impl Into<String>) {
        self.add_message(HandoffMessage::assistant(&self.current_agent, content));
    }

    /// Get last N messages
    pub fn last_messages(&self, n: usize) -> &[HandoffMessage] {
        let start = self.messages.len().saturating_sub(n);
        &self.messages[start..]
    }

    /// Get messages from a specific agent
    pub fn messages_from(&self, agent_id: &str) -> Vec<&HandoffMessage> {
        self.messages
            .iter()
            .filter(|m| m.agent_id.as_deref() == Some(agent_id))
            .collect()
    }

    /// Set custom data
    pub fn set_data(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.data.insert(key.into(), value);
        self.updated_at = Utc::now();
    }

    /// Get custom data
    pub fn get_data<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.data
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Get conversation summary
    pub fn summary(&self) -> String {
        format!(
            "Conversation {} with {} messages, current agent: {}, {} handoffs",
            self.conversation_id,
            self.messages.len(),
            self.current_agent,
            self.handoff_history.len()
        )
    }
}

/// Record of a handoff event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffRecord {
    /// Source agent
    pub from_agent: String,
    /// Target agent
    pub to_agent: String,
    /// Reason for handoff
    pub reason: String,
    /// Whether this is a return handoff
    pub is_return: bool,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Message index when handoff occurred
    pub message_index: usize,
}

/// Trigger condition for a handoff
#[derive(Clone)]
pub enum HandoffTrigger {
    /// Trigger on keyword in message
    Keyword(String),
    /// Trigger on multiple keywords (any match)
    Keywords(Vec<String>),
    /// Trigger on regex pattern
    Pattern(String),
    /// Trigger on custom condition
    Custom(Arc<dyn Fn(&HandoffContext, &str) -> bool + Send + Sync>),
    /// Always trigger (for explicit handoffs)
    Always,
    /// Trigger based on agent decision
    AgentDecision,
}

impl std::fmt::Debug for HandoffTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Keyword(k) => write!(f, "Keyword({})", k),
            Self::Keywords(ks) => write!(f, "Keywords({:?})", ks),
            Self::Pattern(p) => write!(f, "Pattern({})", p),
            Self::Custom(_) => write!(f, "Custom(<fn>)"),
            Self::Always => write!(f, "Always"),
            Self::AgentDecision => write!(f, "AgentDecision"),
        }
    }
}

impl HandoffTrigger {
    /// Create a keyword trigger
    pub fn keyword(keyword: impl Into<String>) -> Self {
        Self::Keyword(keyword.into().to_lowercase())
    }

    /// Create a multi-keyword trigger
    pub fn keywords(keywords: Vec<String>) -> Self {
        Self::Keywords(keywords.into_iter().map(|k| k.to_lowercase()).collect())
    }

    /// Create a pattern trigger
    pub fn pattern(pattern: impl Into<String>) -> Self {
        Self::Pattern(pattern.into())
    }

    /// Create a custom trigger
    pub fn custom<F>(f: F) -> Self
    where
        F: Fn(&HandoffContext, &str) -> bool + Send + Sync + 'static,
    {
        Self::Custom(Arc::new(f))
    }

    /// Check if trigger matches
    pub fn matches(&self, context: &HandoffContext, message: &str) -> bool {
        let lower = message.to_lowercase();
        match self {
            Self::Keyword(k) => lower.contains(k),
            Self::Keywords(ks) => ks.iter().any(|k| lower.contains(k)),
            Self::Pattern(p) => regex::Regex::new(p)
                .map(|re| re.is_match(&lower))
                .unwrap_or(false),
            Self::Custom(f) => f(context, message),
            Self::Always => true,
            Self::AgentDecision => false, // Handled separately
        }
    }
}

/// Handoff instruction from an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffInstruction {
    /// Target agent ID
    pub target_agent: String,
    /// Reason for handoff
    pub reason: String,
    /// Whether to return after target completes
    pub should_return: bool,
    /// Additional data to pass
    pub data: HashMap<String, serde_json::Value>,
}

impl HandoffInstruction {
    /// Create a new handoff instruction
    pub fn to(agent: impl Into<String>) -> Self {
        Self {
            target_agent: agent.into(),
            reason: String::new(),
            should_return: false,
            data: HashMap::new(),
        }
    }

    /// Set reason
    pub fn because(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Request return after completion
    pub fn and_return(mut self) -> Self {
        self.should_return = true;
        self
    }

    /// Add data to pass
    pub fn with_data(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.data.insert(key.into(), value);
        self
    }
}

/// Agent response with optional handoff
#[derive(Debug, Clone)]
pub enum AgentResponse {
    /// Continue conversation with this message
    Message(String),
    /// Hand off to another agent
    Handoff(HandoffInstruction),
    /// Return to previous agent
    Return(String),
    /// End the conversation
    End(String),
}

/// Agent executor function type
pub type AgentExecutor = Arc<
    dyn Fn(HandoffContext) -> futures::future::BoxFuture<'static, Result<AgentResponse, CrewError>>
        + Send
        + Sync,
>;

/// An agent node in the handoff graph
pub struct AgentNode {
    /// Agent ID
    pub id: String,
    /// Agent description
    pub description: String,
    /// Agent executor
    executor: AgentExecutor,
    /// System prompt for this agent
    pub system_prompt: Option<String>,
    /// Whether this agent can return to caller
    pub can_return: bool,
    /// Whether this agent can hand off to others
    pub can_handoff: bool,
    /// Allowed handoff targets (empty = all)
    pub allowed_targets: Vec<String>,
}

impl std::fmt::Debug for AgentNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentNode")
            .field("id", &self.id)
            .field("description", &self.description)
            .field("can_return", &self.can_return)
            .field("can_handoff", &self.can_handoff)
            .finish()
    }
}

impl AgentNode {
    /// Create a new agent node
    pub fn new<F, Fut>(id: impl Into<String>, executor: F) -> Self
    where
        F: Fn(HandoffContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<AgentResponse, CrewError>> + Send + 'static,
    {
        Self {
            id: id.into(),
            description: String::new(),
            executor: Arc::new(move |ctx| Box::pin(executor(ctx))),
            system_prompt: None,
            can_return: false,
            can_handoff: true,
            allowed_targets: Vec::new(),
        }
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Allow this agent to return to caller
    pub fn can_return(mut self) -> Self {
        self.can_return = true;
        self
    }

    /// Prevent this agent from handing off
    pub fn no_handoff(mut self) -> Self {
        self.can_handoff = false;
        self
    }

    /// Restrict handoff targets
    pub fn allowed_targets(mut self, targets: Vec<String>) -> Self {
        self.allowed_targets = targets;
        self
    }

    /// Execute the agent
    pub async fn execute(&self, context: HandoffContext) -> Result<AgentResponse, CrewError> {
        (self.executor)(context).await
    }
}

/// Handoff rule between agents
#[derive(Debug)]
pub struct HandoffRule {
    /// Source agent
    pub from: String,
    /// Target agent
    pub to: String,
    /// Trigger condition
    pub trigger: HandoffTrigger,
    /// Priority (higher = checked first)
    pub priority: i32,
}

/// Handoff router - manages agent network and routes conversations
pub struct HandoffRouter {
    /// Registered agents
    agents: HashMap<String, AgentNode>,
    /// Handoff rules
    rules: Vec<HandoffRule>,
    /// Entry agent
    entry_agent: Option<String>,
    /// Maximum handoffs per conversation
    max_handoffs: u32,
    /// Maximum conversation turns
    max_turns: u32,
}

impl Default for HandoffRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl HandoffRouter {
    /// Create a new handoff router
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            rules: Vec::new(),
            entry_agent: None,
            max_handoffs: 10,
            max_turns: 50,
        }
    }

    /// Add an agent
    pub fn add_agent(mut self, agent: AgentNode) -> Self {
        self.agents.insert(agent.id.clone(), agent);
        self
    }

    /// Add a handoff rule
    pub fn add_handoff(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        trigger: HandoffTrigger,
    ) -> Self {
        self.rules.push(HandoffRule {
            from: from.into(),
            to: to.into(),
            trigger,
            priority: 0,
        });
        self
    }

    /// Add a prioritized handoff rule
    pub fn add_handoff_priority(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        trigger: HandoffTrigger,
        priority: i32,
    ) -> Self {
        self.rules.push(HandoffRule {
            from: from.into(),
            to: to.into(),
            trigger,
            priority,
        });
        self
    }

    /// Set entry agent
    pub fn set_entry(mut self, agent_id: impl Into<String>) -> Self {
        self.entry_agent = Some(agent_id.into());
        self
    }

    /// Set maximum handoffs
    pub fn max_handoffs(mut self, max: u32) -> Self {
        self.max_handoffs = max;
        self
    }

    /// Set maximum turns
    pub fn max_turns(mut self, max: u32) -> Self {
        self.max_turns = max;
        self
    }

    /// Get agent by ID
    pub fn get_agent(&self, id: &str) -> Option<&AgentNode> {
        self.agents.get(id)
    }

    /// List all agents
    pub fn list_agents(&self) -> Vec<&AgentNode> {
        self.agents.values().collect()
    }

    /// Find matching handoff rules for current context
    fn find_matching_rules(&self, context: &HandoffContext, message: &str) -> Vec<&HandoffRule> {
        let mut matches: Vec<_> = self
            .rules
            .iter()
            .filter(|r| r.from == context.current_agent && r.trigger.matches(context, message))
            .collect();

        // Sort by priority (descending)
        matches.sort_by(|a, b| b.priority.cmp(&a.priority));
        matches
    }

    /// Execute handoff to a new agent
    fn execute_handoff(
        &self,
        context: &mut HandoffContext,
        to: &str,
        reason: &str,
        is_return: bool,
    ) {
        let record = HandoffRecord {
            from_agent: context.current_agent.clone(),
            to_agent: to.to_string(),
            reason: reason.to_string(),
            is_return,
            timestamp: Utc::now(),
            message_index: context.messages.len(),
        };

        context.handoff_history.push(record);

        if !is_return {
            context.agent_stack.push(to.to_string());
        } else {
            context.agent_stack.pop();
        }

        context.current_agent = to.to_string();
        context.updated_at = Utc::now();
    }

    /// Run a conversation with automatic handoffs
    pub async fn run(&self, mut context: HandoffContext) -> Result<HandoffResult, CrewError> {
        let entry = self.entry_agent.as_ref().ok_or_else(|| {
            CrewError::InvalidConfiguration("No entry agent specified".to_string())
        })?;

        context.current_agent = entry.clone();
        if context.agent_stack.is_empty() {
            context.agent_stack.push(entry.clone());
        }

        let mut turns = 0;
        let mut handoffs = 0;

        loop {
            // Check limits
            if turns >= self.max_turns {
                return Ok(HandoffResult {
                    context,
                    status: HandoffStatus::MaxTurnsReached,
                    final_message: None,
                });
            }

            if handoffs >= self.max_handoffs {
                return Ok(HandoffResult {
                    context,
                    status: HandoffStatus::MaxHandoffsReached,
                    final_message: None,
                });
            }

            // Get current agent
            let agent = self.agents.get(&context.current_agent).ok_or_else(|| {
                CrewError::TaskNotFound(format!("Agent '{}' not found", context.current_agent))
            })?;

            // Execute agent
            let response = agent.execute(context.clone()).await?;
            turns += 1;

            match response {
                AgentResponse::Message(msg) => {
                    context.agent_message(&msg);

                    // Check for automatic handoff triggers
                    let rules = self.find_matching_rules(&context, &msg);
                    if let Some(rule) = rules.first() {
                        self.execute_handoff(
                            &mut context,
                            &rule.to,
                            &format!("Triggered by rule: {:?}", rule.trigger),
                            false,
                        );
                        handoffs += 1;
                    }
                    // Continue with current agent if no trigger
                }

                AgentResponse::Handoff(instruction) => {
                    // Validate handoff
                    if !agent.can_handoff {
                        return Err(CrewError::ExecutionFailed(format!(
                            "Agent '{}' is not allowed to hand off",
                            context.current_agent
                        )));
                    }

                    if !agent.allowed_targets.is_empty()
                        && !agent.allowed_targets.contains(&instruction.target_agent)
                    {
                        return Err(CrewError::ExecutionFailed(format!(
                            "Agent '{}' cannot hand off to '{}'",
                            context.current_agent, instruction.target_agent
                        )));
                    }

                    if !self.agents.contains_key(&instruction.target_agent) {
                        return Err(CrewError::TaskNotFound(format!(
                            "Target agent '{}' not found",
                            instruction.target_agent
                        )));
                    }

                    // Merge any data from instruction
                    for (key, value) in instruction.data {
                        context.set_data(key, value);
                    }

                    self.execute_handoff(
                        &mut context,
                        &instruction.target_agent,
                        &instruction.reason,
                        false,
                    );
                    handoffs += 1;
                }

                AgentResponse::Return(msg) => {
                    if !agent.can_return {
                        return Err(CrewError::ExecutionFailed(format!(
                            "Agent '{}' is not allowed to return",
                            context.current_agent
                        )));
                    }

                    context.agent_message(&msg);

                    // Return to previous agent in stack
                    if context.agent_stack.len() > 1 {
                        let prev = context.agent_stack[context.agent_stack.len() - 2].clone();
                        self.execute_handoff(&mut context, &prev, "Return to caller", true);
                        handoffs += 1;
                    } else {
                        // No one to return to - end conversation
                        return Ok(HandoffResult {
                            context,
                            status: HandoffStatus::Completed,
                            final_message: Some(msg),
                        });
                    }
                }

                AgentResponse::End(msg) => {
                    context.agent_message(&msg);
                    return Ok(HandoffResult {
                        context,
                        status: HandoffStatus::Completed,
                        final_message: Some(msg),
                    });
                }
            }
        }
    }

    /// Run with a new conversation starting from user message
    pub async fn start(&self, user_message: impl Into<String>) -> Result<HandoffResult, CrewError> {
        let mut context = HandoffContext::new(
            uuid::Uuid::new_v4().to_string(),
            self.entry_agent.as_deref().unwrap_or("default"),
        );
        context.user_message(user_message);
        self.run(context).await
    }
}

/// Result of a handoff conversation
#[derive(Debug, Clone)]
pub struct HandoffResult {
    /// Final conversation context
    pub context: HandoffContext,
    /// Completion status
    pub status: HandoffStatus,
    /// Final message (if completed normally)
    pub final_message: Option<String>,
}

/// Status of handoff completion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandoffStatus {
    /// Conversation completed normally
    Completed,
    /// Hit maximum turns limit
    MaxTurnsReached,
    /// Hit maximum handoffs limit
    MaxHandoffsReached,
    /// Conversation was interrupted
    Interrupted,
}

impl HandoffResult {
    /// Get summary of the conversation
    pub fn summary(&self) -> String {
        format!(
            "Status: {:?}, Agents: {:?}, Messages: {}, Handoffs: {}",
            self.status,
            self.context.agent_stack,
            self.context.messages.len(),
            self.context.handoff_history.len()
        )
    }

    /// Get all handoffs that occurred
    pub fn handoffs(&self) -> &[HandoffRecord] {
        &self.context.handoff_history
    }

    /// Get messages from a specific agent
    pub fn messages_from(&self, agent_id: &str) -> Vec<&HandoffMessage> {
        self.context.messages_from(agent_id)
    }
}

/// Parallel handoff - hand off to multiple agents simultaneously
pub struct ParallelHandoff {
    /// Target agents
    targets: Vec<String>,
    /// State mapping for each target
    #[allow(dead_code)]
    mappings: HashMap<String, HashMap<String, String>>,
}

impl ParallelHandoff {
    /// Create a new parallel handoff
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
            mappings: HashMap::new(),
        }
    }

    /// Add a target agent
    pub fn to(mut self, agent: impl Into<String>) -> Self {
        self.targets.push(agent.into());
        self
    }

    /// Add multiple targets
    pub fn to_all(mut self, agents: Vec<String>) -> Self {
        self.targets.extend(agents);
        self
    }
}

impl Default for ParallelHandoff {
    fn default() -> Self {
        Self::new()
    }
}

/// Handoff chain - sequence of agents to pass through
pub struct HandoffChain {
    /// Ordered list of agents
    agents: Vec<String>,
    /// Whether to return to origin after chain completes
    return_to_origin: bool,
}

impl HandoffChain {
    /// Create a new chain
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            return_to_origin: false,
        }
    }

    /// Add next agent in chain
    pub fn then(mut self, agent: impl Into<String>) -> Self {
        self.agents.push(agent.into());
        self
    }

    /// Return to origin after chain
    pub fn and_return(mut self) -> Self {
        self.return_to_origin = true;
        self
    }
}

impl Default for HandoffChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn create_test_agent(id: &str, response: &'static str) -> AgentNode {
        AgentNode::new(id, move |_ctx| async move {
            Ok(AgentResponse::Message(response.to_string()))
        })
        .description(format!("Test agent {}", id))
    }

    fn create_handoff_agent(id: &str, target: &'static str) -> AgentNode {
        AgentNode::new(id, move |_ctx| async move {
            Ok(AgentResponse::Handoff(
                HandoffInstruction::to(target).because("Need specialist"),
            ))
        })
    }

    fn create_ending_agent(id: &str, response: &'static str) -> AgentNode {
        AgentNode::new(id, move |_ctx| async move {
            Ok(AgentResponse::End(response.to_string()))
        })
    }

    #[allow(dead_code)]
    fn create_returning_agent(id: &str, response: &'static str) -> AgentNode {
        AgentNode::new(id, move |_ctx| async move {
            Ok(AgentResponse::Return(response.to_string()))
        })
        .can_return()
    }

    #[tokio::test]
    async fn test_simple_conversation() {
        let router = HandoffRouter::new()
            .add_agent(create_ending_agent("greeter", "Hello! How can I help?"))
            .set_entry("greeter");

        let result = router.start("Hi there").await.unwrap();

        assert_eq!(result.status, HandoffStatus::Completed);
        assert_eq!(
            result.final_message.as_deref(),
            Some("Hello! How can I help?")
        );
        assert_eq!(result.context.messages.len(), 2); // user + agent
    }

    #[tokio::test]
    async fn test_agent_handoff() {
        let router = HandoffRouter::new()
            .add_agent(create_handoff_agent("triage", "sales"))
            .add_agent(create_ending_agent(
                "sales",
                "I can help with your purchase!",
            ))
            .set_entry("triage");

        let result = router.start("I want to buy something").await.unwrap();

        assert_eq!(result.status, HandoffStatus::Completed);
        assert_eq!(result.context.handoff_history.len(), 1);
        assert_eq!(result.context.handoff_history[0].from_agent, "triage");
        assert_eq!(result.context.handoff_history[0].to_agent, "sales");
    }

    #[tokio::test]
    async fn test_trigger_based_handoff() {
        // Agent that echoes the user's intent, triggering the keyword
        let triage = AgentNode::new("triage", |_ctx| async move {
            Ok(AgentResponse::Message(
                "I understand you want to buy something. Let me help with that.".to_string(),
            ))
        });

        let router = HandoffRouter::new()
            .add_agent(triage)
            .add_agent(create_ending_agent("sales", "Sales here!"))
            .add_handoff("triage", "sales", HandoffTrigger::keyword("buy"))
            .set_entry("triage");

        // This should trigger handoff to sales based on agent's response
        let mut context = HandoffContext::new("test", "triage");
        context.user_message("I want to purchase something");

        let result = router.run(context).await.unwrap();

        assert_eq!(result.context.handoff_history.len(), 1);
        assert_eq!(result.context.handoff_history[0].to_agent, "sales");
    }

    #[tokio::test]
    async fn test_return_handoff() {
        let triage = AgentNode::new("triage", |ctx| async move {
            // First time: hand off to specialist
            // After return: end conversation
            if ctx.handoff_history.is_empty() {
                Ok(AgentResponse::Handoff(
                    HandoffInstruction::to("specialist").because("Need expert"),
                ))
            } else {
                Ok(AgentResponse::End("Thanks for your patience!".to_string()))
            }
        });

        let specialist = AgentNode::new("specialist", |_ctx| async move {
            Ok(AgentResponse::Return("Here's my analysis".to_string()))
        })
        .can_return();

        let router = HandoffRouter::new()
            .add_agent(triage)
            .add_agent(specialist)
            .set_entry("triage");

        let result = router.start("Need help").await.unwrap();

        assert_eq!(result.status, HandoffStatus::Completed);
        assert_eq!(result.context.handoff_history.len(), 2); // forward + return
        assert!(result.context.handoff_history[1].is_return);
    }

    #[tokio::test]
    async fn test_max_handoffs_limit() {
        // Create circular handoff
        let agent_a = create_handoff_agent("a", "b");
        let agent_b = create_handoff_agent("b", "a");

        let router = HandoffRouter::new()
            .add_agent(agent_a)
            .add_agent(agent_b)
            .set_entry("a")
            .max_handoffs(5);

        let result = router.start("Start").await.unwrap();

        assert_eq!(result.status, HandoffStatus::MaxHandoffsReached);
        assert!(result.context.handoff_history.len() <= 5);
    }

    #[tokio::test]
    async fn test_handoff_context_data() {
        let mut context = HandoffContext::new("test-123", "agent1");

        context.set_data("user_id", serde_json::json!("user-456"));
        context.set_data("priority", serde_json::json!(5));

        assert_eq!(
            context.get_data::<String>("user_id"),
            Some("user-456".to_string())
        );
        assert_eq!(context.get_data::<i32>("priority"), Some(5));
    }

    #[tokio::test]
    async fn test_handoff_trigger_keywords() {
        let context = HandoffContext::new("test", "agent");

        let trigger = HandoffTrigger::keywords(vec!["buy".to_string(), "purchase".to_string()]);

        assert!(trigger.matches(&context, "I want to BUY something"));
        assert!(trigger.matches(&context, "Can I purchase this?"));
        assert!(!trigger.matches(&context, "Just browsing"));
    }

    #[tokio::test]
    async fn test_handoff_trigger_pattern() {
        let context = HandoffContext::new("test", "agent");

        let trigger = HandoffTrigger::pattern(r"order\s*#?\d+");

        assert!(trigger.matches(&context, "Check order #12345"));
        assert!(trigger.matches(&context, "order 67890 status"));
        assert!(!trigger.matches(&context, "I want to order"));
    }

    #[tokio::test]
    async fn test_handoff_instruction_builder() {
        let instruction = HandoffInstruction::to("support")
            .because("Technical issue")
            .and_return()
            .with_data("ticket_id", serde_json::json!("TKT-123"));

        assert_eq!(instruction.target_agent, "support");
        assert_eq!(instruction.reason, "Technical issue");
        assert!(instruction.should_return);
        assert_eq!(
            instruction.data.get("ticket_id"),
            Some(&serde_json::json!("TKT-123"))
        );
    }

    #[tokio::test]
    async fn test_handoff_history_tracking() {
        let router = HandoffRouter::new()
            .add_agent(create_handoff_agent("a", "b"))
            .add_agent(create_handoff_agent("b", "c"))
            .add_agent(create_ending_agent("c", "Done!"))
            .set_entry("a");

        let result = router.start("Go").await.unwrap();

        assert_eq!(result.context.handoff_history.len(), 2);
        assert_eq!(result.context.handoff_history[0].from_agent, "a");
        assert_eq!(result.context.handoff_history[0].to_agent, "b");
        assert_eq!(result.context.handoff_history[1].from_agent, "b");
        assert_eq!(result.context.handoff_history[1].to_agent, "c");
    }

    #[tokio::test]
    async fn test_message_history() {
        let router = HandoffRouter::new()
            .add_agent(create_ending_agent("agent", "Response"))
            .set_entry("agent");

        let result = router.start("User message").await.unwrap();

        assert_eq!(result.context.messages.len(), 2);
        assert_eq!(result.context.messages[0].role, MessageRole::User);
        assert_eq!(result.context.messages[0].content, "User message");
        assert_eq!(result.context.messages[1].role, MessageRole::Assistant);
        assert_eq!(
            result.context.messages[1].agent_id,
            Some("agent".to_string())
        );
    }
}
