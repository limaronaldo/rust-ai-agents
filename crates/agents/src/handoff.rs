//! Multi-Agent Handoff Mechanism
//!
//! Enables agents to decline requests outside their domain and forward
//! to a more appropriate peer agent based on capabilities matching.
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents::handoff::{AgentRegistry, AgentCapability, HandoffRouter};
//!
//! // Register agents with their capabilities
//! let mut registry = AgentRegistry::new();
//! registry.register(AgentCapability::new(
//!     "code-expert",
//!     "programming",
//!     vec!["rust", "python", "code", "debug"],
//!     vec!["execute_code", "analyze_code"],
//! ));
//!
//! // Route a task to the best agent
//! let router = HandoffRouter::new(registry);
//! if let Some((agent, score)) = router.route("Help me debug this Rust code") {
//!     println!("Routing to {} (score: {:.2})", agent, score);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Handoff decision from an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffDecision {
    /// Whether the agent can handle this request
    pub can_handle: bool,
    /// If declining, the recommended agent to handle instead
    pub recommended_agent: Option<String>,
    /// Reason for the handoff
    pub reason: Option<String>,
    /// Confidence score (0.0 to 1.0) that this is the right agent
    pub confidence: f32,
}

impl HandoffDecision {
    /// Create a decision indicating the agent can handle the request
    pub fn accept(confidence: f32) -> Self {
        Self {
            can_handle: true,
            recommended_agent: None,
            reason: None,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Create a decision to hand off to another agent
    pub fn handoff(recommended_agent: &str, reason: &str) -> Self {
        Self {
            can_handle: false,
            recommended_agent: Some(recommended_agent.to_string()),
            reason: Some(reason.to_string()),
            confidence: 0.0,
        }
    }

    /// Create a decision indicating uncertainty
    pub fn uncertain(confidence: f32, recommended_agent: Option<&str>) -> Self {
        Self {
            can_handle: confidence > 0.5,
            recommended_agent: recommended_agent.map(|s| s.to_string()),
            reason: Some("Low confidence in task match".to_string()),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Check if handoff is needed
    pub fn needs_handoff(&self) -> bool {
        !self.can_handle && self.recommended_agent.is_some()
    }
}

/// Agent capability descriptor for handoff routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapability {
    /// Unique agent identifier
    pub name: String,
    /// Primary domain/specialty
    pub domain: String,
    /// Keywords that indicate this agent's expertise
    pub keywords: Vec<String>,
    /// Tools this agent can use
    pub tools: Vec<String>,
    /// Priority for tie-breaking (higher = preferred)
    pub priority: u32,
    /// Optional description
    pub description: Option<String>,
}

impl AgentCapability {
    /// Create a new agent capability
    pub fn new(
        name: impl Into<String>,
        domain: impl Into<String>,
        keywords: Vec<impl Into<String>>,
        tools: Vec<impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            domain: domain.into(),
            keywords: keywords.into_iter().map(|s| s.into()).collect(),
            tools: tools.into_iter().map(|s| s.into()).collect(),
            priority: 0,
            description: None,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Score how well this agent matches a task description
    pub fn match_score(&self, task: &str) -> f32 {
        let task_lower = task.to_lowercase();
        let mut score = 0.0f32;
        let mut matches = 0;

        // Check keyword matches
        for keyword in &self.keywords {
            if task_lower.contains(&keyword.to_lowercase()) {
                matches += 1;
            }
        }

        if !self.keywords.is_empty() {
            score = matches as f32 / self.keywords.len() as f32;
        }

        // Boost score if domain is mentioned
        if task_lower.contains(&self.domain.to_lowercase()) {
            score = (score + 0.3).min(1.0);
        }

        // Small boost for priority
        if self.priority > 0 {
            score = (score + (self.priority as f32 * 0.01)).min(1.0);
        }

        score
    }

    /// Check if agent has a specific tool
    pub fn has_tool(&self, tool: &str) -> bool {
        self.tools.iter().any(|t| t == tool)
    }

    /// Check if agent has any of the specified tools
    pub fn has_any_tool(&self, tools: &[&str]) -> bool {
        tools.iter().any(|t| self.has_tool(t))
    }
}

/// Registry of available agents for handoff routing
#[derive(Debug, Clone, Default)]
pub struct AgentRegistry {
    agents: HashMap<String, AgentCapability>,
}

impl AgentRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent's capabilities
    pub fn register(&mut self, capability: AgentCapability) {
        self.agents.insert(capability.name.clone(), capability);
    }

    /// Unregister an agent
    pub fn unregister(&mut self, name: &str) -> Option<AgentCapability> {
        self.agents.remove(name)
    }

    /// Get agent by name
    pub fn get(&self, name: &str) -> Option<&AgentCapability> {
        self.agents.get(name)
    }

    /// Find the best agent for a task
    pub fn find_best_agent(&self, task: &str, exclude: Option<&str>) -> Option<(String, f32)> {
        let mut best: Option<(String, f32)> = None;

        for (name, agent) in &self.agents {
            // Skip excluded agent (usually the current one)
            if let Some(excluded) = exclude {
                if name == excluded {
                    continue;
                }
            }

            let score = agent.match_score(task);
            if score > 0.0 && (best.is_none() || score > best.as_ref().unwrap().1) {
                best = Some((name.clone(), score));
            }
        }

        best
    }

    /// Find agents that can use a specific tool
    pub fn find_by_tool(&self, tool: &str) -> Vec<&AgentCapability> {
        self.agents.values().filter(|a| a.has_tool(tool)).collect()
    }

    /// Find agents by domain
    pub fn find_by_domain(&self, domain: &str) -> Vec<&AgentCapability> {
        let domain_lower = domain.to_lowercase();
        self.agents
            .values()
            .filter(|a| a.domain.to_lowercase() == domain_lower)
            .collect()
    }

    /// Get all registered agents
    pub fn all(&self) -> Vec<&AgentCapability> {
        self.agents.values().collect()
    }

    /// Get agent count
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

/// Router for handling agent handoffs
pub struct HandoffRouter {
    registry: AgentRegistry,
    /// Minimum confidence threshold for accepting a task
    min_confidence: f32,
    /// History of recent handoffs for analysis
    handoff_history: Vec<HandoffRecord>,
    /// Maximum history size
    max_history: usize,
}

/// Record of a handoff event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffRecord {
    /// Source agent
    pub from_agent: String,
    /// Target agent
    pub to_agent: String,
    /// Task description
    pub task: String,
    /// Match score
    pub score: f32,
    /// Timestamp
    pub timestamp: i64,
}

impl HandoffRouter {
    /// Create a new handoff router
    pub fn new(registry: AgentRegistry) -> Self {
        Self {
            registry,
            min_confidence: 0.3,
            handoff_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = threshold.clamp(0.0, 1.0);
        self
    }

    /// Route a task to the best agent
    pub fn route(&self, task: &str) -> Option<(String, f32)> {
        self.registry.find_best_agent(task, None)
    }

    /// Route a task, excluding a specific agent
    pub fn route_excluding(&self, task: &str, exclude: &str) -> Option<(String, f32)> {
        self.registry.find_best_agent(task, Some(exclude))
    }

    /// Evaluate if an agent should handle a task or hand off
    pub fn evaluate(&self, agent: &str, task: &str) -> HandoffDecision {
        let current_agent = self.registry.get(agent);
        let current_score = current_agent.map(|a| a.match_score(task)).unwrap_or(0.0);

        // Check if current agent is confident enough
        if current_score >= self.min_confidence {
            return HandoffDecision::accept(current_score);
        }

        // Find a better agent
        if let Some((best_agent, best_score)) = self.route_excluding(task, agent) {
            if best_score > current_score {
                return HandoffDecision::handoff(
                    &best_agent,
                    &format!(
                        "Agent '{}' is better suited for this task (score: {:.2} vs {:.2})",
                        best_agent, best_score, current_score
                    ),
                );
            }
        }

        // No better option, accept with low confidence
        HandoffDecision::uncertain(current_score, None)
    }

    /// Record a handoff for analysis
    pub fn record_handoff(&mut self, from: &str, to: &str, task: &str, score: f32) {
        let record = HandoffRecord {
            from_agent: from.to_string(),
            to_agent: to.to_string(),
            task: task.to_string(),
            score,
            timestamp: chrono::Utc::now().timestamp(),
        };

        self.handoff_history.push(record);

        // Trim history
        if self.handoff_history.len() > self.max_history {
            self.handoff_history.remove(0);
        }
    }

    /// Get handoff statistics
    pub fn stats(&self) -> HandoffStats {
        let mut stats = HandoffStats::default();
        let mut agent_counts: HashMap<String, u32> = HashMap::new();

        for record in &self.handoff_history {
            stats.total_handoffs += 1;
            stats.total_score += record.score;
            *agent_counts.entry(record.to_agent.clone()).or_insert(0) += 1;
        }

        if stats.total_handoffs > 0 {
            stats.avg_score = stats.total_score / stats.total_handoffs as f32;
        }

        // Find most common target
        if let Some((agent, count)) = agent_counts.iter().max_by_key(|(_, c)| *c) {
            stats.most_common_target = Some(agent.clone());
            stats.most_common_count = *count;
        }

        stats
    }

    /// Get registry reference
    pub fn registry(&self) -> &AgentRegistry {
        &self.registry
    }

    /// Get mutable registry reference
    pub fn registry_mut(&mut self) -> &mut AgentRegistry {
        &mut self.registry
    }
}

/// Statistics about handoffs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HandoffStats {
    pub total_handoffs: u32,
    pub avg_score: f32,
    pub total_score: f32,
    pub most_common_target: Option<String>,
    pub most_common_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_registry() -> AgentRegistry {
        let mut registry = AgentRegistry::new();

        registry.register(AgentCapability::new(
            "code-expert",
            "programming",
            vec!["rust", "python", "code", "debug", "compile"],
            vec!["execute_code", "analyze_code"],
        ));

        registry.register(AgentCapability::new(
            "data-analyst",
            "data",
            vec!["data", "analysis", "statistics", "chart", "graph"],
            vec!["query_database", "create_chart"],
        ));

        registry.register(AgentCapability::new(
            "writer",
            "writing",
            vec!["write", "edit", "document", "essay", "article"],
            vec!["search_web", "create_document"],
        ));

        registry
    }

    #[test]
    fn test_capability_match_score() {
        let agent = AgentCapability::new(
            "code-expert",
            "programming",
            vec!["rust", "python", "code"],
            Vec::<String>::new(),
        );

        assert!(agent.match_score("help me write rust code") > 0.3);
        assert!(agent.match_score("python debugging") > 0.0);
        assert_eq!(agent.match_score("cook me dinner"), 0.0);
    }

    #[test]
    fn test_registry_find_best() {
        let registry = create_test_registry();

        let (agent, score) = registry
            .find_best_agent("help me debug this rust code", None)
            .unwrap();
        assert_eq!(agent, "code-expert");
        assert!(score > 0.3);

        let (agent, _) = registry
            .find_best_agent("analyze the sales data", None)
            .unwrap();
        assert_eq!(agent, "data-analyst");

        let (agent, _) = registry
            .find_best_agent("write an article about AI", None)
            .unwrap();
        assert_eq!(agent, "writer");
    }

    #[test]
    fn test_registry_exclude() {
        let registry = create_test_registry();

        // Should not return code-expert when excluded
        let result = registry.find_best_agent("rust programming", Some("code-expert"));
        assert!(result.is_none() || result.as_ref().unwrap().0 != "code-expert");
    }

    #[test]
    fn test_handoff_decision() {
        let decision = HandoffDecision::accept(0.8);
        assert!(decision.can_handle);
        assert!(!decision.needs_handoff());

        let decision = HandoffDecision::handoff("other-agent", "better match");
        assert!(!decision.can_handle);
        assert!(decision.needs_handoff());
        assert_eq!(decision.recommended_agent, Some("other-agent".to_string()));
    }

    #[test]
    fn test_router_evaluate() {
        let registry = create_test_registry();
        let router = HandoffRouter::new(registry).with_min_confidence(0.3);

        // Code expert should handle code task
        let decision = router.evaluate("code-expert", "debug this rust code");
        assert!(decision.can_handle);
        assert!(decision.confidence > 0.3);

        // Writer should hand off code task
        let decision = router.evaluate("writer", "debug this rust code");
        if decision.needs_handoff() {
            assert_eq!(decision.recommended_agent, Some("code-expert".to_string()));
        }
    }

    #[test]
    fn test_router_route() {
        let registry = create_test_registry();
        let router = HandoffRouter::new(registry);

        let (agent, score) = router.route("analyze the data and create a chart").unwrap();
        assert_eq!(agent, "data-analyst");
        assert!(score > 0.0);
    }

    #[test]
    fn test_find_by_tool() {
        let registry = create_test_registry();

        let agents = registry.find_by_tool("execute_code");
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].name, "code-expert");

        let agents = registry.find_by_tool("search_web");
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].name, "writer");
    }

    #[test]
    fn test_handoff_stats() {
        let registry = create_test_registry();
        let mut router = HandoffRouter::new(registry);

        router.record_handoff("writer", "code-expert", "debug code", 0.8);
        router.record_handoff("data-analyst", "code-expert", "fix bug", 0.9);
        router.record_handoff("writer", "data-analyst", "analyze data", 0.7);

        let stats = router.stats();
        assert_eq!(stats.total_handoffs, 3);
        assert!(stats.avg_score > 0.7);
        assert_eq!(stats.most_common_target, Some("code-expert".to_string()));
        assert_eq!(stats.most_common_count, 2);
    }
}
