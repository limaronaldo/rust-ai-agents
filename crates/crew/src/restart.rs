//! # Agent Restart Logic
//!
//! MassGen-inspired restart system that allows agents to restart when new
//! information arrives or when better answers become available.
//!
//! ## Features
//!
//! - **Context Updates**: Restart agents when relevant new information arrives
//! - **Answer Improvement**: Re-run agents when peer answers suggest better approaches
//! - **Selective Restart**: Only restart agents that would benefit from new info
//! - **Restart Limits**: Prevent infinite restart loops
//! - **State Preservation**: Maintain agent progress across restarts
//!
//! ## Example
//!
//! ```ignore
//! use rust_ai_agents_crew::restart::*;
//!
//! let manager = RestartManager::new(RestartConfig::default());
//!
//! // Register an agent
//! manager.register_agent("searcher", AgentRestartPolicy::default()).await;
//!
//! // Agent produces initial answer
//! manager.set_answer("searcher", serde_json::json!({"results": 5})).await;
//!
//! // New context arrives that might improve the answer
//! let should_restart = manager.notify_new_context(
//!     "searcher",
//!     NewContext::PeerAnswer {
//!         peer_id: "validator".to_string(),
//!         answer: serde_json::json!({"found_more": true}),
//!     }
//! ).await;
//!
//! if should_restart {
//!     // Restart the agent with updated context
//!     let context = manager.get_restart_context("searcher").await;
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};

/// Types of new context that might trigger a restart
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NewContext {
    /// A peer agent produced a new answer
    PeerAnswer {
        peer_id: String,
        answer: serde_json::Value,
    },
    /// User provided additional information
    UserInput { content: String },
    /// External data source provided updates
    ExternalData {
        source: String,
        data: serde_json::Value,
    },
    /// A dependency completed with new information
    DependencyComplete {
        dependency_id: String,
        result: serde_json::Value,
    },
    /// Validation feedback suggesting improvements
    ValidationFeedback {
        validator_id: String,
        feedback: String,
        suggestions: Vec<String>,
    },
    /// Error in current approach requiring retry
    ErrorRecovery {
        error: String,
        recovery_hint: Option<String>,
    },
    /// Custom context type
    Custom {
        context_type: String,
        data: serde_json::Value,
    },
}

impl NewContext {
    /// Get a description of the context type
    pub fn description(&self) -> String {
        match self {
            NewContext::PeerAnswer { peer_id, .. } => {
                format!("Peer answer from {}", peer_id)
            }
            NewContext::UserInput { .. } => "User input".to_string(),
            NewContext::ExternalData { source, .. } => {
                format!("External data from {}", source)
            }
            NewContext::DependencyComplete { dependency_id, .. } => {
                format!("Dependency {} completed", dependency_id)
            }
            NewContext::ValidationFeedback { validator_id, .. } => {
                format!("Validation feedback from {}", validator_id)
            }
            NewContext::ErrorRecovery { .. } => "Error recovery".to_string(),
            NewContext::Custom { context_type, .. } => {
                format!("Custom: {}", context_type)
            }
        }
    }
}

/// Restart trigger conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RestartTrigger {
    /// Restart on any new context
    #[default]
    Always,
    /// Only restart on peer answers
    OnPeerAnswer,
    /// Only restart on user input
    OnUserInput,
    /// Only restart on validation feedback
    OnValidationFeedback,
    /// Only restart on errors
    OnError,
    /// Never restart automatically
    Never,
    /// Custom logic (check with relevance score)
    Custom,
}

/// Policy for agent restarts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRestartPolicy {
    /// When to trigger restarts
    pub trigger: RestartTrigger,
    /// Maximum number of restarts allowed
    pub max_restarts: usize,
    /// Minimum time between restarts
    pub cooldown: Duration,
    /// Whether to preserve previous answer as context
    pub preserve_previous_answer: bool,
    /// Minimum relevance score to trigger restart (0.0 to 1.0)
    pub min_relevance_score: f32,
    /// Whether restarts count against rate limits
    pub count_against_rate_limit: bool,
}

impl Default for AgentRestartPolicy {
    fn default() -> Self {
        Self {
            trigger: RestartTrigger::Always,
            max_restarts: 3,
            cooldown: Duration::from_secs(5),
            preserve_previous_answer: true,
            min_relevance_score: 0.5,
            count_against_rate_limit: true,
        }
    }
}

impl AgentRestartPolicy {
    /// Create a policy that never restarts
    pub fn never() -> Self {
        Self {
            trigger: RestartTrigger::Never,
            ..Default::default()
        }
    }

    /// Create a policy that only restarts on errors
    pub fn on_error_only() -> Self {
        Self {
            trigger: RestartTrigger::OnError,
            max_restarts: 5,
            ..Default::default()
        }
    }

    /// Create a policy for validators (restart on peer answers)
    pub fn validator() -> Self {
        Self {
            trigger: RestartTrigger::OnPeerAnswer,
            max_restarts: 2,
            cooldown: Duration::from_secs(2),
            ..Default::default()
        }
    }
}

/// State of an agent for restart purposes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRestartState {
    /// Agent ID
    pub agent_id: String,
    /// Current restart count
    pub restart_count: usize,
    /// Last restart timestamp
    pub last_restart_at: Option<u64>,
    /// Current answer (if any)
    pub current_answer: Option<serde_json::Value>,
    /// Previous answers (for context)
    pub previous_answers: Vec<serde_json::Value>,
    /// Accumulated context from restarts
    pub accumulated_context: Vec<NewContext>,
    /// Whether a restart is pending
    pub restart_pending: bool,
    /// Reason for pending restart
    pub pending_reason: Option<String>,
}

/// Restart context provided to agent on restart
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartContext {
    /// Previous answer to improve upon
    pub previous_answer: Option<serde_json::Value>,
    /// New context that triggered restart
    pub new_context: Vec<NewContext>,
    /// Restart number (1-indexed)
    pub restart_number: usize,
    /// Maximum restarts remaining
    pub restarts_remaining: usize,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

/// Events from the restart manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestartEvent {
    /// Agent restart triggered
    RestartTriggered {
        agent_id: String,
        reason: String,
        restart_number: usize,
    },
    /// Agent restart completed
    RestartCompleted { agent_id: String, improved: bool },
    /// Restart limit reached
    RestartLimitReached {
        agent_id: String,
        max_restarts: usize,
    },
    /// Restart skipped (cooldown or irrelevant)
    RestartSkipped { agent_id: String, reason: String },
    /// New context received
    NewContextReceived {
        agent_id: String,
        context_type: String,
    },
}

/// Result of evaluating whether to restart
#[derive(Debug, Clone)]
pub struct RestartDecision {
    /// Whether to restart
    pub should_restart: bool,
    /// Reason for decision
    pub reason: String,
    /// Relevance score of new context
    pub relevance_score: f32,
    /// Whether cooldown is active
    pub in_cooldown: bool,
    /// Restarts remaining
    pub restarts_remaining: usize,
}

/// Internal agent state
struct AgentState {
    policy: AgentRestartPolicy,
    restart_state: AgentRestartState,
    #[allow(dead_code)]
    registered_at: Instant,
    last_restart_instant: Option<Instant>,
}

/// Configuration for the restart manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartConfig {
    /// Global maximum restarts per agent
    pub global_max_restarts: usize,
    /// Global cooldown between any restarts
    pub global_cooldown: Duration,
    /// Maximum accumulated context items
    pub max_context_items: usize,
    /// Maximum previous answers to keep
    pub max_previous_answers: usize,
    /// Default relevance score for unscored context
    pub default_relevance_score: f32,
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            global_max_restarts: 10,
            global_cooldown: Duration::from_secs(1),
            max_context_items: 20,
            max_previous_answers: 5,
            default_relevance_score: 0.7,
        }
    }
}

/// Manager for coordinating agent restarts
pub struct RestartManager {
    agents: Arc<RwLock<HashMap<String, AgentState>>>,
    config: RestartConfig,
    event_tx: broadcast::Sender<RestartEvent>,
}

impl RestartManager {
    /// Create a new restart manager
    pub fn new(config: RestartConfig) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            config,
            event_tx,
        }
    }

    /// Subscribe to restart events
    pub fn subscribe(&self) -> broadcast::Receiver<RestartEvent> {
        self.event_tx.subscribe()
    }

    /// Register an agent with restart policy
    pub async fn register_agent(&self, agent_id: impl Into<String>, policy: AgentRestartPolicy) {
        let agent_id = agent_id.into();
        let state = AgentState {
            policy,
            restart_state: AgentRestartState {
                agent_id: agent_id.clone(),
                restart_count: 0,
                last_restart_at: None,
                current_answer: None,
                previous_answers: Vec::new(),
                accumulated_context: Vec::new(),
                restart_pending: false,
                pending_reason: None,
            },
            registered_at: Instant::now(),
            last_restart_instant: None,
        };

        self.agents.write().await.insert(agent_id, state);
    }

    /// Set the current answer for an agent
    pub async fn set_answer(&self, agent_id: &str, answer: serde_json::Value) {
        let mut agents = self.agents.write().await;
        if let Some(state) = agents.get_mut(agent_id) {
            // Move current answer to previous if exists
            if let Some(prev) = state.restart_state.current_answer.take() {
                state.restart_state.previous_answers.push(prev);
                // Trim previous answers
                while state.restart_state.previous_answers.len() > self.config.max_previous_answers
                {
                    state.restart_state.previous_answers.remove(0);
                }
            }
            state.restart_state.current_answer = Some(answer);
            state.restart_state.restart_pending = false;
            state.restart_state.pending_reason = None;
        }
    }

    /// Notify agent of new context, returns whether restart is recommended
    pub async fn notify_new_context(&self, agent_id: &str, context: NewContext) -> RestartDecision {
        let mut agents = self.agents.write().await;

        let Some(state) = agents.get_mut(agent_id) else {
            return RestartDecision {
                should_restart: false,
                reason: "Agent not registered".to_string(),
                relevance_score: 0.0,
                in_cooldown: false,
                restarts_remaining: 0,
            };
        };

        // Broadcast new context event
        let _ = self.event_tx.send(RestartEvent::NewContextReceived {
            agent_id: agent_id.to_string(),
            context_type: context.description(),
        });

        // Check if trigger matches
        let trigger_matches = self.check_trigger_matches(&state.policy.trigger, &context);
        if !trigger_matches {
            return RestartDecision {
                should_restart: false,
                reason: "Context type doesn't match restart trigger".to_string(),
                relevance_score: 0.0,
                in_cooldown: false,
                restarts_remaining: state.policy.max_restarts - state.restart_state.restart_count,
            };
        }

        // Check restart limit
        if state.restart_state.restart_count >= state.policy.max_restarts {
            let _ = self.event_tx.send(RestartEvent::RestartLimitReached {
                agent_id: agent_id.to_string(),
                max_restarts: state.policy.max_restarts,
            });
            return RestartDecision {
                should_restart: false,
                reason: "Restart limit reached".to_string(),
                relevance_score: 0.0,
                in_cooldown: false,
                restarts_remaining: 0,
            };
        }

        // Check cooldown
        let in_cooldown = state
            .last_restart_instant
            .map(|last| last.elapsed() < state.policy.cooldown)
            .unwrap_or(false);

        if in_cooldown {
            let _ = self.event_tx.send(RestartEvent::RestartSkipped {
                agent_id: agent_id.to_string(),
                reason: "Cooldown active".to_string(),
            });
            return RestartDecision {
                should_restart: false,
                reason: "Cooldown period active".to_string(),
                relevance_score: self.config.default_relevance_score,
                in_cooldown: true,
                restarts_remaining: state.policy.max_restarts - state.restart_state.restart_count,
            };
        }

        // Calculate relevance score
        let relevance_score = self.calculate_relevance(&context, &state.restart_state);

        if relevance_score < state.policy.min_relevance_score {
            let _ = self.event_tx.send(RestartEvent::RestartSkipped {
                agent_id: agent_id.to_string(),
                reason: format!(
                    "Relevance score {:.2} below threshold {:.2}",
                    relevance_score, state.policy.min_relevance_score
                ),
            });
            return RestartDecision {
                should_restart: false,
                reason: format!("Relevance score {:.2} below threshold", relevance_score),
                relevance_score,
                in_cooldown: false,
                restarts_remaining: state.policy.max_restarts - state.restart_state.restart_count,
            };
        }

        // Add context to accumulated
        state
            .restart_state
            .accumulated_context
            .push(context.clone());
        while state.restart_state.accumulated_context.len() > self.config.max_context_items {
            state.restart_state.accumulated_context.remove(0);
        }

        // Mark restart as pending
        state.restart_state.restart_pending = true;
        state.restart_state.pending_reason = Some(context.description());

        let _ = self.event_tx.send(RestartEvent::RestartTriggered {
            agent_id: agent_id.to_string(),
            reason: context.description(),
            restart_number: state.restart_state.restart_count + 1,
        });

        RestartDecision {
            should_restart: true,
            reason: context.description(),
            relevance_score,
            in_cooldown: false,
            restarts_remaining: state.policy.max_restarts - state.restart_state.restart_count - 1,
        }
    }

    /// Get context for restarting an agent
    pub async fn get_restart_context(&self, agent_id: &str) -> Option<RestartContext> {
        let mut agents = self.agents.write().await;
        let state = agents.get_mut(agent_id)?;

        if !state.restart_state.restart_pending {
            return None;
        }

        // Increment restart count
        state.restart_state.restart_count += 1;
        state.last_restart_instant = Some(Instant::now());
        state.restart_state.last_restart_at = Some(current_timestamp());

        // Build suggestions from validation feedback
        let suggestions: Vec<String> = state
            .restart_state
            .accumulated_context
            .iter()
            .filter_map(|ctx| {
                if let NewContext::ValidationFeedback { suggestions, .. } = ctx {
                    Some(suggestions.clone())
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        let context = RestartContext {
            previous_answer: if state.policy.preserve_previous_answer {
                state.restart_state.current_answer.clone()
            } else {
                None
            },
            new_context: state.restart_state.accumulated_context.clone(),
            restart_number: state.restart_state.restart_count,
            restarts_remaining: state
                .policy
                .max_restarts
                .saturating_sub(state.restart_state.restart_count),
            suggestions,
        };

        // Clear pending state
        state.restart_state.restart_pending = false;
        state.restart_state.pending_reason = None;
        // Clear accumulated context after providing it
        state.restart_state.accumulated_context.clear();

        Some(context)
    }

    /// Mark restart as completed
    pub async fn complete_restart(&self, agent_id: &str, improved: bool) {
        let _ = self.event_tx.send(RestartEvent::RestartCompleted {
            agent_id: agent_id.to_string(),
            improved,
        });
    }

    /// Check if an agent has a pending restart
    pub async fn has_pending_restart(&self, agent_id: &str) -> bool {
        self.agents
            .read()
            .await
            .get(agent_id)
            .map(|s| s.restart_state.restart_pending)
            .unwrap_or(false)
    }

    /// Get current restart state for an agent
    pub async fn get_state(&self, agent_id: &str) -> Option<AgentRestartState> {
        self.agents
            .read()
            .await
            .get(agent_id)
            .map(|s| s.restart_state.clone())
    }

    /// Reset restart count for an agent
    pub async fn reset_restart_count(&self, agent_id: &str) {
        if let Some(state) = self.agents.write().await.get_mut(agent_id) {
            state.restart_state.restart_count = 0;
            state.restart_state.accumulated_context.clear();
        }
    }

    /// Get all agents with pending restarts
    pub async fn get_pending_restarts(&self) -> Vec<String> {
        self.agents
            .read()
            .await
            .iter()
            .filter(|(_, s)| s.restart_state.restart_pending)
            .map(|(id, _)| id.clone())
            .collect()
    }

    // Internal helpers

    fn check_trigger_matches(&self, trigger: &RestartTrigger, context: &NewContext) -> bool {
        match trigger {
            RestartTrigger::Always => true,
            RestartTrigger::Never => false,
            RestartTrigger::OnPeerAnswer => matches!(context, NewContext::PeerAnswer { .. }),
            RestartTrigger::OnUserInput => matches!(context, NewContext::UserInput { .. }),
            RestartTrigger::OnValidationFeedback => {
                matches!(context, NewContext::ValidationFeedback { .. })
            }
            RestartTrigger::OnError => matches!(context, NewContext::ErrorRecovery { .. }),
            RestartTrigger::Custom => true, // Always check, rely on relevance score
        }
    }

    fn calculate_relevance(&self, context: &NewContext, state: &AgentRestartState) -> f32 {
        // Base relevance by context type
        let base_score = match context {
            NewContext::UserInput { .. } => 1.0, // User input is always highly relevant
            NewContext::ErrorRecovery { .. } => 0.9, // Errors should usually trigger restart
            NewContext::ValidationFeedback { .. } => 0.8,
            NewContext::DependencyComplete { .. } => 0.7,
            NewContext::PeerAnswer { .. } => 0.6,
            NewContext::ExternalData { .. } => 0.5,
            NewContext::Custom { .. } => self.config.default_relevance_score,
        };

        // Reduce score if we've restarted many times already
        let restart_penalty = 0.1 * state.restart_count as f32;

        (base_score - restart_penalty).max(0.0)
    }
}

impl Default for RestartManager {
    fn default() -> Self {
        Self::new(RestartConfig::default())
    }
}

/// Helper function to get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_agent() {
        let manager = RestartManager::default();
        manager
            .register_agent("agent1", AgentRestartPolicy::default())
            .await;

        let state = manager.get_state("agent1").await;
        assert!(state.is_some());
        assert_eq!(state.unwrap().restart_count, 0);
    }

    #[tokio::test]
    async fn test_set_answer() {
        let manager = RestartManager::default();
        manager
            .register_agent("agent1", AgentRestartPolicy::default())
            .await;

        manager
            .set_answer("agent1", serde_json::json!({"result": 42}))
            .await;

        let state = manager.get_state("agent1").await.unwrap();
        assert!(state.current_answer.is_some());
    }

    #[tokio::test]
    async fn test_notify_context_triggers_restart() {
        let manager = RestartManager::default();
        manager
            .register_agent("agent1", AgentRestartPolicy::default())
            .await;

        let decision = manager
            .notify_new_context(
                "agent1",
                NewContext::UserInput {
                    content: "More details".to_string(),
                },
            )
            .await;

        assert!(decision.should_restart);
        assert!(manager.has_pending_restart("agent1").await);
    }

    #[tokio::test]
    async fn test_restart_limit() {
        let policy = AgentRestartPolicy {
            max_restarts: 2,
            cooldown: Duration::ZERO,
            ..Default::default()
        };

        let manager = RestartManager::default();
        manager.register_agent("agent1", policy).await;

        // First two restarts should work
        for _ in 0..2 {
            let decision = manager
                .notify_new_context(
                    "agent1",
                    NewContext::UserInput {
                        content: "test".to_string(),
                    },
                )
                .await;
            assert!(decision.should_restart);
            manager.get_restart_context("agent1").await;
        }

        // Third should fail
        let decision = manager
            .notify_new_context(
                "agent1",
                NewContext::UserInput {
                    content: "test".to_string(),
                },
            )
            .await;
        assert!(!decision.should_restart);
        assert_eq!(decision.restarts_remaining, 0);
    }

    #[tokio::test]
    async fn test_cooldown() {
        let policy = AgentRestartPolicy {
            cooldown: Duration::from_secs(10),
            ..Default::default()
        };

        let manager = RestartManager::default();
        manager.register_agent("agent1", policy).await;

        // First restart works
        let decision = manager
            .notify_new_context(
                "agent1",
                NewContext::UserInput {
                    content: "test".to_string(),
                },
            )
            .await;
        assert!(decision.should_restart);
        manager.get_restart_context("agent1").await;

        // Second should be blocked by cooldown
        let decision = manager
            .notify_new_context(
                "agent1",
                NewContext::UserInput {
                    content: "test2".to_string(),
                },
            )
            .await;
        assert!(!decision.should_restart);
        assert!(decision.in_cooldown);
    }

    #[tokio::test]
    async fn test_trigger_matching() {
        let policy = AgentRestartPolicy {
            trigger: RestartTrigger::OnPeerAnswer,
            ..Default::default()
        };

        let manager = RestartManager::default();
        manager.register_agent("agent1", policy).await;

        // User input should not trigger
        let decision = manager
            .notify_new_context(
                "agent1",
                NewContext::UserInput {
                    content: "test".to_string(),
                },
            )
            .await;
        assert!(!decision.should_restart);

        // Peer answer should trigger
        let decision = manager
            .notify_new_context(
                "agent1",
                NewContext::PeerAnswer {
                    peer_id: "peer".to_string(),
                    answer: serde_json::json!({}),
                },
            )
            .await;
        assert!(decision.should_restart);
    }

    #[tokio::test]
    async fn test_get_restart_context() {
        let manager = RestartManager::default();
        manager
            .register_agent("agent1", AgentRestartPolicy::default())
            .await;

        manager
            .set_answer("agent1", serde_json::json!({"initial": true}))
            .await;

        manager
            .notify_new_context(
                "agent1",
                NewContext::ValidationFeedback {
                    validator_id: "validator".to_string(),
                    feedback: "Could be better".to_string(),
                    suggestions: vec!["Try approach X".to_string()],
                },
            )
            .await;

        let context = manager.get_restart_context("agent1").await.unwrap();
        assert!(context.previous_answer.is_some());
        assert_eq!(context.restart_number, 1);
        assert!(!context.suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_never_restart_policy() {
        let manager = RestartManager::default();
        manager
            .register_agent("agent1", AgentRestartPolicy::never())
            .await;

        let decision = manager
            .notify_new_context(
                "agent1",
                NewContext::UserInput {
                    content: "test".to_string(),
                },
            )
            .await;
        assert!(!decision.should_restart);
    }

    #[tokio::test]
    async fn test_reset_restart_count() {
        let manager = RestartManager::default();
        manager
            .register_agent(
                "agent1",
                AgentRestartPolicy {
                    cooldown: Duration::ZERO,
                    ..Default::default()
                },
            )
            .await;

        // Use up some restarts
        manager
            .notify_new_context(
                "agent1",
                NewContext::UserInput {
                    content: "1".to_string(),
                },
            )
            .await;
        manager.get_restart_context("agent1").await;

        let state = manager.get_state("agent1").await.unwrap();
        assert_eq!(state.restart_count, 1);

        // Reset
        manager.reset_restart_count("agent1").await;

        let state = manager.get_state("agent1").await.unwrap();
        assert_eq!(state.restart_count, 0);
    }

    #[tokio::test]
    async fn test_previous_answers_preserved() {
        let manager = RestartManager::default();
        manager
            .register_agent("agent1", AgentRestartPolicy::default())
            .await;

        // Set multiple answers
        manager
            .set_answer("agent1", serde_json::json!({"v": 1}))
            .await;
        manager
            .set_answer("agent1", serde_json::json!({"v": 2}))
            .await;
        manager
            .set_answer("agent1", serde_json::json!({"v": 3}))
            .await;

        let state = manager.get_state("agent1").await.unwrap();
        assert_eq!(state.previous_answers.len(), 2);
        assert_eq!(state.current_answer, Some(serde_json::json!({"v": 3})));
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let manager = RestartManager::default();
        let mut rx = manager.subscribe();

        manager
            .register_agent("agent1", AgentRestartPolicy::default())
            .await;

        manager
            .notify_new_context(
                "agent1",
                NewContext::UserInput {
                    content: "test".to_string(),
                },
            )
            .await;

        // Should receive context event and trigger event
        let event = rx.try_recv().unwrap();
        assert!(matches!(event, RestartEvent::NewContextReceived { .. }));
    }
}
