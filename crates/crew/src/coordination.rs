//! # Agent Coordination Tracker
//!
//! MassGen-inspired coordination system for tracking agent progress and orchestrating
//! multi-agent workflows with real-time status monitoring.
//!
//! ## Features
//!
//! - **Progress Tracking**: Monitor each agent's status and completion percentage
//! - **Dependency Management**: Track inter-agent dependencies and blocking states
//! - **Timeline Visualization**: View execution timeline and parallel activities
//! - **Bottleneck Detection**: Identify slow agents and coordination issues
//! - **Event Broadcasting**: Real-time updates via channels
//!
//! ## Example
//!
//! ```ignore
//! use rust_ai_agents_crew::coordination::*;
//!
//! let tracker = CoordinationTracker::new("query-123");
//!
//! // Register agents
//! tracker.register_agent("searcher", AgentRole::Primary).await;
//! tracker.register_agent("validator", AgentRole::Validator).await;
//!
//! // Update progress
//! tracker.update_status("searcher", AgentStatus::Working { progress: 0.5 }).await;
//! tracker.mark_complete("searcher", Some("Found 10 results")).await;
//!
//! // Get coordination summary
//! let summary = tracker.summary().await;
//! println!("Progress: {}%", summary.overall_progress * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};

/// Agent status in the coordination workflow
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent registered but not yet started
    Pending,
    /// Agent is starting up
    Starting,
    /// Agent is actively working
    Working {
        /// Progress percentage (0.0 to 1.0)
        progress: f32,
    },
    /// Agent is waiting for dependencies
    Blocked {
        /// IDs of agents this one is waiting for
        waiting_for: Vec<String>,
    },
    /// Agent completed successfully
    Completed {
        /// Optional result summary
        result_summary: Option<String>,
    },
    /// Agent failed
    Failed {
        /// Error message
        error: String,
    },
    /// Agent was cancelled
    Cancelled {
        /// Reason for cancellation
        reason: Option<String>,
    },
    /// Agent is paused
    Paused,
}

impl AgentStatus {
    /// Check if agent is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AgentStatus::Completed { .. }
                | AgentStatus::Failed { .. }
                | AgentStatus::Cancelled { .. }
        )
    }

    /// Check if agent is actively working
    pub fn is_active(&self) -> bool {
        matches!(self, AgentStatus::Starting | AgentStatus::Working { .. })
    }

    /// Get progress value (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        match self {
            AgentStatus::Pending | AgentStatus::Starting => 0.0,
            AgentStatus::Working { progress } => *progress,
            AgentStatus::Blocked { .. } | AgentStatus::Paused => 0.0, // Keep last progress
            AgentStatus::Completed { .. } => 1.0,
            AgentStatus::Failed { .. } | AgentStatus::Cancelled { .. } => 0.0,
        }
    }
}

/// Role of an agent in the coordination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AgentRole {
    /// Primary agent doing main work
    #[default]
    Primary,
    /// Supporting agent providing additional context
    Support,
    /// Validator agent checking results
    Validator,
    /// Synthesizer combining results
    Synthesizer,
    /// Coordinator managing other agents
    Coordinator,
    /// Observer for monitoring/logging
    Observer,
}

/// Information about a tracked agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent identifier
    pub id: String,
    /// Agent role
    pub role: AgentRole,
    /// Current status
    pub status: AgentStatus,
    /// Last progress value (preserved across status changes)
    pub last_progress: f32,
    /// When agent was registered
    pub registered_at: u64,
    /// When agent started working
    pub started_at: Option<u64>,
    /// When agent completed
    pub completed_at: Option<u64>,
    /// Duration of work in milliseconds
    pub duration_ms: Option<u64>,
    /// Dependencies (agents that must complete first)
    pub dependencies: Vec<String>,
    /// Dependents (agents waiting for this one)
    pub dependents: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentInfo {
    fn new(id: String, role: AgentRole) -> Self {
        Self {
            id,
            role,
            status: AgentStatus::Pending,
            last_progress: 0.0,
            registered_at: current_timestamp(),
            started_at: None,
            completed_at: None,
            duration_ms: None,
            dependencies: Vec::new(),
            dependents: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Coordination event for broadcasting updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationEvent {
    /// Agent registered
    AgentRegistered { agent_id: String, role: AgentRole },
    /// Agent status changed
    StatusChanged {
        agent_id: String,
        old_status: AgentStatus,
        new_status: AgentStatus,
    },
    /// Agent completed
    AgentCompleted {
        agent_id: String,
        result_summary: Option<String>,
        duration_ms: u64,
    },
    /// Agent failed
    AgentFailed { agent_id: String, error: String },
    /// Dependency resolved
    DependencyResolved {
        agent_id: String,
        dependency_id: String,
    },
    /// All agents completed
    AllCompleted {
        total_duration_ms: u64,
        success_count: usize,
        failure_count: usize,
    },
    /// Bottleneck detected
    BottleneckDetected {
        agent_id: String,
        blocked_agents: Vec<String>,
        duration_ms: u64,
    },
}

/// Timeline entry for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// Timestamp in milliseconds since coordination start
    pub timestamp_ms: u64,
    /// Agent ID
    pub agent_id: String,
    /// Event type
    pub event: String,
    /// Additional details
    pub details: Option<String>,
}

/// Coordination summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationSummary {
    /// Coordination ID
    pub coordination_id: String,
    /// Overall progress (0.0 to 1.0)
    pub overall_progress: f32,
    /// Number of registered agents
    pub total_agents: usize,
    /// Number of completed agents
    pub completed_agents: usize,
    /// Number of failed agents
    pub failed_agents: usize,
    /// Number of active agents
    pub active_agents: usize,
    /// Number of blocked agents
    pub blocked_agents: usize,
    /// Total duration so far in milliseconds
    pub duration_ms: u64,
    /// Estimated time remaining in milliseconds (if calculable)
    pub estimated_remaining_ms: Option<u64>,
    /// Current bottlenecks (slow agents blocking others)
    pub bottlenecks: Vec<String>,
    /// Is coordination complete?
    pub is_complete: bool,
    /// Was coordination successful?
    pub is_successful: bool,
}

/// Configuration for coordination tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationConfig {
    /// Timeout for the entire coordination
    pub timeout: Duration,
    /// Interval for bottleneck detection
    pub bottleneck_check_interval: Duration,
    /// Threshold for considering an agent a bottleneck (seconds)
    pub bottleneck_threshold_secs: u64,
    /// Maximum event history to keep
    pub max_timeline_entries: usize,
    /// Whether to auto-cancel on timeout
    pub auto_cancel_on_timeout: bool,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            bottleneck_check_interval: Duration::from_secs(5),
            bottleneck_threshold_secs: 30,
            max_timeline_entries: 1000,
            auto_cancel_on_timeout: true,
        }
    }
}

/// Internal state for the tracker
struct TrackerState {
    coordination_id: String,
    agents: HashMap<String, AgentInfo>,
    timeline: Vec<TimelineEntry>,
    started_at: Instant,
    completed_at: Option<Instant>,
}

/// Coordination tracker for monitoring multi-agent workflows
pub struct CoordinationTracker {
    state: Arc<RwLock<TrackerState>>,
    config: CoordinationConfig,
    event_tx: broadcast::Sender<CoordinationEvent>,
}

impl CoordinationTracker {
    /// Create a new coordination tracker
    pub fn new(coordination_id: impl Into<String>) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            state: Arc::new(RwLock::new(TrackerState {
                coordination_id: coordination_id.into(),
                agents: HashMap::new(),
                timeline: Vec::new(),
                started_at: Instant::now(),
                completed_at: None,
            })),
            config: CoordinationConfig::default(),
            event_tx,
        }
    }

    /// Create with custom configuration
    pub fn with_config(coordination_id: impl Into<String>, config: CoordinationConfig) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            state: Arc::new(RwLock::new(TrackerState {
                coordination_id: coordination_id.into(),
                agents: HashMap::new(),
                timeline: Vec::new(),
                started_at: Instant::now(),
                completed_at: None,
            })),
            config,
            event_tx,
        }
    }

    /// Subscribe to coordination events
    pub fn subscribe(&self) -> broadcast::Receiver<CoordinationEvent> {
        self.event_tx.subscribe()
    }

    /// Register an agent for tracking
    pub async fn register_agent(&self, agent_id: impl Into<String>, role: AgentRole) {
        let agent_id = agent_id.into();
        let mut state = self.state.write().await;

        let agent = AgentInfo::new(agent_id.clone(), role);
        state.agents.insert(agent_id.clone(), agent);

        self.add_timeline_entry(&mut state, &agent_id, "registered", None);

        let _ = self
            .event_tx
            .send(CoordinationEvent::AgentRegistered { agent_id, role });
    }

    /// Register an agent with dependencies
    pub async fn register_agent_with_deps(
        &self,
        agent_id: impl Into<String>,
        role: AgentRole,
        dependencies: Vec<String>,
    ) {
        let agent_id = agent_id.into();
        let mut state = self.state.write().await;

        let mut agent = AgentInfo::new(agent_id.clone(), role);
        agent.dependencies = dependencies.clone();

        // Add this agent as dependent to its dependencies
        for dep_id in &dependencies {
            if let Some(dep) = state.agents.get_mut(dep_id) {
                dep.dependents.push(agent_id.clone());
            }
        }

        state.agents.insert(agent_id.clone(), agent);
        self.add_timeline_entry(&mut state, &agent_id, "registered_with_deps", None);

        let _ = self
            .event_tx
            .send(CoordinationEvent::AgentRegistered { agent_id, role });
    }

    /// Update agent status
    pub async fn update_status(&self, agent_id: &str, new_status: AgentStatus) {
        let mut state = self.state.write().await;

        if let Some(agent) = state.agents.get_mut(agent_id) {
            let old_status = agent.status.clone();

            // Track started_at
            if !old_status.is_active() && new_status.is_active() {
                agent.started_at = Some(current_timestamp());
            }

            // Preserve progress
            if let AgentStatus::Working { progress } = &new_status {
                agent.last_progress = *progress;
            }

            agent.status = new_status.clone();

            self.add_timeline_entry(
                &mut state,
                agent_id,
                "status_changed",
                Some(format!("{:?}", new_status)),
            );

            let _ = self.event_tx.send(CoordinationEvent::StatusChanged {
                agent_id: agent_id.to_string(),
                old_status,
                new_status,
            });
        }
    }

    /// Mark agent as completed
    pub async fn mark_complete(&self, agent_id: &str, result_summary: Option<&str>) {
        let mut state = self.state.write().await;

        if let Some(agent) = state.agents.get_mut(agent_id) {
            let now = current_timestamp();
            agent.completed_at = Some(now);
            agent.duration_ms = agent.started_at.map(|s| now.saturating_sub(s));
            agent.status = AgentStatus::Completed {
                result_summary: result_summary.map(String::from),
            };
            agent.last_progress = 1.0;

            let duration_ms = agent.duration_ms.unwrap_or(0);
            let dependents = agent.dependents.clone();

            self.add_timeline_entry(
                &mut state,
                agent_id,
                "completed",
                result_summary.map(String::from),
            );

            // Notify dependents
            for dep_id in &dependents {
                let _ = self.event_tx.send(CoordinationEvent::DependencyResolved {
                    agent_id: dep_id.clone(),
                    dependency_id: agent_id.to_string(),
                });
            }

            let _ = self.event_tx.send(CoordinationEvent::AgentCompleted {
                agent_id: agent_id.to_string(),
                result_summary: result_summary.map(String::from),
                duration_ms,
            });

            // Check if all agents are complete
            self.check_all_complete(&mut state);
        }
    }

    /// Mark agent as failed
    pub async fn mark_failed(&self, agent_id: &str, error: &str) {
        let mut state = self.state.write().await;

        if let Some(agent) = state.agents.get_mut(agent_id) {
            let now = current_timestamp();
            agent.completed_at = Some(now);
            agent.duration_ms = agent.started_at.map(|s| now.saturating_sub(s));
            agent.status = AgentStatus::Failed {
                error: error.to_string(),
            };

            self.add_timeline_entry(&mut state, agent_id, "failed", Some(error.to_string()));

            let _ = self.event_tx.send(CoordinationEvent::AgentFailed {
                agent_id: agent_id.to_string(),
                error: error.to_string(),
            });

            self.check_all_complete(&mut state);
        }
    }

    /// Mark agent as cancelled
    pub async fn mark_cancelled(&self, agent_id: &str, reason: Option<&str>) {
        let mut state = self.state.write().await;

        if let Some(agent) = state.agents.get_mut(agent_id) {
            agent.status = AgentStatus::Cancelled {
                reason: reason.map(String::from),
            };

            self.add_timeline_entry(&mut state, agent_id, "cancelled", reason.map(String::from));
            self.check_all_complete(&mut state);
        }
    }

    /// Get current status of an agent
    pub async fn get_status(&self, agent_id: &str) -> Option<AgentStatus> {
        let state = self.state.read().await;
        state.agents.get(agent_id).map(|a| a.status.clone())
    }

    /// Get agent info
    pub async fn get_agent(&self, agent_id: &str) -> Option<AgentInfo> {
        let state = self.state.read().await;
        state.agents.get(agent_id).cloned()
    }

    /// Get all agents
    pub async fn get_all_agents(&self) -> Vec<AgentInfo> {
        let state = self.state.read().await;
        state.agents.values().cloned().collect()
    }

    /// Get coordination summary
    pub async fn summary(&self) -> CoordinationSummary {
        let state = self.state.read().await;

        let total_agents = state.agents.len();
        let mut completed_agents = 0;
        let mut failed_agents = 0;
        let mut active_agents = 0;
        let mut blocked_agents = 0;
        let mut total_progress = 0.0;

        for agent in state.agents.values() {
            match &agent.status {
                AgentStatus::Completed { .. } => {
                    completed_agents += 1;
                    total_progress += 1.0;
                }
                AgentStatus::Failed { .. } | AgentStatus::Cancelled { .. } => {
                    failed_agents += 1;
                }
                AgentStatus::Working { progress } => {
                    active_agents += 1;
                    total_progress += progress;
                }
                AgentStatus::Blocked { .. } => {
                    blocked_agents += 1;
                    total_progress += agent.last_progress;
                }
                AgentStatus::Starting => {
                    active_agents += 1;
                }
                _ => {}
            }
        }

        let overall_progress = if total_agents > 0 {
            total_progress / total_agents as f32
        } else {
            0.0
        };

        let duration_ms = state.started_at.elapsed().as_millis() as u64;

        // Estimate remaining time based on current progress
        let estimated_remaining_ms = if overall_progress > 0.0 && overall_progress < 1.0 {
            let elapsed = duration_ms as f64;
            let remaining_ratio = (1.0 - overall_progress as f64) / overall_progress as f64;
            Some((elapsed * remaining_ratio) as u64)
        } else {
            None
        };

        // Detect bottlenecks
        let bottlenecks = self.detect_bottlenecks(&state);

        let is_complete =
            state.completed_at.is_some() || state.agents.values().all(|a| a.status.is_terminal());

        let is_successful = is_complete && failed_agents == 0;

        CoordinationSummary {
            coordination_id: state.coordination_id.clone(),
            overall_progress,
            total_agents,
            completed_agents,
            failed_agents,
            active_agents,
            blocked_agents,
            duration_ms,
            estimated_remaining_ms,
            bottlenecks,
            is_complete,
            is_successful,
        }
    }

    /// Get timeline entries
    pub async fn timeline(&self) -> Vec<TimelineEntry> {
        let state = self.state.read().await;
        state.timeline.clone()
    }

    /// Check if all agents can start (dependencies satisfied)
    pub async fn check_dependencies(&self, agent_id: &str) -> bool {
        let state = self.state.read().await;

        if let Some(agent) = state.agents.get(agent_id) {
            for dep_id in &agent.dependencies {
                if let Some(dep) = state.agents.get(dep_id) {
                    if !matches!(dep.status, AgentStatus::Completed { .. }) {
                        return false;
                    }
                } else {
                    return false; // Dependency not registered
                }
            }
            true
        } else {
            false
        }
    }

    /// Get agents that are ready to start (dependencies satisfied)
    pub async fn get_ready_agents(&self) -> Vec<String> {
        let state = self.state.read().await;
        let mut ready = Vec::new();

        for (id, agent) in &state.agents {
            if !matches!(agent.status, AgentStatus::Pending) {
                continue;
            }

            let deps_satisfied = agent.dependencies.iter().all(|dep_id| {
                state
                    .agents
                    .get(dep_id)
                    .map(|d| matches!(d.status, AgentStatus::Completed { .. }))
                    .unwrap_or(false)
            });

            if deps_satisfied {
                ready.push(id.clone());
            }
        }

        ready
    }

    /// Cancel all pending/active agents
    pub async fn cancel_all(&self, reason: &str) {
        let mut state = self.state.write().await;

        let agent_ids: Vec<_> = state
            .agents
            .iter()
            .filter(|(_, a)| !a.status.is_terminal())
            .map(|(id, _)| id.clone())
            .collect();

        for id in agent_ids {
            if let Some(agent) = state.agents.get_mut(&id) {
                agent.status = AgentStatus::Cancelled {
                    reason: Some(reason.to_string()),
                };
                self.add_timeline_entry(&mut state, &id, "cancelled", Some(reason.to_string()));
            }
        }

        state.completed_at = Some(Instant::now());
    }

    // Internal helpers

    fn add_timeline_entry(
        &self,
        state: &mut TrackerState,
        agent_id: &str,
        event: &str,
        details: Option<String>,
    ) {
        let entry = TimelineEntry {
            timestamp_ms: state.started_at.elapsed().as_millis() as u64,
            agent_id: agent_id.to_string(),
            event: event.to_string(),
            details,
        };

        state.timeline.push(entry);

        // Trim timeline if too long
        if state.timeline.len() > self.config.max_timeline_entries {
            state.timeline.remove(0);
        }
    }

    fn detect_bottlenecks(&self, state: &TrackerState) -> Vec<String> {
        let mut bottlenecks = Vec::new();
        let threshold_ms = self.config.bottleneck_threshold_secs * 1000;

        for (id, agent) in &state.agents {
            if !agent.status.is_active() {
                continue;
            }

            if let Some(started) = agent.started_at {
                let now = current_timestamp();
                let duration = now.saturating_sub(started);

                if duration > threshold_ms && !agent.dependents.is_empty() {
                    bottlenecks.push(id.clone());
                }
            }
        }

        bottlenecks
    }

    fn check_all_complete(&self, state: &mut TrackerState) {
        let all_complete = state.agents.values().all(|a| a.status.is_terminal());

        if all_complete && state.completed_at.is_none() {
            state.completed_at = Some(Instant::now());

            let total_duration_ms = state.started_at.elapsed().as_millis() as u64;
            let success_count = state
                .agents
                .values()
                .filter(|a| matches!(a.status, AgentStatus::Completed { .. }))
                .count();
            let failure_count = state
                .agents
                .values()
                .filter(|a| matches!(a.status, AgentStatus::Failed { .. }))
                .count();

            let _ = self.event_tx.send(CoordinationEvent::AllCompleted {
                total_duration_ms,
                success_count,
                failure_count,
            });
        }
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Builder for creating coordinated agent workflows
pub struct CoordinationBuilder {
    coordination_id: String,
    config: CoordinationConfig,
    agents: Vec<(String, AgentRole, Vec<String>)>,
}

impl CoordinationBuilder {
    /// Create a new coordination builder
    pub fn new(coordination_id: impl Into<String>) -> Self {
        Self {
            coordination_id: coordination_id.into(),
            config: CoordinationConfig::default(),
            agents: Vec::new(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: CoordinationConfig) -> Self {
        self.config = config;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Add an agent
    pub fn add_agent(mut self, id: impl Into<String>, role: AgentRole) -> Self {
        self.agents.push((id.into(), role, Vec::new()));
        self
    }

    /// Add an agent with dependencies
    pub fn add_agent_with_deps(
        mut self,
        id: impl Into<String>,
        role: AgentRole,
        deps: Vec<String>,
    ) -> Self {
        self.agents.push((id.into(), role, deps));
        self
    }

    /// Build the coordination tracker with pre-registered agents
    pub async fn build(self) -> CoordinationTracker {
        let tracker = CoordinationTracker::with_config(self.coordination_id, self.config);

        for (id, role, deps) in self.agents {
            if deps.is_empty() {
                tracker.register_agent(id, role).await;
            } else {
                tracker.register_agent_with_deps(id, role, deps).await;
            }
        }

        tracker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_coordination() {
        let tracker = CoordinationTracker::new("test-coord-1");

        tracker.register_agent("agent1", AgentRole::Primary).await;
        tracker.register_agent("agent2", AgentRole::Support).await;

        let agents = tracker.get_all_agents().await;
        assert_eq!(agents.len(), 2);

        // Update status
        tracker
            .update_status("agent1", AgentStatus::Working { progress: 0.5 })
            .await;

        let status = tracker.get_status("agent1").await.unwrap();
        assert!(
            matches!(status, AgentStatus::Working { progress } if (progress - 0.5).abs() < 0.001)
        );
    }

    #[tokio::test]
    async fn test_completion_tracking() {
        let tracker = CoordinationTracker::new("test-coord-2");

        tracker.register_agent("agent1", AgentRole::Primary).await;
        tracker.update_status("agent1", AgentStatus::Starting).await;
        tracker
            .update_status("agent1", AgentStatus::Working { progress: 0.5 })
            .await;
        tracker.mark_complete("agent1", Some("Found results")).await;

        let agent = tracker.get_agent("agent1").await.unwrap();
        assert!(matches!(agent.status, AgentStatus::Completed { .. }));
        assert!(agent.completed_at.is_some());

        let summary = tracker.summary().await;
        assert_eq!(summary.completed_agents, 1);
        assert!(summary.is_complete);
        assert!(summary.is_successful);
    }

    #[tokio::test]
    async fn test_failure_tracking() {
        let tracker = CoordinationTracker::new("test-coord-3");

        tracker.register_agent("agent1", AgentRole::Primary).await;
        tracker.mark_failed("agent1", "Connection error").await;

        let summary = tracker.summary().await;
        assert_eq!(summary.failed_agents, 1);
        assert!(summary.is_complete);
        assert!(!summary.is_successful);
    }

    #[tokio::test]
    async fn test_dependencies() {
        let tracker = CoordinationTracker::new("test-coord-4");

        tracker.register_agent("searcher", AgentRole::Primary).await;
        tracker
            .register_agent_with_deps(
                "validator",
                AgentRole::Validator,
                vec!["searcher".to_string()],
            )
            .await;

        // Validator should not be ready yet
        let ready = tracker.get_ready_agents().await;
        assert!(ready.contains(&"searcher".to_string()));
        assert!(!ready.contains(&"validator".to_string()));

        // Complete searcher
        tracker.mark_complete("searcher", None).await;

        // Now validator should be ready
        let ready = tracker.get_ready_agents().await;
        assert!(ready.contains(&"validator".to_string()));
    }

    #[tokio::test]
    async fn test_summary() {
        let tracker = CoordinationTracker::new("test-coord-5");

        tracker.register_agent("a1", AgentRole::Primary).await;
        tracker.register_agent("a2", AgentRole::Support).await;
        tracker.register_agent("a3", AgentRole::Validator).await;

        tracker
            .update_status("a1", AgentStatus::Working { progress: 0.5 })
            .await;
        tracker.mark_complete("a2", None).await;

        let summary = tracker.summary().await;
        assert_eq!(summary.total_agents, 3);
        assert_eq!(summary.completed_agents, 1);
        assert_eq!(summary.active_agents, 1);
        assert!(!summary.is_complete);

        // Progress should be (0.5 + 1.0 + 0.0) / 3 = 0.5
        assert!((summary.overall_progress - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_timeline() {
        let tracker = CoordinationTracker::new("test-coord-6");

        tracker.register_agent("agent1", AgentRole::Primary).await;
        tracker.update_status("agent1", AgentStatus::Starting).await;
        tracker.mark_complete("agent1", Some("Done")).await;

        let timeline = tracker.timeline().await;
        assert!(!timeline.is_empty());

        let events: Vec<_> = timeline.iter().map(|e| e.event.as_str()).collect();
        assert!(events.contains(&"registered"));
        assert!(events.contains(&"status_changed"));
        assert!(events.contains(&"completed"));
    }

    #[tokio::test]
    async fn test_cancel_all() {
        let tracker = CoordinationTracker::new("test-coord-7");

        tracker.register_agent("a1", AgentRole::Primary).await;
        tracker.register_agent("a2", AgentRole::Support).await;

        tracker
            .update_status("a1", AgentStatus::Working { progress: 0.3 })
            .await;

        tracker.cancel_all("Timeout").await;

        let a1 = tracker.get_agent("a1").await.unwrap();
        let a2 = tracker.get_agent("a2").await.unwrap();

        assert!(matches!(a1.status, AgentStatus::Cancelled { .. }));
        assert!(matches!(a2.status, AgentStatus::Cancelled { .. }));
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let tracker = CoordinationTracker::new("test-coord-8");
        let mut rx = tracker.subscribe();

        tracker.register_agent("agent1", AgentRole::Primary).await;

        // Should receive the registration event
        let event = rx.try_recv().unwrap();
        assert!(matches!(event, CoordinationEvent::AgentRegistered { .. }));
    }

    #[tokio::test]
    async fn test_builder() {
        let tracker = CoordinationBuilder::new("test-coord-9")
            .with_timeout(Duration::from_secs(60))
            .add_agent("searcher", AgentRole::Primary)
            .add_agent_with_deps(
                "validator",
                AgentRole::Validator,
                vec!["searcher".to_string()],
            )
            .build()
            .await;

        let agents = tracker.get_all_agents().await;
        assert_eq!(agents.len(), 2);

        let validator = tracker.get_agent("validator").await.unwrap();
        assert_eq!(validator.dependencies, vec!["searcher".to_string()]);
    }
}
