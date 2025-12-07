//! Integration with the agents crate for real data.
//!
//! This module provides bridging functionality to connect
//! the dashboard to actual agent sessions, trajectories, and factories.
//!
//! # Feature Flag
//!
//! This module requires the `agents` feature:
//!
//! ```toml
//! rust-ai-agents-dashboard = { version = "0.1", features = ["agents"] }
//! ```

use chrono::{TimeZone, Utc};
use rust_ai_agents_agents::{
    factory::AgentFactory,
    session::{ConversationSession, SessionError, SessionStore},
    trajectory::{StepType, Trajectory, TrajectoryStore},
};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::state::{
    AgentStatus, DashboardState, Session, SessionMessage, SessionStatus, TraceEntry, TraceEntryType,
};

/// Convert unix timestamp (millis) to DateTime<Utc>
fn millis_to_datetime(millis: u64) -> chrono::DateTime<Utc> {
    Utc.timestamp_millis_opt(millis as i64)
        .single()
        .unwrap_or_else(Utc::now)
}

/// Convert i64 timestamp to DateTime<Utc>
fn timestamp_to_datetime(ts: i64) -> chrono::DateTime<Utc> {
    Utc.timestamp_millis_opt(ts)
        .single()
        .unwrap_or_else(Utc::now)
}

/// Bridge to sync agent factory data to dashboard state.
///
/// Call `sync_agents` periodically or when agents change to keep
/// the dashboard up-to-date with registered agents.
pub struct AgentBridge {
    factory: Arc<RwLock<AgentFactory>>,
    dashboard: Arc<DashboardState>,
}

impl AgentBridge {
    /// Create a new agent bridge.
    pub fn new(factory: Arc<RwLock<AgentFactory>>, dashboard: Arc<DashboardState>) -> Self {
        Self { factory, dashboard }
    }

    /// Sync all agents from factory to dashboard.
    pub async fn sync_agents(&self) {
        let factory = self.factory.read().await;
        let template_names = factory.list_templates();

        for name in template_names {
            if let Some(template) = factory.get_template(&name) {
                let status = AgentStatus {
                    id: template.name.clone(),
                    name: template.name.clone(),
                    role: template.description.clone(),
                    status: "registered".to_string(),
                    messages_processed: 0,
                    last_activity: None,
                    current_task: None,
                };
                self.dashboard.update_agent(status);
            }
        }
    }

    /// Register a new agent and sync to dashboard.
    pub async fn register_agent(&self, id: &str, name: &str, role: &str) {
        let status = AgentStatus {
            id: id.to_string(),
            name: name.to_string(),
            role: role.to_string(),
            status: "registered".to_string(),
            messages_processed: 0,
            last_activity: None,
            current_task: None,
        };
        self.dashboard.update_agent(status);
    }

    /// Update agent status (running, stopped, etc.).
    pub fn update_agent_status(&self, id: &str, status: &str, current_task: Option<&str>) {
        if let Some(mut agent) = self.dashboard.get_agent(id) {
            agent.status = status.to_string();
            agent.current_task = current_task.map(|s| s.to_string());
            agent.last_activity = Some(Utc::now());
            self.dashboard.update_agent(agent);
        }
    }

    /// Increment message count for an agent.
    pub fn record_agent_message(&self, id: &str) {
        if let Some(mut agent) = self.dashboard.get_agent(id) {
            agent.messages_processed += 1;
            agent.last_activity = Some(Utc::now());
            self.dashboard.update_agent(agent);
        }
    }
}

/// Bridge to sync session store data to dashboard state.
pub struct SessionBridge<S: SessionStore> {
    store: Arc<S>,
    dashboard: Arc<DashboardState>,
}

impl<S: SessionStore> SessionBridge<S> {
    /// Create a new session bridge.
    pub fn new(store: Arc<S>, dashboard: Arc<DashboardState>) -> Self {
        Self { store, dashboard }
    }

    /// Sync all sessions from store to dashboard.
    pub async fn sync_sessions(&self) -> Result<(), SessionError> {
        let session_ids = self.store.list().await?;

        for id in session_ids {
            if let Some(session) = self.store.load(&id).await? {
                self.sync_session(&session);
            }
        }

        Ok(())
    }

    /// Sync a single session to dashboard.
    pub fn sync_session(&self, session: &ConversationSession) {
        let dashboard_session = Session {
            id: session.id.clone(),
            name: session.state.get("name").cloned(),
            created_at: millis_to_datetime(session.created_at),
            updated_at: millis_to_datetime(session.updated_at),
            message_count: session.messages.len() as u32,
            status: if session.state.get("completed").is_some() {
                SessionStatus::Completed
            } else if session.state.get("failed").is_some() {
                SessionStatus::Failed
            } else {
                SessionStatus::Active
            },
            agent_id: session.state.get("agent_id").cloned(),
            metadata: None,
        };

        self.dashboard.update_session(dashboard_session);

        // Sync messages
        for (i, msg) in session.messages.iter().enumerate() {
            let session_msg = SessionMessage {
                id: format!("{}-{}", session.id, i),
                session_id: session.id.clone(),
                role: msg.role.as_str().to_string(),
                content: msg.content.clone(),
                timestamp: millis_to_datetime(msg.timestamp),
                metadata: msg.tool_call_id.as_ref().map(|tc| {
                    serde_json::json!({
                        "tool_call_id": tc
                    })
                }),
            };
            self.dashboard.add_session_message(session_msg);
        }
    }

    /// Called when a new message is added to a session.
    pub fn on_message_added(&self, session: &ConversationSession) {
        self.sync_session(session);
        self.dashboard.record_message();
    }
}

/// Bridge to sync trajectory store data to dashboard state.
///
/// Uses the concrete `TrajectoryStore` type from the agents crate.
pub struct TrajectoryBridge {
    store: Arc<TrajectoryStore>,
    dashboard: Arc<DashboardState>,
}

impl TrajectoryBridge {
    /// Create a new trajectory bridge.
    pub fn new(store: Arc<TrajectoryStore>, dashboard: Arc<DashboardState>) -> Self {
        Self { store, dashboard }
    }

    /// Sync all trajectories from store to dashboard as traces.
    pub fn sync_trajectories(&self) {
        let trajectories = self.store.all();

        for traj in trajectories {
            self.sync_trajectory(&traj);
        }
    }

    /// Sync a single trajectory to dashboard as trace entries.
    pub fn sync_trajectory(&self, trajectory: &Trajectory) {
        for (i, step) in trajectory.steps.iter().enumerate() {
            let entry_type = match &step.step_type {
                StepType::LlmCall => {
                    // Extract details from step.details JSON
                    let model = step
                        .details
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let prompt_tokens = step
                        .details
                        .get("prompt_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let completion_tokens = step
                        .details
                        .get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let cost = step
                        .details
                        .get("cost")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);

                    TraceEntryType::LlmRequest {
                        model,
                        prompt_tokens,
                        completion_tokens,
                        cost,
                    }
                }
                StepType::ToolCall => {
                    let tool_name = step
                        .details
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let arguments = step
                        .details
                        .get("arguments")
                        .cloned()
                        .unwrap_or(serde_json::json!({}));

                    TraceEntryType::ToolCall {
                        tool_name,
                        arguments,
                    }
                }
                StepType::ToolResult => {
                    let tool_name = step
                        .details
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let result = step
                        .details
                        .get("result")
                        .cloned()
                        .unwrap_or(serde_json::json!({}));

                    TraceEntryType::ToolResult {
                        tool_name,
                        result,
                        success: step.success,
                    }
                }
                StepType::TaskFailed => TraceEntryType::Error {
                    message: step.error.clone().unwrap_or_default(),
                    error_type: "TaskFailed".to_string(),
                },
                StepType::Planning => {
                    let thought = step
                        .details
                        .get("plan")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    TraceEntryType::AgentThought { thought }
                }
                _ => {
                    // For other step types, create a generic thought entry
                    let thought = format!(
                        "{:?}: {}",
                        step.step_type,
                        step.details
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                    );
                    TraceEntryType::AgentThought { thought }
                }
            };

            let trace = TraceEntry {
                id: format!("{}-{}", trajectory.task_id, i),
                session_id: trajectory
                    .metadata
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&trajectory.task_id)
                    .to_string(),
                timestamp: timestamp_to_datetime(step.timestamp),
                entry_type,
                duration_ms: Some(step.duration_ms as f64),
                metadata: Some(step.details.clone()),
            };

            self.dashboard.add_trace(trace);
        }
    }

    /// Called when a new step is added to a trajectory.
    pub fn on_step_added(&self, trajectory: &Trajectory) {
        self.sync_trajectory(trajectory);
    }
}

/// Combined bridge for all agent data sources.
pub struct DashboardBridge<S: SessionStore> {
    pub agents: AgentBridge,
    pub sessions: SessionBridge<S>,
    pub trajectories: TrajectoryBridge,
}

impl<S: SessionStore> DashboardBridge<S> {
    /// Create a new combined bridge.
    pub fn new(
        factory: Arc<RwLock<AgentFactory>>,
        session_store: Arc<S>,
        trajectory_store: Arc<TrajectoryStore>,
        dashboard: Arc<DashboardState>,
    ) -> Self {
        Self {
            agents: AgentBridge::new(factory, dashboard.clone()),
            sessions: SessionBridge::new(session_store, dashboard.clone()),
            trajectories: TrajectoryBridge::new(trajectory_store, dashboard),
        }
    }

    /// Sync all data sources to dashboard.
    pub async fn sync_all(&self) -> Result<(), SessionError> {
        self.agents.sync_agents().await;
        self.sessions.sync_sessions().await?;
        self.trajectories.sync_trajectories();
        Ok(())
    }
}

/// Add demo data to dashboard for development/testing.
pub fn add_demo_data(dashboard: &DashboardState) {
    // Add demo agents
    let agents = vec![
        AgentStatus {
            id: "agent-1".to_string(),
            name: "Research Agent".to_string(),
            role: "researcher".to_string(),
            status: "running".to_string(),
            messages_processed: 42,
            last_activity: Some(Utc::now()),
            current_task: Some("Analyzing market trends".to_string()),
        },
        AgentStatus {
            id: "agent-2".to_string(),
            name: "Code Agent".to_string(),
            role: "developer".to_string(),
            status: "idle".to_string(),
            messages_processed: 156,
            last_activity: Some(Utc::now() - chrono::Duration::minutes(5)),
            current_task: None,
        },
        AgentStatus {
            id: "agent-3".to_string(),
            name: "Data Agent".to_string(),
            role: "analyst".to_string(),
            status: "stopped".to_string(),
            messages_processed: 23,
            last_activity: Some(Utc::now() - chrono::Duration::hours(1)),
            current_task: None,
        },
    ];

    for agent in agents {
        dashboard.update_agent(agent);
    }

    // Add demo sessions
    let sessions = vec![
        Session {
            id: "session-abc123".to_string(),
            name: Some("Market Research Chat".to_string()),
            created_at: Utc::now() - chrono::Duration::hours(2),
            updated_at: Utc::now() - chrono::Duration::minutes(10),
            message_count: 12,
            status: SessionStatus::Active,
            agent_id: Some("agent-1".to_string()),
            metadata: None,
        },
        Session {
            id: "session-def456".to_string(),
            name: Some("Code Review Session".to_string()),
            created_at: Utc::now() - chrono::Duration::days(1),
            updated_at: Utc::now() - chrono::Duration::hours(3),
            message_count: 28,
            status: SessionStatus::Completed,
            agent_id: Some("agent-2".to_string()),
            metadata: None,
        },
        Session {
            id: "session-ghi789".to_string(),
            name: None,
            created_at: Utc::now() - chrono::Duration::hours(5),
            updated_at: Utc::now() - chrono::Duration::hours(4),
            message_count: 5,
            status: SessionStatus::Failed,
            agent_id: Some("agent-3".to_string()),
            metadata: None,
        },
    ];

    for session in sessions {
        dashboard.update_session(session);
    }

    // Add demo messages
    let messages = vec![
        SessionMessage {
            id: "msg-1".to_string(),
            session_id: "session-abc123".to_string(),
            role: "user".to_string(),
            content: "What are the current market trends for AI?".to_string(),
            timestamp: Utc::now() - chrono::Duration::hours(2),
            metadata: None,
        },
        SessionMessage {
            id: "msg-2".to_string(),
            session_id: "session-abc123".to_string(),
            role: "assistant".to_string(),
            content: "Based on my analysis, here are the key AI market trends:\n\n1. **Generative AI Growth**: The market is expected to reach $1.3 trillion by 2032.\n2. **Enterprise Adoption**: 65% of enterprises are now using AI in production.\n3. **Edge AI**: Growing demand for on-device AI processing.".to_string(),
            timestamp: Utc::now() - chrono::Duration::hours(2) + chrono::Duration::seconds(30),
            metadata: None,
        },
        SessionMessage {
            id: "msg-3".to_string(),
            session_id: "session-abc123".to_string(),
            role: "user".to_string(),
            content: "Can you provide more details on generative AI?".to_string(),
            timestamp: Utc::now() - chrono::Duration::minutes(30),
            metadata: None,
        },
    ];

    for msg in messages {
        dashboard.add_session_message(msg);
    }

    // Add demo traces
    let traces = vec![
        TraceEntry {
            id: "trace-1".to_string(),
            session_id: "session-abc123".to_string(),
            timestamp: Utc::now() - chrono::Duration::minutes(5),
            entry_type: TraceEntryType::LlmRequest {
                model: "claude-3-5-sonnet".to_string(),
                prompt_tokens: 1250,
                completion_tokens: 450,
                cost: 0.0065,
            },
            duration_ms: Some(1234.5),
            metadata: None,
        },
        TraceEntry {
            id: "trace-2".to_string(),
            session_id: "session-abc123".to_string(),
            timestamp: Utc::now() - chrono::Duration::minutes(4),
            entry_type: TraceEntryType::ToolCall {
                tool_name: "web_search".to_string(),
                arguments: serde_json::json!({
                    "query": "AI market trends 2024"
                }),
            },
            duration_ms: Some(856.2),
            metadata: None,
        },
        TraceEntry {
            id: "trace-3".to_string(),
            session_id: "session-abc123".to_string(),
            timestamp: Utc::now() - chrono::Duration::minutes(3),
            entry_type: TraceEntryType::ToolResult {
                tool_name: "web_search".to_string(),
                result: serde_json::json!({
                    "results": ["Market analysis report", "Industry forecast"]
                }),
                success: true,
            },
            duration_ms: None,
            metadata: None,
        },
        TraceEntry {
            id: "trace-4".to_string(),
            session_id: "session-abc123".to_string(),
            timestamp: Utc::now() - chrono::Duration::minutes(2),
            entry_type: TraceEntryType::AgentThought {
                thought: "I should synthesize the search results with my knowledge to provide a comprehensive answer.".to_string(),
            },
            duration_ms: None,
            metadata: None,
        },
        TraceEntry {
            id: "trace-5".to_string(),
            session_id: "session-def456".to_string(),
            timestamp: Utc::now() - chrono::Duration::hours(3),
            entry_type: TraceEntryType::Error {
                message: "Rate limit exceeded".to_string(),
                error_type: "RateLimitError".to_string(),
            },
            duration_ms: None,
            metadata: None,
        },
    ];

    for trace in traces {
        dashboard.add_trace(trace);
    }
}
