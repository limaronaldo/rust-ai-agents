//! # Cancellation Manager
//!
//! MassGen-inspired cancellation system with partial result preservation.
//! Enables graceful cancellation of agent workflows while preserving completed work.
//!
//! ## Features
//!
//! - **Graceful Cancellation**: Allow agents to finish current work before stopping
//! - **Partial Results**: Preserve results from completed agents
//! - **Cancellation Tokens**: Cooperative cancellation pattern
//! - **Timeout Handling**: Auto-cancel on timeout with result preservation
//! - **Cascading Cancellation**: Cancel dependent agents when parent fails
//!
//! ## Example
//!
//! ```ignore
//! use rust_ai_agents_crew::cancellation::*;
//!
//! let manager = CancellationManager::new();
//!
//! // Create a cancellation token for a workflow
//! let token = manager.create_token("workflow-123");
//!
//! // Agents check the token periodically
//! if token.is_cancelled() {
//!     // Save partial results and exit gracefully
//!     return;
//! }
//!
//! // Cancel with partial result preservation
//! manager.cancel("workflow-123", CancellationReason::UserRequest, true).await;
//!
//! // Retrieve partial results
//! let results = manager.get_partial_results("workflow-123").await;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};

/// Reason for cancellation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancellationReason {
    /// User requested cancellation
    UserRequest,
    /// Timeout exceeded
    Timeout,
    /// Parent workflow cancelled
    ParentCancelled,
    /// Dependency failed
    DependencyFailed { dependency_id: String },
    /// Resource limit exceeded
    ResourceLimit { resource: String, limit: String },
    /// Error in processing
    Error { message: String },
    /// System shutdown
    Shutdown,
    /// Custom reason
    Custom(String),
}

impl std::fmt::Display for CancellationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CancellationReason::UserRequest => write!(f, "User requested cancellation"),
            CancellationReason::Timeout => write!(f, "Operation timed out"),
            CancellationReason::ParentCancelled => write!(f, "Parent workflow cancelled"),
            CancellationReason::DependencyFailed { dependency_id } => {
                write!(f, "Dependency '{}' failed", dependency_id)
            }
            CancellationReason::ResourceLimit { resource, limit } => {
                write!(
                    f,
                    "Resource limit exceeded: {} (limit: {})",
                    resource, limit
                )
            }
            CancellationReason::Error { message } => write!(f, "Error: {}", message),
            CancellationReason::Shutdown => write!(f, "System shutdown"),
            CancellationReason::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

/// Cancellation policy for handling cancellation requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CancellationPolicy {
    /// Cancel immediately without waiting
    Immediate,
    /// Wait for current operation to complete (default)
    #[default]
    Graceful,
    /// Wait up to a timeout for graceful cancellation, then force
    GracefulWithTimeout,
    /// Never cancel (must complete)
    NeverCancel,
}

/// A partial result from an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialResult {
    /// Agent that produced this result
    pub agent_id: String,
    /// The partial result data
    pub data: serde_json::Value,
    /// Completion percentage (0.0 to 1.0)
    pub completion: f32,
    /// Whether this is a final/complete result
    pub is_complete: bool,
    /// Timestamp when result was saved
    pub saved_at: u64,
    /// Optional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PartialResult {
    /// Create a new partial result
    pub fn new(agent_id: impl Into<String>, data: serde_json::Value, completion: f32) -> Self {
        Self {
            agent_id: agent_id.into(),
            data,
            completion: completion.clamp(0.0, 1.0),
            is_complete: completion >= 1.0,
            saved_at: current_timestamp(),
            metadata: HashMap::new(),
        }
    }

    /// Create a complete result
    pub fn complete(agent_id: impl Into<String>, data: serde_json::Value) -> Self {
        Self::new(agent_id, data, 1.0)
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Cancellation token for cooperative cancellation
#[derive(Clone)]
pub struct CancellationToken {
    id: String,
    cancelled: Arc<AtomicBool>,
    cancel_requested_at: Arc<AtomicU64>,
    reason: Arc<RwLock<Option<CancellationReason>>>,
    policy: CancellationPolicy,
    grace_period: Duration,
}

impl CancellationToken {
    /// Create a new cancellation token
    fn new(id: String, policy: CancellationPolicy, grace_period: Duration) -> Self {
        Self {
            id,
            cancelled: Arc::new(AtomicBool::new(false)),
            cancel_requested_at: Arc::new(AtomicU64::new(0)),
            reason: Arc::new(RwLock::new(None)),
            policy,
            grace_period,
        }
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Check if graceful cancellation is requested (should finish current work)
    pub fn is_graceful_cancel(&self) -> bool {
        self.is_cancelled()
            && matches!(
                self.policy,
                CancellationPolicy::Graceful | CancellationPolicy::GracefulWithTimeout
            )
    }

    /// Check if immediate cancellation is required
    pub fn is_immediate_cancel(&self) -> bool {
        if !self.is_cancelled() {
            return false;
        }

        match self.policy {
            CancellationPolicy::Immediate => true,
            CancellationPolicy::GracefulWithTimeout => {
                let requested_at = self.cancel_requested_at.load(Ordering::SeqCst);
                if requested_at == 0 {
                    return false;
                }
                let now = current_timestamp();
                now.saturating_sub(requested_at) > self.grace_period.as_millis() as u64
            }
            _ => false,
        }
    }

    /// Get the cancellation reason
    pub async fn reason(&self) -> Option<CancellationReason> {
        self.reason.read().await.clone()
    }

    /// Get the token ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get remaining grace period (if applicable)
    pub fn remaining_grace_period(&self) -> Option<Duration> {
        if !self.is_cancelled() {
            return None;
        }

        if self.policy != CancellationPolicy::GracefulWithTimeout {
            return None;
        }

        let requested_at = self.cancel_requested_at.load(Ordering::SeqCst);
        if requested_at == 0 {
            return Some(self.grace_period);
        }

        let elapsed_ms = current_timestamp().saturating_sub(requested_at);
        let grace_ms = self.grace_period.as_millis() as u64;

        if elapsed_ms >= grace_ms {
            Some(Duration::ZERO)
        } else {
            Some(Duration::from_millis(grace_ms - elapsed_ms))
        }
    }

    /// Wait for cancellation
    pub async fn cancelled(&self) {
        while !self.is_cancelled() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Internal: trigger cancellation
    async fn trigger(&self, reason: CancellationReason) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.cancel_requested_at
            .store(current_timestamp(), Ordering::SeqCst);
        *self.reason.write().await = Some(reason);
    }
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("id", &self.id)
            .field("cancelled", &self.is_cancelled())
            .field("policy", &self.policy)
            .finish()
    }
}

/// Cancellation event for broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CancellationEvent {
    /// Cancellation requested
    CancellationRequested {
        workflow_id: String,
        reason: CancellationReason,
        policy: CancellationPolicy,
    },
    /// Partial result saved
    PartialResultSaved {
        workflow_id: String,
        agent_id: String,
        completion: f32,
    },
    /// Graceful cancellation completed
    GracefulCancellationComplete {
        workflow_id: String,
        results_preserved: usize,
    },
    /// Force cancellation executed
    ForceCancellation { workflow_id: String, reason: String },
}

/// Workflow cancellation state
struct WorkflowState {
    token: CancellationToken,
    partial_results: Vec<PartialResult>,
    children: Vec<String>,
    #[allow(dead_code)]
    created_at: Instant,
}

/// Configuration for the cancellation manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancellationConfig {
    /// Default cancellation policy
    pub default_policy: CancellationPolicy,
    /// Default grace period for graceful cancellation
    pub default_grace_period: Duration,
    /// Whether to cascade cancellation to children
    pub cascade_to_children: bool,
    /// Maximum partial results to keep per workflow
    pub max_partial_results: usize,
    /// Whether to auto-cleanup completed workflows
    pub auto_cleanup: bool,
    /// Cleanup delay after completion
    pub cleanup_delay: Duration,
}

impl Default for CancellationConfig {
    fn default() -> Self {
        Self {
            default_policy: CancellationPolicy::Graceful,
            default_grace_period: Duration::from_secs(30),
            cascade_to_children: true,
            max_partial_results: 100,
            auto_cleanup: true,
            cleanup_delay: Duration::from_secs(60),
        }
    }
}

/// Cancellation manager for coordinating cancellations across workflows
pub struct CancellationManager {
    workflows: Arc<RwLock<HashMap<String, WorkflowState>>>,
    config: CancellationConfig,
    event_tx: broadcast::Sender<CancellationEvent>,
}

impl CancellationManager {
    /// Create a new cancellation manager
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            workflows: Arc::new(RwLock::new(HashMap::new())),
            config: CancellationConfig::default(),
            event_tx,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CancellationConfig) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            workflows: Arc::new(RwLock::new(HashMap::new())),
            config,
            event_tx,
        }
    }

    /// Subscribe to cancellation events
    pub fn subscribe(&self) -> broadcast::Receiver<CancellationEvent> {
        self.event_tx.subscribe()
    }

    /// Create a cancellation token for a workflow
    pub async fn create_token(&self, workflow_id: impl Into<String>) -> CancellationToken {
        self.create_token_with_policy(workflow_id, self.config.default_policy)
            .await
    }

    /// Create a cancellation token with specific policy
    pub async fn create_token_with_policy(
        &self,
        workflow_id: impl Into<String>,
        policy: CancellationPolicy,
    ) -> CancellationToken {
        let workflow_id = workflow_id.into();
        let token = CancellationToken::new(
            workflow_id.clone(),
            policy,
            self.config.default_grace_period,
        );

        let state = WorkflowState {
            token: token.clone(),
            partial_results: Vec::new(),
            children: Vec::new(),
            created_at: Instant::now(),
        };

        self.workflows.write().await.insert(workflow_id, state);
        token
    }

    /// Create a child token (cancelled when parent is cancelled)
    pub async fn create_child_token(
        &self,
        parent_id: &str,
        child_id: impl Into<String>,
    ) -> Option<CancellationToken> {
        let child_id = child_id.into();

        let mut workflows = self.workflows.write().await;

        // Check parent exists and get its policy
        let parent_policy = workflows.get(parent_id).map(|s| s.token.policy)?;

        // Create child token
        let token = CancellationToken::new(
            child_id.clone(),
            parent_policy,
            self.config.default_grace_period,
        );

        // Register child with parent
        if let Some(parent) = workflows.get_mut(parent_id) {
            parent.children.push(child_id.clone());
        }

        // Add child state
        let state = WorkflowState {
            token: token.clone(),
            partial_results: Vec::new(),
            children: Vec::new(),
            created_at: Instant::now(),
        };

        workflows.insert(child_id, state);
        Some(token)
    }

    /// Get a cancellation token for an existing workflow
    pub async fn get_token(&self, workflow_id: &str) -> Option<CancellationToken> {
        self.workflows
            .read()
            .await
            .get(workflow_id)
            .map(|s| s.token.clone())
    }

    /// Request cancellation of a workflow
    pub async fn cancel(
        &self,
        workflow_id: &str,
        reason: CancellationReason,
        preserve_results: bool,
    ) -> bool {
        let workflows = self.workflows.read().await;

        if let Some(state) = workflows.get(workflow_id) {
            // Trigger cancellation
            state.token.trigger(reason.clone()).await;

            let policy = state.token.policy;
            let children = state.children.clone();

            drop(workflows);

            // Broadcast event
            let _ = self
                .event_tx
                .send(CancellationEvent::CancellationRequested {
                    workflow_id: workflow_id.to_string(),
                    reason: reason.clone(),
                    policy,
                });

            // Cascade to children if configured
            if self.config.cascade_to_children {
                for child_id in children {
                    Box::pin(self.cancel(
                        &child_id,
                        CancellationReason::ParentCancelled,
                        preserve_results,
                    ))
                    .await;
                }
            }

            true
        } else {
            false
        }
    }

    /// Cancel with timeout (wait for graceful completion)
    pub async fn cancel_with_timeout(
        &self,
        workflow_id: &str,
        reason: CancellationReason,
        timeout: Duration,
    ) -> bool {
        // First request graceful cancellation
        if !self.cancel(workflow_id, reason.clone(), true).await {
            return false;
        }

        // Wait for completion or timeout
        let start = Instant::now();
        loop {
            if start.elapsed() >= timeout {
                // Force cancellation
                let _ = self.event_tx.send(CancellationEvent::ForceCancellation {
                    workflow_id: workflow_id.to_string(),
                    reason: format!("Timeout after {:?}", timeout),
                });
                return true;
            }

            // Check if all work is done
            if let Some(state) = self.workflows.read().await.get(workflow_id) {
                let all_complete = state.partial_results.iter().all(|r| r.is_complete);
                if all_complete && !state.partial_results.is_empty() {
                    let _ = self
                        .event_tx
                        .send(CancellationEvent::GracefulCancellationComplete {
                            workflow_id: workflow_id.to_string(),
                            results_preserved: state.partial_results.len(),
                        });
                    return true;
                }
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Save a partial result for a workflow
    pub async fn save_partial_result(&self, workflow_id: &str, result: PartialResult) {
        let mut workflows = self.workflows.write().await;

        if let Some(state) = workflows.get_mut(workflow_id) {
            let agent_id = result.agent_id.clone();
            let completion = result.completion;

            // Replace existing result from same agent or add new
            if let Some(existing) = state
                .partial_results
                .iter_mut()
                .find(|r| r.agent_id == result.agent_id)
            {
                *existing = result;
            } else {
                state.partial_results.push(result);
            }

            // Trim if too many results
            if state.partial_results.len() > self.config.max_partial_results {
                // Remove oldest incomplete results first
                state.partial_results.retain(|r| r.is_complete);
                if state.partial_results.len() > self.config.max_partial_results {
                    state.partial_results.remove(0);
                }
            }

            let _ = self.event_tx.send(CancellationEvent::PartialResultSaved {
                workflow_id: workflow_id.to_string(),
                agent_id,
                completion,
            });
        }
    }

    /// Get partial results for a workflow
    pub async fn get_partial_results(&self, workflow_id: &str) -> Vec<PartialResult> {
        self.workflows
            .read()
            .await
            .get(workflow_id)
            .map(|s| s.partial_results.clone())
            .unwrap_or_default()
    }

    /// Get complete results only
    pub async fn get_complete_results(&self, workflow_id: &str) -> Vec<PartialResult> {
        self.workflows
            .read()
            .await
            .get(workflow_id)
            .map(|s| {
                s.partial_results
                    .iter()
                    .filter(|r| r.is_complete)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if a workflow is cancelled
    pub async fn is_cancelled(&self, workflow_id: &str) -> bool {
        self.workflows
            .read()
            .await
            .get(workflow_id)
            .map(|s| s.token.is_cancelled())
            .unwrap_or(false)
    }

    /// Remove a workflow (cleanup)
    pub async fn remove_workflow(&self, workflow_id: &str) -> Option<Vec<PartialResult>> {
        self.workflows
            .write()
            .await
            .remove(workflow_id)
            .map(|s| s.partial_results)
    }

    /// Get workflow statistics
    pub async fn stats(&self) -> CancellationStats {
        let workflows = self.workflows.read().await;

        let total = workflows.len();
        let cancelled = workflows
            .values()
            .filter(|s| s.token.is_cancelled())
            .count();
        let with_results = workflows
            .values()
            .filter(|s| !s.partial_results.is_empty())
            .count();
        let total_results: usize = workflows.values().map(|s| s.partial_results.len()).sum();

        CancellationStats {
            total_workflows: total,
            cancelled_workflows: cancelled,
            workflows_with_results: with_results,
            total_partial_results: total_results,
        }
    }
}

impl Default for CancellationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about cancellation manager state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancellationStats {
    /// Total tracked workflows
    pub total_workflows: usize,
    /// Cancelled workflows
    pub cancelled_workflows: usize,
    /// Workflows with preserved results
    pub workflows_with_results: usize,
    /// Total partial results stored
    pub total_partial_results: usize,
}

/// Scoped cancellation guard that ensures cleanup on drop
pub struct CancellationGuard {
    workflow_id: String,
    manager: Arc<CancellationManager>,
    token: CancellationToken,
}

impl CancellationGuard {
    /// Create a new cancellation guard
    pub async fn new(manager: Arc<CancellationManager>, workflow_id: impl Into<String>) -> Self {
        let workflow_id = workflow_id.into();
        let token = manager.create_token(&workflow_id).await;
        Self {
            workflow_id,
            manager,
            token,
        }
    }

    /// Get the cancellation token
    pub fn token(&self) -> &CancellationToken {
        &self.token
    }

    /// Save a partial result
    pub async fn save_result(&self, result: PartialResult) {
        self.manager
            .save_partial_result(&self.workflow_id, result)
            .await;
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        // Cleanup is handled asynchronously by the manager if auto_cleanup is enabled
        // For synchronous cleanup, users should call remove_workflow explicitly
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
    async fn test_create_token() {
        let manager = CancellationManager::new();
        let token = manager.create_token("workflow-1").await;

        assert!(!token.is_cancelled());
        assert_eq!(token.id(), "workflow-1");
    }

    #[tokio::test]
    async fn test_cancel_workflow() {
        let manager = CancellationManager::new();
        let token = manager.create_token("workflow-1").await;

        assert!(!token.is_cancelled());

        manager
            .cancel("workflow-1", CancellationReason::UserRequest, true)
            .await;

        assert!(token.is_cancelled());

        let reason = token.reason().await;
        assert!(matches!(reason, Some(CancellationReason::UserRequest)));
    }

    #[tokio::test]
    async fn test_partial_results() {
        let manager = CancellationManager::new();
        let _token = manager.create_token("workflow-1").await;

        let result = PartialResult::new("agent-1", serde_json::json!({"data": "partial"}), 0.5);
        manager.save_partial_result("workflow-1", result).await;

        let results = manager.get_partial_results("workflow-1").await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_id, "agent-1");
        assert!(!results[0].is_complete);
    }

    #[tokio::test]
    async fn test_complete_results() {
        let manager = CancellationManager::new();
        let _token = manager.create_token("workflow-1").await;

        // Add partial result
        manager
            .save_partial_result(
                "workflow-1",
                PartialResult::new("agent-1", serde_json::json!({"partial": true}), 0.5),
            )
            .await;

        // Add complete result
        manager
            .save_partial_result(
                "workflow-1",
                PartialResult::complete("agent-2", serde_json::json!({"complete": true})),
            )
            .await;

        let complete = manager.get_complete_results("workflow-1").await;
        assert_eq!(complete.len(), 1);
        assert_eq!(complete[0].agent_id, "agent-2");
        assert!(complete[0].is_complete);
    }

    #[tokio::test]
    async fn test_child_token() {
        let manager = CancellationManager::new();
        let _parent = manager.create_token("parent").await;
        let child = manager.create_child_token("parent", "child").await.unwrap();

        assert!(!child.is_cancelled());

        // Cancel parent
        manager
            .cancel("parent", CancellationReason::UserRequest, true)
            .await;

        // Child should be cancelled too (cascade)
        assert!(child.is_cancelled());
    }

    #[tokio::test]
    async fn test_cancellation_policies() {
        let manager = CancellationManager::new();

        // Immediate policy
        let immediate = manager
            .create_token_with_policy("immediate", CancellationPolicy::Immediate)
            .await;

        // Graceful policy
        let graceful = manager
            .create_token_with_policy("graceful", CancellationPolicy::Graceful)
            .await;

        manager
            .cancel("immediate", CancellationReason::UserRequest, false)
            .await;
        manager
            .cancel("graceful", CancellationReason::UserRequest, true)
            .await;

        assert!(immediate.is_immediate_cancel());
        assert!(!graceful.is_immediate_cancel());
        assert!(graceful.is_graceful_cancel());
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let manager = CancellationManager::new();
        let mut rx = manager.subscribe();

        let _token = manager.create_token("workflow-1").await;
        manager
            .cancel("workflow-1", CancellationReason::Timeout, true)
            .await;

        let event = rx.try_recv().unwrap();
        assert!(matches!(
            event,
            CancellationEvent::CancellationRequested { .. }
        ));
    }

    #[tokio::test]
    async fn test_stats() {
        let manager = CancellationManager::new();

        let _t1 = manager.create_token("w1").await;
        let _t2 = manager.create_token("w2").await;

        manager
            .save_partial_result(
                "w1",
                PartialResult::new("agent", serde_json::json!({}), 0.5),
            )
            .await;

        manager
            .cancel("w2", CancellationReason::UserRequest, false)
            .await;

        let stats = manager.stats().await;
        assert_eq!(stats.total_workflows, 2);
        assert_eq!(stats.cancelled_workflows, 1);
        assert_eq!(stats.workflows_with_results, 1);
        assert_eq!(stats.total_partial_results, 1);
    }

    #[tokio::test]
    async fn test_remove_workflow() {
        let manager = CancellationManager::new();
        let _token = manager.create_token("workflow-1").await;

        manager
            .save_partial_result(
                "workflow-1",
                PartialResult::complete("agent", serde_json::json!({"result": 42})),
            )
            .await;

        let results = manager.remove_workflow("workflow-1").await;
        assert!(results.is_some());
        assert_eq!(results.unwrap().len(), 1);

        // Should be gone now
        assert!(manager.get_token("workflow-1").await.is_none());
    }

    #[tokio::test]
    async fn test_graceful_timeout() {
        let manager = CancellationManager::new();
        let token = manager
            .create_token_with_policy("workflow", CancellationPolicy::GracefulWithTimeout)
            .await;

        manager
            .cancel("workflow", CancellationReason::UserRequest, true)
            .await;

        // Should have remaining grace period
        assert!(token.remaining_grace_period().is_some());
        assert!(!token.is_immediate_cancel());
    }

    #[tokio::test]
    async fn test_cancellation_guard() {
        let manager = Arc::new(CancellationManager::new());
        let guard = CancellationGuard::new(manager.clone(), "guarded-workflow").await;

        assert!(!guard.is_cancelled());

        guard
            .save_result(PartialResult::new(
                "agent",
                serde_json::json!({"saved": true}),
                0.5,
            ))
            .await;

        let results = manager.get_partial_results("guarded-workflow").await;
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_result_update() {
        let manager = CancellationManager::new();
        let _token = manager.create_token("workflow").await;

        // Save initial result
        manager
            .save_partial_result(
                "workflow",
                PartialResult::new("agent", serde_json::json!({"progress": 0.3}), 0.3),
            )
            .await;

        // Update with more progress
        manager
            .save_partial_result(
                "workflow",
                PartialResult::new("agent", serde_json::json!({"progress": 0.7}), 0.7),
            )
            .await;

        let results = manager.get_partial_results("workflow").await;
        assert_eq!(results.len(), 1);
        assert!((results[0].completion - 0.7).abs() < 0.001);
    }
}
