//! Agent Discovery Module
//!
//! Enables dynamic agent discovery through heartbeat and capability broadcasting.
//!
//! ## Features
//!
//! - **Heartbeat**: Agents periodically announce their presence
//! - **Capabilities**: Agents advertise what they can do
//! - **Registry**: Central registry for discovering agents
//! - **Events**: Subscribe to agent join/leave events
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_agents::discovery::{DiscoveryRegistry, AgentCapabilities};
//!
//! let registry = DiscoveryRegistry::new();
//!
//! // Register agent with capabilities
//! registry.register(AgentInfo {
//!     id: AgentId::new("researcher"),
//!     capabilities: vec![Capability::Analysis, Capability::WebSearch],
//!     ..Default::default()
//! }).await;
//!
//! // Find agents by capability
//! let agents = registry.find_by_capability(Capability::Analysis).await;
//!
//! // Subscribe to discovery events
//! let mut rx = registry.subscribe();
//! while let Some(event) = rx.recv().await {
//!     match event {
//!         DiscoveryEvent::AgentJoined(info) => println!("Agent joined: {}", info.id),
//!         DiscoveryEvent::AgentLeft(id) => println!("Agent left: {}", id),
//!         DiscoveryEvent::HeartbeatReceived(id) => println!("Heartbeat from: {}", id),
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};

use rust_ai_agents_core::types::{AgentId, AgentRole, Capability};

/// Information about a discovered agent
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Agent identifier
    pub id: AgentId,
    /// Agent name
    pub name: String,
    /// Agent role
    pub role: AgentRole,
    /// Agent capabilities
    pub capabilities: Vec<Capability>,
    /// Agent status
    pub status: AgentDiscoveryStatus,
    /// Last heartbeat time
    pub last_heartbeat: Instant,
    /// Agent metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Agent endpoint (for remote agents)
    pub endpoint: Option<String>,
}

impl AgentInfo {
    pub fn new(id: AgentId, name: impl Into<String>, role: AgentRole) -> Self {
        Self {
            id,
            name: name.into(),
            role,
            capabilities: Vec::new(),
            status: AgentDiscoveryStatus::Online,
            last_heartbeat: Instant::now(),
            metadata: HashMap::new(),
            endpoint: None,
        }
    }

    pub fn with_capabilities(mut self, capabilities: Vec<Capability>) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    pub fn add_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Check if agent has a specific capability
    pub fn has_capability(&self, capability: &Capability) -> bool {
        self.capabilities.contains(capability)
    }

    /// Update heartbeat timestamp
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
        self.status = AgentDiscoveryStatus::Online;
    }

    /// Check if agent is considered stale (no heartbeat for too long)
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_heartbeat.elapsed() > timeout
    }
}

impl Default for AgentInfo {
    fn default() -> Self {
        Self {
            id: AgentId::generate(),
            name: "Unknown".to_string(),
            role: AgentRole::Executor,
            capabilities: Vec::new(),
            status: AgentDiscoveryStatus::Unknown,
            last_heartbeat: Instant::now(),
            metadata: HashMap::new(),
            endpoint: None,
        }
    }
}

/// Agent discovery status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentDiscoveryStatus {
    /// Agent is online and responding
    Online,
    /// Agent missed recent heartbeats
    Degraded,
    /// Agent is offline/unreachable
    Offline,
    /// Agent status is unknown
    Unknown,
}

/// Events emitted by the discovery system
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    /// A new agent has joined
    AgentJoined(AgentInfo),
    /// An agent has left or timed out
    AgentLeft(AgentId),
    /// Heartbeat received from an agent
    HeartbeatReceived(AgentId),
    /// Agent capabilities updated
    CapabilitiesUpdated(AgentId, Vec<Capability>),
    /// Agent status changed
    StatusChanged(AgentId, AgentDiscoveryStatus),
}

/// Configuration for the discovery registry
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Timeout before marking agent as degraded
    pub degraded_timeout: Duration,
    /// Timeout before marking agent as offline
    pub offline_timeout: Duration,
    /// Whether to auto-cleanup offline agents
    pub auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(30),
            degraded_timeout: Duration::from_secs(60),
            offline_timeout: Duration::from_secs(120),
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

/// Central registry for agent discovery
pub struct DiscoveryRegistry {
    agents: Arc<RwLock<HashMap<AgentId, AgentInfo>>>,
    #[allow(dead_code)]
    config: DiscoveryConfig,
    event_tx: broadcast::Sender<DiscoveryEvent>,
    _cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

impl DiscoveryRegistry {
    /// Create a new discovery registry with default config
    pub fn new() -> Self {
        Self::with_config(DiscoveryConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DiscoveryConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let agents = Arc::new(RwLock::new(HashMap::new()));

        let cleanup_handle = if config.auto_cleanup {
            let agents_clone = agents.clone();
            let event_tx_clone = event_tx.clone();
            let cleanup_interval = config.cleanup_interval;
            let offline_timeout = config.offline_timeout;

            Some(tokio::spawn(async move {
                let mut interval = tokio::time::interval(cleanup_interval);
                loop {
                    interval.tick().await;
                    Self::cleanup_stale_agents(&agents_clone, &event_tx_clone, offline_timeout)
                        .await;
                }
            }))
        } else {
            None
        };

        Self {
            agents,
            config,
            event_tx,
            _cleanup_handle: cleanup_handle,
        }
    }

    /// Register a new agent
    pub async fn register(&self, info: AgentInfo) {
        let id = info.id.clone();
        let is_new = {
            let agents = self.agents.read().await;
            !agents.contains_key(&id)
        };

        {
            let mut agents = self.agents.write().await;
            agents.insert(id.clone(), info.clone());
        }

        if is_new {
            info!(agent = %id, "Agent registered");
            let _ = self.event_tx.send(DiscoveryEvent::AgentJoined(info));
        }
    }

    /// Unregister an agent
    pub async fn unregister(&self, id: &AgentId) {
        let removed = {
            let mut agents = self.agents.write().await;
            agents.remove(id)
        };

        if removed.is_some() {
            info!(agent = %id, "Agent unregistered");
            let _ = self.event_tx.send(DiscoveryEvent::AgentLeft(id.clone()));
        }
    }

    /// Record a heartbeat from an agent
    pub async fn heartbeat(&self, id: &AgentId) {
        let mut agents = self.agents.write().await;
        if let Some(agent) = agents.get_mut(id) {
            let old_status = agent.status.clone();
            agent.heartbeat();

            if old_status != AgentDiscoveryStatus::Online {
                drop(agents);
                let _ = self.event_tx.send(DiscoveryEvent::StatusChanged(
                    id.clone(),
                    AgentDiscoveryStatus::Online,
                ));
            }

            debug!(agent = %id, "Heartbeat received");
            let _ = self
                .event_tx
                .send(DiscoveryEvent::HeartbeatReceived(id.clone()));
        }
    }

    /// Update agent capabilities
    pub async fn update_capabilities(&self, id: &AgentId, capabilities: Vec<Capability>) {
        let mut agents = self.agents.write().await;
        if let Some(agent) = agents.get_mut(id) {
            agent.capabilities = capabilities.clone();
            drop(agents);

            info!(agent = %id, caps = ?capabilities, "Capabilities updated");
            let _ = self.event_tx.send(DiscoveryEvent::CapabilitiesUpdated(
                id.clone(),
                capabilities,
            ));
        }
    }

    /// Get agent info by ID
    pub async fn get(&self, id: &AgentId) -> Option<AgentInfo> {
        let agents = self.agents.read().await;
        agents.get(id).cloned()
    }

    /// List all registered agents
    pub async fn list_all(&self) -> Vec<AgentInfo> {
        let agents = self.agents.read().await;
        agents.values().cloned().collect()
    }

    /// List online agents only
    pub async fn list_online(&self) -> Vec<AgentInfo> {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|a| a.status == AgentDiscoveryStatus::Online)
            .cloned()
            .collect()
    }

    /// Find agents by capability
    pub async fn find_by_capability(&self, capability: &Capability) -> Vec<AgentInfo> {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|a| a.has_capability(capability))
            .cloned()
            .collect()
    }

    /// Find agents by role
    pub async fn find_by_role(&self, role: &AgentRole) -> Vec<AgentInfo> {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|a| std::mem::discriminant(&a.role) == std::mem::discriminant(role))
            .cloned()
            .collect()
    }

    /// Find agents matching multiple capabilities (AND)
    pub async fn find_by_capabilities(&self, capabilities: &[Capability]) -> Vec<AgentInfo> {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|a| capabilities.iter().all(|c| a.has_capability(c)))
            .cloned()
            .collect()
    }

    /// Find agents matching any of the capabilities (OR)
    pub async fn find_by_any_capability(&self, capabilities: &[Capability]) -> Vec<AgentInfo> {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|a| capabilities.iter().any(|c| a.has_capability(c)))
            .cloned()
            .collect()
    }

    /// Subscribe to discovery events
    pub fn subscribe(&self) -> broadcast::Receiver<DiscoveryEvent> {
        self.event_tx.subscribe()
    }

    /// Get the number of registered agents
    pub async fn agent_count(&self) -> usize {
        let agents = self.agents.read().await;
        agents.len()
    }

    /// Get the number of online agents
    pub async fn online_count(&self) -> usize {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|a| a.status == AgentDiscoveryStatus::Online)
            .count()
    }

    /// Cleanup stale agents
    async fn cleanup_stale_agents(
        agents: &Arc<RwLock<HashMap<AgentId, AgentInfo>>>,
        event_tx: &broadcast::Sender<DiscoveryEvent>,
        offline_timeout: Duration,
    ) {
        let mut agents_write = agents.write().await;

        // Collect IDs of agents to remove
        let stale_ids: Vec<AgentId> = agents_write
            .iter()
            .filter(|(_, a)| a.is_stale(offline_timeout))
            .map(|(id, _)| id.clone())
            .collect();

        for id in stale_ids {
            if let Some(agent) = agents_write.remove(&id) {
                warn!(agent = %id, "Removing stale agent");
                let _ = event_tx.send(DiscoveryEvent::AgentLeft(agent.id));
            }
        }

        // Update status for degraded agents
        let degraded_timeout = offline_timeout / 2;
        for agent in agents_write.values_mut() {
            if agent.status == AgentDiscoveryStatus::Online && agent.is_stale(degraded_timeout) {
                agent.status = AgentDiscoveryStatus::Degraded;
                let _ = event_tx.send(DiscoveryEvent::StatusChanged(
                    agent.id.clone(),
                    AgentDiscoveryStatus::Degraded,
                ));
            }
        }
    }
}

impl Default for DiscoveryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Heartbeat sender for an agent
pub struct HeartbeatSender {
    registry: Arc<DiscoveryRegistry>,
    agent_id: AgentId,
    interval: Duration,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl HeartbeatSender {
    /// Create a new heartbeat sender
    pub fn new(registry: Arc<DiscoveryRegistry>, agent_id: AgentId) -> Self {
        Self {
            registry,
            agent_id,
            interval: Duration::from_secs(30),
            handle: None,
        }
    }

    /// Set heartbeat interval
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Start sending heartbeats
    pub fn start(&mut self) {
        let registry = self.registry.clone();
        let agent_id = self.agent_id.clone();
        let interval = self.interval;

        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                registry.heartbeat(&agent_id).await;
            }
        });

        self.handle = Some(handle);
        info!(agent = %self.agent_id, interval = ?self.interval, "Heartbeat sender started");
    }

    /// Stop sending heartbeats
    pub fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
            info!(agent = %self.agent_id, "Heartbeat sender stopped");
        }
    }
}

impl Drop for HeartbeatSender {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_and_get() {
        let registry = DiscoveryRegistry::new();

        let info = AgentInfo::new(AgentId::new("test"), "Test Agent", AgentRole::Executor)
            .with_capabilities(vec![Capability::Analysis]);

        registry.register(info.clone()).await;

        let retrieved = registry.get(&AgentId::new("test")).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test Agent");
    }

    #[tokio::test]
    async fn test_find_by_capability() {
        let registry = DiscoveryRegistry::new();

        let agent1 = AgentInfo::new(AgentId::new("a1"), "Agent 1", AgentRole::Researcher)
            .with_capabilities(vec![Capability::Analysis, Capability::WebSearch]);

        let agent2 = AgentInfo::new(AgentId::new("a2"), "Agent 2", AgentRole::Writer)
            .with_capabilities(vec![Capability::ContentGeneration]);

        registry.register(agent1).await;
        registry.register(agent2).await;

        let analysts = registry.find_by_capability(&Capability::Analysis).await;
        assert_eq!(analysts.len(), 1);
        assert_eq!(analysts[0].id, AgentId::new("a1"));

        let writers = registry
            .find_by_capability(&Capability::ContentGeneration)
            .await;
        assert_eq!(writers.len(), 1);
        assert_eq!(writers[0].id, AgentId::new("a2"));
    }

    #[tokio::test]
    async fn test_unregister() {
        let registry = DiscoveryRegistry::new();

        let info = AgentInfo::new(AgentId::new("test"), "Test Agent", AgentRole::Executor);
        registry.register(info).await;

        assert_eq!(registry.agent_count().await, 1);

        registry.unregister(&AgentId::new("test")).await;
        assert_eq!(registry.agent_count().await, 0);
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let registry = DiscoveryRegistry::new();

        let info = AgentInfo::new(AgentId::new("test"), "Test Agent", AgentRole::Executor);
        registry.register(info).await;

        // Send heartbeat
        registry.heartbeat(&AgentId::new("test")).await;

        let agent = registry.get(&AgentId::new("test")).await.unwrap();
        assert_eq!(agent.status, AgentDiscoveryStatus::Online);
    }

    #[tokio::test]
    async fn test_update_capabilities() {
        let registry = DiscoveryRegistry::new();

        let info = AgentInfo::new(AgentId::new("test"), "Test Agent", AgentRole::Executor);
        registry.register(info).await;

        registry
            .update_capabilities(
                &AgentId::new("test"),
                vec![Capability::Analysis, Capability::Prediction],
            )
            .await;

        let agent = registry.get(&AgentId::new("test")).await.unwrap();
        assert_eq!(agent.capabilities.len(), 2);
        assert!(agent.has_capability(&Capability::Analysis));
    }

    #[test]
    fn test_agent_info_stale_check() {
        let mut info = AgentInfo::new(AgentId::new("test"), "Test", AgentRole::Executor);

        // Fresh agent should not be stale
        assert!(!info.is_stale(Duration::from_secs(60)));

        // Simulate old heartbeat by creating with old timestamp
        info.last_heartbeat = Instant::now() - Duration::from_secs(120);
        assert!(info.is_stale(Duration::from_secs(60)));

        // Heartbeat should refresh
        info.heartbeat();
        assert!(!info.is_stale(Duration::from_secs(60)));
    }
}
