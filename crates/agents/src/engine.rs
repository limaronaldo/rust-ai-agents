//! Agent execution engine with ReACT loop
//!
//! Features:
//! - Tracing instrumentation for observability (Jaeger, Honeycomb compatible)
//! - Configurable timeout per agent to prevent hangs
//! - Error injection into context for self-correction
//! - Cost tracking integration for LLM API usage monitoring

use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::timeout;
use tracing::{debug, debug_span, error, info, instrument, warn, Instrument};

use rust_ai_agents_core::*;
use rust_ai_agents_monitoring::CostTracker;
use rust_ai_agents_providers::*;
use rust_ai_agents_tools::ToolRegistry;

use crate::executor::ToolExecutor;
use crate::memory::AgentMemory;

/// Agent engine managing multiple agents
#[derive(Clone)]
pub struct AgentEngine {
    agents: Arc<DashMap<AgentId, Arc<AgentRuntime>>>,
    metrics: Arc<EngineMetrics>,
    cost_tracker: Option<Arc<CostTracker>>,
}

/// Engine metrics with atomic counters for lock-free updates
#[derive(Default)]
pub struct EngineMetrics {
    pub agents_spawned: std::sync::atomic::AtomicU64,
    pub agents_active: std::sync::atomic::AtomicU64,
    pub messages_processed: std::sync::atomic::AtomicU64,
    pub messages_failed: std::sync::atomic::AtomicU64,
    pub total_tool_calls: std::sync::atomic::AtomicU64,
    pub timeouts: std::sync::atomic::AtomicU64,
}

/// Agent runtime state
pub struct AgentRuntime {
    pub config: AgentConfig,
    pub state: Arc<RwLock<AgentState>>,
    pub memory: Arc<AgentMemory>,
    pub tool_registry: Arc<ToolRegistry>,
    pub backend: Arc<dyn LLMBackend>,
    pub executor: ToolExecutor,
    pub inbox_tx: mpsc::UnboundedSender<Message>,
    processing_task: parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl AgentEngine {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(DashMap::new()),
            metrics: Arc::new(EngineMetrics::default()),
            cost_tracker: None,
        }
    }

    /// Create an engine with cost tracking enabled
    pub fn with_cost_tracker(cost_tracker: Arc<CostTracker>) -> Self {
        Self {
            agents: Arc::new(DashMap::new()),
            metrics: Arc::new(EngineMetrics::default()),
            cost_tracker: Some(cost_tracker),
        }
    }

    /// Set or replace the cost tracker
    pub fn set_cost_tracker(&mut self, cost_tracker: Arc<CostTracker>) {
        self.cost_tracker = Some(cost_tracker);
    }

    /// Get the cost tracker (if configured)
    pub fn cost_tracker(&self) -> Option<&Arc<CostTracker>> {
        self.cost_tracker.as_ref()
    }

    /// Spawn a new agent with tracing instrumentation
    #[instrument(skip(self, tool_registry, backend), fields(agent_id = %config.id, agent_name = %config.name))]
    pub async fn spawn_agent(
        &self,
        config: AgentConfig,
        tool_registry: Arc<ToolRegistry>,
        backend: Arc<dyn LLMBackend>,
    ) -> Result<AgentId, AgentError> {
        use std::sync::atomic::Ordering;

        let (inbox_tx, mut inbox_rx) = mpsc::unbounded_channel::<Message>();

        let agent_id = config.id.clone();
        let state = Arc::new(RwLock::new(AgentState::new()));
        let memory = Arc::new(AgentMemory::new(config.memory_config.clone()));
        let executor = ToolExecutor::new(10); // 10 concurrent tool calls

        // Clone everything needed for the processing task
        let config_clone = config.clone();
        let state_clone = state.clone();
        let memory_clone = memory.clone();
        let registry_clone = tool_registry.clone();
        let backend_clone = backend.clone();
        let executor_clone = executor.clone();
        let engine_clone = self.clone();
        let metrics_clone = self.metrics.clone();
        let cost_tracker_clone = self.cost_tracker.clone();

        // Create a span for the agent's processing loop
        let agent_loop_span = tracing::info_span!(
            "agent_loop",
            agent_id = %agent_id,
            agent_name = %config.name
        );

        // Spawn processing task with tracing
        let processing_task = tokio::spawn(
            async move {
                while let Some(message) = inbox_rx.recv().await {
                    // Create a span for each message processing
                    let msg_span = debug_span!(
                        "process_message",
                        from = %message.from,
                        iteration = tracing::field::Empty
                    );

                    async {
                        let start = std::time::Instant::now();
                        let timeout_duration = Duration::from_secs(config_clone.timeout_secs);

                        info!(
                            timeout_secs = config_clone.timeout_secs,
                            "Processing message from {}", message.from
                        );

                        // Wrap processing in a timeout
                        let process_result = timeout(
                            timeout_duration,
                            Self::process_message(
                                &config_clone,
                                &state_clone,
                                &memory_clone,
                                &registry_clone,
                                &backend_clone,
                                &executor_clone,
                                &metrics_clone,
                                cost_tracker_clone.as_ref(),
                                message.clone(),
                            ),
                        )
                        .await;

                        match process_result {
                            Ok(Ok(responses)) => {
                                // Successfully processed
                                for response in responses {
                                    if let Err(e) = engine_clone.send_message(response) {
                                        error!("Failed to route response: {}", e);
                                    }
                                }
                                metrics_clone
                                    .messages_processed
                                    .fetch_add(1, Ordering::Relaxed);

                                let latency = start.elapsed();
                                if latency.as_millis() > 500 {
                                    warn!(
                                        latency_ms = latency.as_millis(),
                                        "Slow processing detected"
                                    );
                                }
                            }
                            Ok(Err(e)) => {
                                // Processing error - restore state to Idle
                                error!("Processing error: {}", e);
                                {
                                    let mut state_write = state_clone.write().await;
                                    state_write.status = AgentStatus::Idle;
                                }
                                metrics_clone
                                    .messages_failed
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                            Err(_) => {
                                // Timeout - restore state to Idle
                                error!(
                                    timeout_secs = config_clone.timeout_secs,
                                    "Agent timeout processing message"
                                );
                                {
                                    let mut state_write = state_clone.write().await;
                                    state_write.status = AgentStatus::Idle;
                                }
                                metrics_clone.timeouts.fetch_add(1, Ordering::Relaxed);
                                metrics_clone
                                    .messages_failed
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    .instrument(msg_span)
                    .await;
                }

                info!("Agent processing task stopped");
            }
            .instrument(agent_loop_span),
        );

        let runtime = Arc::new(AgentRuntime {
            config: config.clone(),
            state,
            memory,
            tool_registry,
            backend,
            executor,
            inbox_tx,
            processing_task: parking_lot::Mutex::new(Some(processing_task)),
        });

        self.agents.insert(agent_id.clone(), runtime);

        self.metrics.agents_spawned.fetch_add(1, Ordering::Relaxed);
        self.metrics.agents_active.fetch_add(1, Ordering::Relaxed);

        info!("Agent spawned successfully");

        Ok(agent_id)
    }

    /// Process a single message with ReACT loop (Reason -> Act -> Observe)
    #[instrument(
        skip(state, memory, tool_registry, backend, executor, metrics, cost_tracker, message),
        fields(agent_id = %config.id, max_iterations = config.max_iterations)
    )]
    async fn process_message(
        config: &AgentConfig,
        state: &Arc<RwLock<AgentState>>,
        memory: &Arc<AgentMemory>,
        tool_registry: &Arc<ToolRegistry>,
        backend: &Arc<dyn LLMBackend>,
        executor: &ToolExecutor,
        metrics: &Arc<EngineMetrics>,
        cost_tracker: Option<&Arc<CostTracker>>,
        message: Message,
    ) -> Result<Vec<Message>, AgentError> {
        let mut responses = Vec::new();

        // Update state
        {
            let mut state_write = state.write().await;
            state_write.status = AgentStatus::Processing;
            state_write.iteration = 0;
        }

        // Add message to memory
        memory.add_message(message.clone()).await?;

        // ReACT Loop: Reason -> Act -> Observe
        let mut iteration = 0;
        let max_iterations = config.max_iterations;

        while iteration < max_iterations {
            iteration += 1;

            // Create a span for each iteration
            let iter_span = debug_span!("react_iteration", n = iteration, max = max_iterations);
            let _iter_guard = iter_span.enter();

            {
                let mut state_write = state.write().await;
                state_write.iteration = iteration;
                state_write.status = AgentStatus::Thinking;
            }

            debug!(iteration, max_iterations, "ReACT iteration");

            // 1. REASON: Get conversation history and available tools
            let history = memory.get_history().await?;
            let tool_schemas = tool_registry.list_schemas();

            // Convert to LLM format
            let mut llm_messages = Vec::new();

            // Add system prompt
            if let Some(system_prompt) = &config.system_prompt {
                llm_messages.push(LLMMessage::system(system_prompt));
            }

            // Add conversation history
            for msg in history {
                match &msg.content {
                    Content::Text(text) => {
                        let role = if msg.from == config.id {
                            MessageRole::Assistant
                        } else {
                            MessageRole::User
                        };
                        llm_messages.push(LLMMessage {
                            role,
                            content: text.clone(),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        });
                    }
                    Content::ToolCall(calls) => {
                        llm_messages.push(LLMMessage::assistant_with_tools(calls.clone()));
                    }
                    Content::ToolResult(results) => {
                        for result in results {
                            let content = if result.success {
                                serde_json::to_string_pretty(&result.data).unwrap_or_default()
                            } else {
                                result.error.clone().unwrap_or_default()
                            };
                            llm_messages.push(LLMMessage::tool(
                                result.call_id.clone(),
                                "tool_result".to_string(),
                                content,
                            ));
                        }
                    }
                    _ => {}
                }
            }

            // 2. Call LLM with cost tracking
            let infer_start = std::time::Instant::now();
            let inference = backend
                .infer(&llm_messages, &tool_schemas, config.temperature)
                .await;
            let infer_latency_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

            // Record cost if tracker is configured
            if let Some(tracker) = cost_tracker {
                let model_info = backend.model_info();
                match &inference {
                    Ok(inf) => {
                        tracker.record_request_detailed(
                            &model_info.model,
                            inf.token_usage.prompt_tokens as u64,
                            inf.token_usage.completion_tokens as u64,
                            inf.token_usage.cached_tokens.unwrap_or(0) as u64,
                            infer_latency_ms,
                            Some(config.id.0.as_str()),
                            true,
                        );
                    }
                    Err(_) => {
                        tracker.record_request_detailed(
                            &model_info.model,
                            0,
                            0,
                            0,
                            infer_latency_ms,
                            Some(config.id.0.as_str()),
                            false,
                        );
                    }
                }
            }

            let inference = inference?;

            debug!(
                content_len = inference.content.len(),
                tool_calls = inference.tool_calls.as_ref().map(|c| c.len()).unwrap_or(0),
                "LLM inference complete"
            );

            // 3. ACT: Check if agent wants to use tools
            if let Some(tool_calls) = &inference.tool_calls {
                if !tool_calls.is_empty() {
                    {
                        let mut state_write = state.write().await;
                        state_write.status = AgentStatus::ExecutingTool;
                    }

                    info!(num_calls = tool_calls.len(), "Executing tool calls");

                    // Track tool call metrics
                    metrics.total_tool_calls.fetch_add(
                        tool_calls.len() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );

                    // Store tool call in memory
                    let tool_call_msg = Message::new(
                        config.id.clone(),
                        config.id.clone(),
                        Content::ToolCall(tool_calls.clone()),
                    );
                    memory.add_message(tool_call_msg).await?;

                    // Execute tools in parallel
                    let results = executor
                        .execute_tools(tool_calls, tool_registry, &config.id)
                        .await;

                    // Store results in memory
                    let result_msg = Message::new(
                        AgentId::new("system"),
                        config.id.clone(),
                        Content::ToolResult(results),
                    );
                    memory.add_message(result_msg).await?;

                    // Continue loop to process tool results
                    continue;
                }
            }

            // 4. RESPOND: Agent has final answer
            if !inference.content.is_empty() {
                let response = Message::new(
                    config.id.clone(),
                    message.from.clone(),
                    Content::Text(inference.content.clone()),
                );

                // Store own response in memory
                memory.add_message(response.clone()).await?;

                responses.push(response);
                break;
            }

            // Safety: if LLM returns nothing, break
            if inference.content.is_empty() && inference.tool_calls.is_none() {
                warn!("Agent {} LLM returned empty response", config.id);
                break;
            }
        }

        // Update state to idle before returning (success or error)
        {
            let mut state_write = state.write().await;
            state_write.status = AgentStatus::Idle;
        }

        if iteration >= max_iterations {
            warn!(iterations = iteration, "Max iterations reached");
            return Err(AgentError::MaxIterationsExceeded);
        }

        Ok(responses)
    }

    /// Send a message to an agent
    #[instrument(skip(self), fields(to = %message.to, from = %message.from))]
    pub fn send_message(&self, message: Message) -> Result<(), AgentError> {
        if let Some(agent) = self.agents.get(&message.to) {
            agent
                .inbox_tx
                .send(message)
                .map_err(|e| AgentError::SendError(e.to_string()))?;

            debug!("Message sent successfully");
            Ok(())
        } else {
            warn!("Agent not found");
            Err(AgentError::AgentNotFound(message.to))
        }
    }

    /// Get agent by ID
    pub fn get_agent(&self, id: &AgentId) -> Option<Arc<AgentRuntime>> {
        self.agents.get(id).map(|r| r.clone())
    }

    /// Get agent count
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Get metrics
    pub fn metrics(&self) -> EngineMetricsSnapshot {
        use std::sync::atomic::Ordering;
        EngineMetricsSnapshot {
            agents_spawned: self.metrics.agents_spawned.load(Ordering::Relaxed),
            agents_active: self.metrics.agents_active.load(Ordering::Relaxed),
            messages_processed: self.metrics.messages_processed.load(Ordering::Relaxed),
            messages_failed: self.metrics.messages_failed.load(Ordering::Relaxed),
            total_tool_calls: self.metrics.total_tool_calls.load(Ordering::Relaxed),
            timeouts: self.metrics.timeouts.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of engine metrics
#[derive(Debug, Clone)]
pub struct EngineMetricsSnapshot {
    pub agents_spawned: u64,
    pub agents_active: u64,
    pub messages_processed: u64,
    pub messages_failed: u64,
    pub total_tool_calls: u64,
    pub timeouts: u64,
}

impl EngineMetricsSnapshot {
    pub fn success_rate(&self) -> f64 {
        let total = self.messages_processed + self.messages_failed;
        if total > 0 {
            self.messages_processed as f64 / total as f64
        } else {
            1.0
        }
    }
}

impl AgentEngine {
    /// Stop an agent
    #[instrument(skip(self), fields(agent_id = %id))]
    pub async fn stop_agent(&self, id: &AgentId) -> Result<(), AgentError> {
        use std::sync::atomic::Ordering;

        if let Some((_, runtime)) = self.agents.remove(id) {
            // Abort the processing task
            if let Some(handle) = runtime.processing_task.lock().take() {
                handle.abort();
            }
            self.metrics.agents_active.fetch_sub(1, Ordering::Relaxed);
            info!("Agent stopped");
            Ok(())
        } else {
            warn!("Agent not found");
            Err(AgentError::AgentNotFound(id.clone()))
        }
    }

    /// Stop all agents
    #[instrument(skip(self))]
    pub async fn shutdown(&self) {
        let ids: Vec<AgentId> = self.agents.iter().map(|r| r.key().clone()).collect();
        info!(agent_count = ids.len(), "Shutting down all agents");
        for id in ids {
            let _ = self.stop_agent(&id).await;
        }
    }
}

impl Default for AgentEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AgentEngine {
    fn drop(&mut self) {
        // Note: We can't use async in Drop, so agents will be aborted when tasks are dropped
        debug!("AgentEngine dropped");
    }
}
