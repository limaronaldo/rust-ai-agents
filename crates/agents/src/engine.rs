//! Agent execution engine with ReACT loop

use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, warn};

use rust_ai_agents_core::*;
use rust_ai_agents_providers::*;
use rust_ai_agents_tools::ToolRegistry;

use crate::executor::ToolExecutor;
use crate::memory::AgentMemory;

/// Agent engine managing multiple agents
#[derive(Clone)]
pub struct AgentEngine {
    agents: Arc<DashMap<AgentId, Arc<AgentRuntime>>>,
    metrics: Arc<EngineMetrics>,
}

/// Engine metrics
#[derive(Default)]
pub struct EngineMetrics {
    pub agents_spawned: parking_lot::Mutex<u64>,
    pub messages_processed: parking_lot::Mutex<u64>,
    pub total_tool_calls: parking_lot::Mutex<u64>,
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
        }
    }

    /// Spawn a new agent
    pub async fn spawn_agent(
        &self,
        config: AgentConfig,
        tool_registry: Arc<ToolRegistry>,
        backend: Arc<dyn LLMBackend>,
    ) -> Result<AgentId, AgentError> {
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

        // Spawn processing task
        let processing_task = tokio::spawn(async move {
            while let Some(message) = inbox_rx.recv().await {
                let start = std::time::Instant::now();

                debug!(
                    "Agent {} processing message from {}",
                    config_clone.id, message.from
                );

                match Self::process_message(
                    &config_clone,
                    &state_clone,
                    &memory_clone,
                    &registry_clone,
                    &backend_clone,
                    &executor_clone,
                    message,
                )
                .await
                {
                    Ok(responses) => {
                        for response in responses {
                            if let Err(e) = engine_clone.send_message(response) {
                                error!("Failed to send response: {}", e);
                            }
                        }

                        let latency = start.elapsed();
                        if latency.as_millis() > 500 {
                            warn!("Agent {} slow processing: {:?}", config_clone.id, latency);
                        }
                    }
                    Err(e) => {
                        error!("Agent {} processing error: {}", config_clone.id, e);
                    }
                }
            }

            debug!("Agent {} processing task stopped", config_clone.id);
        });

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

        {
            let mut count = self.metrics.agents_spawned.lock();
            *count += 1;
        }

        debug!("Agent {} spawned successfully", agent_id);

        Ok(agent_id)
    }

    /// Process a single message with ReACT loop
    async fn process_message(
        config: &AgentConfig,
        state: &Arc<RwLock<AgentState>>,
        memory: &Arc<AgentMemory>,
        tool_registry: &Arc<ToolRegistry>,
        backend: &Arc<dyn LLMBackend>,
        executor: &ToolExecutor,
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

            {
                let mut state_write = state.write().await;
                state_write.iteration = iteration;
                state_write.status = AgentStatus::Thinking;
            }

            debug!(
                "Agent {} iteration {}/{}",
                config.id, iteration, max_iterations
            );

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

            // 2. Call LLM
            let inference = backend
                .infer(&llm_messages, &tool_schemas, config.temperature)
                .await?;

            debug!(
                "Agent {} LLM response: content_len={}, tool_calls={}",
                config.id,
                inference.content.len(),
                inference.tool_calls.as_ref().map(|c| c.len()).unwrap_or(0)
            );

            // 3. ACT: Check if agent wants to use tools
            if let Some(tool_calls) = &inference.tool_calls {
                if !tool_calls.is_empty() {
                    {
                        let mut state_write = state.write().await;
                        state_write.status = AgentStatus::ExecutingTool;
                    }

                    debug!(
                        "Agent {} executing {} tool calls",
                        config.id,
                        tool_calls.len()
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

        if iteration >= max_iterations {
            warn!("Agent {} reached max iterations", config.id);
            return Err(AgentError::MaxIterationsExceeded);
        }

        // Update state to idle
        {
            let mut state_write = state.write().await;
            state_write.status = AgentStatus::Idle;
        }

        Ok(responses)
    }

    /// Send a message to an agent
    pub fn send_message(&self, message: Message) -> Result<(), AgentError> {
        if let Some(agent) = self.agents.get(&message.to) {
            agent
                .inbox_tx
                .send(message)
                .map_err(|e| AgentError::SendError(e.to_string()))?;

            {
                let mut count = self.metrics.messages_processed.lock();
                *count += 1;
            }

            Ok(())
        } else {
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
    pub fn metrics(&self) -> (u64, u64) {
        let spawned = *self.metrics.agents_spawned.lock();
        let processed = *self.metrics.messages_processed.lock();
        (spawned, processed)
    }

    /// Stop an agent
    pub async fn stop_agent(&self, id: &AgentId) -> Result<(), AgentError> {
        if let Some((_, runtime)) = self.agents.remove(id) {
            // Abort the processing task
            if let Some(handle) = runtime.processing_task.lock().take() {
                handle.abort();
            }
            debug!("Agent {} stopped", id);
            Ok(())
        } else {
            Err(AgentError::AgentNotFound(id.clone()))
        }
    }

    /// Stop all agents
    pub async fn shutdown(&self) {
        let ids: Vec<AgentId> = self.agents.iter().map(|r| r.key().clone()).collect();
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
