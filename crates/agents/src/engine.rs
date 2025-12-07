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

use rust_ai_agents_core::types::PlanningMode;
use rust_ai_agents_core::*;
use rust_ai_agents_monitoring::CostTracker;
use rust_ai_agents_providers::*;
use rust_ai_agents_tools::ToolRegistry;

use crate::executor::ToolExecutor;
use crate::memory::AgentMemory;
use crate::planning::{check_stop_words, PlanGenerator, StepExecutionContext};

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
    /// Supports planning mode for structured task execution
    #[instrument(
        skip(state, memory, tool_registry, backend, executor, metrics, cost_tracker, message),
        fields(agent_id = %config.id, max_iterations = config.max_iterations, planning_mode = ?config.planning_mode)
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

        // Check if planning mode is enabled
        if config.planning_mode.is_enabled() {
            return Self::process_with_planning(
                config,
                state,
                memory,
                tool_registry,
                backend,
                executor,
                metrics,
                cost_tracker,
                message,
            )
            .await;
        }

        // ReACT Loop: Reason -> Act -> Observe (no planning)
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
                // Check for stop words
                if !config.stop_words.is_empty() {
                    if let Some(stop_word) =
                        check_stop_words(&inference.content, &config.stop_words)
                    {
                        info!(stop_word = %stop_word, "Stop word detected, terminating");
                        let mut state_write = state.write().await;
                        state_write.status = AgentStatus::StoppedByStopWord;
                        drop(state_write);

                        let response = Message::new(
                            config.id.clone(),
                            message.from.clone(),
                            Content::Text(inference.content.clone()),
                        );
                        memory.add_message(response.clone()).await?;
                        responses.push(response);
                        return Ok(responses);
                    }
                }

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

    /// Process message with planning mode enabled
    #[instrument(
        skip(state, memory, tool_registry, backend, executor, metrics, cost_tracker, message),
        fields(agent_id = %config.id, planning_mode = ?config.planning_mode)
    )]
    async fn process_with_planning(
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

        // Extract goal from message
        let goal = match &message.content {
            Content::Text(text) => text.clone(),
            _ => {
                return Err(AgentError::ProcessingError(
                    "Planning requires text message".into(),
                ))
            }
        };

        // 1. PLANNING PHASE: Generate execution plan
        {
            let mut state_write = state.write().await;
            state_write.status = AgentStatus::Planning;
        }

        info!(goal = %goal, "Generating execution plan");

        let tool_schemas = tool_registry.list_schemas();
        let planning_prompt = PlanGenerator::create_planning_prompt(&goal, &tool_schemas);

        let plan_messages = vec![
            config
                .system_prompt
                .as_ref()
                .map(LLMMessage::system)
                .unwrap_or_else(|| {
                    LLMMessage::system("You are a planning agent. Create detailed execution plans.")
                }),
            LLMMessage::user(&planning_prompt),
        ];

        let infer_start = std::time::Instant::now();
        let plan_inference = backend
            .infer(&plan_messages, &[], config.temperature)
            .await?;
        let infer_latency_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

        // Record cost for planning
        if let Some(tracker) = cost_tracker {
            let model_info = backend.model_info();
            tracker.record_request_detailed(
                &model_info.model,
                plan_inference.token_usage.prompt_tokens as u64,
                plan_inference.token_usage.completion_tokens as u64,
                plan_inference.token_usage.cached_tokens.unwrap_or(0) as u64,
                infer_latency_ms,
                Some(config.id.0.as_str()),
                true,
            );
        }

        // Parse the plan
        let mut plan = match PlanGenerator::parse_plan(&goal, &plan_inference.content) {
            Ok(p) => {
                info!(steps = p.steps.len(), "Plan generated successfully");
                p
            }
            Err(e) => {
                warn!(error = %e, "Failed to parse plan, returning raw response");
                // Return the LLM response directly instead of recursing
                let response = Message::new(
                    config.id.clone(),
                    message.from.clone(),
                    Content::Text(plan_inference.content.clone()),
                );
                memory.add_message(response.clone()).await?;

                let mut state_write = state.write().await;
                state_write.status = AgentStatus::Idle;
                drop(state_write);

                return Ok(vec![response]);
            }
        };

        // Store plan in state
        {
            let mut state_write = state.write().await;
            state_write.set_plan(plan.clone());
            state_write.status = AgentStatus::ExecutingPlan;
        }

        // 2. EXECUTION PHASE: Execute each step
        while plan.advance() {
            let step = plan.current_step().unwrap().clone();

            info!(
                step = step.step_number,
                total = plan.steps.len(),
                description = %step.description,
                "Executing plan step"
            );

            let step_ctx = StepExecutionContext::from_step(&step, &plan);

            // Create messages for step execution
            let mut step_messages = vec![];

            if let Some(sys_prompt) = &config.system_prompt {
                step_messages.push(LLMMessage::system(sys_prompt));
            }
            step_messages.push(LLMMessage::user(&step_ctx.prompt));

            // Execute step with ReACT loop
            let step_result = Self::execute_step(
                config,
                state,
                memory,
                tool_registry,
                backend,
                executor,
                metrics,
                cost_tracker,
                step_messages,
            )
            .await?;

            // Mark step completed
            plan.mark_current_completed(&step_result);

            // Update state with progress
            {
                let mut state_write = state.write().await;
                state_write.current_plan = Some(plan.clone());
            }

            // Check for stop words in step result
            if !config.stop_words.is_empty() {
                if let Some(stop_word) = check_stop_words(&step_result, &config.stop_words) {
                    info!(stop_word = %stop_word, "Stop word detected during plan execution");
                    let mut state_write = state.write().await;
                    state_write.status = AgentStatus::StoppedByStopWord;
                    break;
                }
            }

            // Adaptive re-planning
            if config.planning_mode == PlanningMode::Adaptive && !plan.is_complete() {
                let replan_prompt = PlanGenerator::create_replan_prompt(&plan, &step_result);
                let replan_messages = vec![
                    LLMMessage::system(
                        "You are a planning agent. Review progress and adjust the plan if needed.",
                    ),
                    LLMMessage::user(&replan_prompt),
                ];

                if let Ok(replan_inference) = backend
                    .infer(&replan_messages, &[], config.temperature)
                    .await
                {
                    if let Ok(modified) =
                        PlanGenerator::apply_replan(&mut plan, &replan_inference.content)
                    {
                        if modified {
                            info!("Plan was modified based on step results");
                            let mut state_write = state.write().await;
                            state_write.current_plan = Some(plan.clone());
                        }
                    }
                }
            }
        }

        // 3. COMPLETION: Generate final response
        plan.completed = true;
        plan.success = plan.steps.iter().all(|s| s.completed);

        let summary = format!(
            "Plan completed. Goal: {}\nSteps executed: {}/{}\nSuccess: {}",
            plan.goal,
            plan.steps.iter().filter(|s| s.completed).count(),
            plan.steps.len(),
            plan.success
        );

        let response = Message::new(
            config.id.clone(),
            message.from.clone(),
            Content::Text(summary),
        );

        memory.add_message(response.clone()).await?;
        responses.push(response);

        // Update final state
        {
            let mut state_write = state.write().await;
            state_write.status = AgentStatus::Idle;
            state_write.current_plan = Some(plan);
        }

        Ok(responses)
    }

    /// Execute a single step using ReACT loop
    async fn execute_step(
        config: &AgentConfig,
        state: &Arc<RwLock<AgentState>>,
        memory: &Arc<AgentMemory>,
        tool_registry: &Arc<ToolRegistry>,
        backend: &Arc<dyn LLMBackend>,
        executor: &ToolExecutor,
        metrics: &Arc<EngineMetrics>,
        cost_tracker: Option<&Arc<CostTracker>>,
        mut messages: Vec<LLMMessage>,
    ) -> Result<String, AgentError> {
        let tool_schemas = tool_registry.list_schemas();
        let mut iteration = 0;
        let max_iterations = config.max_iterations.min(5); // Limit per-step iterations

        loop {
            iteration += 1;
            if iteration > max_iterations {
                return Err(AgentError::MaxIterationsExceeded);
            }

            let infer_start = std::time::Instant::now();
            let inference = backend
                .infer(&messages, &tool_schemas, config.temperature)
                .await?;
            let infer_latency_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

            if let Some(tracker) = cost_tracker {
                let model_info = backend.model_info();
                tracker.record_request_detailed(
                    &model_info.model,
                    inference.token_usage.prompt_tokens as u64,
                    inference.token_usage.completion_tokens as u64,
                    inference.token_usage.cached_tokens.unwrap_or(0) as u64,
                    infer_latency_ms,
                    Some(config.id.0.as_str()),
                    true,
                );
            }

            // Handle tool calls
            if let Some(tool_calls) = &inference.tool_calls {
                if !tool_calls.is_empty() {
                    {
                        let mut state_write = state.write().await;
                        state_write.status = AgentStatus::ExecutingTool;
                    }

                    metrics.total_tool_calls.fetch_add(
                        tool_calls.len() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );

                    // Store tool call
                    let tool_call_msg = Message::new(
                        config.id.clone(),
                        config.id.clone(),
                        Content::ToolCall(tool_calls.clone()),
                    );
                    memory.add_message(tool_call_msg).await?;

                    // Execute tools
                    let results = executor
                        .execute_tools(tool_calls, tool_registry, &config.id)
                        .await;

                    // Add tool results to messages
                    messages.push(LLMMessage::assistant_with_tools(tool_calls.clone()));
                    for result in &results {
                        let content = if result.success {
                            serde_json::to_string_pretty(&result.data).unwrap_or_default()
                        } else {
                            result.error.clone().unwrap_or_default()
                        };
                        messages.push(LLMMessage::tool(
                            result.call_id.clone(),
                            "tool_result".to_string(),
                            content,
                        ));
                    }

                    // Store results
                    let result_msg = Message::new(
                        AgentId::new("system"),
                        config.id.clone(),
                        Content::ToolResult(results),
                    );
                    memory.add_message(result_msg).await?;

                    {
                        let mut state_write = state.write().await;
                        state_write.status = AgentStatus::ExecutingPlan;
                    }

                    continue;
                }
            }

            // Return response
            if !inference.content.is_empty() {
                return Ok(inference.content);
            }

            // Empty response
            return Ok("Step completed without explicit output".to_string());
        }
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
