//! Integration tests for AgentEngine with MockBackend
//!
//! These tests verify the complete ReACT loop using deterministic responses.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use rust_ai_agents_core::types::{
        AgentRole, Capability, MemoryConfig, PlanningMode, RoutingStrategy,
    };
    use rust_ai_agents_core::*;
    use rust_ai_agents_providers::{MockBackend, MockResponse};
    use rust_ai_agents_tools::{Tool, ToolRegistry};
    use serde_json::json;

    use crate::engine::AgentEngine;

    /// Simple calculator tool for testing
    struct CalculatorTool;

    #[async_trait::async_trait]
    impl Tool for CalculatorTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "calculate".to_string(),
                description: "Performs basic arithmetic".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }),
                dangerous: false,
                metadata: HashMap::new(),
            }
        }

        async fn execute(
            &self,
            _context: &ExecutionContext,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, ToolError> {
            let expr = input["expression"]
                .as_str()
                .ok_or_else(|| ToolError::ExecutionFailed("missing expression".into()))?;

            // Simple evaluation for test
            let result = match expr {
                "2 + 2" => 4,
                "10 * 5" => 50,
                "100 / 4" => 25,
                _ => 0,
            };

            Ok(json!({ "result": result }))
        }
    }

    /// Echo tool that returns its input
    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "echo".to_string(),
                description: "Echoes back the input".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                }),
                dangerous: false,
                metadata: HashMap::new(),
            }
        }

        async fn execute(
            &self,
            _context: &ExecutionContext,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, ToolError> {
            let message = input["message"].as_str().unwrap_or("no message");
            Ok(json!({ "echoed": message }))
        }
    }

    fn create_test_config(id: &str) -> AgentConfig {
        AgentConfig {
            id: AgentId::new(id),
            name: format!("Test Agent {}", id),
            role: AgentRole::Executor,
            capabilities: vec![Capability::Analysis],
            system_prompt: Some("You are a helpful test agent.".to_string()),
            max_iterations: 10,
            timeout_secs: 30,
            temperature: 0.0,
            planning_mode: PlanningMode::Disabled,
            stop_words: vec![],
            memory_config: MemoryConfig::new(1024 * 1024),
            routing_strategy: RoutingStrategy::Direct,
        }
    }

    #[tokio::test]
    async fn test_simple_text_response() {
        // Setup: Mock returns a simple text response
        let backend = Arc::new(
            MockBackend::new().with_response(MockResponse::text("Hello! How can I help you?")),
        );

        let registry = Arc::new(ToolRegistry::new());
        let engine = AgentEngine::new();

        let config = create_test_config("agent-1");
        let agent_id = engine
            .spawn_agent(config.clone(), registry, backend)
            .await
            .unwrap();

        // Send message
        let message = Message::new(
            AgentId::new("user"),
            agent_id.clone(),
            Content::Text("Hi there!".to_string()),
        );
        engine.send_message(message).unwrap();

        // Wait for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify metrics
        let metrics = engine.metrics();
        assert_eq!(metrics.messages_processed, 1);
        assert_eq!(metrics.messages_failed, 0);
    }

    #[tokio::test]
    async fn test_tool_execution_react_loop() {
        // Setup: Mock first returns a tool call, then a final response
        let backend = Arc::new(
            MockBackend::new()
                .with_response(MockResponse::tool_call(
                    "calculate",
                    json!({"expression": "2 + 2"}),
                ))
                .with_response(MockResponse::text("The result of 2 + 2 is 4.")),
        );

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(CalculatorTool));
        let registry = Arc::new(registry);

        let engine = AgentEngine::new();

        let config = create_test_config("agent-2");
        let agent_id = engine
            .spawn_agent(config.clone(), registry, backend)
            .await
            .unwrap();

        // Send message
        let message = Message::new(
            AgentId::new("user"),
            agent_id.clone(),
            Content::Text("What is 2 + 2?".to_string()),
        );
        engine.send_message(message).unwrap();

        // Wait for processing (tool execution + final response)
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Verify metrics
        let metrics = engine.metrics();
        assert_eq!(metrics.messages_processed, 1);
        assert_eq!(metrics.total_tool_calls, 1);
    }

    #[tokio::test]
    async fn test_multiple_tool_calls() {
        // Mock returns multiple tool calls in sequence
        let backend = Arc::new(
            MockBackend::new()
                .with_response(MockResponse::tool_call("echo", json!({"message": "first"})))
                .with_response(MockResponse::tool_call(
                    "echo",
                    json!({"message": "second"}),
                ))
                .with_response(MockResponse::text("Done with both calls.")),
        );

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let registry = Arc::new(registry);

        let engine = AgentEngine::new();

        let config = create_test_config("agent-3");
        let agent_id = engine
            .spawn_agent(config.clone(), registry, backend)
            .await
            .unwrap();

        let message = Message::new(
            AgentId::new("user"),
            agent_id.clone(),
            Content::Text("Echo two messages".to_string()),
        );
        engine.send_message(message).unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let metrics = engine.metrics();
        assert_eq!(metrics.messages_processed, 1);
        assert_eq!(metrics.total_tool_calls, 2);
    }

    #[tokio::test]
    async fn test_echo_mode() {
        // Echo mode returns the user's message
        let backend = Arc::new(MockBackend::echo());
        let registry = Arc::new(ToolRegistry::new());
        let engine = AgentEngine::new();

        let config = create_test_config("agent-4");
        let agent_id = engine
            .spawn_agent(config.clone(), registry, backend)
            .await
            .unwrap();

        let message = Message::new(
            AgentId::new("user"),
            agent_id.clone(),
            Content::Text("Echo this back to me".to_string()),
        );
        engine.send_message(message).unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let metrics = engine.metrics();
        assert_eq!(metrics.messages_processed, 1);
    }

    #[tokio::test]
    async fn test_stop_word_detection() {
        // Response contains a stop word
        let backend =
            Arc::new(MockBackend::new().with_response(MockResponse::text("Task complete. DONE.")));

        let registry = Arc::new(ToolRegistry::new());
        let engine = AgentEngine::new();

        let mut config = create_test_config("agent-5");
        config.stop_words = vec!["DONE".to_string()];

        let agent_id = engine
            .spawn_agent(config.clone(), registry, backend)
            .await
            .unwrap();

        let message = Message::new(
            AgentId::new("user"),
            agent_id.clone(),
            Content::Text("Complete the task".to_string()),
        );
        engine.send_message(message).unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check agent state
        let agent = engine.get_agent(&agent_id).unwrap();
        let state = agent.state.read().await;
        assert_eq!(state.status, AgentStatus::StoppedByStopWord);
    }

    #[tokio::test]
    async fn test_max_iterations_exceeded() {
        // Mock always returns tool calls, never a final response
        let backend = Arc::new(
            MockBackend::new()
                .with_response(MockResponse::tool_call("echo", json!({"message": "loop"})))
                .with_response(MockResponse::tool_call("echo", json!({"message": "loop"})))
                .with_response(MockResponse::tool_call("echo", json!({"message": "loop"})))
                .with_response(MockResponse::tool_call("echo", json!({"message": "loop"})))
                .with_response(MockResponse::tool_call("echo", json!({"message": "loop"}))),
        );

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let registry = Arc::new(registry);

        let engine = AgentEngine::new();

        let mut config = create_test_config("agent-6");
        config.max_iterations = 3; // Low limit

        let agent_id = engine
            .spawn_agent(config.clone(), registry, backend)
            .await
            .unwrap();

        let message = Message::new(
            AgentId::new("user"),
            agent_id.clone(),
            Content::Text("Keep looping".to_string()),
        );
        engine.send_message(message).unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let metrics = engine.metrics();
        assert_eq!(metrics.messages_failed, 1); // Should fail due to max iterations
    }

    #[tokio::test]
    async fn test_multiple_agents() {
        let backend1 = Arc::new(MockBackend::new().with_response(MockResponse::text("Agent 1")));
        let backend2 = Arc::new(MockBackend::new().with_response(MockResponse::text("Agent 2")));

        let registry = Arc::new(ToolRegistry::new());
        let engine = AgentEngine::new();

        let config1 = create_test_config("agent-a");
        let config2 = create_test_config("agent-b");

        let id1 = engine
            .spawn_agent(config1, registry.clone(), backend1)
            .await
            .unwrap();
        let id2 = engine
            .spawn_agent(config2, registry.clone(), backend2)
            .await
            .unwrap();

        assert_eq!(engine.agent_count(), 2);

        // Send messages to both
        engine
            .send_message(Message::new(
                AgentId::new("user"),
                id1.clone(),
                Content::Text("Hello agent 1".to_string()),
            ))
            .unwrap();

        engine
            .send_message(Message::new(
                AgentId::new("user"),
                id2.clone(),
                Content::Text("Hello agent 2".to_string()),
            ))
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let metrics = engine.metrics();
        assert_eq!(metrics.messages_processed, 2);
        assert_eq!(metrics.agents_spawned, 2);
        assert_eq!(metrics.agents_active, 2);
    }

    #[tokio::test]
    async fn test_agent_shutdown() {
        let backend = Arc::new(MockBackend::new().with_response(MockResponse::text("Goodbye")));
        let registry = Arc::new(ToolRegistry::new());
        let engine = AgentEngine::new();

        let config = create_test_config("agent-shutdown");
        let agent_id = engine.spawn_agent(config, registry, backend).await.unwrap();

        assert_eq!(engine.agent_count(), 1);

        engine.stop_agent(&agent_id).await.unwrap();

        assert_eq!(engine.agent_count(), 0);

        let metrics = engine.metrics();
        assert_eq!(metrics.agents_active, 0);
    }

    #[tokio::test]
    async fn test_send_to_nonexistent_agent() {
        let engine = AgentEngine::new();

        let message = Message::new(
            AgentId::new("user"),
            AgentId::new("nonexistent"),
            Content::Text("Hello?".to_string()),
        );

        let result = engine.send_message(message);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_recorded_calls() {
        use rust_ai_agents_providers::MockConfig;

        let config = MockConfig {
            record_calls: true,
            ..Default::default()
        };

        let backend = MockBackend::with_config(config)
            .with_response(MockResponse::text("Response 1"))
            .with_response(MockResponse::text("Response 2"));

        let backend = Arc::new(backend);

        let registry = Arc::new(ToolRegistry::new());
        let engine = AgentEngine::new();

        let agent_config = create_test_config("agent-record");
        let agent_id = engine
            .spawn_agent(agent_config, registry, backend.clone())
            .await
            .unwrap();

        engine
            .send_message(Message::new(
                AgentId::new("user"),
                agent_id.clone(),
                Content::Text("First message".to_string()),
            ))
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        engine
            .send_message(Message::new(
                AgentId::new("user"),
                agent_id.clone(),
                Content::Text("Second message".to_string()),
            ))
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check recorded calls
        let calls = backend.get_recorded_calls();
        assert_eq!(calls.len(), 2);
    }

    #[tokio::test]
    async fn test_success_rate_calculation() {
        let backend = Arc::new(MockBackend::new().with_response(MockResponse::text("Success")));
        let registry = Arc::new(ToolRegistry::new());
        let engine = AgentEngine::new();

        let config = create_test_config("agent-rate");
        let agent_id = engine.spawn_agent(config, registry, backend).await.unwrap();

        // Send 3 successful messages
        for i in 0..3 {
            engine
                .send_message(Message::new(
                    AgentId::new("user"),
                    agent_id.clone(),
                    Content::Text(format!("Message {}", i)),
                ))
                .unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let metrics = engine.metrics();
        assert_eq!(metrics.success_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_engine_shutdown_all() {
        let backend = Arc::new(MockBackend::new().with_response(MockResponse::text("Hi")));
        let registry = Arc::new(ToolRegistry::new());
        let engine = AgentEngine::new();

        // Spawn multiple agents
        for i in 0..3 {
            let config = create_test_config(&format!("agent-{}", i));
            engine
                .spawn_agent(config, registry.clone(), backend.clone())
                .await
                .unwrap();
        }

        assert_eq!(engine.agent_count(), 3);

        // Shutdown all
        engine.shutdown().await;

        assert_eq!(engine.agent_count(), 0);
    }
}
