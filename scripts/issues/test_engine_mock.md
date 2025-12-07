## Description

Comprehensive engine tests using `MockBackend` to validate the ReACT loop without external API calls.

## Test Scenarios

### 1. `test_engine_simple_response`
Direct text response, no tool calls.

```rust
#[tokio::test]
async fn test_engine_simple_response() {
    let backend = MockBackend::text("Hello! I'm here to help.");
    let engine = AgentEngine::new(Arc::new(backend), Default::default());
    
    let result = engine.run("Say hello").await.unwrap();
    
    assert!(result.contains("Hello"));
}
```

### 2. `test_engine_tool_call_flow`
Single tool call followed by response.

```rust
#[tokio::test]
async fn test_engine_tool_call_flow() {
    let backend = MockBackend::new(vec![
        MockStep::ToolCall {
            name: "search_entity".into(),
            arguments: json!({"query": "ACME Corp"}),
        },
        MockStep::Text("Found: ACME Corp, founded in 1990.".into()),
    ]);
    
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(MockSearchTool::new()));
    
    let engine = AgentEngine::new(Arc::new(backend), config).with_tools(tools);
    let result = engine.run("Search for ACME Corp").await.unwrap();
    
    assert!(result.contains("ACME Corp"));
}
```

### 3. `test_engine_multiple_tool_calls`
Sequential tool calls.

```rust
#[tokio::test]
async fn test_engine_multiple_tool_calls() {
    let backend = MockBackend::new(vec![
        MockStep::ToolCall { name: "step1".into(), arguments: json!({}) },
        MockStep::ToolCall { name: "step2".into(), arguments: json!({}) },
        MockStep::Text("Completed both steps.".into()),
    ]);
    // ...
}
```

### 4. `test_engine_tool_error_handling`
Tool returns an error.

```rust
#[tokio::test]
async fn test_engine_tool_error_handling() {
    // Backend returns tool call, but tool execution fails
    let backend = MockBackend::tool_then_text("failing_tool", json!({}), "...");
    
    // Register tool that returns error
    tools.register(Arc::new(FailingTool::new()));
    
    // Engine should handle gracefully
}
```

### 5. `test_engine_max_iterations`
Ensure iteration limit is respected.

```rust
#[tokio::test]
async fn test_engine_max_iterations() {
    // Backend always returns tool calls (infinite loop)
    let backend = MockBackend::new(vec![
        MockStep::ToolCall { name: "loop".into(), arguments: json!({}) },
    ]);
    
    let config = EngineConfig {
        max_iterations: 5,
        ..Default::default()
    };
    
    let result = engine.run("...").await;
    // Should stop after 5 iterations
}
```

## Checklist

- [ ] Create `tests/engine_mock_integration.rs`
- [ ] Implement all 5 test scenarios
- [ ] Verify trajectory events are recorded correctly
- [ ] Test edge cases (empty response, malformed tool call)
- [ ] Ensure tests are fast (no real API calls)
