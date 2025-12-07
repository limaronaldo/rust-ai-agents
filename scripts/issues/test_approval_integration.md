## Description

Comprehensive tests for all approval flows using `MockBackend` + `TestApprovalHandler`.

## Test Scenarios

### 1. `test_dangerous_tool_requires_approval`
```rust
#[tokio::test]
async fn test_dangerous_tool_requires_approval() {
    let backend = MockBackend::tool_then_text(
        "delete_all_data",
        json!({"confirm": true}),
        "Data deleted",
    );
    
    let handler = TestApprovalHandler::always_approve();
    let config = EngineConfig {
        approval: Some(ApprovalConfig {
            handler: Arc::new(handler.clone()),
            ..Default::default()
        }),
        ..Default::default()
    };
    
    // Register dangerous tool
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(DangerousTool::new())); // dangerous=true
    
    let engine = AgentEngine::new(Arc::new(backend), config).with_tools(tools);
    let result = engine.run("Delete all data").await.unwrap();
    
    // Verify approval was requested
    handler.assert_request_count(1);
    assert!(handler.requests()[0].is_dangerous);
}
```

### 2. `test_rejected_tool_not_executed`
```rust
#[tokio::test]
async fn test_rejected_tool_not_executed() {
    let handler = TestApprovalHandler::always_reject("User declined");
    // ... setup ...
    
    let result = engine.run("Delete all data").await;
    
    // Should fail with rejection
    assert!(matches!(result, Err(EngineError::ToolError(
        ToolError::ApprovalRejected(_)
    ))));
    
    // Verify tool was NOT executed (check trajectory)
}
```

### 3. `test_modified_args_used`
```rust
#[tokio::test]
async fn test_modified_args_used() {
    let modified_args = json!({"safe_mode": true});
    let handler = TestApprovalHandler::always_modify(modified_args.clone());
    // ... verify tool receives modified_args ...
}
```

### 4. `test_non_dangerous_tool_no_approval`
```rust
#[tokio::test]
async fn test_non_dangerous_tool_no_approval() {
    let handler = TestApprovalHandler::always_approve();
    // Register safe tool (dangerous=false)
    // ... run engine ...
    
    // No approval should be requested
    handler.assert_request_count(0);
}
```

### 5. `test_auto_approve_timeout`
```rust
#[tokio::test]
async fn test_auto_approve_timeout() {
    // Handler that never responds
    let handler = SlowApprovalHandler::new(Duration::from_secs(60));
    
    let config = ApprovalConfig {
        auto_approve_timeout: Some(Duration::from_millis(100)),
        handler: Arc::new(handler),
        ..Default::default()
    };
    
    // Should auto-approve after 100ms
    let result = engine.run("...").await;
    assert!(result.is_ok());
}
```

## Checklist

- [ ] Create `tests/approval_integration.rs`
- [ ] Implement all 5 test scenarios
- [ ] Verify trajectory events are emitted correctly
- [ ] Test edge cases (empty decisions, handler errors)
- [ ] Test with additional_sensitive_tools config
