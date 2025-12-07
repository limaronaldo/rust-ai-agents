## Description

Document the approval system, including `ToolSchema.dangerous` integration and the Safety Sandwich pattern.

## Content Outline

### 1. Overview
- Why human-in-the-loop?
- When to use approvals
- The Safety Sandwich pattern

### 2. Quick Start
```rust
use rust_ai_agents::approval::{ApprovalConfig, TerminalApprovalHandler};

let config = EngineConfig {
    approval: Some(ApprovalConfig {
        handler: Arc::new(TerminalApprovalHandler::new()),
        auto_approve_timeout: Some(Duration::from_secs(300)),
        ..Default::default()
    }),
    ..Default::default()
};

let engine = AgentEngine::new(backend, config);
```

### 3. Marking Tools as Dangerous
```rust
ToolSchema::new("delete_user", "Delete a user account")
    .with_dangerous(true)  // Will require approval
```

### 4. The Safety Sandwich

```
┌─────────────────────────────────────────┐
│           SAFETY SANDWICH               │
├─────────────────────────────────────────┤
│ 1. Guardrails (input validation)        │
│         ↓                               │
│ 2. Approvals (if tool.dangerous=true)   │
│         ↓                               │
│ 3. Executor (run tool)                  │
│         ↓                               │
│ 4. Guardrails (output validation)       │
└─────────────────────────────────────────┘
```

### 5. Available Handlers

| Handler | Use Case |
|---------|----------|
| `TerminalApprovalHandler` | CLI/dev interactive |
| `TestApprovalHandler` | Automated testing |
| Custom | Slack, webhook, queue, etc. |

### 6. Configuration Options

```rust
ApprovalConfig {
    // Tools beyond dangerous=true that need approval
    additional_sensitive_tools: vec!["send_email".into()],
    
    // Require approval for external API calls
    require_approval_for_external: true,
    
    // Auto-approve after timeout (None = wait forever)
    auto_approve_timeout: Some(Duration::from_secs(300)),
    
    handler: Arc::new(TerminalApprovalHandler::new()),
}
```

### 7. Creating a Custom Handler
```rust
#[async_trait]
impl ApprovalHandler for SlackApprovalHandler {
    async fn request_approval(&self, request: ApprovalRequest) 
        -> Result<ApprovalDecision, ApprovalError> 
    {
        // Send Slack message
        // Wait for button click
        // Return decision
    }
}
```

### 8. Testing Approvals
```rust
let handler = TestApprovalHandler::always_approve();
// or
let handler = TestApprovalHandler::always_reject("test");
// or
let handler = TestApprovalHandler::always_modify(json!({"safe": true}));

handler.assert_request_count(1);
handler.assert_tool_requested("dangerous_tool");
```

## Checklist

- [ ] Create `docs/approvals.md`
- [ ] Document ToolSchema.dangerous usage
- [ ] Include Safety Sandwich diagram
- [ ] Document all handlers
- [ ] Add configuration examples
- [ ] Add custom handler example
- [ ] Add testing section
- [ ] Link from README.md and ROADMAP.md
