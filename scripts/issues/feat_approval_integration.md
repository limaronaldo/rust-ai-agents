## Description

Integrate `ApprovalConfig` into the tool execution flow, implementing the "Safety Sandwich" pattern.

## Safety Sandwich Architecture

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

## Integration Points

The approval check should happen in `engine.rs` or `executor.rs`, after input guardrails but before tool execution.

## Acceptance Criteria

- [ ] Approval check triggers for `ToolSchema.dangerous = true`
- [ ] Approval check triggers for tools in `additional_sensitive_tools`
- [ ] `Approved` → tool executes normally
- [ ] `Rejected` → tool NOT executed, error returned
- [ ] `Modified` → tool executes with `new_args`
- [ ] Trajectory events emitted for observability
- [ ] `auto_approve_timeout` respected

## Implementation

```rust
// In engine.rs or executor.rs, before tool execution:

async fn execute_tool_with_approval(
    &self,
    tool: &dyn Tool,
    schema: &ToolSchema,
    arguments: serde_json::Value,
    context: &ExecutionContext,
) -> Result<serde_json::Value, ToolError> {
    // Check if approval needed
    if let Some(ref approval_config) = self.config.approval {
        if approval_config.requires_approval(schema) {
            // Emit trajectory event
            self.emit_event(AgentEvent::ApprovalRequested {
                tool: schema.name.clone(),
                request_id: uuid::Uuid::new_v4().to_string(),
                is_dangerous: schema.dangerous,
            });
            
            // Request approval (with optional timeout)
            let request = ApprovalRequest {
                id: uuid::Uuid::new_v4().to_string(),
                tool_name: schema.name.clone(),
                tool_description: schema.description.clone(),
                arguments: arguments.clone(),
                context: None,
                timestamp: now(),
                is_dangerous: schema.dangerous,
            };
            
            let decision = if let Some(timeout) = approval_config.auto_approve_timeout {
                tokio::time::timeout(timeout, approval_config.handler.request_approval(request))
                    .await
                    .unwrap_or(Ok(ApprovalDecision::Approved))?
            } else {
                approval_config.handler.request_approval(request).await?
            };
            
            // Emit decision event
            self.emit_event(AgentEvent::ApprovalDecision {
                request_id: request.id.clone(),
                decision: format!("{:?}", decision),
            });
            
            match decision {
                ApprovalDecision::Approved => {
                    // Continue with original args
                }
                ApprovalDecision::Rejected { reason } => {
                    return Err(ToolError::ApprovalRejected(reason));
                }
                ApprovalDecision::Modified { new_args } => {
                    // Use modified args
                    return tool.execute(context, new_args).await;
                }
            }
        }
    }
    
    // Execute tool
    tool.execute(context, arguments).await
}
```

## Trajectory Events to Add

```rust
// In trajectory.rs, add to AgentEvent enum:

pub enum AgentEvent {
    // ... existing variants ...
    
    /// Approval requested for a tool
    ApprovalRequested {
        tool: String,
        request_id: String,
        is_dangerous: bool,
    },
    
    /// Approval decision received
    ApprovalDecision {
        request_id: String,
        decision: String, // "Approved", "Rejected", "Modified"
    },
}
```

## Checklist

- [ ] Add `Option<ApprovalConfig>` to `EngineConfig` or similar
- [ ] Implement approval check before tool execution
- [ ] Handle all three `ApprovalDecision` variants
- [ ] Implement timeout with `tokio::time::timeout`
- [ ] Add `ApprovalRequested` event to `trajectory.rs`
- [ ] Add `ApprovalDecision` event to `trajectory.rs`
- [ ] Add `ToolError::ApprovalRejected` variant
- [ ] Document the safety sandwich pattern in code comments
- [ ] Add integration test with real approval flow
