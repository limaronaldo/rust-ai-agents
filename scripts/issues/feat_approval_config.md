## Description

Introduce human approval system for sensitive tools, leveraging the existing `ToolSchema.dangerous` flag from `crates/core/src/tool.rs`.

## Key Design Decision

Use existing `ToolSchema.dangerous: bool` to trigger approvals:
```rust
// Already exists in crates/core/src/tool.rs
pub struct ToolSchema {
    pub dangerous: bool,  // <- Use this!
    pub metadata: HashMap<String, serde_json::Value>,
}
```

## Implementation

```rust
// crates/agents/src/approval/mod.rs

use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use rust_ai_agents_core::ToolSchema;

/// Configuration for the approval system
pub struct ApprovalConfig {
    /// Additional tools requiring approval (beyond those with dangerous=true)
    pub additional_sensitive_tools: Vec<String>,
    
    /// Require approval for ALL external API calls
    pub require_approval_for_external: bool,
    
    /// Auto-approve after timeout (None = wait forever)
    pub auto_approve_timeout: Option<Duration>,
    
    /// The approval handler
    pub handler: Arc<dyn ApprovalHandler>,
}

impl ApprovalConfig {
    /// Check if a tool requires approval
    pub fn requires_approval(&self, schema: &ToolSchema) -> bool {
        schema.dangerous 
            || self.additional_sensitive_tools.contains(&schema.name)
            || (self.require_approval_for_external 
                && schema.metadata.get("external").and_then(|v| v.as_bool()).unwrap_or(false))
    }
}

#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    async fn request_approval(&self, request: ApprovalRequest) -> Result<ApprovalDecision, ApprovalError>;
    async fn on_status_change(&self, request_id: &str, status: ApprovalStatus);
}

pub struct ApprovalRequest {
    pub id: String,
    pub tool_name: String,
    pub tool_description: String,
    pub arguments: serde_json::Value,
    pub context: Option<String>,
    pub timestamp: u64,
    pub is_dangerous: bool,
}

pub enum ApprovalDecision {
    Approved,
    Rejected { reason: String },
    Modified { new_args: serde_json::Value },
}

pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    TimedOut,
}

#[derive(Debug, thiserror::Error)]
pub enum ApprovalError {
    #[error("Approval timed out after {0:?}")]
    Timeout(Duration),
    
    #[error("Handler error: {0}")]
    HandlerError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
```

## Checklist

- [ ] Create `crates/agents/src/approval/mod.rs`
- [ ] Create `crates/agents/src/approval/config.rs` (optional, can be in mod.rs)
- [ ] Implement `ApprovalConfig::requires_approval()` using `ToolSchema.dangerous`
- [ ] Define `ApprovalHandler` trait
- [ ] Define `ApprovalRequest`, `ApprovalDecision`, `ApprovalStatus`
- [ ] Define `ApprovalError`
- [ ] Add `pub mod approval;` to `crates/agents/src/lib.rs`
- [ ] Re-export key types from `lib.rs`
- [ ] Add unit tests for `requires_approval()` logic
