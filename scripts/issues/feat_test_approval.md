## Description

Create a test-friendly approval handler with pre-configured decisions for automated testing.

## Implementation

```rust
// crates/agents/src/approval/test_handler.rs

use super::{ApprovalHandler, ApprovalRequest, ApprovalDecision, ApprovalError, ApprovalStatus};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Approval handler for tests with pre-configured decisions
pub struct TestApprovalHandler {
    decisions: Vec<ApprovalDecision>,
    current: AtomicUsize,
    requests: Mutex<Vec<ApprovalRequest>>,
}

impl TestApprovalHandler {
    /// Create with a sequence of decisions
    pub fn new(decisions: Vec<ApprovalDecision>) -> Self {
        Self {
            decisions,
            current: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
        }
    }
    
    /// Always approve
    pub fn always_approve() -> Self {
        Self::new(vec![ApprovalDecision::Approved])
    }
    
    /// Always reject with given reason
    pub fn always_reject(reason: impl Into<String>) -> Self {
        Self::new(vec![ApprovalDecision::Rejected { reason: reason.into() }])
    }
    
    /// Always modify with given args
    pub fn always_modify(new_args: serde_json::Value) -> Self {
        Self::new(vec![ApprovalDecision::Modified { new_args }])
    }
    
    /// Get all requests that were made
    pub fn requests(&self) -> Vec<ApprovalRequest> {
        self.requests.lock().clone()
    }
    
    /// Get count of approval requests
    pub fn request_count(&self) -> usize {
        self.requests.lock().len()
    }
    
    /// Assert that exactly N approvals were requested
    pub fn assert_request_count(&self, expected: usize) {
        let actual = self.request_count();
        assert_eq!(
            actual, expected,
            "Expected {} approval requests, got {}",
            expected, actual
        );
    }
    
    /// Assert that a specific tool was requested for approval
    pub fn assert_tool_requested(&self, tool_name: &str) {
        let requests = self.requests.lock();
        assert!(
            requests.iter().any(|r| r.tool_name == tool_name),
            "Expected tool '{}' to be requested for approval",
            tool_name
        );
    }
}

#[async_trait]
impl ApprovalHandler for TestApprovalHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> Result<ApprovalDecision, ApprovalError> {
        // Record the request
        self.requests.lock().push(request);
        
        // Return next decision (cycling if needed)
        let idx = self.current.fetch_add(1, Ordering::SeqCst);
        let decision = self.decisions
            .get(idx % self.decisions.len())
            .cloned()
            .unwrap_or(ApprovalDecision::Approved);
        
        Ok(decision)
    }
    
    async fn on_status_change(&self, _request_id: &str, _status: ApprovalStatus) {
        // No-op for tests
    }
}
```

## Checklist

- [ ] Create `crates/agents/src/approval/test_handler.rs`
- [ ] Implement pre-configured decision sequences
- [ ] Add request tracking with assertion helpers
- [ ] Add `pub mod test_handler;` to approval/mod.rs
- [ ] Add unit tests for the handler itself
- [ ] Export for use in integration tests
