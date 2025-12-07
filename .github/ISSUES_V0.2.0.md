# Issues for Milestone v0.2.0 – LLM Backends & Approvals (REVISED)

**Last Updated:** December 7, 2025

This document contains revised issues for the v0.2.0 milestone, accounting for existing code in:
- `crates/providers/` - Already has `LLMBackend` trait, OpenAI, Anthropic, OpenRouter backends
- `crates/core/` - Already has `LLMMessage`, `ToolCall`, `ToolSchema` with `dangerous` flag

---

## Existing Infrastructure (No Changes Needed)

| Component | Location | Status |
|-----------|----------|--------|
| `LLMBackend` trait | `crates/providers/src/backend.rs` | ✅ Exists |
| `InferenceOutput`, `TokenUsage` | `crates/providers/src/backend.rs` | ✅ Exists |
| `OpenAIBackend` | `crates/providers/src/openai.rs` | ✅ Exists |
| `AnthropicBackend` | `crates/providers/src/anthropic.rs` | ✅ Exists |
| `LLMMessage`, `MessageRole` | `crates/core/src/message.rs` | ✅ Exists |
| `ToolCall`, `ToolResult` | `crates/core/src/message.rs` | ✅ Exists |
| `ToolSchema` with `dangerous` flag | `crates/core/src/tool.rs` | ✅ Exists |
| `Tool` trait, `ToolRegistry` | `crates/core/src/tool.rs` | ✅ Exists |
| Streaming (`infer_stream`) | `crates/providers/src/backend.rs` | ✅ Exists |
| `RateLimiter` | `crates/providers/src/backend.rs` | ✅ Exists |

---

## Revised Issues for v0.2.0

### Issue 1: feat: implement MockBackend for tests

**Labels:** `enhancement`, `testing`, `priority-high`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Create a `MockBackend` that implements the existing `LLMBackend` trait for deterministic testing without API calls.

#### Acceptance Criteria

- [ ] `MockBackend` implements `LLMBackend` from `crates/providers`
- [ ] Configurable scripts (tool call → response sequences)
- [ ] Engine tests can use `MockBackend` instead of real providers

#### Implementation

```rust
// crates/providers/src/mock.rs

use crate::{LLMBackend, InferenceOutput, TokenUsage};
use rust_ai_agents_core::{LLMMessage, ToolSchema, ToolCall};
use std::sync::atomic::{AtomicUsize, Ordering};

pub enum MockStep {
    /// Return a text response
    Text(String),
    /// Return a tool call
    ToolCall { name: String, arguments: serde_json::Value },
    /// Return an error
    Error(LLMError),
}

pub struct MockBackend {
    steps: Vec<MockStep>,
    current: AtomicUsize,
    model_info: ModelInfo,
}

impl MockBackend {
    pub fn new(steps: Vec<MockStep>) -> Self;
    
    /// Single text response
    pub fn text(response: impl Into<String>) -> Self;
    
    /// Tool call then text response
    pub fn tool_then_text(tool: &str, args: Value, response: impl Into<String>) -> Self;
    
    /// Always return error
    pub fn error(error: LLMError) -> Self;
}

#[async_trait]
impl LLMBackend for MockBackend {
    async fn infer(&self, messages: &[LLMMessage], tools: &[ToolSchema], temperature: f32) 
        -> Result<InferenceOutput, LLMError>;
    
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError>;
    
    fn model_info(&self) -> ModelInfo;
}
```

#### Checklist

- [ ] Create `crates/providers/src/mock.rs`
- [ ] Implement script-based `MockBackend`
- [ ] Add convenience constructors (`text()`, `tool_then_text()`, `error()`)
- [ ] Add to `crates/providers/src/lib.rs` exports
- [ ] Add tests in `crates/providers/src/mock.rs`
- [ ] Update some engine tests to use `MockBackend`

---

### Issue 2: refactor: wire engine.rs to use LLMBackend from providers

**Labels:** `refactor`, `core`, `priority-high`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Ensure `crates/agents/src/engine.rs` uses `LLMBackend` trait from `crates/providers` instead of any hardcoded provider logic.

#### Current State Analysis Needed

- [ ] Check if `engine.rs` already uses `LLMBackend` trait
- [ ] Identify any provider-specific code that should be abstracted
- [ ] Ensure `Arc<dyn LLMBackend>` is injectable

#### Acceptance Criteria

- [ ] `AgentEngine` accepts `Arc<dyn LLMBackend>` in constructor
- [ ] No direct OpenAI/Anthropic imports in `engine.rs`
- [ ] Tests can inject `MockBackend`

#### Checklist

- [ ] Audit `engine.rs` for provider coupling
- [ ] Refactor to use `Arc<dyn LLMBackend>`
- [ ] Update engine initialization in examples
- [ ] Verify all engine tests pass with `MockBackend`

---

### Issue 3: feat: ApprovalConfig and ApprovalHandler (human-in-the-loop)

**Labels:** `enhancement`, `safety`, `priority-high`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Introduce human approval system for sensitive tools. Leverage existing `ToolSchema.dangerous` flag.

#### Key Decision: Use Existing `dangerous` Flag

The `ToolSchema` in `crates/core/src/tool.rs` already has:
```rust
pub struct ToolSchema {
    // ...
    /// Whether tool is dangerous and requires confirmation
    pub dangerous: bool,
    /// Tool metadata
    pub metadata: HashMap<String, serde_json::Value>,
}
```

We should use `dangerous: true` to trigger approvals, plus `metadata` for additional config.

#### Implementation

```rust
// crates/agents/src/approval/mod.rs

use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;

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
                && schema.metadata.get("external").map(|v| v.as_bool()).flatten().unwrap_or(false))
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
```

#### Checklist

- [ ] Create `crates/agents/src/approval/mod.rs`
- [ ] Create `crates/agents/src/approval/config.rs`
- [ ] Implement `ApprovalConfig::requires_approval()` using `ToolSchema.dangerous`
- [ ] Define `ApprovalHandler` trait
- [ ] Define `ApprovalRequest`, `ApprovalDecision`, `ApprovalStatus`
- [ ] Add `ApprovalError` to `crates/core/src/errors.rs`
- [ ] Export from `crates/agents/src/lib.rs`
- [ ] Add unit tests

---

### Issue 4: feat: implement TerminalApprovalHandler

**Labels:** `enhancement`, `safety`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Implement CLI approval handler for development and debugging.

#### Implementation

```rust
// crates/agents/src/approval/terminal.rs

pub struct TerminalApprovalHandler {
    pub verbose: bool,
}

impl TerminalApprovalHandler {
    pub fn new() -> Self { Self { verbose: false } }
    pub fn verbose() -> Self { Self { verbose: true } }
}

#[async_trait]
impl ApprovalHandler for TerminalApprovalHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> Result<ApprovalDecision, ApprovalError> {
        println!("\n╔══════════════════════════════════════════════╗");
        println!("║           APPROVAL REQUIRED                   ║");
        println!("╠══════════════════════════════════════════════╣");
        println!("║ Tool: {:<39} ║", request.tool_name);
        if request.is_dangerous {
            println!("║ ⚠️  DANGEROUS TOOL                            ║");
        }
        println!("╠══════════════════════════════════════════════╣");
        if self.verbose {
            println!("Arguments:\n{}", serde_json::to_string_pretty(&request.arguments)?);
        }
        println!("╚══════════════════════════════════════════════╝");
        println!("\nApprove? [y]es / [n]o / [m]odify: ");
        
        // Read from stdin...
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => Ok(ApprovalDecision::Approved),
            "n" | "no" => Ok(ApprovalDecision::Rejected { 
                reason: "User rejected".into() 
            }),
            "m" | "modify" => {
                // Prompt for new JSON
                println!("Enter new arguments (JSON):");
                let mut json_input = String::new();
                std::io::stdin().read_line(&mut json_input)?;
                let new_args: serde_json::Value = serde_json::from_str(&json_input)?;
                Ok(ApprovalDecision::Modified { new_args })
            }
            _ => Ok(ApprovalDecision::Rejected { 
                reason: "Invalid input".into() 
            }),
        }
    }
    
    async fn on_status_change(&self, request_id: &str, status: ApprovalStatus) {
        println!("[Approval {}] Status: {:?}", request_id, status);
    }
}
```

#### Checklist

- [ ] Create `crates/agents/src/approval/terminal.rs`
- [ ] Implement stdin reading with nice formatting
- [ ] Handle Ctrl+C / EOF gracefully
- [ ] Create `examples/human_in_loop.rs`
- [ ] Test with a mock "dangerous" tool

---

### Issue 5: feat: implement TestApprovalHandler for automated tests

**Labels:** `enhancement`, `testing`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Create a test-friendly approval handler that can be pre-configured with decisions.

#### Implementation

```rust
// crates/agents/src/approval/test_handler.rs

use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TestApprovalHandler {
    decisions: Vec<ApprovalDecision>,
    current: AtomicUsize,
    requests: parking_lot::Mutex<Vec<ApprovalRequest>>,
}

impl TestApprovalHandler {
    pub fn new(decisions: Vec<ApprovalDecision>) -> Self;
    
    pub fn always_approve() -> Self {
        Self::new(vec![ApprovalDecision::Approved])
    }
    
    pub fn always_reject(reason: &str) -> Self {
        Self::new(vec![ApprovalDecision::Rejected { 
            reason: reason.to_string() 
        }])
    }
    
    pub fn always_modify(new_args: Value) -> Self {
        Self::new(vec![ApprovalDecision::Modified { new_args }])
    }
    
    /// Get all requests that were made
    pub fn requests(&self) -> Vec<ApprovalRequest> {
        self.requests.lock().clone()
    }
    
    /// Assert that N approvals were requested
    pub fn assert_request_count(&self, expected: usize) {
        assert_eq!(self.requests.lock().len(), expected);
    }
}
```

#### Checklist

- [ ] Create `crates/agents/src/approval/test_handler.rs`
- [ ] Implement pre-configured decision sequences
- [ ] Add request tracking for assertions
- [ ] Export for use in tests

---

### Issue 6: feat: integrate approvals into executor

**Labels:** `enhancement`, `safety`, `priority-high`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Integrate `ApprovalConfig` into the tool execution flow, between guardrails and actual execution.

#### Flow Diagram

```
User → Engine (ReACT) → ToolCall
            ↓
       guardrails.rs (validate input)
            ↓
       approval.rs (check ToolSchema.dangerous + ApprovalConfig)
            ↓
       executor.rs (execute tool)
            ↓
       guardrails.rs (validate output)
```

#### Implementation Points

1. **Where to integrate:** `crates/agents/src/executor.rs` or `engine.rs` before tool execution
2. **How to check:** Use `ApprovalConfig::requires_approval(&tool_schema)`
3. **Events:** Add to `trajectory.rs`: `ApprovalRequested`, `ApprovalGranted`, `ApprovalRejected`

#### Acceptance Criteria

- [ ] Approval check happens before dangerous tool execution
- [ ] All `ApprovalDecision` variants handled correctly
- [ ] Trajectory events emitted
- [ ] `auto_approve_timeout` respected

#### Checklist

- [ ] Add `Option<ApprovalConfig>` to engine/executor config
- [ ] Call `requires_approval()` before tool execution
- [ ] Await `handler.request_approval()` when needed
- [ ] Handle `Approved`, `Rejected`, `Modified` decisions
- [ ] Implement timeout with `tokio::time::timeout`
- [ ] Add events to `trajectory.rs`:
  ```rust
  pub enum AgentEvent {
      // ... existing ...
      ApprovalRequested { tool: String, request_id: String, is_dangerous: bool },
      ApprovalDecision { request_id: String, decision: String },
  }
  ```
- [ ] Document the "safety sandwich" in code comments

---

### Issue 7: test: integration tests for approval flows

**Labels:** `testing`, `safety`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Comprehensive tests for all approval scenarios using `MockBackend` + `TestApprovalHandler`.

#### Test Scenarios

```rust
#[tokio::test]
async fn test_dangerous_tool_requires_approval() {
    let backend = MockBackend::tool_then_text(
        "delete_all_data",  // dangerous tool
        json!({"confirm": true}),
        "Data deleted successfully",
    );
    
    let handler = TestApprovalHandler::always_approve();
    let config = ApprovalConfig {
        handler: Arc::new(handler.clone()),
        ..Default::default()
    };
    
    let engine = AgentEngine::new(backend, config);
    let result = engine.run("Delete all data").await.unwrap();
    
    // Verify approval was requested
    handler.assert_request_count(1);
    assert!(handler.requests()[0].is_dangerous);
}

#[tokio::test]
async fn test_rejected_tool_not_executed() {
    let backend = MockBackend::tool_then_text(
        "send_email",
        json!({"to": "test@example.com"}),
        "Email sent",
    );
    
    let handler = TestApprovalHandler::always_reject("User declined");
    // ...
    
    // Verify tool was NOT executed
    // Verify rejection event in trajectory
}

#[tokio::test]
async fn test_modified_args_used() {
    // Test that Modified { new_args } actually uses the new arguments
}

#[tokio::test]
async fn test_non_dangerous_tool_no_approval() {
    // Test that tools with dangerous=false skip approval
}

#[tokio::test]
async fn test_auto_approve_timeout() {
    // Test that auto_approve_timeout works
}
```

#### Checklist

- [ ] Create `tests/approval_integration.rs`
- [ ] Test dangerous tool → approval required
- [ ] Test rejected → tool not executed
- [ ] Test modified → new args used
- [ ] Test non-dangerous → no approval
- [ ] Test timeout behavior
- [ ] Verify trajectory events

---

### Issue 8: test: engine integration tests with MockBackend

**Labels:** `testing`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Create comprehensive engine tests using `MockBackend` to validate the ReACT loop.

#### Test Scenarios

```rust
#[tokio::test]
async fn test_engine_simple_response() {
    let backend = MockBackend::text("Hello, I'm an AI assistant!");
    let engine = AgentEngine::new(Arc::new(backend), config);
    
    let result = engine.run("Say hello").await.unwrap();
    assert!(result.contains("Hello"));
}

#[tokio::test]
async fn test_engine_tool_call_flow() {
    let backend = MockBackend::new(vec![
        MockStep::ToolCall {
            name: "search_entity".into(),
            arguments: json!({"query": "ACME Corp"}),
        },
        MockStep::Text("Found: ACME Corp, founded 1990".into()),
    ]);
    
    // Register the tool
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(SearchEntityTool::new()));
    
    let engine = AgentEngine::new(Arc::new(backend), config).with_tools(tools);
    let result = engine.run("Search for ACME Corp").await.unwrap();
    
    assert!(result.contains("ACME Corp"));
}

#[tokio::test]
async fn test_engine_multiple_tool_calls() {
    // Test scenario with 2+ sequential tool calls
}

#[tokio::test]
async fn test_engine_tool_error_handling() {
    // Test what happens when a tool returns an error
}
```

#### Checklist

- [ ] Create `tests/engine_mock_integration.rs`
- [ ] Test simple text response
- [ ] Test single tool call flow
- [ ] Test multiple sequential tool calls
- [ ] Test tool execution errors
- [ ] Test max iterations limit
- [ ] Verify trajectory recording

---

### Issue 9: docs: document LLM Backends and MockBackend

**Labels:** `documentation`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Document the existing `LLMBackend` trait and new `MockBackend`.

#### Content

```markdown
# LLM Backends

HyperAgent uses a pluggable backend system for LLM providers via the `LLMBackend` trait.

## Available Backends

| Backend | Provider | Location |
|---------|----------|----------|
| `OpenAIBackend` | OpenAI | `crates/providers/src/openai.rs` |
| `AnthropicBackend` | Anthropic | `crates/providers/src/anthropic.rs` |
| `OpenRouterBackend` | OpenRouter | `crates/providers/src/openrouter.rs` |
| `MockBackend` | Testing | `crates/providers/src/mock.rs` |

## Quick Start

\`\`\`rust
use rust_ai_agents_providers::{OpenAIBackend, OpenAIConfig};

let backend = OpenAIBackend::new(OpenAIConfig::default());
let engine = AgentEngine::new(Arc::new(backend), config);
\`\`\`

## Testing with MockBackend

\`\`\`rust
use rust_ai_agents_providers::MockBackend;

// Simple text response
let backend = MockBackend::text("Hello!");

// Tool call then response
let backend = MockBackend::tool_then_text(
    "search",
    json!({"query": "test"}),
    "Found: test result",
);

// Custom sequence
let backend = MockBackend::new(vec![
    MockStep::ToolCall { name: "step1".into(), arguments: json!({}) },
    MockStep::ToolCall { name: "step2".into(), arguments: json!({}) },
    MockStep::Text("Final answer".into()),
]);
\`\`\`

## The LLMBackend Trait

\`\`\`rust
#[async_trait]
pub trait LLMBackend: Send + Sync {
    async fn infer(
        &self,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError>;

    async fn infer_stream(...) -> Result<StreamResponse, LLMError>;
    
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError>;
    
    fn model_info(&self) -> ModelInfo;
}
\`\`\`
```

#### Checklist

- [ ] Create `docs/llm_backends.md`
- [ ] Document existing backends
- [ ] Document `MockBackend` usage
- [ ] Add examples
- [ ] Link from README.md

---

### Issue 10: docs: document Human-in-the-loop & Approvals

**Labels:** `documentation`, `safety`  
**Milestone:** v0.2.0 – LLM Backends & Approvals

#### Description

Document the approval system and how it integrates with `ToolSchema.dangerous`.

#### Content

```markdown
# Human-in-the-Loop Approvals

HyperAgent supports human approval for sensitive tool operations.

## Quick Start

\`\`\`rust
use rust_ai_agents::approval::{ApprovalConfig, TerminalApprovalHandler};

let config = ApprovalConfig {
    handler: Arc::new(TerminalApprovalHandler::new()),
    auto_approve_timeout: Some(Duration::from_secs(300)),
    ..Default::default()
};

let engine = AgentEngine::new(backend, engine_config)
    .with_approval(config);
\`\`\`

## Marking Tools as Dangerous

Use `ToolSchema.dangerous = true`:

\`\`\`rust
ToolSchema::new("delete_user", "Delete a user account")
    .with_dangerous(true)  // Will require approval
\`\`\`

## The Safety Sandwich

\`\`\`
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
\`\`\`

## Approval Handlers

| Handler | Use Case |
|---------|----------|
| `TerminalApprovalHandler` | CLI/dev interactive |
| `TestApprovalHandler` | Automated testing |
| Custom | Slack, webhook, queue, etc. |
```

#### Checklist

- [ ] Create `docs/approvals.md`
- [ ] Document `ToolSchema.dangerous` usage
- [ ] Include safety sandwich diagram
- [ ] Document approval handlers
- [ ] Add examples
- [ ] Link from README.md and ROADMAP.md

---

## Summary

| # | Issue | Type | Priority | Notes |
|---|-------|------|----------|-------|
| 1 | MockBackend for tests | feature | high | NEW - doesn't exist yet |
| 2 | Wire engine to LLMBackend | refactor | high | Verify current state |
| 3 | ApprovalConfig/Handler | feature | high | Use existing `dangerous` flag |
| 4 | TerminalApprovalHandler | feature | medium | CLI approval |
| 5 | TestApprovalHandler | feature | medium | For automated tests |
| 6 | Integrate approvals into executor | feature | high | Main integration work |
| 7 | Approval integration tests | test | medium | Full flow tests |
| 8 | Engine integration tests | test | medium | With MockBackend |
| 9 | Docs: LLM Backends | docs | medium | Document existing + MockBackend |
| 10 | Docs: Approvals | docs | medium | New feature docs |

**Total: 10 issues** (reduced from 13 - removed duplicates of existing code)

---

## What We're NOT Doing (Already Exists)

- ❌ Creating new `ReasoningBackend` trait (use existing `LLMBackend`)
- ❌ Creating new Message/ToolCall types (use existing from `crates/core`)
- ❌ Implementing OpenAIBackend (already in `crates/providers/src/openai.rs`)
- ❌ Implementing AnthropicBackend (already in `crates/providers/src/anthropic.rs`)

---

## Suggested Implementation Order

1. **Issue 1** - MockBackend (enables testing everything else)
2. **Issue 2** - Verify/wire engine to LLMBackend
3. **Issue 8** - Engine integration tests (validate MockBackend works)
4. **Issue 3** - ApprovalConfig/Handler types
5. **Issue 5** - TestApprovalHandler (for testing approvals)
6. **Issue 4** - TerminalApprovalHandler
7. **Issue 6** - Integrate approvals into executor
8. **Issue 7** - Approval integration tests
9. **Issue 9** - Docs: LLM Backends
10. **Issue 10** - Docs: Approvals
