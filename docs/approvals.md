# Approvals and Safety Sandwich Guide

This guide covers the human-in-the-loop approval system and the Safety Sandwich architecture pattern in HyperAgent.

## Overview

HyperAgent implements a multi-layered safety system that intercepts dangerous operations and requires human approval before execution.

## Safety Sandwich Architecture

The Safety Sandwich wraps tool execution with input/output validation and approval gates:

```
┌─────────────────────────────────────────────────────────────┐
│                     SAFETY SANDWICH                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INPUT GUARDRAILS                                        │
│     ├── Validate user input                                 │
│     ├── Check for prompt injection                          │
│     └── Sanitize dangerous patterns                         │
│                         ↓                                   │
│  2. APPROVAL GATE (if tool.dangerous = true)                │
│     ├── Show tool call details to human                     │
│     ├── Wait for approval/denial                            │
│     └── Allow argument modification                         │
│                         ↓                                   │
│  3. TOOL EXECUTOR                                           │
│     ├── Execute approved tool                               │
│     └── Capture output                                      │
│                         ↓                                   │
│  4. OUTPUT GUARDRAILS                                       │
│     ├── Validate tool output                                │
│     ├── Check for sensitive data leakage                    │
│     └── Sanitize response                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```rust
use rust_ai_agents_agents::approvals::{ApprovalConfig, TerminalApprovalHandler};

// Configure approval requirements
let config = ApprovalConfig::builder()
    .require_approval_for_dangerous(true)
    .always_approve_tools(vec!["read_file", "search"])
    .always_deny_tools(vec!["delete_all", "format_disk"])
    .build();

// Create handler for terminal interaction
let handler = TerminalApprovalHandler::new(config);

// Use with agent
let agent = AgentBuilder::new()
    .backend(backend)
    .approval_handler(handler)
    .build();
```

## Approval Configuration

### ApprovalConfig

```rust
pub struct ApprovalConfig {
    /// Require approval for tools marked as dangerous
    pub require_approval_for_dangerous: bool,
    
    /// Require approval for ALL tool calls
    pub require_approval_for_all: bool,
    
    /// Tools that bypass approval (whitelist)
    pub always_approve_tools: HashSet<String>,
    
    /// Tools that are always denied (blacklist)
    pub always_deny_tools: HashSet<String>,
    
    /// Regex patterns requiring approval
    pub approval_patterns: Vec<String>,
    
    /// Timeout in seconds (0 = no timeout)
    pub timeout_seconds: u64,
    
    /// Max pending approvals before auto-deny
    pub max_pending_approvals: usize,
}
```

### Builder Pattern

```rust
let config = ApprovalConfig::builder()
    // Require approval for dangerous tools (default: true)
    .require_approval_for_dangerous(true)
    
    // Or require approval for ALL tools
    .require_approval_for_all(true)
    
    // Whitelist safe tools
    .always_approve_tools(vec![
        "read_file",
        "list_directory",
        "search",
    ])
    
    // Blacklist dangerous tools
    .always_deny_tools(vec![
        "delete_all",
        "execute_arbitrary_code",
    ])
    
    // Pattern-based rules
    .approval_pattern(".*_delete$")  // Anything ending in _delete
    .approval_pattern("^admin_.*")   // Anything starting with admin_
    
    // Timeout (0 = wait forever)
    .timeout_seconds(60)
    
    // Max pending approvals
    .max_pending_approvals(10)
    
    .build();
```

## Approval Handlers

### TerminalApprovalHandler

Interactive approval via command line:

```rust
use rust_ai_agents_agents::approvals::TerminalApprovalHandler;

let handler = TerminalApprovalHandler::new(config);

// When a dangerous tool is called, user sees:
// ┌─────────────────────────────────────────────────────────┐
// │  APPROVAL REQUIRED                                      │
// ├─────────────────────────────────────────────────────────┤
// │  Tool: delete_file                                      │
// │  Reason: Tool is marked as dangerous                    │
// │                                                         │
// │  Arguments:                                             │
// │    {                                                    │
// │      "path": "/etc/passwd"                              │
// │    }                                                    │
// │                                                         │
// │  [A]pprove  [D]eny  [M]odify  [S]kip                   │
// └─────────────────────────────────────────────────────────┘
```

### TestApprovalHandler

For automated testing:

```rust
use rust_ai_agents_agents::approvals::TestApprovalHandler;

// Auto-approve everything
let handler = TestApprovalHandler::auto_approve();

// Auto-deny everything  
let handler = TestApprovalHandler::auto_deny();

// Approve first N, then deny
let handler = TestApprovalHandler::approve_first(3);

// Custom decisions
let handler = TestApprovalHandler::with_decisions(vec![
    ApprovalDecision::Approved,
    ApprovalDecision::Denied { reason: "Test".to_string() },
    ApprovalDecision::Approved,
]);

// Verify approvals in tests
#[tokio::test]
async fn test_approval_flow() {
    let handler = TestApprovalHandler::auto_approve();
    
    // ... run agent ...
    
    assert_eq!(handler.approval_count(), 2);
    assert!(handler.was_tool_approved("dangerous_tool"));
}
```

### AutoApproveHandler

Bypass approvals entirely (use with caution):

```rust
use rust_ai_agents_agents::approvals::AutoApproveHandler;

// Approve everything automatically
let handler = AutoApproveHandler::new();

// With logging
let handler = AutoApproveHandler::with_logging();
```

### Custom Handler

Implement your own approval logic:

```rust
use rust_ai_agents_agents::approvals::{ApprovalHandler, ApprovalRequest, ApprovalDecision};

struct SlackApprovalHandler {
    webhook_url: String,
}

#[async_trait]
impl ApprovalHandler for SlackApprovalHandler {
    async fn request_approval(
        &self,
        request: ApprovalRequest,
    ) -> Result<ApprovalDecision, ApprovalError> {
        // Send to Slack
        let message = format!(
            "Approval needed for tool `{}`\nArguments: {}",
            request.tool_call.name,
            serde_json::to_string_pretty(&request.tool_call.arguments)?
        );
        
        // Post to Slack and wait for response
        let response = self.post_and_wait(&message).await?;
        
        match response.as_str() {
            "approve" => Ok(ApprovalDecision::Approved),
            "deny" => Ok(ApprovalDecision::denied("Denied via Slack")),
            _ => Ok(ApprovalDecision::Skip),
        }
    }
}
```

## Approval Decisions

```rust
pub enum ApprovalDecision {
    /// Proceed with execution
    Approved,
    
    /// Do not execute
    Denied { reason: String },
    
    /// Execute with modified arguments
    Modify { new_arguments: serde_json::Value },
    
    /// Skip this tool but continue agent loop
    Skip,
}
```

### Modifying Arguments

Users can modify tool arguments before approval:

```rust
// Original call: delete_file("/important/file.txt")
// User modifies to: delete_file("/tmp/safe_file.txt")

ApprovalDecision::Modify {
    new_arguments: json!({
        "path": "/tmp/safe_file.txt"
    })
}
```

## Marking Tools as Dangerous

```rust
use rust_ai_agents_core::Tool;

// Using the dangerous flag
let tool = Tool::new("delete_file", "Deletes a file")
    .dangerous(true)  // Requires approval
    .handler(|args| async { /* ... */ });

// Safe tools don't require approval
let tool = Tool::new("read_file", "Reads a file")
    .dangerous(false)  // No approval needed (default)
    .handler(|args| async { /* ... */ });
```

## Approval Reasons

```rust
pub enum ApprovalReason {
    /// Tool has dangerous=true
    DangerousTool,
    
    /// Tool name matches approval pattern
    MatchesPattern(String),
    
    /// All tools require approval (config setting)
    AllToolsRequireApproval,
    
    /// Custom reason
    Custom(String),
}
```

## Integration with AgentEngine

```rust
use rust_ai_agents_agents::{AgentEngine, AgentConfig};
use rust_ai_agents_agents::approvals::{ApprovalConfig, TerminalApprovalHandler};

// Configure agent with approvals
let approval_config = ApprovalConfig::builder()
    .require_approval_for_dangerous(true)
    .timeout_seconds(120)
    .build();

let agent_config = AgentConfig::builder()
    .name("secure-agent")
    .approval_handler(TerminalApprovalHandler::new(approval_config))
    .build();

let engine = AgentEngine::new(backend, agent_config);

// When agent calls a dangerous tool, approval is requested
let result = engine.run("Delete the temp files").await?;
```

## Best Practices

### 1. Defense in Depth

Combine multiple safety layers:

```rust
let config = ApprovalConfig::builder()
    // Layer 1: Blacklist known dangerous tools
    .always_deny_tools(vec!["execute_shell", "sudo"])
    
    // Layer 2: Require approval for dangerous flag
    .require_approval_for_dangerous(true)
    
    // Layer 3: Pattern matching for suspicious operations
    .approval_pattern(".*delete.*")
    .approval_pattern(".*remove.*")
    .approval_pattern(".*drop.*")
    
    .build();
```

### 2. Principle of Least Privilege

Only whitelist tools that are truly safe:

```rust
// Good: Specific whitelist
.always_approve_tools(vec!["read_file", "search", "calculate"])

// Bad: Too permissive
.always_approve_tools(vec!["file_*"])  // Don't do this
```

### 3. Timeouts for Production

Always set timeouts in production:

```rust
let config = ApprovalConfig::builder()
    .timeout_seconds(300)  // 5 minute timeout
    .build();
```

### 4. Audit Logging

Log all approval decisions:

```rust
struct AuditedApprovalHandler<H: ApprovalHandler> {
    inner: H,
    logger: Logger,
}

#[async_trait]
impl<H: ApprovalHandler> ApprovalHandler for AuditedApprovalHandler<H> {
    async fn request_approval(&self, request: ApprovalRequest) -> Result<ApprovalDecision, ApprovalError> {
        let decision = self.inner.request_approval(request.clone()).await?;
        
        self.logger.log(AuditEvent {
            tool: request.tool_call.name,
            arguments: request.tool_call.arguments,
            decision: decision.clone(),
            timestamp: Utc::now(),
        });
        
        Ok(decision)
    }
}
```

## Example: Secure File Operations Agent

```rust
use rust_ai_agents_agents::approvals::{ApprovalConfig, TerminalApprovalHandler};

// Define tools with appropriate danger levels
let read_tool = Tool::new("read_file", "Read file contents")
    .dangerous(false);

let write_tool = Tool::new("write_file", "Write to file")
    .dangerous(true);  // Requires approval

let delete_tool = Tool::new("delete_file", "Delete a file")
    .dangerous(true);  // Requires approval

// Configure approvals
let config = ApprovalConfig::builder()
    .require_approval_for_dangerous(true)
    .always_approve_tools(vec!["read_file"])  // Reading is safe
    .always_deny_tools(vec!["delete_system_files"])  // Never allow
    .timeout_seconds(60)
    .build();

// Build agent
let agent = AgentBuilder::new()
    .name("file-agent")
    .backend(backend)
    .tool(read_tool)
    .tool(write_tool)
    .tool(delete_tool)
    .approval_handler(TerminalApprovalHandler::new(config))
    .build();

// Run - write and delete will prompt for approval
agent.run("Organize my documents folder").await?;
```

## Error Handling

```rust
match approval_result {
    Ok(ApprovalDecision::Approved) => {
        // Execute tool
    }
    Ok(ApprovalDecision::Denied { reason }) => {
        // Log denial, inform agent
        agent.add_message(format!("Tool denied: {}", reason));
    }
    Ok(ApprovalDecision::Skip) => {
        // Continue without this tool
    }
    Err(ApprovalError::Timeout(secs)) => {
        // Handle timeout
        log::warn!("Approval timed out after {}s", secs);
    }
    Err(ApprovalError::Denied(reason)) => {
        // Handle explicit denial
    }
    Err(e) => {
        // Handle other errors
    }
}
```

## Next Steps

- [LLM Backends Guide](./backends.md) - Configure LLM providers
- [MCP Integration](../crates/mcp/README.md) - Model Context Protocol
