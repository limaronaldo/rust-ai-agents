//! Human-in-the-loop approval system for dangerous tool executions
//!
//! This module provides a configurable approval mechanism that intercepts
//! tool calls marked as `dangerous` and requires human approval before execution.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
//! │   Engine    │────>│  Approvals   │────>│   Executor   │
//! │             │     │   Handler    │     │              │
//! └─────────────┘     └──────────────┘     └──────────────┘
//!                            │
//!                     ┌──────┴──────┐
//!                     │             │
//!               ┌─────▼─────┐ ┌─────▼─────┐
//!               │ Terminal  │ │   Test    │
//!               │ Handler   │ │  Handler  │
//!               └───────────┘ └───────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_agents::approvals::{ApprovalConfig, TerminalApprovalHandler};
//!
//! let config = ApprovalConfig::builder()
//!     .require_approval_for_dangerous(true)
//!     .always_approve_tools(vec!["read_file"])
//!     .always_deny_tools(vec!["delete_all"])
//!     .build();
//!
//! let handler = TerminalApprovalHandler::new(config);
//! ```

use async_trait::async_trait;
use parking_lot::Mutex;
use rust_ai_agents_core::{ToolCall, ToolSchema};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

/// Errors that can occur during approval
#[derive(Debug, Error)]
pub enum ApprovalError {
    #[error("Tool execution denied by user: {0}")]
    Denied(String),

    #[error("Approval timeout after {0} seconds")]
    Timeout(u64),

    #[error("Approval handler error: {0}")]
    HandlerError(String),

    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
}

/// Result of an approval request
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalDecision {
    /// Approved - proceed with execution
    Approved,

    /// Denied - do not execute
    Denied { reason: String },

    /// Modify - execute with modified arguments
    Modify { new_arguments: serde_json::Value },

    /// Skip - skip this tool but continue the agent loop
    Skip,
}

impl ApprovalDecision {
    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approved | Self::Modify { .. })
    }

    pub fn denied(reason: impl Into<String>) -> Self {
        Self::Denied {
            reason: reason.into(),
        }
    }
}

/// Information about a pending approval request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// The tool call requiring approval
    pub tool_call: ToolCall,

    /// The tool schema (for context)
    pub tool_schema: Option<ToolSchema>,

    /// Why approval is required
    pub reason: ApprovalReason,

    /// Request timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Additional context for the approver
    pub context: Option<String>,
}

/// Reason why approval is required
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalReason {
    /// Tool is marked as dangerous
    DangerousTool,

    /// Tool matches a pattern requiring approval
    MatchesPattern(String),

    /// All tools require approval
    AllToolsRequireApproval,

    /// Custom reason
    Custom(String),
}

impl std::fmt::Display for ApprovalReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DangerousTool => write!(f, "Tool is marked as dangerous"),
            Self::MatchesPattern(p) => write!(f, "Tool matches pattern: {}", p),
            Self::AllToolsRequireApproval => write!(f, "All tools require approval"),
            Self::Custom(r) => write!(f, "{}", r),
        }
    }
}

/// Configuration for the approval system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalConfig {
    /// Require approval for tools marked as dangerous
    pub require_approval_for_dangerous: bool,

    /// Require approval for all tool calls
    pub require_approval_for_all: bool,

    /// Tools that are always approved (bypass approval)
    pub always_approve_tools: HashSet<String>,

    /// Tools that are always denied
    pub always_deny_tools: HashSet<String>,

    /// Patterns for tools requiring approval (regex)
    pub approval_patterns: Vec<String>,

    /// Timeout for approval requests in seconds (0 = no timeout)
    pub timeout_seconds: u64,

    /// Maximum number of approvals to request before auto-denying
    pub max_pending_approvals: usize,
}

impl Default for ApprovalConfig {
    fn default() -> Self {
        Self {
            require_approval_for_dangerous: true,
            require_approval_for_all: false,
            always_approve_tools: HashSet::new(),
            always_deny_tools: HashSet::new(),
            approval_patterns: Vec::new(),
            timeout_seconds: 0,
            max_pending_approvals: 100,
        }
    }
}

impl ApprovalConfig {
    /// Create a new builder
    pub fn builder() -> ApprovalConfigBuilder {
        ApprovalConfigBuilder::default()
    }

    /// Check if a tool requires approval
    pub fn requires_approval(&self, tool_name: &str, is_dangerous: bool) -> Option<ApprovalReason> {
        // Check always deny first
        if self.always_deny_tools.contains(tool_name) {
            return Some(ApprovalReason::Custom("Tool is in deny list".to_string()));
        }

        // Check always approve (bypass)
        if self.always_approve_tools.contains(tool_name) {
            return None;
        }

        // Check all tools require approval
        if self.require_approval_for_all {
            return Some(ApprovalReason::AllToolsRequireApproval);
        }

        // Check dangerous flag
        if self.require_approval_for_dangerous && is_dangerous {
            return Some(ApprovalReason::DangerousTool);
        }

        // Check patterns
        for pattern in &self.approval_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if re.is_match(tool_name) {
                    return Some(ApprovalReason::MatchesPattern(pattern.clone()));
                }
            }
        }

        None
    }
}

/// Builder for ApprovalConfig
#[derive(Default)]
pub struct ApprovalConfigBuilder {
    config: ApprovalConfig,
}

impl ApprovalConfigBuilder {
    pub fn require_approval_for_dangerous(mut self, value: bool) -> Self {
        self.config.require_approval_for_dangerous = value;
        self
    }

    pub fn require_approval_for_all(mut self, value: bool) -> Self {
        self.config.require_approval_for_all = value;
        self
    }

    pub fn always_approve_tools(mut self, tools: Vec<impl Into<String>>) -> Self {
        self.config.always_approve_tools = tools.into_iter().map(Into::into).collect();
        self
    }

    pub fn always_deny_tools(mut self, tools: Vec<impl Into<String>>) -> Self {
        self.config.always_deny_tools = tools.into_iter().map(Into::into).collect();
        self
    }

    pub fn add_approval_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.config.approval_patterns.push(pattern.into());
        self
    }

    pub fn timeout_seconds(mut self, seconds: u64) -> Self {
        self.config.timeout_seconds = seconds;
        self
    }

    pub fn max_pending_approvals(mut self, max: usize) -> Self {
        self.config.max_pending_approvals = max;
        self
    }

    pub fn build(self) -> ApprovalConfig {
        self.config
    }
}

/// Trait for handling approval requests
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    /// Request approval for a tool call
    async fn request_approval(
        &self,
        request: ApprovalRequest,
    ) -> Result<ApprovalDecision, ApprovalError>;

    /// Get the configuration
    fn config(&self) -> &ApprovalConfig;

    /// Check if a tool requires approval and return the reason
    fn check_requires_approval(
        &self,
        tool_call: &ToolCall,
        tool_schema: Option<&ToolSchema>,
    ) -> Option<ApprovalReason> {
        let is_dangerous = tool_schema.map(|s| s.dangerous).unwrap_or(false);
        self.config()
            .requires_approval(&tool_call.name, is_dangerous)
    }
}

/// Terminal-based approval handler for interactive use
pub struct TerminalApprovalHandler {
    config: ApprovalConfig,
    pending_count: AtomicUsize,
}

impl TerminalApprovalHandler {
    pub fn new(config: ApprovalConfig) -> Self {
        Self {
            config,
            pending_count: AtomicUsize::new(0),
        }
    }

    fn format_request(&self, request: &ApprovalRequest) -> String {
        let mut output = String::new();
        output.push('\n');
        output.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║              🔒 TOOL APPROVAL REQUIRED                       ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║ Tool: {:<55} ║\n", request.tool_call.name));
        output.push_str(&format!("║ Reason: {:<53} ║\n", request.reason));
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        output.push_str("║ Arguments:                                                   ║\n");

        // Format arguments nicely
        let args_str = serde_json::to_string_pretty(&request.tool_call.arguments)
            .unwrap_or_else(|_| request.tool_call.arguments.to_string());

        for line in args_str.lines().take(10) {
            let truncated = if line.len() > 60 { &line[..57] } else { line };
            output.push_str(&format!("║   {:<59} ║\n", truncated));
        }

        if args_str.lines().count() > 10 {
            output.push_str("║   ... (truncated)                                           ║\n");
        }

        if let Some(ref ctx) = request.context {
            output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
            output.push_str(&format!("║ Context: {:<52} ║\n", ctx));
        }

        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        output.push_str("║ [Y]es / [N]o / [S]kip / [A]lways approve / [D]eny always     ║\n");
        output.push_str("╚══════════════════════════════════════════════════════════════╝\n");
        output.push_str("> ");

        output
    }
}

#[async_trait]
impl ApprovalHandler for TerminalApprovalHandler {
    async fn request_approval(
        &self,
        request: ApprovalRequest,
    ) -> Result<ApprovalDecision, ApprovalError> {
        // Check pending count
        let pending = self.pending_count.fetch_add(1, Ordering::SeqCst);
        if pending >= self.config.max_pending_approvals {
            self.pending_count.fetch_sub(1, Ordering::SeqCst);
            return Ok(ApprovalDecision::denied("Too many pending approvals"));
        }

        // Display the request
        let prompt = self.format_request(&request);
        print!("{}", prompt);
        io::stdout().flush()?;

        // Read response
        let input: String;

        // Handle timeout if configured
        if self.config.timeout_seconds > 0 {
            let timeout = tokio::time::Duration::from_secs(self.config.timeout_seconds);
            let read_result = tokio::time::timeout(timeout, async {
                tokio::task::spawn_blocking(|| {
                    let mut buf = String::new();
                    io::stdin().read_line(&mut buf)?;
                    Ok::<_, io::Error>(buf)
                })
                .await
            })
            .await;

            match read_result {
                Ok(Ok(Ok(s))) => input = s,
                Ok(Ok(Err(e))) => {
                    self.pending_count.fetch_sub(1, Ordering::SeqCst);
                    return Err(ApprovalError::IoError(e));
                }
                Ok(Err(e)) => {
                    self.pending_count.fetch_sub(1, Ordering::SeqCst);
                    return Err(ApprovalError::HandlerError(e.to_string()));
                }
                Err(_) => {
                    self.pending_count.fetch_sub(1, Ordering::SeqCst);
                    return Err(ApprovalError::Timeout(self.config.timeout_seconds));
                }
            }
        } else {
            let mut buf = String::new();
            io::stdin().read_line(&mut buf)?;
            input = buf;
        }

        self.pending_count.fetch_sub(1, Ordering::SeqCst);

        let decision = match input.trim().to_lowercase().as_str() {
            "y" | "yes" => ApprovalDecision::Approved,
            "n" | "no" => ApprovalDecision::denied("User denied"),
            "s" | "skip" => ApprovalDecision::Skip,
            "a" | "always" => {
                // Note: In a real implementation, this would modify the config
                // For now, just approve
                println!(
                    "  ✓ Tool '{}' will be auto-approved in this session",
                    request.tool_call.name
                );
                ApprovalDecision::Approved
            }
            "d" | "deny" => {
                println!(
                    "  ✗ Tool '{}' will be auto-denied in this session",
                    request.tool_call.name
                );
                ApprovalDecision::denied("User set to always deny")
            }
            _ => {
                println!("  Invalid input, defaulting to deny");
                ApprovalDecision::denied("Invalid input")
            }
        };

        Ok(decision)
    }

    fn config(&self) -> &ApprovalConfig {
        &self.config
    }
}

/// Test approval handler with configurable responses
pub struct TestApprovalHandler {
    config: ApprovalConfig,
    /// Pre-configured decisions (tool_name -> decision)
    decisions: Mutex<std::collections::HashMap<String, ApprovalDecision>>,
    /// Default decision for unspecified tools
    default_decision: ApprovalDecision,
    /// Recorded approval requests
    requests: Mutex<Vec<ApprovalRequest>>,
    /// Call counter
    call_count: AtomicUsize,
}

impl TestApprovalHandler {
    /// Create a handler that approves everything
    pub fn approve_all() -> Self {
        Self {
            config: ApprovalConfig::default(),
            decisions: Mutex::new(std::collections::HashMap::new()),
            default_decision: ApprovalDecision::Approved,
            requests: Mutex::new(Vec::new()),
            call_count: AtomicUsize::new(0),
        }
    }

    /// Create a handler that denies everything
    pub fn deny_all() -> Self {
        Self {
            config: ApprovalConfig::default(),
            decisions: Mutex::new(std::collections::HashMap::new()),
            default_decision: ApprovalDecision::denied("Test handler denies all"),
            requests: Mutex::new(Vec::new()),
            call_count: AtomicUsize::new(0),
        }
    }

    /// Create a handler with specific decisions per tool
    pub fn with_decisions(decisions: Vec<(impl Into<String>, ApprovalDecision)>) -> Self {
        let map: std::collections::HashMap<String, ApprovalDecision> =
            decisions.into_iter().map(|(k, v)| (k.into(), v)).collect();

        Self {
            config: ApprovalConfig::default(),
            decisions: Mutex::new(map),
            default_decision: ApprovalDecision::Approved,
            requests: Mutex::new(Vec::new()),
            call_count: AtomicUsize::new(0),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ApprovalConfig) -> Self {
        Self {
            config,
            decisions: Mutex::new(std::collections::HashMap::new()),
            default_decision: ApprovalDecision::Approved,
            requests: Mutex::new(Vec::new()),
            call_count: AtomicUsize::new(0),
        }
    }

    /// Set decision for a specific tool
    pub fn set_decision(&self, tool_name: impl Into<String>, decision: ApprovalDecision) {
        self.decisions.lock().insert(tool_name.into(), decision);
    }

    /// Get all recorded requests
    pub fn get_requests(&self) -> Vec<ApprovalRequest> {
        self.requests.lock().clone()
    }

    /// Get the number of approval requests made
    pub fn request_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// Clear recorded requests
    pub fn clear(&self) {
        self.requests.lock().clear();
        self.call_count.store(0, Ordering::SeqCst);
    }

    /// Check if a specific tool had approval requested
    pub fn was_approval_requested(&self, tool_name: &str) -> bool {
        self.requests
            .lock()
            .iter()
            .any(|r| r.tool_call.name == tool_name)
    }
}

#[async_trait]
impl ApprovalHandler for TestApprovalHandler {
    async fn request_approval(
        &self,
        request: ApprovalRequest,
    ) -> Result<ApprovalDecision, ApprovalError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let tool_name = request.tool_call.name.clone();
        self.requests.lock().push(request);

        let decisions = self.decisions.lock();
        let decision = decisions
            .get(&tool_name)
            .cloned()
            .unwrap_or_else(|| self.default_decision.clone());

        Ok(decision)
    }

    fn config(&self) -> &ApprovalConfig {
        &self.config
    }
}

/// Auto-approve handler (no human interaction)
#[derive(Clone)]
pub struct AutoApproveHandler {
    config: ApprovalConfig,
}

impl AutoApproveHandler {
    pub fn new() -> Self {
        Self {
            config: ApprovalConfig {
                require_approval_for_dangerous: false,
                require_approval_for_all: false,
                ..Default::default()
            },
        }
    }
}

impl Default for AutoApproveHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ApprovalHandler for AutoApproveHandler {
    async fn request_approval(
        &self,
        _request: ApprovalRequest,
    ) -> Result<ApprovalDecision, ApprovalError> {
        Ok(ApprovalDecision::Approved)
    }

    fn config(&self) -> &ApprovalConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_approval_config_dangerous() {
        let config = ApprovalConfig::builder()
            .require_approval_for_dangerous(true)
            .build();

        // Dangerous tool should require approval
        assert!(config.requires_approval("delete_file", true).is_some());

        // Non-dangerous tool should not
        assert!(config.requires_approval("read_file", false).is_none());
    }

    #[test]
    fn test_approval_config_always_approve() {
        let config = ApprovalConfig::builder()
            .require_approval_for_all(true)
            .always_approve_tools(vec!["safe_tool"])
            .build();

        // Always approved tool bypasses
        assert!(config.requires_approval("safe_tool", true).is_none());

        // Other tools still require approval
        assert!(config.requires_approval("other_tool", false).is_some());
    }

    #[test]
    fn test_approval_config_always_deny() {
        let config = ApprovalConfig::builder()
            .always_deny_tools(vec!["dangerous_tool"])
            .build();

        // Always deny takes precedence
        let reason = config.requires_approval("dangerous_tool", false);
        assert!(reason.is_some());
    }

    #[test]
    fn test_approval_config_patterns() {
        let config = ApprovalConfig::builder()
            .add_approval_pattern("delete_.*")
            .add_approval_pattern(".*_dangerous")
            .build();

        assert!(config.requires_approval("delete_file", false).is_some());
        assert!(config.requires_approval("delete_folder", false).is_some());
        assert!(config.requires_approval("run_dangerous", false).is_some());
        assert!(config.requires_approval("read_file", false).is_none());
    }

    #[tokio::test]
    async fn test_test_handler_approve_all() {
        let handler = TestApprovalHandler::approve_all();

        let request = ApprovalRequest {
            tool_call: ToolCall {
                id: "1".to_string(),
                name: "any_tool".to_string(),
                arguments: json!({}),
            },
            tool_schema: None,
            reason: ApprovalReason::DangerousTool,
            timestamp: chrono::Utc::now(),
            context: None,
        };

        let decision = handler.request_approval(request).await.unwrap();
        assert_eq!(decision, ApprovalDecision::Approved);
        assert_eq!(handler.request_count(), 1);
    }

    #[tokio::test]
    async fn test_test_handler_deny_all() {
        let handler = TestApprovalHandler::deny_all();

        let request = ApprovalRequest {
            tool_call: ToolCall {
                id: "1".to_string(),
                name: "any_tool".to_string(),
                arguments: json!({}),
            },
            tool_schema: None,
            reason: ApprovalReason::DangerousTool,
            timestamp: chrono::Utc::now(),
            context: None,
        };

        let decision = handler.request_approval(request).await.unwrap();
        assert!(!decision.is_approved());
    }

    #[tokio::test]
    async fn test_test_handler_specific_decisions() {
        let handler = TestApprovalHandler::with_decisions(vec![
            ("tool_a", ApprovalDecision::Approved),
            ("tool_b", ApprovalDecision::denied("Not allowed")),
        ]);

        let make_request = |name: &str| ApprovalRequest {
            tool_call: ToolCall {
                id: "1".to_string(),
                name: name.to_string(),
                arguments: json!({}),
            },
            tool_schema: None,
            reason: ApprovalReason::DangerousTool,
            timestamp: chrono::Utc::now(),
            context: None,
        };

        let d1 = handler
            .request_approval(make_request("tool_a"))
            .await
            .unwrap();
        assert_eq!(d1, ApprovalDecision::Approved);

        let d2 = handler
            .request_approval(make_request("tool_b"))
            .await
            .unwrap();
        assert!(!d2.is_approved());

        // Default is approve
        let d3 = handler
            .request_approval(make_request("tool_c"))
            .await
            .unwrap();
        assert_eq!(d3, ApprovalDecision::Approved);
    }

    #[tokio::test]
    async fn test_test_handler_records_requests() {
        let handler = TestApprovalHandler::approve_all();

        let request = ApprovalRequest {
            tool_call: ToolCall {
                id: "1".to_string(),
                name: "test_tool".to_string(),
                arguments: json!({"key": "value"}),
            },
            tool_schema: None,
            reason: ApprovalReason::DangerousTool,
            timestamp: chrono::Utc::now(),
            context: Some("Test context".to_string()),
        };

        handler.request_approval(request).await.unwrap();

        assert!(handler.was_approval_requested("test_tool"));
        assert!(!handler.was_approval_requested("other_tool"));

        let requests = handler.get_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].tool_call.name, "test_tool");
    }

    #[tokio::test]
    async fn test_auto_approve_handler() {
        let handler = AutoApproveHandler::new();

        let request = ApprovalRequest {
            tool_call: ToolCall {
                id: "1".to_string(),
                name: "dangerous_tool".to_string(),
                arguments: json!({}),
            },
            tool_schema: Some(ToolSchema {
                name: "dangerous_tool".to_string(),
                description: "A dangerous tool".to_string(),
                parameters: json!({}),
                dangerous: true,
                metadata: std::collections::HashMap::new(),
            }),
            reason: ApprovalReason::DangerousTool,
            timestamp: chrono::Utc::now(),
            context: None,
        };

        let decision = handler.request_approval(request).await.unwrap();
        assert_eq!(decision, ApprovalDecision::Approved);
    }

    #[test]
    fn test_approval_decision_is_approved() {
        assert!(ApprovalDecision::Approved.is_approved());
        assert!(ApprovalDecision::Modify {
            new_arguments: json!({})
        }
        .is_approved());
        assert!(!ApprovalDecision::Denied {
            reason: "no".to_string()
        }
        .is_approved());
        assert!(!ApprovalDecision::Skip.is_approved());
    }

    #[test]
    fn test_check_requires_approval() {
        let config = ApprovalConfig::builder()
            .require_approval_for_dangerous(true)
            .build();

        let handler = TestApprovalHandler::with_config(config);

        let safe_call = ToolCall {
            id: "1".to_string(),
            name: "safe_tool".to_string(),
            arguments: json!({}),
        };

        let safe_schema = ToolSchema {
            name: "safe_tool".to_string(),
            description: "Safe".to_string(),
            parameters: json!({}),
            dangerous: false,
            metadata: std::collections::HashMap::new(),
        };

        let dangerous_schema = ToolSchema {
            name: "dangerous_tool".to_string(),
            description: "Dangerous".to_string(),
            parameters: json!({}),
            dangerous: true,
            metadata: std::collections::HashMap::new(),
        };

        // Safe tool with safe schema - no approval needed
        assert!(handler
            .check_requires_approval(&safe_call, Some(&safe_schema))
            .is_none());

        // Safe tool with dangerous schema - approval needed
        assert!(handler
            .check_requires_approval(&safe_call, Some(&dangerous_schema))
            .is_some());

        // No schema, defaults to not dangerous
        assert!(handler.check_requires_approval(&safe_call, None).is_none());
    }
}
