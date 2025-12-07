## Description

Implement a CLI approval handler for development and debugging use.

## Behavior

- Print tool name, arguments, and dangerous flag
- Prompt: `Approve? [y]es / [n]o / [m]odify`
- Handle Ctrl+C / EOF as rejection

## Implementation

```rust
// crates/agents/src/approval/terminal.rs

use super::{ApprovalHandler, ApprovalRequest, ApprovalDecision, ApprovalError};
use async_trait::async_trait;

pub struct TerminalApprovalHandler {
    /// Show full arguments or just summary
    pub verbose: bool,
}

impl TerminalApprovalHandler {
    pub fn new() -> Self {
        Self { verbose: false }
    }
    
    pub fn verbose() -> Self {
        Self { verbose: true }
    }
}

impl Default for TerminalApprovalHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ApprovalHandler for TerminalApprovalHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> Result<ApprovalDecision, ApprovalError> {
        use std::io::{self, Write};
        
        println!();
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║                   APPROVAL REQUIRED                       ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ Tool: {:<52} ║", request.tool_name);
        if request.is_dangerous {
            println!("║ ⚠️  MARKED AS DANGEROUS                                   ║");
        }
        println!("╠══════════════════════════════════════════════════════════╣");
        
        if self.verbose {
            println!("║ Arguments:                                                ║");
            let args_str = serde_json::to_string_pretty(&request.arguments)
                .unwrap_or_else(|_| request.arguments.to_string());
            for line in args_str.lines() {
                println!("║   {:<56} ║", line);
            }
        }
        
        println!("╚══════════════════════════════════════════════════════════╝");
        println!();
        print!("Approve? [y]es / [n]o / [m]odify: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => Ok(ApprovalDecision::Approved),
            "n" | "no" => Ok(ApprovalDecision::Rejected {
                reason: "User rejected via terminal".into(),
            }),
            "m" | "modify" => {
                println!("Enter new arguments (JSON):");
                let mut json_input = String::new();
                io::stdin().read_line(&mut json_input)?;
                let new_args: serde_json::Value = serde_json::from_str(&json_input)
                    .map_err(|e| ApprovalError::HandlerError(format!("Invalid JSON: {}", e)))?;
                Ok(ApprovalDecision::Modified { new_args })
            }
            _ => Ok(ApprovalDecision::Rejected {
                reason: format!("Invalid input: {}", input.trim()),
            }),
        }
    }
    
    async fn on_status_change(&self, request_id: &str, status: super::ApprovalStatus) {
        println!("[Approval {}] Status changed: {:?}", request_id, status);
    }
}
```

## Checklist

- [ ] Create `crates/agents/src/approval/terminal.rs`
- [ ] Implement stdin reading with nice box formatting
- [ ] Handle EOF/Ctrl+C gracefully (return Rejected)
- [ ] Add `pub mod terminal;` to approval/mod.rs
- [ ] Create `examples/human_in_loop.rs` demonstrating usage
- [ ] Test with a mock dangerous tool
