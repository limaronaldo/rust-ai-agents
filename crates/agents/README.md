# rust-ai-agents-agents

Core agent implementation with ReACT loop (Reasoning + Acting) for intelligent task execution.

## Features

- **ReACT Loop**: Reason -> Act -> Observe cycle for intelligent task execution
- **Planning Mode**: Optional planning before execution for complex tasks
- **Tool Execution**: Parallel tool execution with timeout handling
- **Memory**: Conversation history and context management
- **Approvals**: Human-in-the-loop approval for dangerous operations (Safety Sandwich)
- **Checkpointing**: State persistence for recovery and time-travel
- **Guardrails**: Input/output validation with tripwire functionality
- **Discovery**: Dynamic agent discovery via heartbeat/capabilities
- **Handoff**: Capability-based routing between agents
- **Multi-Memory**: Hierarchical memory (short-term, long-term, entity)
- **Agent-as-Tool**: Compose agents as callable tools
- **Self-Correcting**: LLM-as-Judge pattern for automatic quality improvement
- **Durable Execution**: Fault-tolerant execution with recovery

## Installation

```toml
[dependencies]
rust-ai-agents-agents = "0.1"
```

## Quick Start

### Basic Agent

```rust
use rust_ai_agents_agents::{AgentEngine, AgentConfig};
use rust_ai_agents_providers::AnthropicProvider;
use rust_ai_agents_tools::ToolRegistry;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create LLM backend
    let backend = Arc::new(AnthropicProvider::claude_35_sonnet("your-api-key"));

    // Create tool registry
    let tools = Arc::new(ToolRegistry::new());

    // Configure agent
    let config = AgentConfig::builder()
        .id("agent-1")
        .name("Assistant")
        .system_prompt("You are a helpful assistant.")
        .timeout_secs(300)
        .build();

    // Create engine and spawn agent
    let engine = AgentEngine::new();
    let agent_id = engine.spawn_agent(config, tools, backend).await?;

    // Send message
    engine.send_message(agent_id, "Hello!").await?;

    Ok(())
}
```

### With Planning Mode

```rust
use rust_ai_agents_core::types::PlanningMode;

let config = AgentConfig::builder()
    .id("planner")
    .name("Planning Agent")
    .system_prompt("You are a methodical assistant.")
    .planning_mode(PlanningMode::Always)
    .build();
```

### With Stop Words

```rust
let config = AgentConfig::builder()
    .id("agent")
    .name("Agent")
    .stop_words(vec!["TASK_COMPLETE".into(), "DONE".into()])
    .build();
```

## Human-in-the-Loop Approvals

The Safety Sandwich pattern ensures dangerous operations require human approval:

```rust
use rust_ai_agents_agents::approvals::{ApprovalConfig, TerminalApprovalHandler};

let approval_config = ApprovalConfig::builder()
    .require_approval_for_dangerous(true)
    .always_approve_tools(vec!["read_file", "list_files"])
    .always_deny_tools(vec!["delete_all", "format_disk"])
    .timeout_secs(60)
    .build();

let handler = TerminalApprovalHandler::new(approval_config);
```

### Approval Flow

```
1. Agent wants to call a dangerous tool
2. ApprovalHandler intercepts the call
3. User sees: "[APPROVAL REQUIRED] delete_file(path: /important.txt)"
4. User responds: y/n/m (yes/no/modify)
5. If approved, tool executes; if denied, agent receives denial message
```

See [approvals guide](../../docs/approvals.md) for complete documentation.

## Checkpointing & Recovery

Save and restore agent state for fault tolerance:

```rust
use rust_ai_agents_agents::{CheckpointStore, SqliteCheckpointStore};

// Create checkpoint store
let store = SqliteCheckpointStore::new("checkpoints.db").await?;

// Save checkpoint
let checkpoint_id = store.save(agent_id, &state).await?;

// Restore from checkpoint
let state = store.load(checkpoint_id).await?;

// Time-travel: list all checkpoints
let history = store.list(agent_id).await?;
```

## Guardrails

Validate inputs and outputs with tripwire functionality:

```rust
use rust_ai_agents_agents::guardrails::{Guardrail, InputGuardrail, OutputGuardrail};

// Input validation
let input_guard = InputGuardrail::builder()
    .max_length(10000)
    .block_patterns(vec![r"(?i)ignore.*instructions".into()])
    .build();

// Output validation
let output_guard = OutputGuardrail::builder()
    .block_patterns(vec![r"(?i)api[_-]?key".into()])
    .require_patterns(vec![r"(?i)(hello|hi|greetings)".into()])
    .build();
```

## Memory Management

### Basic Memory

```rust
use rust_ai_agents_agents::memory::{AgentMemory, MemoryConfig};

let config = MemoryConfig {
    max_messages: 100,
    max_tokens: 8000,
    summarize_threshold: 50,
};

let memory = AgentMemory::new(config);
memory.add_message(message).await;
```

### Multi-Memory (Hierarchical)

```rust
use rust_ai_agents_agents::multi_memory::{MultiMemory, MemoryType};

let memory = MultiMemory::new();

// Short-term (conversation)
memory.store(MemoryType::ShortTerm, "user asked about weather").await;

// Long-term (persistent facts)
memory.store(MemoryType::LongTerm, "user prefers metric units").await;

// Entity (structured data)
memory.store_entity("user", "name", "Alice").await;
```

## Agent Discovery & Handoff

### Register Agent Capabilities

```rust
use rust_ai_agents_agents::discovery::{AgentRegistry, AgentCapabilities};

let registry = AgentRegistry::new();

let capabilities = AgentCapabilities::builder()
    .skills(vec!["code_review", "testing"])
    .languages(vec!["rust", "python"])
    .build();

registry.register(agent_id, capabilities).await;
```

### Capability-Based Routing

```rust
use rust_ai_agents_agents::handoff::HandoffRouter;

let router = HandoffRouter::new(registry);

// Find agent that can handle Python code review
let target = router.find_agent(&["code_review", "python"]).await?;
router.handoff(source_agent, target, context).await?;
```

## Agent-as-Tool

Compose agents as callable tools for hierarchical execution:

```rust
use rust_ai_agents_agents::agent_tool::AgentTool;

// Wrap an agent as a tool
let code_reviewer = AgentTool::new(
    "code_review",
    "Reviews code for bugs and style issues",
    code_review_agent,
);

// Add to another agent's tool registry
parent_tools.register(code_reviewer);
```

## Self-Correcting Agents

Use LLM-as-Judge pattern for automatic quality improvement:

```rust
use rust_ai_agents_agents::self_correct::{SelfCorrector, CorrectionConfig};

let corrector = SelfCorrector::new(CorrectionConfig {
    max_iterations: 3,
    quality_threshold: 0.8,
    judge_prompt: "Rate the response quality from 0-1...",
});

let improved_response = corrector.correct(agent, initial_response).await?;
```

## Durable Execution

Fault-tolerant execution with automatic recovery:

```rust
use rust_ai_agents_agents::durable::{DurableExecutor, DurabilityConfig};

let executor = DurableExecutor::new(DurabilityConfig {
    checkpoint_interval: Duration::from_secs(30),
    max_retries: 3,
    recovery_strategy: RecoveryStrategy::FromLastCheckpoint,
});

executor.run(agent, task).await?;
```

## Session Management

Structured conversation sessions with persistence:

```rust
use rust_ai_agents_agents::session::{Session, SessionStore};

// Create session
let session = Session::new("user-123");

// Add messages
session.add_user_message("Hello!").await;
session.add_assistant_message("Hi! How can I help?").await;

// Persist
let store = SqliteSessionStore::new("sessions.db").await?;
store.save(&session).await?;

// Resume later
let session = store.load("user-123").await?;
```

## Cost Tracking Integration

Track LLM API costs per agent:

```rust
use rust_ai_agents_monitoring::CostTracker;

let cost_tracker = Arc::new(CostTracker::new());
let engine = AgentEngine::with_cost_tracker(cost_tracker.clone());

// After running agents
let total_cost = cost_tracker.total_cost();
let by_model = cost_tracker.cost_by_model();
```

## Tracing & Observability

All operations are instrumented with `tracing`:

```rust
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_env_filter("rust_ai_agents_agents=debug")
    .init();

// Spans created:
// - agent_loop (per agent)
// - process_message (per message)
// - tool_execution (per tool call)
// - planning_step (when planning enabled)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentEngine                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Agent 1   │  │   Agent 2   │  │   Agent N   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Message Router                      │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ReACT Loop (per agent)                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │   │
│  │  │ Reason  │─▶│   Act   │─▶│ Observe │─┐           │   │
│  │  └─────────┘  └────┬────┘  └─────────┘ │           │   │
│  │       ▲            │                    │           │   │
│  │       └────────────┴────────────────────┘           │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Approvals  │  │  Guardrails  │  │ Checkpoints  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `engine` | Core AgentEngine with ReACT loop |
| `executor` | Parallel tool execution |
| `approvals` | Human-in-the-loop approval system |
| `checkpoint` | State persistence and recovery |
| `guardrails` | Input/output validation |
| `memory` | Conversation history management |
| `multi_memory` | Hierarchical memory system |
| `discovery` | Agent capability registration |
| `handoff` | Capability-based routing |
| `agent_tool` | Agent composition as tools |
| `self_correct` | LLM-as-Judge quality improvement |
| `durable` | Fault-tolerant execution |
| `session` | Conversation session management |
| `planning` | Planning mode implementation |
| `trajectory` | Execution trajectory tracking |

## Related Crates

- [`rust-ai-agents-core`](../core) - Shared types and traits
- [`rust-ai-agents-providers`](../providers) - LLM backends
- [`rust-ai-agents-tools`](../tools) - Tool implementations
- [`rust-ai-agents-monitoring`](../monitoring) - Cost tracking and metrics

## License

Apache-2.0
