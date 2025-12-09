# rust-ai-agents-crew

Multi-agent orchestration system for coordinating tasks across multiple agents.

## Features

- **Crew**: Traditional multi-agent coordination with sequential/parallel/hierarchical processes
- **Graph**: LangGraph-style graph execution with cycles, conditionals, and checkpointing
- **Workflow**: DAG-based workflow system with conditional branching
- **Orchestra**: Multi-perspective analysis with parallel execution and synthesis
- **Handoff**: Agent-to-agent handoffs with context passing
- **Human Loop**: Human-in-the-loop with approval gates and breakpoints
- **Time Travel**: State history, replay, fork, and debugging
- **Streaming**: Unified streaming events across all execution modes
- **Voting**: Consensus mechanisms for multi-agent decisions
- **Novelty Detection**: Detect answer diversity and avoid repetition

## Installation

```toml
[dependencies]
rust-ai-agents-crew = "0.1"
```

## Quick Start

### Basic Crew

```rust
use rust_ai_agents_crew::{Crew, CrewConfig, Process, Task};
use rust_ai_agents_agents::AgentEngine;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = Arc::new(AgentEngine::new());

    // Configure crew
    let config = CrewConfig::new("Research Team")
        .with_description("Team for research tasks")
        .with_process(Process::Sequential)
        .with_max_concurrency(4);

    let mut crew = Crew::new(config, engine);

    // Add agents
    crew.add_agent(researcher_config);
    crew.add_agent(writer_config);
    crew.add_agent(reviewer_config);

    // Add tasks
    crew.add_task(Task::new("research", "Research the topic")
        .with_agent("researcher"))?;
    crew.add_task(Task::new("write", "Write the report")
        .with_agent("writer")
        .depends_on("research"))?;
    crew.add_task(Task::new("review", "Review the report")
        .with_agent("reviewer")
        .depends_on("write"))?;

    // Execute
    let results = crew.kickoff().await?;

    Ok(())
}
```

## Execution Processes

### Sequential

Tasks execute one after another in order:

```rust
let config = CrewConfig::new("Sequential Crew")
    .with_process(Process::Sequential);
```

### Parallel

Independent tasks execute concurrently:

```rust
let config = CrewConfig::new("Parallel Crew")
    .with_process(Process::Parallel)
    .with_max_concurrency(4);
```

### Hierarchical

Manager agent delegates to workers:

```rust
let config = CrewConfig::new("Hierarchical Crew")
    .with_process(Process::Hierarchical {
        manager: "coordinator".into(),
    });
```

## Graph Execution (LangGraph-style)

Build complex workflows with cycles and conditionals:

```rust
use rust_ai_agents_crew::graph::{GraphBuilder, GraphState};

// Create a reasoning loop
let graph = GraphBuilder::new("reasoning_loop")
    .add_node("think", |state| async move {
        // Reasoning step
        state.set("thought", "I should search for more info");
        Ok(state)
    })
    .add_node("act", |state| async move {
        // Action step
        let result = execute_action(&state).await?;
        state.set("action_result", result);
        Ok(state)
    })
    .add_node("evaluate", |state| async move {
        // Check if done
        let done = check_completion(&state);
        state.set("done", done);
        Ok(state)
    })
    // Define edges
    .add_edge("think", "act")
    .add_edge("act", "evaluate")
    // Conditional: loop back or finish
    .add_conditional_edge("evaluate", |state| {
        if state.get::<bool>("done").unwrap_or(false) {
            "END".to_string()
        } else {
            "think".to_string()  // Loop back
        }
    })
    .set_entry("think")
    .set_finish("END")
    .max_iterations(10)  // Prevent infinite loops
    .build()?;

// Execute
let initial = GraphState::from_json(json!({"query": "research topic"}));
let result = graph.invoke(initial).await?;
```

### Parallel Branches

```rust
let graph = GraphBuilder::new("parallel_research")
    .add_node("start", start_node)
    .add_node("search_web", web_search_node)
    .add_node("search_docs", doc_search_node)
    .add_node("merge", merge_results_node)
    // Fan out
    .add_edge("start", "search_web")
    .add_edge("start", "search_docs")
    // Fan in
    .add_edge("search_web", "merge")
    .add_edge("search_docs", "merge")
    .set_entry("start")
    .set_finish("merge")
    .build()?;
```

## Workflow (DAG-based)

Simpler DAG workflows without cycles:

```rust
use rust_ai_agents_crew::workflow::{Workflow, WorkflowBuilder, WorkflowStep};

let workflow = WorkflowBuilder::new("data_pipeline")
    .add_step(WorkflowStep::new("fetch", fetch_data))
    .add_step(WorkflowStep::new("transform", transform_data)
        .depends_on("fetch"))
    .add_step(WorkflowStep::new("validate", validate_data)
        .depends_on("transform"))
    .add_step(WorkflowStep::new("store", store_data)
        .depends_on("validate"))
    // Conditional step
    .add_step(WorkflowStep::new("notify", send_notification)
        .depends_on("store")
        .when(|ctx| ctx.get("notify_on_complete").unwrap_or(false)))
    .build()?;

let result = workflow.execute(context).await?;
```

## Orchestra (Multi-Perspective)

Get multiple perspectives and synthesize:

```rust
use rust_ai_agents_crew::orchestra::{Orchestra, OrchestraConfig, Perspective};

let orchestra = Orchestra::new(OrchestraConfig {
    name: "Analysis Orchestra".into(),
    perspectives: vec![
        Perspective::new("technical", technical_agent),
        Perspective::new("business", business_agent),
        Perspective::new("legal", legal_agent),
    ],
    synthesizer: synthesis_agent,
    parallel: true,
});

let analysis = orchestra.analyze("Should we adopt this technology?").await?;
// analysis.perspectives: Vec<PerspectiveResult>
// analysis.synthesis: String
```

## Human-in-the-Loop

Add human approval gates and input collection:

```rust
use rust_ai_agents_crew::human_loop::{HumanLoop, ApprovalGate, Breakpoint};

let graph = GraphBuilder::new("with_human")
    .add_node("draft", draft_node)
    .add_node("human_review", HumanLoop::approval_gate(
        "Please review and approve this draft",
        ApprovalGate::Required,
    ))
    .add_node("finalize", finalize_node)
    .add_edge("draft", "human_review")
    .add_edge("human_review", "finalize")
    .build()?;

// During execution, human_review will pause and wait for approval
let result = graph.invoke_with_handler(state, human_handler).await?;
```

### Breakpoints

```rust
// Pause at specific nodes for debugging
let graph = GraphBuilder::new("debuggable")
    .add_breakpoint("critical_step")
    .build()?;
```

## Time Travel

Debug and replay graph executions:

```rust
use rust_ai_agents_crew::time_travel::{TimeTravelDebugger, ExecutionHistory};

// Enable history tracking
let graph = GraphBuilder::new("tracked")
    .enable_time_travel()
    .build()?;

let result = graph.invoke(state).await?;

// Access history
let history = graph.history();

// Replay from a checkpoint
let replayed = history.replay_from(checkpoint_id).await?;

// Fork from a point in history
let forked = history.fork(checkpoint_id, modified_state).await?;

// Visualize execution
history.print_timeline();
```

## Agent Handoff

Transfer context between agents:

```rust
use rust_ai_agents_crew::handoff::{HandoffManager, HandoffContext};

let manager = HandoffManager::new(engine);

// Register agents with capabilities
manager.register("researcher", vec!["research", "analysis"]);
manager.register("writer", vec!["writing", "editing"]);

// Handoff with context
let context = HandoffContext::new()
    .with_data("research_results", results)
    .with_instructions("Write a summary based on the research");

let result = manager.handoff("researcher", "writer", context).await?;

// Handoff with return (agent returns control)
let result = manager.handoff_with_return("main", "specialist", context).await?;
```

## Voting & Consensus

Multi-agent voting for decisions:

```rust
use rust_ai_agents_crew::voting::{VotingSystem, VotingStrategy, Vote};

let voting = VotingSystem::new(VotingStrategy::Majority);

// Collect votes from agents
let votes = vec![
    Vote::new("agent-1", "option_a", 0.9),
    Vote::new("agent-2", "option_b", 0.7),
    Vote::new("agent-3", "option_a", 0.8),
];

let result = voting.tally(votes)?;
// result.winner: "option_a"
// result.confidence: 0.85
```

### Voting Strategies

```rust
// Simple majority
VotingStrategy::Majority

// Weighted by confidence
VotingStrategy::WeightedMajority

// Unanimous required
VotingStrategy::Unanimous

// Custom threshold
VotingStrategy::Threshold(0.75)
```

## Novelty Detection

Detect and encourage answer diversity:

```rust
use rust_ai_agents_crew::novelty::{NoveltyDetector, NoveltyConfig};

let detector = NoveltyDetector::new(NoveltyConfig {
    similarity_threshold: 0.8,
    min_unique_concepts: 3,
});

// Check if response adds new information
let is_novel = detector.is_novel(&new_response, &previous_responses)?;

// Get novelty score
let score = detector.novelty_score(&response, &history)?;
```

## Rate Limiting

Per-model rate limiting for crew execution:

```rust
use rust_ai_agents_crew::rate_limit::{CrewRateLimiter, ModelLimits};

let limiter = CrewRateLimiter::new()
    .add_limit("claude-3-5-sonnet", ModelLimits {
        requests_per_minute: 60,
        tokens_per_minute: 100_000,
    })
    .add_limit("gpt-4o", ModelLimits {
        requests_per_minute: 500,
        tokens_per_minute: 150_000,
    });

let crew = Crew::new(config, engine)
    .with_rate_limiter(limiter);
```

## Streaming

Stream events from crew execution:

```rust
use rust_ai_agents_crew::streaming::{CrewStream, StreamEvent};
use futures::StreamExt;

let stream = crew.kickoff_stream().await;

while let Some(event) = stream.next().await {
    match event {
        StreamEvent::TaskStarted { task_id, agent_id } => {
            println!("Task {} started by {}", task_id, agent_id);
        }
        StreamEvent::AgentThinking { agent_id, thought } => {
            println!("{} thinking: {}", agent_id, thought);
        }
        StreamEvent::ToolCall { agent_id, tool, args } => {
            println!("{} calling {}({:?})", agent_id, tool, args);
        }
        StreamEvent::TaskCompleted { task_id, result } => {
            println!("Task {} completed: {}", task_id, result);
        }
        StreamEvent::Error { task_id, error } => {
            eprintln!("Task {} failed: {}", task_id, error);
        }
    }
}
```

## Cancellation

Cancel crew execution gracefully:

```rust
use rust_ai_agents_crew::cancellation::CancellationToken;

let token = CancellationToken::new();
let token_clone = token.clone();

// Start crew in background
tokio::spawn(async move {
    crew.kickoff_with_cancellation(token_clone).await
});

// Cancel after timeout
tokio::time::sleep(Duration::from_secs(30)).await;
token.cancel();
```

## Coordination Tracking

Track multi-agent coordination:

```rust
use rust_ai_agents_crew::coordination::{CoordinationTracker, CoordinationEvent};

let tracker = CoordinationTracker::new();

// Track events
tracker.record(CoordinationEvent::Handoff {
    from: "agent-1".into(),
    to: "agent-2".into(),
    context_size: 1500,
});

// Get coordination summary
let summary = tracker.summary();
// summary.total_handoffs: 5
// summary.avg_context_size: 1200
// summary.bottleneck_agents: ["agent-3"]
```

## Task Management

```rust
use rust_ai_agents_crew::task_manager::{TaskManager, Task, TaskPriority};

let mut manager = TaskManager::new(4); // max 4 concurrent

// Add tasks with priority
manager.add_task(Task::new("urgent", "Handle urgent request")
    .with_priority(TaskPriority::High))?;

manager.add_task(Task::new("background", "Background processing")
    .with_priority(TaskPriority::Low))?;

// Get next task
let next = manager.next_task()?;
```

## Restart & Recovery

Handle failures gracefully:

```rust
use rust_ai_agents_crew::restart::{RestartPolicy, RestartConfig};

let config = CrewConfig::new("Resilient Crew")
    .with_restart_policy(RestartPolicy::OnFailure {
        max_retries: 3,
        backoff: Duration::from_secs(5),
    });

// Or exponential backoff
let config = CrewConfig::new("Crew")
    .with_restart_policy(RestartPolicy::ExponentialBackoff {
        initial_delay: Duration::from_secs(1),
        max_delay: Duration::from_secs(60),
        multiplier: 2.0,
    });
```

## Subgraphs

Nest graphs for modularity:

```rust
use rust_ai_agents_crew::subgraph::Subgraph;

// Create reusable subgraph
let research_subgraph = GraphBuilder::new("research")
    .add_node("search", search_node)
    .add_node("analyze", analyze_node)
    .add_edge("search", "analyze")
    .build()?;

// Use in parent graph
let main_graph = GraphBuilder::new("main")
    .add_node("start", start_node)
    .add_subgraph("research", research_subgraph)
    .add_node("report", report_node)
    .add_edge("start", "research")
    .add_edge("research", "report")
    .build()?;
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Crew                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Task Manager                        │   │
│  │  [Task 1] → [Task 2] → [Task 3] → [Task 4]         │   │
│  └─────────────────────────────────────────────────────┘   │
│         │           │           │           │               │
│         ▼           ▼           ▼           ▼               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │ Agent 4 │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│         │           │           │           │               │
│         └───────────┴─────┬─────┴───────────┘               │
│                           │                                  │
│  ┌────────────────────────▼────────────────────────────┐   │
│  │              Coordination Layer                      │   │
│  │  Handoff | Voting | HumanLoop | TimeTravel          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `crew` | Core crew coordination |
| `graph` | LangGraph-style execution |
| `workflow` | DAG workflow system |
| `orchestra` | Multi-perspective analysis |
| `handoff` | Agent-to-agent handoffs |
| `human_loop` | Human-in-the-loop support |
| `time_travel` | State history and debugging |
| `voting` | Consensus mechanisms |
| `novelty` | Answer diversity detection |
| `streaming` | Execution event streaming |
| `subgraph` | Nested graph support |
| `rate_limit` | Per-model rate limiting |
| `cancellation` | Graceful cancellation |
| `coordination` | Coordination tracking |
| `task_manager` | Task scheduling |
| `restart` | Failure recovery |

## Related Crates

- [`rust-ai-agents-agents`](../agents) - Agent implementation
- [`rust-ai-agents-core`](../core) - Shared types
- [`rust-ai-agents-providers`](../providers) - LLM backends

## License

Apache-2.0
