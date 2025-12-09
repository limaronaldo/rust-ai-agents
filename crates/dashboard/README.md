# rust-ai-agents-dashboard

Axum-based web dashboard for monitoring and controlling AI agents in real-time.

## Features

- **REST API**: Endpoints for metrics, sessions, traces, and agent control
- **WebSocket**: Real-time updates for live monitoring
- **SSE Streaming**: Server-Sent Events for chat responses
- **Prometheus Metrics**: `/metrics` endpoint for Grafana integration
- **Agent Studio**: Leptos WASM frontend served at `/studio`
- **Audit Logging**: Request/response logging middleware

## Installation

```toml
[dependencies]
rust-ai-agents-dashboard = "0.1"

# Optional features
rust-ai-agents-dashboard = { version = "0.1", features = ["agents", "audit", "streaming"] }
```

## Quick Start

```rust
use rust_ai_agents_dashboard::DashboardServer;
use rust_ai_agents_monitoring::CostTracker;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create cost tracker
    let cost_tracker = Arc::new(CostTracker::new());

    // Create and run dashboard
    let server = DashboardServer::new(cost_tracker);
    server.run("127.0.0.1:3000").await?;

    Ok(())
}
```

## API Endpoints

### Metrics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/metrics` | GET | Current metrics snapshot |
| `/api/stats` | GET | Aggregated statistics |
| `/metrics` | GET | Prometheus metrics (text format) |

```bash
# Get current metrics
curl http://localhost:3000/api/metrics

# Response
{
  "total_requests": 1523,
  "total_tokens": 245000,
  "total_cost_usd": 12.45,
  "active_agents": 3,
  "requests_per_minute": 15.2,
  "avg_latency_ms": 1250
}
```

### Sessions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET | List all sessions |
| `/api/sessions/:id` | GET | Get session details |
| `/api/sessions/:id/messages` | GET | Get session messages |
| `/api/sessions/:id/traces` | GET | Get session traces |

```bash
# List sessions
curl http://localhost:3000/api/sessions

# Get session detail
curl http://localhost:3000/api/sessions/sess-123

# Get messages
curl http://localhost:3000/api/sessions/sess-123/messages
```

### Traces

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/traces` | GET | List execution traces |

```bash
# Get traces with pagination
curl "http://localhost:3000/api/traces?limit=50&offset=0"
```

### Agents

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents` | GET | List all agents |
| `/api/agents/:id` | GET | Get agent details |
| `/api/agents/:id/start` | POST | Start an agent |
| `/api/agents/:id/stop` | POST | Stop an agent |
| `/api/agents/:id/restart` | POST | Restart an agent |

```bash
# List agents
curl http://localhost:3000/api/agents

# Response
[
  {
    "id": "agent-1",
    "name": "Assistant",
    "status": "running",
    "messages_processed": 156,
    "uptime_secs": 3600
  }
]

# Control agent
curl -X POST http://localhost:3000/api/agents/agent-1/stop
curl -X POST http://localhost:3000/api/agents/agent-1/start
curl -X POST http://localhost:3000/api/agents/agent-1/restart
```

### Streaming (SSE)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat/stream` | POST | Stream chat response |

```bash
# Stream a chat response
curl -X POST http://localhost:3000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "claude-3-5-sonnet-20241022"
  }'

# SSE Response
data: {"type": "text", "content": "Hello"}
data: {"type": "text", "content": "!"}
data: {"type": "text", "content": " How"}
data: {"type": "done", "usage": {"input": 10, "output": 25}}
```

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |

```bash
curl http://localhost:3000/health
# {"status": "ok"}
```

## WebSocket

Real-time updates via WebSocket at `/ws`:

```javascript
const ws = new WebSocket('ws://localhost:3000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'metrics':
      updateDashboard(data.metrics);
      break;
    case 'agent_status':
      updateAgentStatus(data.agent_id, data.status);
      break;
    case 'trace':
      addTrace(data.trace);
      break;
  }
};
```

### Message Types

```typescript
// Metrics update (every second)
{
  "type": "metrics",
  "metrics": {
    "total_requests": 1523,
    "total_tokens": 245000,
    "total_cost_usd": 12.45
  }
}

// Agent status change
{
  "type": "agent_status",
  "agent_id": "agent-1",
  "status": "running" | "stopped" | "error"
}

// New trace entry
{
  "type": "trace",
  "trace": {
    "id": "trace-123",
    "agent_id": "agent-1",
    "entry_type": "tool_call",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Pages

| URL | Description |
|-----|-------------|
| `/` | Legacy dashboard (simple HTML) |
| `/studio` | Agent Studio (Leptos WASM app) |

## Agent Studio Integration

The dashboard serves the Agent Studio WASM app:

```rust
use rust_ai_agents_dashboard::{DashboardServer, DashboardBridge};

// Create server with agent integration
let server = DashboardServer::new(cost_tracker);

// Get state for external updates
let state = server.state();

// Update from your agent engine
state.update_agent_status("agent-1", AgentStatus::Running);
state.add_trace(trace_entry);
state.update_session(session);
```

## Audit Logging

Enable audit logging middleware (requires `audit` feature):

```rust
use rust_ai_agents_dashboard::{DashboardServer, audit_middleware};
use rust_ai_agents_audit::AuditLogger;

let logger = Arc::new(AuditLogger::new("api-audit.log"));

// The middleware logs all API requests
let server = DashboardServer::new(cost_tracker)
    .with_audit(logger);
```

Audit endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/audit/logs` | GET | Get audit logs |
| `/api/audit/stats` | GET | Get audit statistics |

```bash
# Query audit logs
curl "http://localhost:3000/api/audit/logs?limit=100&level=warn"
```

## Prometheus Metrics

The `/metrics` endpoint exposes Prometheus-format metrics:

```
# HELP agent_requests_total Total number of agent requests
# TYPE agent_requests_total counter
agent_requests_total{agent="agent-1"} 156

# HELP agent_tokens_total Total tokens used
# TYPE agent_tokens_total counter
agent_tokens_total{agent="agent-1",type="input"} 45000
agent_tokens_total{agent="agent-1",type="output"} 12000

# HELP agent_latency_seconds Request latency
# TYPE agent_latency_seconds histogram
agent_latency_seconds_bucket{le="0.5"} 100
agent_latency_seconds_bucket{le="1.0"} 140
agent_latency_seconds_bucket{le="2.0"} 155
```

See [prometheus-metrics.md](../../docs/prometheus-metrics.md) for Grafana dashboard setup.

## Configuration

```rust
use rust_ai_agents_dashboard::{DashboardServer, DashboardConfig};

let config = DashboardConfig {
    // WebSocket broadcast interval
    broadcast_interval: Duration::from_secs(1),
    
    // Session retention
    max_sessions: 1000,
    session_ttl: Duration::from_hours(24),
    
    // Trace retention
    max_traces: 10000,
    
    // CORS settings
    cors_origins: vec!["http://localhost:3000"],
};

let server = DashboardServer::with_config(cost_tracker, config);
```

## State Management

```rust
use rust_ai_agents_dashboard::{DashboardState, Session, TraceEntry};

// Access state
let state = server.state();

// Add session
state.add_session(Session {
    id: "sess-123".into(),
    agent_id: "agent-1".into(),
    status: SessionStatus::Active,
    messages: vec![],
    created_at: Utc::now(),
});

// Add trace
state.add_trace(TraceEntry {
    id: "trace-456".into(),
    session_id: Some("sess-123".into()),
    agent_id: "agent-1".into(),
    entry_type: TraceEntryType::ToolCall,
    content: json!({"tool": "search", "args": {"q": "test"}}),
    timestamp: Utc::now(),
});

// Update metrics
state.increment_requests();
state.add_tokens(1500);
state.record_latency(Duration::from_millis(850));
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DashboardServer                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   Axum Router                         │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │  │
│  │  │  REST   │ │   WS    │ │   SSE   │ │ Static  │    │  │
│  │  │ /api/*  │ │  /ws    │ │/stream  │ │/studio  │    │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │  │
│  └───────┼───────────┼───────────┼───────────┼──────────┘  │
│          │           │           │           │              │
│  ┌───────▼───────────▼───────────▼───────────▼──────────┐  │
│  │                  DashboardState                       │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │  │
│  │  │ Metrics  │ │ Sessions │ │  Traces  │ │ Agents  │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│          │                                                  │
│  ┌───────▼──────────────────────────────────────────────┐  │
│  │              Integration Layer                        │  │
│  │  AgentBridge | SessionBridge | TrajectoryBridge      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Related Crates

- [`rust-ai-agents-studio`](../studio) - Leptos WASM frontend
- [`rust-ai-agents-monitoring`](../monitoring) - Metrics and cost tracking
- [`rust-ai-agents-agents`](../agents) - Agent implementation
- [`rust-ai-agents-audit`](../audit) - Audit logging

## License

Apache-2.0
