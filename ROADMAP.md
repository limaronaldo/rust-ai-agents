# HyperAgent Roadmap

**Last Updated:** December 7, 2025  
**Status:** Active Development  
**License:** Apache-2.0

---

## Vision

HyperAgent is a **Rust-native AI agent framework** designed for high-performance, data-intensive applications. Unlike Python-based frameworks that focus primarily on LLM orchestration, HyperAgent combines:

- **Deterministic Data Plane**: Native integration with PostgreSQL, DuckDB, Meilisearch
- **High Performance**: Rust's zero-cost abstractions for production workloads
- **Enterprise Ready**: Built-in observability, durability, and security features
- **Interoperability**: Designed to work alongside LangChain, CrewAI, and other frameworks

---

## Design Principles

1. **Data-first**: Every agent has a deterministic Data Plane before the LLM reasoning layer
2. **Safe-by-default**: Guardrails, approvals, and tracing are defaults, not add-ons
3. **Interop over lock-in**: MCP, HTTP APIs, and pluggable backends for coexistence with other frameworks
4. **Rust-native**: Zero-cost abstractions, no `unsafe`, focus on predictable performance

---

## Who is this for?

| Persona | How they use HyperAgent |
|---------|------------------------|
| **Data Engineer** | Connect PostgreSQL/DuckDB/Meilisearch and expose investigations via API |
| **MLOps/Platform** | Orchestrate multiple models/LLMs with observability and quotas |
| **Compliance/Anti-fraud** | Investigate entities, track history, validate sensitive actions |
| **Product/AI Engineer** | Create specialized agents pluggable into CRM, backoffice, etc. |

---

## Interoperability Story

HyperAgent integrates with your existing stack in three ways:

- **MCP Client/Server** → Integrates with Claude Desktop, VS Code, and MCP-compatible tools
- **HTTP API** → Integrates with LangChain/CrewAI as an "external tool" via REST
- **Rust crate** → Integrates with other Rust services (Axum, Tonic, etc.) as a library

---

## Current State (v0.1.0) ✅

### Agent Core (19 modules, 142 tests passing)

| Category | Module | Description | Tests |
|----------|--------|-------------|-------|
| **Core Loop** | `engine.rs` | ReACT Loop with Planning Mode support | ✅ |
| **Planning** | `planning.rs` | Plan generation and step execution | ✅ |
| **Memory** | `memory.rs` | Basic conversation memory | ✅ |
| **Memory** | `multi_memory.rs` | Short-term (LRU+TTL), Long-term (vector), Entity | 13 |
| **Discovery** | `discovery.rs` | Agent discovery via heartbeat/capabilities | ✅ |
| **Routing** | `handoff.rs` | Capability-based routing between agents | ✅ |
| **Safety** | `guardrails.rs` | Input/output validation with tripwire | 12 |
| **Persistence** | `checkpoint.rs` | State persistence, time-travel, fork/rewind | 11 |
| **Composition** | `agent_tool.rs` | Agent-as-Tool, DelegationChain | 10 |
| **Quality** | `self_correct.rs` | LLM-as-Judge, auto-retry with feedback | 11 |
| **Factory** | `factory.rs` | Runtime instantiation, templates, pooling | 12 |
| **Durability** | `durable.rs` | Fault-tolerant execution, resume from failure | 10 |
| **Sessions** | `session.rs` | Conversation memory, turns, context window | 14 |
| **Observability** | `trajectory.rs` | Execution trace (events, steps, tools) | ✅ |
| **RAG** | `vector_store.rs` | Semantic search, chunking, embeddings | ✅ |
| **Storage** | `persistence.rs` | Session persistence (memory, sled) | ✅ |
| **Storage** | `sqlite_store.rs` | SQLite backend for sessions/vectors | ✅ |

### LLM Providers (`crates/providers`)

| Backend | Location | Status |
|---------|----------|--------|
| `LLMBackend` trait | `backend.rs` | ✅ Implemented |
| `OpenAIBackend` | `openai.rs` | ✅ Implemented |
| `AnthropicBackend` | `anthropic.rs` | ✅ Implemented |
| `OpenRouterBackend` | `openrouter.rs` | ✅ Implemented |
| Streaming (`infer_stream`) | `backend.rs` | ✅ Implemented |
| `RateLimiter` | `backend.rs` | ✅ Implemented |

### Core Types (`crates/core`)

| Type | Location | Status |
|------|----------|--------|
| `LLMMessage`, `MessageRole` | `message.rs` | ✅ Implemented |
| `ToolCall`, `ToolResult` | `message.rs` | ✅ Implemented |
| `ToolSchema` with `dangerous` flag | `tool.rs` | ✅ Implemented |
| `Tool` trait, `ToolRegistry` | `tool.rs` | ✅ Implemented |
| `PlanningMode` (Disabled/BeforeTask/FullPlan/Adaptive) | `types.rs` | ✅ Implemented |
| `ExecutionPlan`, `PlanStep` | `types.rs` | ✅ Implemented |

### MCP Support (`crates/mcp`) ✅ NEW

| Component | File | Status |
|-----------|------|--------|
| `McpClient` | `client.rs` | ✅ Implemented |
| `StdioTransport` | `transport.rs` | ✅ Implemented |
| `SseTransport` | `transport.rs` | ✅ Implemented |
| `McpToolBridge` | `bridge.rs` | ✅ Implemented |
| Protocol types | `protocol.rs` | ✅ Implemented |

### Workflow DSL (`crates/workflow`) ✅ NEW

| Component | File | Status |
|-----------|------|--------|
| YAML Parser | `parser.rs` | ✅ Implemented |
| Workflow Schema | `schema.rs` | ✅ Implemented |
| Workflow Runner | `runner.rs` | ✅ Implemented |
| Variable interpolation `{{var}}` | `runner.rs` | ✅ Implemented |

### Additional Crates

| Crate | Description | Status |
|-------|-------------|--------|
| `crates/cache` | Caching layer | ✅ Implemented |
| `crates/resilience` | Resilience patterns | ✅ Implemented |
| `crates/github-ops` | GitHub operations | ✅ Implemented |
| `crates/crew` | Multi-agent crews | ✅ Implemented |
| `crates/dashboard` | Dashboard API | ✅ Implemented |
| `crates/wasm` | WASM support | 🔲 Scaffolded |

### Data Plane & APIs (Brazil-First)

| Category | Module / Example | Description |
|----------|------------------|-------------|
| **Data Plane** | `data/matcher.rs`, `data/pipeline.rs` | Entity matching, cross-source pipeline |
| **Data Plane** | `data/extractors/brazilian.rs` | Parties, People, IPTU extractors (mock BR) |
| **API** | `examples/brazilian_entity_api.rs` | Axum API + Swagger for investigation |
| **CLI** | `examples/brazilian_entity_investigation.rs` | Offline investigation CLI |
| **MCP** | `examples/mcp_integration.rs` | MCP server integration example |

---

## Milestone: v0.2.0 – MockBackend & Approvals ✅

> **Status:** ✅ Completed  
> **Completed:** December 7, 2025

This milestone delivered:
1. `MockBackend` for deterministic testing without API calls
2. **Human-in-the-loop** approvals for sensitive tools (using existing `ToolSchema.dangerous`)

### 🧪 Testing Infrastructure

| Status | Issue | Priority | Description |
|:------:|:------|:--------:|:------------|
| ✅ | **feat: implement MockBackend for tests** | 🔥 High | `crates/providers/src/mock.rs` - 13 tests |
| ✅ | **audit: verify engine uses LLMBackend correctly** | 🔹 Med | Confirmed clean abstraction |
| 🔴 | **test: engine integration with MockBackend** | 🔹 Med | End-to-end tests of ReACT loop using `MockBackend` |

### 🛡️ Human-in-the-Loop (Approvals)

Leverages existing `ToolSchema.dangerous: bool` flag from `crates/core/src/tool.rs`.

| Status | Issue | Priority | Description |
|:------:|:------|:--------:|:------------|
| ✅ | **feat: ApprovalConfig and ApprovalHandler** | 🔥 High | `crates/agents/src/approvals.rs` |
| ✅ | **feat: TerminalApprovalHandler** | 🔹 Med | CLI handler (Y/N/S/A/D) for local development |
| ✅ | **feat: TestApprovalHandler** | 🔹 Med | Pre-configured handler for automated tests |
| ✅ | **feat: AutoApproveHandler** | 🔹 Med | Bypass handler for non-interactive use |
| ✅ | **feat: integrate approvals into executor** | 🔥 High | `executor.rs` updated - 6 tests |
| ✅ | **test: approval flows** | 🔹 Med | 11 tests covering all decision paths |

### 🔐 Safety Sandwich (Architecture)

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

### 📚 Documentation

| Status | Issue | Priority | Description |
|:------:|:------|:--------:|:------------|
| 🔴 | **docs: LLM Backends guide** | 🟢 Low | Document existing backends + MockBackend |
| 🔴 | **docs: Approvals guide** | 🟢 Low | Document Safety Sandwich, handlers |

---

## Phase 2: MCP Server & Polish (Q1 2026)

> **Status:** 🚧 In Progress  
> **Updated:** December 7, 2025

**Goal:** Expose HyperAgent as an MCP Server for external tools to call.

### MCP Client ✅ (Completed in v0.1.0)

- `McpClient` - Connect to external MCP servers
- `StdioTransport` - Subprocess-based MCP
- `SseTransport` - HTTP SSE-based MCP
- `McpToolBridge` - Convert MCP tools to agent tools

### MCP Server ✅ (Completed December 7, 2025)

| Status | Task | Description |
|:------:|:-----|:------------|
| ✅ | `McpServer` | `crates/mcp/src/server.rs` - Full MCP server implementation |
| ✅ | Stdio server transport | Accept connections via stdin/stdout |
| ✅ | Tool handlers | `ToolHandler`, `FnTool`, `AsyncFnTool` wrappers |
| ✅ | Resources support | `ResourceHandler` trait for exposing resources |
| ✅ | Prompts support | `PromptHandler` trait for prompt templates |
| ✅ | Example server | `examples/mcp_server.rs` - 5 demo tools |
| ✅ | Documentation | `crates/mcp/README.md` with Claude Desktop config |

### Remaining Work

| Status | Task | Description |
|:------:|:-----|:------------|
| 🔴 | SSE server transport | HTTP SSE-based server for web clients |
| 🔴 | Agent-as-MCP-tool | Expose agents directly as MCP tools |
| 🔴 | Session resources | Expose conversation sessions via MCP resources |
| 🔴 | Trace resources | Expose execution traces via MCP resources |

### Integration Example (Target)

**Claude Desktop (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "hyperagent": {
      "command": "hyperagent-mcp",
      "args": ["--config", "~/.hyperagent/config.toml"]
    }
  }
}
```

---

## Phase 3: Agent Studio UI (Q2 2026)

**Goal:** Web-based interface for monitoring, debugging, and managing agents.

### Architecture

- **Trace Viewer** → Built on `trajectory.rs`
- **Session Browser** → Built on `durable.rs` + `session.rs`
- **Live Execution** → WebSocket streaming from `engine.rs`

### Tech Stack

- **Frontend:** Next.js 14 + React + Tailwind + shadcn/ui
- **Backend:** Axum (Rust) via `crates/dashboard`
- **Real-time:** WebSocket for live updates

---

## Phase 4: Production Hardening (Q3 2026)

**Goal:** Enterprise-ready features for production deployments.

### Features

| Feature | Status | Description |
|---------|--------|-------------|
| Audit Logging | 🔴 | All tool calls and LLM requests logged |
| Rate Limiting | ✅ | Already in `crates/providers` |
| Resilience | ✅ | `crates/resilience` implemented |
| Caching | ✅ | `crates/cache` implemented |
| Encryption | 🔴 | At-rest encryption for sensitive data |

---

## Phase 5: Edge & WASM (Q4 2026)

**Goal:** Run HyperAgent in edge environments and browsers.

| Platform | Status |
|----------|--------|
| WASM crate scaffolded | ✅ `crates/wasm` exists |
| Cloudflare Workers | 🔲 Planned |
| Browser (WASM) | 🔲 Planned |

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Test Coverage | >80% | ~75% |
| Documentation Coverage | 100% public API | ~60% |
| Crates | 10+ | 17 ✅ |
| Benchmark: Simple agent execution | <50ms | TBD |
| Benchmark: BR entity investigation (3 sources) | <80ms | TBD |

---

## GitHub Milestones

| Milestone | Target | Focus |
|-----------|--------|-------|
| `v0.2.0` | Q4 2025 | MockBackend + Approvals |
| `v0.3.0` | Q1 2026 | MCP Server |
| `v0.4.0` | Q2 2026 | Agent Studio |
| `v0.5.0` | Q3 2026 | Production Hardening |
| `v0.6.0` | Q4 2026 | Edge & WASM |

---

## Contributing

We welcome contributions! Priority areas:

1. **MockBackend** - Deterministic testing infrastructure
2. **MCP Server** - Expose agents via MCP
3. **Examples** - Real-world use cases
4. **Documentation** - API docs and guides

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## References

This roadmap was informed by analysis of leading AI agent frameworks:

| Framework | Key Learnings |
|-----------|---------------|
| **LangChain/LangGraph** | Tool abstraction, chain composition |
| **CrewAI** | Role-Goal-Backstory pattern |
| **AutoGen** | Multi-agent conversations |
| **OpenAI Agents SDK** | Handoffs, guardrails |
| **Mastra** | Developer experience, observability |
| **swarms-rs** | Rust patterns for agents |

---

**Maintained by:** Ronaldo Lima
**License:** Apache-2.0
