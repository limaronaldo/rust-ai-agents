# rust-ai-agents-mcp

Model Context Protocol (MCP) support for rust-ai-agents.

## Overview

This crate provides both **client** and **server** implementations of the [Model Context Protocol](https://modelcontextprotocol.io/), enabling:

- **Client**: Connect to MCP servers and use their tools in your agents
- **Server**: Expose your tools and agents as MCP servers for use with Claude Desktop, VS Code, etc.

## Features

- Full MCP protocol implementation (JSON-RPC 2.0)
- STDIO and SSE transports
- Tool, Resource, and Prompt support
- Easy tool bridging between MCP and rust-ai-agents

## Quick Start

### Client: Connect to an MCP Server

```rust
use rust_ai_agents_mcp::{McpClient, StdioTransport};

// Connect to an MCP filesystem server
let transport = StdioTransport::spawn(
    "npx",
    &["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
).await?;

let mut client = McpClient::new(transport).await?;

// List available tools
let tools = client.list_tools().await?;

// Call a tool
let result = client.call_tool("read_file", json!({"path": "/tmp/test.txt"})).await?;
```

### Server: Create an MCP Server

```rust
use rust_ai_agents_mcp::{McpServer, ToolHandler, McpTool, CallToolResult, ToolContent};

struct MyTool;

#[async_trait]
impl ToolHandler for MyTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "my_tool".to_string(),
            description: Some("Does something useful".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }),
        }
    }

    async fn execute(&self, args: serde_json::Value) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult {
            content: vec![ToolContent::text("Result!")],
            is_error: false,
        })
    }
}

// Build and run the server
let server = McpServer::builder()
    .name("my-server")
    .version("1.0.0")
    .add_tool(MyTool)
    .build();

server.run_stdio().await?;
```

### Server: HTTP/SSE Transport

For web-based clients, use the SSE transport:

```rust
use rust_ai_agents_mcp::{McpServer, SseServerConfig};

let server = McpServer::builder()
    .name("my-sse-server")
    .version("1.0.0")
    .add_tool(MyTool)
    .build();

// Configure SSE server
let config = SseServerConfig {
    host: "127.0.0.1".to_string(),
    port: 3000,
    sse_path: "/sse".to_string(),
    message_path: "/message".to_string(),
    enable_cors: true,
    keep_alive_secs: 30,
};

// Run HTTP server (blocks until shutdown)
server.run_sse(config).await?;
```

Endpoints:
- `GET /sse` - SSE stream for server-to-client messages
- `POST /message?sessionId=xxx` - JSON-RPC requests from client

## Using with Claude Desktop

1. Build your MCP server:
```bash
cargo build --release --example mcp_server
```

2. Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):
```json
{
  "mcpServers": {
    "rust-ai-agents": {
      "command": "/path/to/target/release/examples/mcp_server"
    }
  }
}
```

3. Restart Claude Desktop - your tools will be available!

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Crate                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │   McpClient     │          │   McpServer     │          │
│  │                 │          │                 │          │
│  │ - list_tools()  │          │ - add_tool()    │          │
│  │ - call_tool()   │          │ - run_stdio()   │          │
│  │ - list_resources│          │ - handle_*()    │          │
│  └────────┬────────┘          └────────┬────────┘          │
│           │                            │                    │
│  ┌────────┴────────────────────────────┴────────┐          │
│  │              McpTransport Trait               │          │
│  ├──────────────────┬───────────────────────────┤          │
│  │  StdioTransport  │      SseTransport         │          │
│  │  (subprocess)    │      (HTTP/SSE)           │          │
│  └──────────────────┴───────────────────────────┘          │
│                                                             │
│  ┌─────────────────────────────────────────────┐           │
│  │              McpToolBridge                   │           │
│  │  (Convert MCP tools to rust-ai-agents Tool) │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Modules

### `client`
MCP client for connecting to external servers.

### `server`
MCP server for exposing tools to clients.

### `transport`
Transport implementations (STDIO, SSE).

### `bridge`
Bridge MCP tools to rust-ai-agents Tool trait.

### `protocol`
MCP protocol types (JSON-RPC, capabilities, tool definitions).

## Tool Handler Trait

Implement `ToolHandler` to create MCP tools:

```rust
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Get the tool definition (name, description, schema)
    fn definition(&self) -> McpTool;

    /// Execute the tool with the given arguments
    async fn execute(&self, arguments: serde_json::Value) -> Result<CallToolResult, McpError>;
}
```

## Function-Based Tools

For simple tools, use `FnTool` or `AsyncFnTool`:

```rust
use rust_ai_agents_mcp::FnTool;

let add_tool = FnTool::new(
    "add",
    "Adds two numbers",
    json!({
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        }
    }),
    |args| {
        let a = args["a"].as_f64().unwrap_or(0.0);
        let b = args["b"].as_f64().unwrap_or(0.0);
        Ok(json!(a + b))
    },
);
```

## Resource and Prompt Handlers

For resources and prompts, implement the respective traits:

```rust
#[async_trait]
pub trait ResourceHandler: Send + Sync {
    fn list(&self) -> Vec<McpResource>;
    async fn read(&self, uri: &str) -> Result<ResourceContent, McpError>;
}

#[async_trait]
pub trait PromptHandler: Send + Sync {
    fn list(&self) -> Vec<McpPrompt>;
    async fn get(&self, name: &str, arguments: HashMap<String, String>) 
        -> Result<Vec<PromptMessage>, McpError>;
}
```

## Examples

See the examples directory:

- `examples/mcp_server.rs` - MCP server with STDIO transport (for Claude Desktop)
- `examples/mcp_sse_server.rs` - MCP server with HTTP/SSE transport (for web clients)
- `examples/mcp_integration.rs` - Client connecting to external MCP servers

Run examples:
```bash
# STDIO server example (for Claude Desktop)
cargo run --example mcp_server

# SSE server example (HTTP on port 3000)
cargo run --example mcp_sse_server

# Client example (requires Node.js/npx)
cargo run --example mcp_integration
```

## Protocol Version

This implementation supports MCP protocol version `2024-11-05`.

## License

Apache-2.0
