//! MCP Integration Example
//!
//! This example demonstrates how to use the Model Context Protocol (MCP)
//! to connect to external tool servers and use their tools with agents.
//!
//! # Running the Example
//!
//! 1. Make sure you have Node.js installed for npx
//! 2. Run: `cargo run --example mcp_integration`
//!
//! # What This Example Does
//!
//! 1. Connects to the MCP filesystem server via STDIO transport
//! 2. Lists available tools from the server
//! 3. Demonstrates calling tools directly
//! 4. Shows how to bridge MCP tools to rust-ai-agents Tool trait

use anyhow::Result;
use rust_ai_agents_core::tool::{ExecutionContext, ToolRegistry};
use rust_ai_agents_core::types::AgentId;
use rust_ai_agents_mcp::{McpClient, McpToolBridge, StdioTransport};
use tracing::info;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("MCP Integration Example");
    info!("========================");

    // Example 1: Direct MCP Client Usage
    info!("\n--- Example 1: Direct MCP Client ---");
    direct_client_example().await?;

    // Example 2: Tool Bridge (convert MCP tools to agent tools)
    info!("\n--- Example 2: MCP Tool Bridge ---");
    tool_bridge_example().await?;

    info!("\nMCP Integration Example completed!");
    Ok(())
}

/// Example 1: Using MCP client directly
async fn direct_client_example() -> Result<()> {
    // Create a temporary directory for the filesystem server
    let temp_dir = std::env::temp_dir().join("mcp_example");
    std::fs::create_dir_all(&temp_dir)?;

    // Create a test file
    let test_file = temp_dir.join("hello.txt");
    std::fs::write(&test_file, "Hello from MCP!")?;

    info!("Created test file at: {}", test_file.display());

    // Spawn the MCP filesystem server
    // Note: This requires npx and the @modelcontextprotocol/server-filesystem package
    info!("Spawning MCP filesystem server...");

    let transport = match StdioTransport::spawn(
        "npx",
        &[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            temp_dir.to_str().unwrap(),
        ],
    )
    .await
    {
        Ok(t) => t,
        Err(e) => {
            info!("Could not spawn MCP server (npx not available?): {}", e);
            info!("Skipping direct client example - this requires Node.js/npx");
            return Ok(());
        }
    };

    // Create MCP client
    let mut client = McpClient::new(transport).await?;

    // Get server info
    if let Some(info) = client.server_info() {
        info!("Connected to: {} v{}", info.name, info.version);
    }

    // List available tools
    info!("Listing available tools...");
    let tools = client.list_tools().await?;

    for tool in &tools {
        info!(
            "  - {}: {}",
            tool.name,
            tool.description.as_deref().unwrap_or("No description")
        );
    }

    // Call the read_file tool
    info!("\nReading test file via MCP...");
    let result = client
        .call_tool(
            "read_file",
            serde_json::json!({
                "path": test_file.to_str().unwrap()
            }),
        )
        .await?;

    if !result.is_error {
        for content in &result.content {
            if let Some(text) = content.as_text() {
                info!("File content: {}", text);
            }
        }
    }

    // List directory contents
    info!("\nListing directory via MCP...");
    let result = client
        .call_tool(
            "list_directory",
            serde_json::json!({
                "path": temp_dir.to_str().unwrap()
            }),
        )
        .await?;

    if !result.is_error {
        for content in &result.content {
            if let Some(text) = content.as_text() {
                info!("Directory contents:\n{}", text);
            }
        }
    }

    // Close connection
    client.close().await?;
    info!("MCP client closed");

    // Cleanup
    std::fs::remove_dir_all(&temp_dir)?;

    Ok(())
}

/// Example 2: Using MCP Tool Bridge to convert MCP tools to agent tools
async fn tool_bridge_example() -> Result<()> {
    // Create a temporary directory
    let temp_dir = std::env::temp_dir().join("mcp_bridge_example");
    std::fs::create_dir_all(&temp_dir)?;

    // Create test files
    std::fs::write(temp_dir.join("config.json"), r#"{"key": "value"}"#)?;
    std::fs::write(temp_dir.join("data.txt"), "Some important data")?;

    info!("Created test files in: {}", temp_dir.display());

    // Spawn MCP server
    let transport = match StdioTransport::spawn(
        "npx",
        &[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            temp_dir.to_str().unwrap(),
        ],
    )
    .await
    {
        Ok(t) => t,
        Err(e) => {
            info!("Could not spawn MCP server: {}", e);
            info!("Skipping tool bridge example");
            return Ok(());
        }
    };

    // Create client and bridge
    let client = McpClient::new(transport).await?;
    let bridge = McpToolBridge::new(client);

    info!("MCP server: {}", bridge.server_name());

    // Get all tools as rust-ai-agents Tools
    let tools = bridge.create_tools().await?;
    info!("Converted {} MCP tools to agent tools", tools.len());

    // Or register into a ToolRegistry
    let mut registry = ToolRegistry::new();
    // Note: We need a new bridge for the registry since create_tools consumed tools
    let transport2 = StdioTransport::spawn(
        "npx",
        &[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            temp_dir.to_str().unwrap(),
        ],
    )
    .await?;
    let client2 = McpClient::new(transport2).await?;
    let bridge2 = McpToolBridge::new(client2);
    let count = bridge2.register_tools(&mut registry).await?;
    info!("Registered {} tools in registry", count);

    // Use a tool through the Tool trait
    let read_file_tool = registry.get("read_file").unwrap();
    info!("\nUsing read_file tool via Tool trait:");
    info!("  Name: {}", read_file_tool.schema().name);
    info!("  Description: {}", read_file_tool.schema().description);

    // Execute the tool
    let ctx = ExecutionContext::new(AgentId::new("test-agent"));
    let result = read_file_tool
        .execute(
            &ctx,
            serde_json::json!({
                "path": temp_dir.join("config.json").to_str().unwrap()
            }),
        )
        .await?;

    info!("  Result: {}", result);

    // Demonstrate tool validation
    info!("\nTesting tool validation:");
    let validation_result = read_file_tool.validate(&serde_json::json!({}));
    match validation_result {
        Ok(_) => info!("  Validation passed (unexpected)"),
        Err(e) => info!("  Validation failed as expected: {}", e),
    }

    // List all schemas (useful for LLM function calling)
    info!("\nAll tool schemas for LLM:");
    for schema in registry.list_schemas() {
        info!(
            "  - {} (mcp={})",
            schema.name,
            schema.metadata.contains_key("mcp_tool")
        );
    }

    // Cleanup
    std::fs::remove_dir_all(&temp_dir)?;

    Ok(())
}

/// Example showing how to use MCP tools with an actual agent
#[allow(dead_code)]
async fn agent_with_mcp_tools() -> Result<()> {
    // This is a sketch of how you would integrate MCP tools with an agent
    // The actual agent implementation would use the tools from the registry

    /*
    use rust_ai_agents_agents::Agent;
    use rust_ai_agents_providers::OpenRouterProvider;

    // Setup MCP
    let transport = StdioTransport::spawn("npx", &["-y", "some-mcp-server"]).await?;
    let client = McpClient::new(transport).await?;
    let bridge = McpToolBridge::new(client);

    // Get tools
    let mcp_tools = bridge.create_tools().await?;

    // Create agent with MCP tools
    let provider = OpenRouterProvider::new()?;
    let agent = Agent::builder()
        .name("mcp-agent")
        .model("anthropic/claude-3-5-sonnet")
        .system_prompt("You have access to filesystem tools via MCP.")
        .tools(mcp_tools)
        .provider(provider)
        .build()?;

    // The agent can now use MCP tools in its responses
    let response = agent.chat("List all files in the current directory").await?;
    */

    Ok(())
}
