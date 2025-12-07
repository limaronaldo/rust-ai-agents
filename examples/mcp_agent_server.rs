//! MCP Agent Server Example
//!
//! This example demonstrates how to expose rust-ai-agents as MCP tools,
//! allowing external MCP clients (like Claude Desktop) to invoke agents.
//!
//! # Running the Example
//!
//! ```bash
//! cargo run --example mcp_agent_server
//! ```
//!
//! The server will communicate over STDIO, making it compatible with
//! Claude Desktop and other MCP clients.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   MCP Client (Claude Desktop)               │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                    tools/call "agent_research"
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     McpServer (STDIO)                       │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │  agent_research  │  agent_code_review │  agent_qa   │   │
//! │  │  (AgentMcpHandler instances)                        │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Available Agents
//!
//! - `agent_research`: Researches topics and provides summaries
//! - `agent_code_review`: Reviews code snippets for issues
//! - `agent_qa`: Question answering with context
//! - `agent_echo`: Simple echo for testing
//!
//! # Claude Desktop Configuration
//!
//! Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "rust-ai-agents": {
//!       "command": "cargo",
//!       "args": ["run", "--example", "mcp_agent_server"],
//!       "cwd": "/path/to/rust-ai-agents"
//!     }
//!   }
//! }
//! ```

use rust_ai_agents_mcp::{simple_agent, AgentMcpHandler, AgentMcpInput, AgentMcpOutput, McpServer};
use std::collections::HashMap;
use tracing::info;

// =============================================================================
// Agent Implementations
// =============================================================================

/// Research agent - simulates topic research and summarization
fn create_research_agent() -> AgentMcpHandler {
    AgentMcpHandler::builder("research")
        .description(
            "Researches topics and provides comprehensive summaries. \
            Useful for gathering information about technical concepts, \
            comparing technologies, or exploring new domains.",
        )
        .capability("summarization")
        .capability("analysis")
        .capability("comparison")
        .handler(|input: AgentMcpInput| async move {
            let start = std::time::Instant::now();

            // Simulate research process
            let topic = &input.query;
            let depth = input
                .context
                .get("depth")
                .map(|s| s.as_str())
                .unwrap_or("medium");

            // Simulated research (in real implementation, this would use tools/LLM)
            let summary = match depth {
                "shallow" => format!(
                    "Quick overview of '{}':\n\n\
                    This topic involves several key concepts that are important \
                    for understanding the broader context.",
                    topic
                ),
                "deep" => format!(
                    "In-depth analysis of '{}':\n\n\
                    ## Overview\n\
                    This comprehensive research covers multiple aspects of the topic.\n\n\
                    ## Key Points\n\
                    1. Foundational concepts and terminology\n\
                    2. Historical context and evolution\n\
                    3. Current state and best practices\n\
                    4. Future trends and implications\n\n\
                    ## Conclusion\n\
                    The research indicates significant developments in this area.",
                    topic
                ),
                _ => format!(
                    "Research summary for '{}':\n\n\
                    ## Key Findings\n\
                    - Primary concepts have been identified\n\
                    - Relevant connections established\n\
                    - Practical applications noted\n\n\
                    ## Recommendations\n\
                    Further investigation recommended for specific use cases.",
                    topic
                ),
            };

            let mut metadata = HashMap::new();
            metadata.insert("depth".to_string(), depth.to_string());
            metadata.insert("sources_consulted".to_string(), "3".to_string());

            Ok(AgentMcpOutput {
                content: summary,
                success: true,
                confidence: 0.85,
                metadata,
                duration_ms: start.elapsed().as_millis() as u64,
                tools_used: vec!["web_search".to_string(), "summarizer".to_string()],
            })
        })
}

/// Code review agent - analyzes code for issues and improvements
fn create_code_review_agent() -> AgentMcpHandler {
    AgentMcpHandler::builder("code_review")
        .description(
            "Reviews code snippets for potential issues, bugs, and improvements. \
            Supports multiple programming languages and provides actionable feedback.",
        )
        .capability("code_analysis")
        .capability("bug_detection")
        .capability("style_review")
        .handler(|input: AgentMcpInput| async move {
            let start = std::time::Instant::now();

            let code = &input.query;
            let language = input
                .context
                .get("language")
                .map(|s| s.as_str())
                .unwrap_or("unknown");

            // Simulated code review (real implementation would use LLM)
            let review = format!(
                "## Code Review Results\n\n\
                **Language:** {}\n\
                **Lines analyzed:** {}\n\n\
                ### Findings\n\n\
                #### Potential Issues\n\
                - Consider error handling for edge cases\n\
                - Variable naming could be more descriptive\n\n\
                #### Suggestions\n\
                - Add documentation comments\n\
                - Consider breaking into smaller functions\n\n\
                #### Positive Aspects\n\
                - Clear logic flow\n\
                - Good separation of concerns\n\n\
                ### Summary\n\
                The code is generally well-structured with minor improvements suggested.",
                language,
                code.lines().count()
            );

            let mut metadata = HashMap::new();
            metadata.insert("language".to_string(), language.to_string());
            metadata.insert("issues_found".to_string(), "2".to_string());
            metadata.insert("suggestions".to_string(), "2".to_string());

            Ok(AgentMcpOutput {
                content: review,
                success: true,
                confidence: 0.9,
                metadata,
                duration_ms: start.elapsed().as_millis() as u64,
                tools_used: vec!["static_analyzer".to_string(), "linter".to_string()],
            })
        })
}

/// QA agent - question answering with context support
fn create_qa_agent() -> AgentMcpHandler {
    AgentMcpHandler::builder("qa")
        .description(
            "Answers questions using provided context. Supports follow-up questions \
            and maintains conversation history for coherent multi-turn interactions.",
        )
        .capability("question_answering")
        .capability("context_awareness")
        .capability("follow_up")
        .handler(|input: AgentMcpInput| async move {
            let start = std::time::Instant::now();

            let question = &input.query;
            let has_context = !input.context.is_empty();
            let has_history = !input.history.is_empty();

            // Build response based on available context
            let answer = if has_context {
                let context_summary: String = input
                    .context
                    .iter()
                    .map(|(k, v)| format!("- {}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join("\n");

                format!(
                    "Based on the provided context:\n{}\n\n\
                    **Question:** {}\n\n\
                    **Answer:** Given the context provided, the most relevant information \
                    suggests that the answer involves consideration of the key factors \
                    mentioned above. The specific details depend on the domain context.",
                    context_summary, question
                )
            } else if has_history {
                format!(
                    "Following up on our previous conversation ({} turns):\n\n\
                    **Question:** {}\n\n\
                    **Answer:** Building on what we discussed earlier, \
                    this question relates to the ongoing topic. The answer \
                    should be considered in the context of our previous exchanges.",
                    input.history.len(),
                    question
                )
            } else {
                format!(
                    "**Question:** {}\n\n\
                    **Answer:** This is a standalone question. For more accurate \
                    answers, consider providing context or relevant background \
                    information in the context field.",
                    question
                )
            };

            let mut metadata = HashMap::new();
            metadata.insert("has_context".to_string(), has_context.to_string());
            metadata.insert("history_turns".to_string(), input.history.len().to_string());

            Ok(AgentMcpOutput {
                content: answer,
                success: true,
                confidence: if has_context { 0.92 } else { 0.75 },
                metadata,
                duration_ms: start.elapsed().as_millis() as u64,
                tools_used: vec!["knowledge_base".to_string()],
            })
        })
}

/// Simple echo agent using the helper function
fn create_echo_agent() -> AgentMcpHandler {
    simple_agent(
        "echo",
        "Simple echo agent that returns the input query. Useful for testing.",
        |query: String| async move { Ok(format!("Agent echo: {}", query)) },
    )
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging to stderr (stdout is used for MCP protocol)
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .init();

    info!("Starting rust-ai-agents MCP Agent Server");

    // Create the agents
    let research_agent = create_research_agent();
    let code_review_agent = create_code_review_agent();
    let qa_agent = create_qa_agent();
    let echo_agent = create_echo_agent();

    info!("Created 4 agent handlers:");
    info!("  - {}", research_agent.name());
    info!("  - {}", code_review_agent.name());
    info!("  - {}", qa_agent.name());
    info!("  - {}", echo_agent.name());

    // Build the MCP server with agent tools
    let server = McpServer::builder()
        .name("rust-ai-agents-server")
        .version(env!("CARGO_PKG_VERSION"))
        .instructions(
            "This MCP server exposes AI agents as tools. Available agents:\n\
            - agent_research: Research and summarize topics\n\
            - agent_code_review: Review code for issues\n\
            - agent_qa: Answer questions with context\n\
            - agent_echo: Simple echo for testing\n\n\
            Each agent accepts a 'query' parameter and optional 'context' for additional info.",
        )
        .add_tool(research_agent)
        .add_tool(code_review_agent)
        .add_tool(qa_agent)
        .add_tool(echo_agent)
        .build();

    info!("Server configured with 4 agent tools");
    info!("Running on STDIO - ready for MCP client connections");

    // Run the server over STDIO (blocks until client disconnects)
    server.run_stdio().await?;

    info!("Server shutting down");
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_ai_agents_mcp::ToolHandler;
    use serde_json::json;

    #[tokio::test]
    async fn test_research_agent() {
        let agent = create_research_agent();
        let result = agent
            .execute(json!({
                "query": "Rust programming language",
                "context": {"depth": "shallow"}
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        let text = result.content[0].as_text().unwrap();
        assert!(text.contains("Rust programming language"));
    }

    #[tokio::test]
    async fn test_code_review_agent() {
        let agent = create_code_review_agent();
        let result = agent
            .execute(json!({
                "query": "fn main() { println!(\"Hello\"); }",
                "context": {"language": "rust"}
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        let text = result.content[0].as_text().unwrap();
        assert!(text.contains("Code Review Results"));
        assert!(text.contains("rust"));
    }

    #[tokio::test]
    async fn test_qa_agent_with_context() {
        let agent = create_qa_agent();
        let result = agent
            .execute(json!({
                "query": "What is the capital?",
                "context": {"country": "France", "topic": "geography"}
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        let text = result.content[0].as_text().unwrap();
        assert!(text.contains("context"));
    }

    #[tokio::test]
    async fn test_qa_agent_with_history() {
        let agent = create_qa_agent();
        let result = agent
            .execute(json!({
                "query": "Can you elaborate?",
                "history": ["What is Rust?", "It's a systems programming language."]
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        let text = result.content[0].as_text().unwrap();
        assert!(text.contains("previous conversation"));
    }

    #[tokio::test]
    async fn test_echo_agent() {
        let agent = create_echo_agent();
        let result = agent
            .execute(json!({"query": "test message"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        let text = result.content[0].as_text().unwrap();
        assert!(text.contains("Agent echo: test message"));
    }

    #[test]
    fn test_agent_definitions() {
        let research = create_research_agent();
        let def = research.definition();

        assert_eq!(def.name, "agent_research");
        let desc = def.description.unwrap();
        assert!(desc.contains("summarization"));
        assert!(desc.contains("analysis"));
    }
}
