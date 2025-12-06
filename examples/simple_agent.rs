//! Simple single agent example

use rust_ai_agents_core::*;
use rust_ai_agents_providers::*;
use rust_ai_agents_agents::*;
use rust_ai_agents_tools::*;

use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Starting Simple Agent Example\n");

    // Create agent engine
    let engine = Arc::new(AgentEngine::new());

    // Create tool registry and add a calculator
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool::new()));

    let registry = Arc::new(registry);

    // Create LLM backend (using OpenRouter with a free model)
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY must be set");

    let backend = Arc::new(OpenRouterProvider::new(
        api_key,
        "openai/gpt-3.5-turbo".to_string(),
    )) as Arc<dyn LLMBackend>;

    // Configure agent
    let config = AgentConfig::new("Math Assistant", AgentRole::Executor)
        .with_capabilities(vec![Capability::Analysis])
        .with_system_prompt(
            "You are a helpful math assistant. When users ask math questions, \
             use the calculator tool to compute accurate results."
        )
        .with_temperature(0.7);

    // Spawn agent
    println!("📦 Spawning agent...");
    let agent_id = engine.spawn_agent(
        config,
        registry.clone(),
        backend.clone(),
    ).await?;

    println!("✅ Agent spawned: {}\n", agent_id);

    // Send a message to the agent
    let user_message = Message::user(
        agent_id.clone(),
        "What is 234 multiplied by 567?"
    );

    println!("💬 Sending message: {}", user_message.as_text().unwrap());
    engine.send_message(user_message)?;

    // Wait for processing
    println!("⏳ Waiting for agent to process...\n");
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    // Get metrics
    let (spawned, processed) = engine.metrics();
    println!("📊 Metrics:");
    println!("   Agents spawned: {}", spawned);
    println!("   Messages processed: {}", processed);

    // Shutdown
    println!("\n🛑 Shutting down...");
    engine.shutdown().await;

    println!("✨ Done!");

    Ok(())
}
