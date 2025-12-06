//! Multi-agent crew example with task orchestration

use rust_ai_agents_core::*;
use rust_ai_agents_providers::*;
use rust_ai_agents_agents::*;
use rust_ai_agents_crew::*;
use rust_ai_agents_tools::*;

use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("🚀 Starting Multi-Agent Crew Example\n");

    // Create engine
    let engine = Arc::new(AgentEngine::new());

    // Create shared tool registry
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool::new()));
    registry.register(Arc::new(WebSearchTool::new()));
    let registry = Arc::new(registry);

    // Create LLM backend
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY must be set");

    let backend = Arc::new(OpenRouterProvider::new(
        api_key,
        "openai/gpt-3.5-turbo".to_string(),
    )) as Arc<dyn LLMBackend>;

    // Configure agents
    println!("📦 Creating agent configurations...\n");

    let researcher_config = AgentConfig::new("Research Agent", AgentRole::Researcher)
        .with_capabilities(vec![Capability::WebSearch, Capability::Analysis])
        .with_system_prompt(
            "You are a research specialist. Your job is to gather information \
             and provide well-researched insights."
        );

    let analyst_config = AgentConfig::new("Data Analyst", AgentRole::Executor)
        .with_capabilities(vec![Capability::Analysis, Capability::Prediction])
        .with_system_prompt(
            "You are a data analyst. You analyze data and provide insights \
             based on research findings."
        );

    let writer_config = AgentConfig::new("Content Writer", AgentRole::Writer)
        .with_capabilities(vec![Capability::ContentGeneration])
        .with_system_prompt(
            "You are a content writer. You create clear, engaging content \
             based on research and analysis."
        );

    // Spawn agents
    println!("🤖 Spawning agents...");
    let researcher_id = engine.spawn_agent(
        researcher_config.clone(),
        registry.clone(),
        backend.clone(),
    ).await?;
    println!("   ✅ Researcher spawned: {}", researcher_id);

    let analyst_id = engine.spawn_agent(
        analyst_config.clone(),
        registry.clone(),
        backend.clone(),
    ).await?;
    println!("   ✅ Analyst spawned: {}", analyst_id);

    let writer_id = engine.spawn_agent(
        writer_config.clone(),
        registry.clone(),
        backend.clone(),
    ).await?;
    println!("   ✅ Writer spawned: {}", writer_id);

    // Create crew
    println!("\n👥 Creating crew...");
    let crew_config = CrewConfig::new("Content Creation Crew")
        .with_description("A team that researches, analyzes, and writes content")
        .with_process(Process::Sequential)
        .with_max_concurrency(2)
        .with_verbose(true);

    let mut crew = Crew::new(crew_config, engine.clone());
    crew.add_agent(researcher_config);
    crew.add_agent(analyst_config);
    crew.add_agent(writer_config);

    // Define tasks
    println!("📋 Adding tasks...");

    let research_task = Task::new("Research the latest trends in AI agents")
        .with_expected_output("A summary of recent AI agent developments")
        .with_agent(researcher_id.clone());

    let analysis_task = Task::new("Analyze the research findings")
        .with_expected_output("Key insights and trends")
        .with_agent(analyst_id.clone())
        .with_dependencies(vec![research_task.id.clone()]);

    let writing_task = Task::new("Write a blog post based on the analysis")
        .with_expected_output("A 500-word blog post")
        .with_agent(writer_id.clone())
        .with_dependencies(vec![analysis_task.id.clone()]);

    crew.add_task(research_task)?;
    crew.add_task(analysis_task)?;
    crew.add_task(writing_task)?;

    println!("   ✅ {} tasks added\n", crew.stats().task_count);

    // Execute crew
    println!("🎬 Starting crew execution...\n");
    let results = crew.kickoff().await?;

    println!("\n✅ Crew execution completed!");
    println!("📊 Results: {} tasks completed\n", results.len());

    for (idx, result) in results.iter().enumerate() {
        println!("Task {}: {}", idx + 1, if result.success { "✅ Success" } else { "❌ Failed" });
    }

    // Print crew stats
    let stats = crew.stats();
    println!("\n📈 Crew Statistics:");
    println!("   Name: {}", stats.name);
    println!("   Agents: {}", stats.agent_count);
    println!("   Tasks: {}", stats.task_count);

    // Shutdown
    println!("\n🛑 Shutting down...");
    engine.shutdown().await;

    println!("✨ Done!");

    Ok(())
}
