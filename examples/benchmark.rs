use async_trait::async_trait;
use rust_ai_agents_agents::*;
use rust_ai_agents_core::{AgentConfig, *};
use rust_ai_agents_monitoring::CostTracker;
use rust_ai_agents_providers::{InferenceOutput, LLMBackend, ModelInfo, TokenUsage};
use rust_ai_agents_tools::ToolRegistry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Mock LLM backend to benchmark a full ReACT loop sem custo de rede.
struct MockBackend {
    latency_ms: u64,
}

#[async_trait]
impl LLMBackend for MockBackend {
    async fn infer(
        &self,
        _messages: &[LLMMessage],
        _tools: &[ToolSchema],
        _temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
        Ok(InferenceOutput {
            content: "ok".to_string(),
            tool_calls: None,
            reasoning: None,
            confidence: 1.0,
            token_usage: TokenUsage::new(64, 32),
            metadata: HashMap::new(),
        })
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, LLMError> {
        Ok(vec![0.0; 3])
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            model: "mock-llm".to_string(),
            provider: "mock".to_string(),
            max_tokens: 8192,
            input_cost_per_1m: 0.0,
            output_cost_per_1m: 0.0,
            supports_functions: true,
            supports_vision: false,
        }
    }
}

fn parse_arg(name: &str, default: usize) -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            if let Some(val) = args.next() {
                if let Ok(parsed) = val.parse::<usize>() {
                    return parsed;
                }
            }
        }
    }
    default
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agents = parse_arg("--agents", 5);
    let messages_per_agent = parse_arg("--messages", 50);
    let mock_latency_ms = parse_arg("--latency-ms", 5) as u64;

    println!(
        "Benchmark: agents={}, messages/agent={}, mock latency={}ms",
        agents, messages_per_agent, mock_latency_ms
    );

    let cost_tracker = Arc::new(CostTracker::new());
    let engine = Arc::new(AgentEngine::with_cost_tracker(cost_tracker.clone()));
    let tools = Arc::new(ToolRegistry::new());
    let backend = Arc::new(MockBackend {
        latency_ms: mock_latency_ms,
    }) as Arc<dyn LLMBackend>;

    // Spawn agents
    let mut agent_ids = Vec::with_capacity(agents);
    for idx in 0..agents {
        let cfg = AgentConfig::new(format!("bench-agent-{}", idx), AgentRole::Executor)
            .with_temperature(0.0);
        let id: AgentId = engine
            .spawn_agent(cfg, tools.clone(), backend.clone())
            .await?;
        agent_ids.push(id);
    }

    // Fire messages
    let total_messages = (agents * messages_per_agent) as u64;
    let start = Instant::now();
    for agent_id in &agent_ids {
        for _ in 0..messages_per_agent {
            engine.send_message(Message::user(agent_id.clone(), "benchmark ping"))?;
        }
    }

    // Wait until all messages are processed or timeout
    let timeout = Duration::from_secs(30);
    let poll = Duration::from_millis(50);
    let wait_start = Instant::now();
    loop {
        let metrics = engine.metrics();
        if metrics.messages_processed >= total_messages {
            break;
        }
        if wait_start.elapsed() > timeout {
            println!(
                "Timeout waiting for messages. processed={}/{}",
                metrics.messages_processed, total_messages
            );
            break;
        }
        tokio::time::sleep(poll).await;
    }

    let elapsed = start.elapsed();
    let metrics = engine.metrics();
    let throughput = (metrics.messages_processed as f64) / elapsed.as_secs_f64();

    let cost_stats = cost_tracker.stats();
    println!("--- Benchmark result ---");
    println!("Processed:   {}", metrics.messages_processed);
    println!("Elapsed:     {:.2?}", elapsed);
    println!("Throughput:  {:.2} msg/s", throughput);
    println!(
        "Avg latency: {:.2} ms (CostTracker)",
        cost_stats.avg_latency_ms
    );
    println!(
        "Tokens in/out: {}/{}",
        cost_stats.total_input_tokens, cost_stats.total_output_tokens
    );
    println!("Total cost:  ${:.6}", cost_stats.total_cost);

    engine.shutdown().await;
    Ok(())
}
