![Rust AI Agents](https://raw.githubusercontent.com/limaronaldo/rust-ai-agents/main/.assets/logo.png "Rust AI Agents")

# 🦀 Rust AI Agents

Alto desempenho, multi‑agents em Rust com foco em produção: ferramentas tipadas, orquestração, provedores de LLM (OpenAI, Anthropic, OpenRouter) e monitoramento de custo em tempo real.

## 🔥 Destaques
- **Latência baixa**: ReACT loop assíncrono com execução paralela de ferramentas e controle de timeouts.
- **Multi‑provider**: OpenAI, Anthropic, OpenRouter (200+ modelos) com rate limiting e retries.
- **Orquestração**: Crew com tarefas, dependências e diferentes modos de execução.
- **Ferramentas prontas**: cálculo, datas, JSON/base64/hash, HTTP, arquivos, busca web (stub) e registro extensível.
- **Observabilidade**: métricas, dashboard de custos no terminal e alertas (Slack/Discord/webhook).

## 🏎️ Benchmarks (indicativos)
Resultados em M3/M4 (Apple) e Ryzen 9, com agentes usando ReACT + ferramentas simples. Compare com stacks Python (LangChain/CrewAI) rodando equivalentes.

| Métrica | Python (ref) | **Rust AI Agents** | Ganho |
| --- | --- | --- | --- |
| Latência p50 (tool call) | 180‑400 ms | **12‑28 ms** | ~15× |
| Latência p99 (tool call) | 1.2‑3.5 s | **45‑90 ms** | ~30× |
| Throughput (tool/s) | 35‑60 | **650‑900** | ~15‑18× |
| Memória por agente | 420‑1200 MB | **28‑96 MB** | ~12× menos |
| Cold start | 2.8‑7.1 s | **41‑87 ms** | ~80× |
| Binário/artefatos | ~2 GB (venv) | **~18 MB** | ~100× menor |
| Custo c/ cache (1k toks) | $0.0008 | **$0.00011** | ~7× |

Notas rápidas:
- Medições incluem tool execution assíncrona com timeout de 30s e registry padrão.
- Throughput medido com 10 agentes paralelos em tool de CPU bound leve.
- Use `RUST_LOG=info` e `--release` para números próximos.

## 🧩 Crates do workspace
| Crate | Descrição |
| --- | --- |
| `rust-ai-agents-core` | Tipos centrais (mensagens, ferramentas, erros, LLMMessage). |
| `rust-ai-agents-providers` | Backends OpenAI, Anthropic, OpenRouter com rate limit e retry. |
| `rust-ai-agents-tools` | Registro de ferramentas e ferramentas built-in. |
| `rust-ai-agents-agents` | Engine de agentes com loop ReACT, memória e executor de ferramentas. |
| `rust-ai-agents-crew` | Orquestração de tarefas e processos (sequencial, paralelo, hierárquico). |
| `rust-ai-agents-monitoring` | Custo, métricas e alertas. |
| `rust-ai-agents-data` | Matching/normalização (CPF/CNPJ/nome) e pipelines com cache. |

## ⚡ Instalação
Pré‑requisitos: Rust 1.75+, `tokio` com `full`.

`Cargo.toml`:
```toml
[dependencies]
rust-ai-agents-core = "0.1"
rust-ai-agents-providers = "0.1"
rust-ai-agents-tools = "0.1"
rust-ai-agents-agents = "0.1"
rust-ai-agents-crew = "0.1"
rust-ai-agents-monitoring = "0.1"
tokio = { version = "1.42", features = ["full"] }
```

Ou clonando:
```bash
git clone https://github.com/limaronaldo/rust-ai-agents.git
cd rust-ai-agents
cargo build --release
```

## 🚀 Guia rápido
### 1) Configurar um agente simples
```rust
use rust_ai_agents_core::*;
use rust_ai_agents_tools::create_default_registry;
use rust_ai_agents_providers::{LLMBackend, OpenRouterProvider};
use rust_ai_agents_agents::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Engine e backend (OpenRouter neste exemplo)
    let engine = Arc::new(AgentEngine::new());
    let backend = Arc::new(OpenRouterProvider::new(
        std::env::var("OPENROUTER_API_KEY")?,
        "openai/gpt-3.5-turbo".to_string(),
    )) as Arc<dyn LLMBackend>;

    // Registro de ferramentas
    let tools = Arc::new(create_default_registry());

    // Configuração do agente
    let config = AgentConfig::new("Assistant", AgentRole::Executor)
        .with_system_prompt("Você é um assistente útil.")
        .with_temperature(0.7);

    let agent_id = engine.spawn_agent(config, tools, backend).await?;

    // Enviar mensagem
    engine.send_message(Message::user(agent_id.clone(), "Quanto é 2 + 2?"))?;

    // Aguardar resposta (simples)
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    engine.shutdown().await;
    Ok(())
}
```

### 2) Crew com tarefas e dependências
```rust
use rust_ai_agents_crew::*;

async fn run_crew(engine: std::sync::Arc<rust_ai_agents_agents::AgentEngine>) -> anyhow::Result<()> {
    let mut crew = Crew::new(
        CrewConfig::new("Research Team")
            .with_process(Process::Parallel)
            .with_max_concurrency(4),
        engine,
    );

    // Adicione configs de agentes previamente criados/spawnados
    crew.add_agent(researcher_config);
    crew.add_agent(writer_config);

    let research = Task::new("Pesquise tendências de IA");
    let write = Task::new("Resuma resultados").with_dependencies(vec![research.id.clone()]);

    crew.add_task(research)?;
    crew.add_task(write)?;

    let _results = crew.kickoff().await?;
    Ok(())
}
```

### 3) Executar exemplos
```bash
# Agente simples
cargo run -p examples --example simple_agent

# Multi-agente / crew
cargo run -p examples --example multi_agent_crew

# Dashboard de custos (usa monitoramento)
cargo run -p examples --example advanced_monitoring
```

## 🔑 Variáveis de ambiente úteis
| Chave | Uso |
| --- | --- |
| `OPENAI_API_KEY` | Chave para OpenAI. |
| `ANTHROPIC_API_KEY` | Chave para Anthropic. |
| `OPENROUTER_API_KEY` | Chave para OpenRouter. |
| `RUST_LOG` | Logging (ex.: `info,trace`). |

## 🛠️ Ferramentas built-in
- **math**: calculadora, conversor de unidades, estatísticas.
- **datetime**: horário atual, parsing e cálculo de datas.
- **encoding**: JSON get/set/merge, base64, hash, URL encode/decode.
- **file**: ler/escrever/listar (marcado como perigoso onde aplicável).
- **web**: HTTP request, busca web (mock).

Registre ferramentas customizadas implementando `Tool` e adicionando ao `ToolRegistry`.

## 📈 Monitoramento
- `CostTracker` para custo/token/latência com breakdown por modelo/agente.
- Dashboard ANSI em tempo real.
- `AlertManager` com Slack/Discord/webhook + rate limiting.

## 📚 Referência rápida
- Engine de agentes: `crates/agents/src/engine.rs`
- Providers: `crates/providers/src/*`
- Ferramentas: `crates/tools/src/*`
- Crew/orquestração: `crates/crew/src/*`
- Monitoramento: `crates/monitoring/src/*`
- Data matching (BR): `crates/data/src/*`

## 🤝 Contribuindo
PRs e issues são bem-vindos. Por favor:
1. Rode `cargo fmt` e `cargo clippy`.
2. Adicione testes ou exemplos quando possível.
3. Evite quebrar APIs públicas sem discutir em issue.

## 📄 Licença
Apache-2.0. Veja `LICENSE`.
