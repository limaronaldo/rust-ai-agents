# rust-ai-agents-cloudflare

Run AI agents on Cloudflare Workers edge computing platform.

## Features

- **CloudflareAgent** - Edge-native AI agent with chat and streaming
- **Multi-Provider** - OpenAI, Anthropic, OpenRouter support
- **KV Persistence** - Conversation history via Cloudflare KV
- **SSE Streaming** - Real-time streaming responses
- **Session Management** - Persistent conversation sessions
- **Workers-Native** - Uses Workers fetch API, no external dependencies

## Installation

```toml
[dependencies]
rust-ai-agents-cloudflare = "0.1"
worker = "0.4"
```

## Quick Start

### Basic Chat

```rust
use rust_ai_agents_cloudflare::{CloudflareAgent, CloudflareConfig, Provider};
use worker::*;

#[event(fetch)]
async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let api_key = env.secret("ANTHROPIC_API_KEY")?.to_string();

    let config = CloudflareConfig::builder()
        .provider(Provider::Anthropic)
        .api_key(api_key)
        .model("claude-3-5-sonnet-20241022")
        .system_prompt("You are a helpful assistant.")
        .build();

    let mut agent = CloudflareAgent::new(config);
    let response = agent.chat("Hello!").await?;

    Response::ok(response.content)
}
```

### With KV Persistence

```rust
use rust_ai_agents_cloudflare::{CloudflareAgent, CloudflareConfig, KvStore, Provider};

#[event(fetch)]
async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let config = CloudflareConfig::builder()
        .provider(Provider::Anthropic)
        .api_key(env.secret("ANTHROPIC_API_KEY")?.to_string())
        .model("claude-3-5-sonnet-20241022")
        .build();

    // Get KV namespace
    let kv = env.kv("CONVERSATIONS")?;

    // Create agent with KV persistence
    let mut agent = CloudflareAgent::new(config)
        .with_kv(KvStore::new(kv));

    // Get session ID from request
    let session_id = req.headers()
        .get("X-Session-ID")?
        .unwrap_or_else(|| "default".to_string());

    // Chat with session - history is automatically persisted
    let response = agent.chat_with_session(&session_id, "Hello!").await?;

    Response::ok(response.content)
}
```

### Streaming Response

```rust
use rust_ai_agents_cloudflare::{CloudflareAgent, CloudflareConfig, Provider};
use futures::StreamExt;

#[event(fetch)]
async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let config = CloudflareConfig::builder()
        .provider(Provider::OpenAI)
        .api_key(env.secret("OPENAI_API_KEY")?.to_string())
        .model("gpt-4o")
        .build();

    let mut agent = CloudflareAgent::new(config);

    // Get streaming response
    let stream = agent.chat_stream("Tell me a story").await?;

    // Return as SSE
    let response_stream = stream.map(|chunk| {
        match chunk {
            Ok(text) => format!("data: {}\n\n", text),
            Err(e) => format!("data: [ERROR] {}\n\n", e),
        }
    });

    Response::from_stream(response_stream)
        .map(|mut r| {
            r.headers_mut()
                .set("Content-Type", "text/event-stream")?;
            r.headers_mut()
                .set("Cache-Control", "no-cache")?;
            Ok(r)
        })?
}
```

## Configuration

### CloudflareConfig

```rust
let config = CloudflareConfig::builder()
    // Required
    .provider(Provider::Anthropic)
    .api_key("sk-...")
    .model("claude-3-5-sonnet-20241022")

    // Optional
    .system_prompt("You are a helpful assistant.")
    .max_tokens(4096)
    .temperature(0.7)

    // Anthropic-specific
    .api_version("2024-01-01")

    .build();
```

### Providers

```rust
use rust_ai_agents_cloudflare::Provider;

// Anthropic Claude
Provider::Anthropic

// OpenAI GPT
Provider::OpenAI

// OpenRouter (200+ models)
Provider::OpenRouter
```

### Model Examples

```rust
// Anthropic
.model("claude-3-5-sonnet-20241022")
.model("claude-3-opus-20240229")
.model("claude-3-haiku-20240307")

// OpenAI
.model("gpt-4o")
.model("gpt-4-turbo")
.model("gpt-3.5-turbo")

// OpenRouter
.model("anthropic/claude-3.5-sonnet")
.model("meta-llama/llama-3.1-405b-instruct")
.model("mistralai/mistral-large")
```

## KV Store

### Setup

1. Create KV namespace in Cloudflare dashboard
2. Bind in `wrangler.toml`:

```toml
[[kv_namespaces]]
binding = "CONVERSATIONS"
id = "your-namespace-id"
```

### Usage

```rust
use rust_ai_agents_cloudflare::KvStore;

let kv = env.kv("CONVERSATIONS")?;
let store = KvStore::new(kv);

// Manual conversation management
let history = store.get_conversation("session-123").await?;
store.save_conversation("session-123", &messages).await?;
store.delete_conversation("session-123").await?;

// List sessions
let sessions = store.list_sessions(100).await?;
```

### Session Metadata

```rust
// Sessions include metadata
let metadata = store.get_session_metadata("session-123").await?;
// metadata.created_at: DateTime
// metadata.updated_at: DateTime
// metadata.message_count: usize
```

## Response Types

### AgentResponse

```rust
pub struct AgentResponse {
    /// The text content of the response
    pub content: String,

    /// Tool calls (if any)
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Token usage
    pub usage: Option<AgentUsage>,
}

pub struct AgentUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
```

## Error Handling

```rust
use rust_ai_agents_cloudflare::CloudflareError;

match agent.chat("Hello").await {
    Ok(response) => Response::ok(response.content),
    Err(CloudflareError::ApiError { status, message }) => {
        Response::error(message, status)
    }
    Err(CloudflareError::RateLimited { retry_after }) => {
        let mut response = Response::error("Rate limited", 429)?;
        response.headers_mut()
            .set("Retry-After", &retry_after.to_string())?;
        Ok(response)
    }
    Err(CloudflareError::ConfigError(msg)) => {
        Response::error(msg, 400)
    }
    Err(e) => Response::error(e.to_string(), 500),
}
```

## Complete Worker Example

```rust
use rust_ai_agents_cloudflare::{CloudflareAgent, CloudflareConfig, KvStore, Provider};
use serde::{Deserialize, Serialize};
use worker::*;

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    session_id: Option<String>,
}

#[derive(Serialize)]
struct ChatResponse {
    content: String,
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[event(fetch)]
async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    // CORS preflight
    if req.method() == Method::Options {
        return Response::empty()
            .map(|r| {
                let mut r = r;
                r.headers_mut().set("Access-Control-Allow-Origin", "*")?;
                r.headers_mut().set("Access-Control-Allow-Methods", "POST")?;
                r.headers_mut().set("Access-Control-Allow-Headers", "Content-Type")?;
                Ok(r)
            })?;
    }

    // Parse request
    let body: ChatRequest = req.json().await?;

    // Build config
    let config = CloudflareConfig::builder()
        .provider(Provider::Anthropic)
        .api_key(env.secret("ANTHROPIC_API_KEY")?.to_string())
        .model("claude-3-5-sonnet-20241022")
        .system_prompt("You are a helpful assistant.")
        .max_tokens(1024)
        .build();

    // Create agent
    let mut agent = CloudflareAgent::new(config);

    // Add KV if session requested
    if body.session_id.is_some() {
        let kv = env.kv("CONVERSATIONS")?;
        agent = agent.with_kv(KvStore::new(kv));
    }

    // Chat
    let response = match body.session_id {
        Some(session_id) => {
            agent.chat_with_session(&session_id, &body.message).await?
        }
        None => {
            agent.chat(&body.message).await?
        }
    };

    // Build response
    let chat_response = ChatResponse {
        content: response.content,
        usage: response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
        }),
    };

    Response::from_json(&chat_response)
        .map(|mut r| {
            r.headers_mut().set("Access-Control-Allow-Origin", "*")?;
            Ok(r)
        })?
}
```

## wrangler.toml

```toml
name = "ai-agent"
main = "src/lib.rs"
compatibility_date = "2024-01-01"

[build]
command = "cargo install -q worker-build && worker-build --release"

[[kv_namespaces]]
binding = "CONVERSATIONS"
id = "your-namespace-id"

[vars]
# Non-secret config vars

# Secrets (set via `wrangler secret put`)
# ANTHROPIC_API_KEY
# OPENAI_API_KEY
```

## Deployment

```bash
# Install wrangler
npm install -g wrangler

# Login
wrangler login

# Set secrets
wrangler secret put ANTHROPIC_API_KEY

# Deploy
wrangler deploy
```

## Performance Tips

1. **Reuse agents** - Create agent once per request, not per message
2. **Session persistence** - Use KV for multi-turn conversations
3. **Streaming** - Use streaming for long responses to avoid timeouts
4. **Model selection** - Use Haiku/3.5-turbo for simple tasks

## Limitations

- Workers have 30-second CPU time limit
- Use streaming for long responses
- KV has eventual consistency (usually <1s)
- Request body limit: 100MB

## Related Crates

- [`rust-ai-agents-llm-client`](../llm-client) - Core LLM client
- [`rust-ai-agents-wasm`](../wasm) - Browser WASM support
- [`rust-ai-agents-fastly`](../fastly) - Fastly Compute@Edge support

## License

Apache-2.0
