//! Basic Cloudflare Worker example with AI chat
//!
//! Deploy with:
//! ```bash
//! cd crates/cloudflare
//! wrangler deploy
//! ```
//!
//! Set your API key:
//! ```bash
//! wrangler secret put ANTHROPIC_API_KEY
//! ```

use rust_ai_agents_cloudflare::{CloudflareAgent, CloudflareConfig, KvStore, Provider};
use worker::*;

#[event(fetch)]
async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    // Set up panic hook for better error messages
    console_error_panic_hook::set_once();

    // Route handling
    let router = Router::new();

    router
        .post_async("/chat", |mut req, ctx| async move {
            handle_chat(req, ctx).await
        })
        .post_async("/chat/:session_id", |mut req, ctx| async move {
            handle_chat_with_session(req, ctx).await
        })
        .get("/health", |_, _| Response::ok("OK"))
        .run(req, env)
        .await
}

/// Handle a simple chat request (no session persistence)
async fn handle_chat(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Parse request body
    let body: ChatRequest = req.json().await?;

    // Get API key from secrets
    let api_key = ctx.secret("ANTHROPIC_API_KEY")?.to_string();

    // Get model from env or use default
    let model = ctx
        .var("DEFAULT_MODEL")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());

    // Create agent configuration
    let config = CloudflareConfig::builder()
        .provider(Provider::Anthropic)
        .api_key(api_key)
        .model(model)
        .system_prompt("You are a helpful AI assistant.")
        .temperature(0.7)
        .build();

    // Create agent and chat
    let mut agent = CloudflareAgent::new(config);

    match agent.chat(&body.message).await {
        Ok(response) => {
            let chat_response = ChatResponse {
                content: response.content,
                usage: response.usage.map(|u| UsageResponse {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                }),
            };
            Response::from_json(&chat_response)
        }
        Err(e) => Response::error(format!("Chat error: {}", e), 500),
    }
}

/// Handle a chat request with session persistence
async fn handle_chat_with_session(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get session ID from path
    let session_id = ctx.param("session_id").unwrap_or("default");

    // Parse request body
    let body: ChatRequest = req.json().await?;

    // Get API key and KV store
    let api_key = ctx.secret("ANTHROPIC_API_KEY")?.to_string();
    let kv = ctx.kv("CONVERSATIONS")?;

    let model = ctx
        .var("DEFAULT_MODEL")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());

    // Create agent with KV persistence
    let config = CloudflareConfig::builder()
        .provider(Provider::Anthropic)
        .api_key(api_key)
        .model(model)
        .system_prompt("You are a helpful AI assistant.")
        .build();

    let mut agent = CloudflareAgent::new(config).with_kv(KvStore::new(kv));

    match agent.chat_with_session(session_id, &body.message).await {
        Ok(response) => {
            let chat_response = ChatResponse {
                content: response.content,
                usage: response.usage.map(|u| UsageResponse {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                }),
            };
            Response::from_json(&chat_response)
        }
        Err(e) => Response::error(format!("Chat error: {}", e), 500),
    }
}

/// Request body for chat endpoint
#[derive(serde::Deserialize)]
struct ChatRequest {
    message: String,
}

/// Response body for chat endpoint
#[derive(serde::Serialize)]
struct ChatResponse {
    content: String,
    usage: Option<UsageResponse>,
}

/// Token usage in response
#[derive(serde::Serialize)]
struct UsageResponse {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}
