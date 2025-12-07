# dioxus-ai

AI hooks for Dioxus applications - chat, completions, and streaming.

## Features

- `use_chat` - Reactive chat with message history and streaming
- `use_completion` - Single completion requests
- Built on `llm-client` for provider support (OpenAI, Anthropic, OpenRouter)
- Real-time streaming with `streaming_content()` method
- Stop generation support

## Installation

```toml
[dependencies]
dioxus-ai = { git = "https://github.com/limaronaldo/rust-ai-agents" }
```

## Usage

### Chat Interface

```rust
use dioxus::prelude::*;
use dioxus_ai::{use_chat, ChatOptions};

#[component]
fn Chat() -> Element {
    let mut chat = use_chat(ChatOptions {
        provider: "openai".to_string(),
        api_key: "sk-...".to_string(),
        model: "gpt-4o-mini".to_string(),
        system_prompt: Some("You are a helpful assistant.".to_string()),
        ..Default::default()
    });

    let mut input = use_signal(String::new);

    rsx! {
        div { class: "chat",
            // Message list
            for msg in chat.messages().iter() {
                div { class: "message {msg.role}",
                    "{msg.content}"
                }
            }

            // Streaming indicator
            if !chat.streaming_content().is_empty() {
                div { class: "message assistant streaming",
                    "{chat.streaming_content()}"
                }
            }

            // Loading indicator
            if chat.is_loading() {
                div { class: "loading", "..." }
            }

            // Error display
            if let Some(err) = chat.error() {
                div { class: "error", "{err}" }
            }

            // Input form
            form {
                onsubmit: move |e| {
                    e.prevent_default();
                    if !input().is_empty() {
                        chat.send(&input());
                        input.set(String::new());
                    }
                },
                input {
                    r#type: "text",
                    value: "{input}",
                    oninput: move |e| input.set(e.value().clone()),
                    placeholder: "Type a message...",
                    disabled: chat.is_loading()
                }
                button { r#type: "submit", disabled: chat.is_loading(), "Send" }
                button {
                    r#type: "button",
                    onclick: move |_| chat.stop(),
                    disabled: !chat.is_loading(),
                    "Stop"
                }
            }
        }
    }
}
```

### Single Completion

```rust
use dioxus::prelude::*;
use dioxus_ai::{use_completion, CompletionOptions};

#[component]
fn Translator() -> Element {
    let mut completion = use_completion(CompletionOptions {
        provider: "openai".to_string(),
        api_key: "sk-...".to_string(),
        model: "gpt-4o-mini".to_string(),
        system_prompt: Some("You are a translator. Translate to Spanish.".to_string()),
        ..Default::default()
    });

    let mut input = use_signal(String::new);

    rsx! {
        div {
            input {
                r#type: "text",
                value: "{input}",
                oninput: move |e| input.set(e.value().clone()),
                placeholder: "Enter text..."
            }
            button {
                onclick: move |_| completion.complete(&input()),
                disabled: completion.is_loading(),
                "Translate"
            }

            if let Some(result) = completion.completion() {
                p { strong { "Translation: " } "{result}" }
            }
        }
    }
}
```

## API Reference

### `use_chat(options: ChatOptions) -> UseChatState`

Creates a reactive chat interface.

**ChatOptions:**
- `provider` - "openai", "anthropic", or "openrouter"
- `api_key` - Your API key
- `model` - Model identifier (e.g., "gpt-4o-mini")
- `system_prompt` - Optional system prompt
- `temperature` - 0.0 to 2.0 (default: 0.7)
- `max_tokens` - Maximum tokens (default: 4096)
- `stream` - Enable streaming (default: true)
- `initial_messages` - Pre-populate chat history

**UseChatState:**
- `messages()` - Get chat history
- `is_loading()` - Check if request in progress
- `error()` - Get current error
- `streaming_content()` - Get real-time streaming text
- `send(&str)` - Send a message
- `clear()` - Clear history
- `stop()` - Stop generation

### `use_completion(options: CompletionOptions) -> UseCompletionState`

Creates a single completion interface.

**UseCompletionState:**
- `completion()` - Get the result
- `is_loading()` - Check if request in progress
- `error()` - Get current error
- `complete(&str)` - Request completion

## Providers

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku |
| OpenRouter | 100+ models from various providers |

## Platform Support

- **Web** (default) - WASM with web-sys fetch
- **Desktop** - Coming soon (requires different HTTP client)

## License

Apache-2.0
