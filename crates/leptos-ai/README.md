# leptos-ai

AI hooks for Leptos applications - chat, completions, and streaming.

## Features

- `use_chat` - Reactive chat with message history and streaming
- `use_completion` - Single completion requests  
- Built on `llm-client` for provider support (OpenAI, Anthropic, OpenRouter)
- Real-time streaming with `streaming_content` signal
- Stop generation support

## Installation

```toml
[dependencies]
leptos-ai = { git = "https://github.com/limaronaldo/rust-ai-agents" }
```

## Usage

### Chat Interface

```rust
use leptos::prelude::*;
use leptos_ai::{use_chat, ChatOptions};

#[component]
fn Chat() -> impl IntoView {
    let chat = use_chat(ChatOptions {
        provider: "openai".to_string(),
        api_key: "sk-...".to_string(),
        model: "gpt-4o-mini".to_string(),
        system_prompt: Some("You are a helpful assistant.".to_string()),
        ..Default::default()
    });

    let input_ref = NodeRef::<leptos::html::Input>::new();

    let on_submit = move |ev: leptos::ev::SubmitEvent| {
        ev.prevent_default();
        if let Some(input) = input_ref.get() {
            let value = input.value();
            if !value.is_empty() {
                chat.send(&value);
                input.set_value("");
            }
        }
    };

    view! {
        <div class="chat">
            // Message list
            <For
                each=move || chat.messages.get()
                key=|msg| msg.id.clone()
                children=move |msg| {
                    view! {
                        <div class=format!("message {}", msg.role)>
                            {msg.content.clone()}
                        </div>
                    }
                }
            />

            // Streaming indicator
            <Show when=move || !chat.streaming_content.get().is_empty()>
                <div class="message assistant streaming">
                    {move || chat.streaming_content.get()}
                </div>
            </Show>

            // Loading indicator
            <Show when=move || chat.is_loading.get()>
                <div class="loading">"..."</div>
            </Show>

            // Error display
            <Show when=move || chat.error.get().is_some()>
                <div class="error">
                    {move || chat.error.get().unwrap_or_default()}
                </div>
            </Show>

            // Input form
            <form on:submit=on_submit>
                <input 
                    type="text" 
                    node_ref=input_ref 
                    placeholder="Type a message..."
                    disabled=move || chat.is_loading.get()
                />
                <button type="submit" disabled=move || chat.is_loading.get()>
                    "Send"
                </button>
                <button 
                    type="button" 
                    on:click=move |_| chat.stop()
                    disabled=move || !chat.is_loading.get()
                >
                    "Stop"
                </button>
            </form>
        </div>
    }
}
```

### Single Completion

```rust
use leptos::prelude::*;
use leptos_ai::{use_completion, CompletionOptions};

#[component]
fn Translator() -> impl IntoView {
    let completion = use_completion(CompletionOptions {
        provider: "openai".to_string(),
        api_key: "sk-...".to_string(),
        model: "gpt-4o-mini".to_string(),
        system_prompt: Some("You are a translator. Translate to Spanish.".to_string()),
        ..Default::default()
    });

    let input_ref = NodeRef::<leptos::html::Input>::new();

    let on_translate = move |_| {
        if let Some(input) = input_ref.get() {
            completion.complete(&input.value());
        }
    };

    view! {
        <div>
            <input type="text" node_ref=input_ref placeholder="Enter text..." />
            <button on:click=on_translate disabled=move || completion.is_loading.get()>
                "Translate"
            </button>
            
            <Show when=move || completion.completion.get().is_some()>
                <p><strong>"Translation: "</strong>{move || completion.completion.get()}</p>
            </Show>
        </div>
    }
}
```

## API Reference

### `use_chat(options: ChatOptions) -> UseChat`

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

**UseChat:**
- `messages: RwSignal<Vec<ChatMessage>>` - Chat history
- `is_loading: RwSignal<bool>` - Request in progress
- `error: RwSignal<Option<String>>` - Current error
- `streaming_content: RwSignal<String>` - Real-time streaming text
- `send(&str)` - Send a message
- `clear()` - Clear history
- `stop()` - Stop generation

### `use_completion(options: CompletionOptions) -> UseCompletion`

Creates a single completion interface.

**UseCompletion:**
- `completion: RwSignal<Option<String>>` - The result
- `is_loading: RwSignal<bool>` - Request in progress
- `error: RwSignal<Option<String>>` - Current error
- `complete(&str)` - Request completion

## Providers

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku |
| OpenRouter | 100+ models from various providers |

## License

Apache-2.0
