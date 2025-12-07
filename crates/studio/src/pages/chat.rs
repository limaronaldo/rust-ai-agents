//! Chat page with streaming LLM responses

use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Message in the chat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Stream event from SSE
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum StreamEvent {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "tool_call_start")]
    ToolCallStart { id: String, name: String },
    #[serde(rename = "tool_call_delta")]
    ToolCallDelta { id: String, arguments_delta: String },
    #[serde(rename = "tool_call_end")]
    ToolCallEnd { id: String },
    #[serde(rename = "done")]
    Done {
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Chat page component
#[component]
pub fn ChatPage() -> impl IntoView {
    let (messages, set_messages) = signal(Vec::<ChatMessage>::new());
    let (input, set_input) = signal(String::new());
    let (is_streaming, set_streaming) = signal(false);
    let (current_response, set_current_response) = signal(String::new());
    let (error, set_error) = signal(Option::<String>::None);

    // Core send logic (extracted to avoid event type issues)
    let do_send = move || {
        let user_input = input.get();
        if user_input.trim().is_empty() || is_streaming.get() {
            return;
        }

        // Add user message
        set_messages.update(|msgs| {
            msgs.push(ChatMessage {
                role: "user".to_string(),
                content: user_input.clone(),
            });
        });

        // Clear input and start streaming
        set_input.set(String::new());
        set_streaming.set(true);
        set_current_response.set(String::new());
        set_error.set(None);

        // Build request body
        let all_messages: Vec<_> = messages
            .get()
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content
                })
            })
            .collect();

        let request_body = serde_json::json!({
            "messages": all_messages,
            "temperature": 0.7
        });

        // Use fetch with ReadableStream for SSE
        spawn_local(async move {
            match fetch_stream(&request_body.to_string(), set_current_response, set_error).await {
                Ok(final_content) => {
                    // Add assistant message when done
                    set_messages.update(|msgs| {
                        msgs.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: final_content,
                        });
                    });
                    set_current_response.set(String::new());
                }
                Err(e) => {
                    set_error.set(Some(e));
                }
            }
            set_streaming.set(false);
        });
    };

    // Button click handler
    let on_click = move |_: web_sys::MouseEvent| {
        do_send();
    };

    // Handle Enter key
    let on_keydown = move |e: web_sys::KeyboardEvent| {
        if e.key() == "Enter" && !e.shift_key() {
            e.prevent_default();
            do_send();
        }
    };

    view! {
        <div class="flex flex-col h-full">
            <div class="flex items-center justify-between mb-4">
                <h1 class="text-2xl font-bold text-white">"Chat"</h1>
                <div class="text-sm text-gray-400">
                    {move || if is_streaming.get() { "Streaming..." } else { "Ready" }}
                </div>
            </div>

            // Error display
            {move || error.get().map(|e| view! {
                <div class="bg-red-900/50 border border-red-700 text-red-200 px-4 py-2 rounded-lg mb-4">
                    {e}
                </div>
            })}

            // Messages container
            <div class="flex-1 overflow-y-auto bg-gray-800 rounded-lg p-4 mb-4 space-y-4">
                {move || {
                    let msgs = messages.get();
                    if msgs.is_empty() && current_response.get().is_empty() {
                        view! {
                            <div class="flex flex-col items-center justify-center h-full text-gray-500">
                                <span class="text-4xl mb-2">"💬"</span>
                                <p>"Start a conversation with the AI"</p>
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <div class="space-y-4">
                                {msgs.into_iter().map(|msg| {
                                    let is_user = msg.role == "user";
                                    view! {
                                        <MessageBubble
                                            _role=msg.role
                                            content=msg.content
                                            is_user=is_user
                                        />
                                    }
                                }).collect::<Vec<_>>()}

                                // Streaming response
                                {move || {
                                    let response = current_response.get();
                                    if !response.is_empty() {
                                        Some(view! {
                                            <MessageBubble
                                                _role="assistant".to_string()
                                                content=response
                                                is_user=false
                                            />
                                        })
                                    } else {
                                        None
                                    }
                                }}
                            </div>
                        }.into_any()
                    }
                }}
            </div>

            // Input area
            <div class="flex gap-2">
                <textarea
                    class="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 resize-none"
                    placeholder="Type your message..."
                    rows=2
                    prop:value=move || input.get()
                    on:input=move |e| set_input.set(event_target_value(&e))
                    on:keydown=on_keydown
                    disabled=move || is_streaming.get()
                />
                <button
                    class="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
                    on:click=on_click
                    disabled=move || is_streaming.get() || input.get().trim().is_empty()
                >
                    {move || if is_streaming.get() { "..." } else { "Send" }}
                </button>
            </div>
        </div>
    }
}

/// Message bubble component
#[component]
fn MessageBubble(#[prop(into)] _role: String, content: String, is_user: bool) -> impl IntoView {
    let bg_class = if is_user {
        "bg-blue-600"
    } else {
        "bg-gray-700"
    };
    let align_class = if is_user { "ml-auto" } else { "mr-auto" };

    view! {
        <div class=format!("max-w-[80%] {} {} rounded-lg p-3", bg_class, align_class)>
            <div class="text-xs text-gray-400 mb-1">
                {if is_user { "You" } else { "Assistant" }}
            </div>
            <div class="text-white whitespace-pre-wrap">
                {content}
            </div>
        </div>
    }
}

/// Fetch with streaming support
async fn fetch_stream(
    body: &str,
    set_response: WriteSignal<String>,
    set_error: WriteSignal<Option<String>>,
) -> Result<String, String> {
    use js_sys::{Reflect, Uint8Array};
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let window = web_sys::window().ok_or("No window")?;
    let origin = window.location().origin().map_err(|_| "No origin")?;
    let url = format!("{}/api/inference/stream", origin);

    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::Cors);
    opts.set_body(&JsValue::from_str(body));

    let request = Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{:?}", e))?;

    request
        .headers()
        .set("Content-Type", "application/json")
        .map_err(|e| format!("{:?}", e))?;

    request
        .headers()
        .set("Accept", "text/event-stream")
        .map_err(|e| format!("{:?}", e))?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("{:?}", e))?;

    let resp: Response = resp_value.dyn_into().map_err(|_| "Not a Response")?;

    if !resp.ok() {
        return Err(format!("HTTP {}: {}", resp.status(), resp.status_text()));
    }

    let body = resp.body().ok_or("No body")?;
    let reader = body
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
        .map_err(|_| "Not a reader")?;

    let mut accumulated = String::new();
    let decoder = web_sys::TextDecoder::new().map_err(|e| format!("{:?}", e))?;

    loop {
        let result = JsFuture::from(reader.read())
            .await
            .map_err(|e| format!("{:?}", e))?;

        let done = Reflect::get(&result, &JsValue::from_str("done"))
            .map_err(|_| "No done")?
            .as_bool()
            .unwrap_or(true);

        if done {
            break;
        }

        let value = Reflect::get(&result, &JsValue::from_str("value")).map_err(|_| "No value")?;

        if value.is_undefined() {
            continue;
        }

        let chunk = Uint8Array::new(&value);
        let text = decoder
            .decode_with_buffer_source(&chunk)
            .map_err(|e| format!("{:?}", e))?;

        // Parse SSE lines
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                    match event {
                        StreamEvent::TextDelta { text } => {
                            accumulated.push_str(&text);
                            set_response.set(accumulated.clone());
                        }
                        StreamEvent::Error { message } => {
                            set_error.set(Some(message.clone()));
                            return Err(message);
                        }
                        StreamEvent::Done { .. } => {
                            // Stream complete
                        }
                        _ => {
                            // Handle tool calls etc. in future
                        }
                    }
                }
            }
        }
    }

    Ok(accumulated)
}
