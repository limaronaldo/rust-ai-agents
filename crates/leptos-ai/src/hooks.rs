//! Reactive hooks for Leptos AI

use leptos::prelude::*;
use rust_ai_agents_llm_client::{Message, Provider, RequestBuilder, ResponseParser};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

use crate::{ChatMessage, ChatOptions, CompletionOptions, LeptosAiError, Result};

/// Return type for use_chat hook
#[derive(Clone, Copy)]
pub struct UseChat {
    /// Reactive list of messages
    pub messages: RwSignal<Vec<ChatMessage>>,
    /// Whether a request is in progress
    pub is_loading: RwSignal<bool>,
    /// Current error, if any
    pub error: RwSignal<Option<String>>,
    /// Current streaming content (updated in real-time)
    pub streaming_content: RwSignal<String>,
    /// Internal: options stored
    options: RwSignal<ChatOptions>,
    /// Internal: stop flag
    should_stop: RwSignal<bool>,
}

impl UseChat {
    /// Send a message to the chat
    pub fn send(&self, message: &str) {
        let opts = self.options.get_untracked();
        let messages_signal = self.messages;
        let is_loading = self.is_loading;
        let error = self.error;
        let streaming_content = self.streaming_content;
        let should_stop = self.should_stop;

        // Add user message
        let user_msg = ChatMessage::user(message);
        messages_signal.update(|msgs| msgs.push(user_msg));

        // Reset state
        is_loading.set(true);
        error.set(None);
        streaming_content.set(String::new());
        should_stop.set(false);

        let current_messages = messages_signal.get_untracked();

        // Spawn the async request
        leptos::task::spawn_local(async move {
            let result = if opts.stream {
                send_streaming_request(&opts, current_messages, streaming_content, should_stop)
                    .await
            } else {
                send_request(&opts, current_messages).await
            };

            match result {
                Ok(content) => {
                    let assistant_msg = ChatMessage::assistant(content);
                    messages_signal.update(|msgs| msgs.push(assistant_msg));
                }
                Err(e) => {
                    error.set(Some(e.to_string()));
                }
            }

            is_loading.set(false);
            streaming_content.set(String::new());
        });
    }

    /// Clear all messages
    pub fn clear(&self) {
        self.messages.set(Vec::new());
        self.error.set(None);
        self.streaming_content.set(String::new());
    }

    /// Stop the current generation
    pub fn stop(&self) {
        self.should_stop.set(true);
    }
}

/// Create a reactive chat interface
///
/// # Example
///
/// ```rust,ignore
/// let chat = use_chat(ChatOptions {
///     provider: "openai".to_string(),
///     api_key: "sk-...".to_string(),
///     model: "gpt-4o-mini".to_string(),
///     ..Default::default()
/// });
///
/// // Send a message
/// chat.send("Hello!");
///
/// // Access messages reactively
/// let messages = chat.messages.get();
/// ```
pub fn use_chat(options: ChatOptions) -> UseChat {
    let messages = RwSignal::new(options.initial_messages.clone());
    let is_loading = RwSignal::new(false);
    let error = RwSignal::new(None::<String>);
    let streaming_content = RwSignal::new(String::new());
    let should_stop = RwSignal::new(false);
    let options_signal = RwSignal::new(options);

    UseChat {
        messages,
        is_loading,
        error,
        streaming_content,
        options: options_signal,
        should_stop,
    }
}

/// Return type for use_completion hook
#[derive(Clone, Copy)]
pub struct UseCompletion {
    /// The completion result
    pub completion: RwSignal<Option<String>>,
    /// Whether a request is in progress
    pub is_loading: RwSignal<bool>,
    /// Current error, if any
    pub error: RwSignal<Option<String>>,
    /// Internal: options
    options: RwSignal<CompletionOptions>,
}

impl UseCompletion {
    /// Request a completion
    pub fn complete(&self, prompt: &str) {
        let opts = self.options.get_untracked();
        let completion = self.completion;
        let is_loading = self.is_loading;
        let error = self.error;
        let prompt = prompt.to_string();

        is_loading.set(true);
        error.set(None);
        completion.set(None);

        leptos::task::spawn_local(async move {
            let messages = vec![ChatMessage::user(&prompt)];

            let chat_opts = ChatOptions {
                provider: opts.provider,
                api_key: opts.api_key,
                model: opts.model,
                system_prompt: opts.system_prompt,
                temperature: opts.temperature,
                max_tokens: opts.max_tokens,
                stream: false,
                initial_messages: Vec::new(),
            };

            match send_request(&chat_opts, messages).await {
                Ok(content) => {
                    completion.set(Some(content));
                }
                Err(e) => {
                    error.set(Some(e.to_string()));
                }
            }

            is_loading.set(false);
        });
    }
}

/// Create a completion interface for single requests
///
/// # Example
///
/// ```rust,ignore
/// let completion = use_completion(CompletionOptions {
///     provider: "openai".to_string(),
///     api_key: "sk-...".to_string(),
///     model: "gpt-4o-mini".to_string(),
///     ..Default::default()
/// });
///
/// // Request a completion
/// completion.complete("Translate 'hello' to Spanish");
///
/// // Access result reactively
/// if let Some(result) = completion.completion.get() {
///     println!("{}", result);
/// }
/// ```
pub fn use_completion(options: CompletionOptions) -> UseCompletion {
    let completion = RwSignal::new(None::<String>);
    let is_loading = RwSignal::new(false);
    let error = RwSignal::new(None::<String>);
    let options_signal = RwSignal::new(options);

    UseCompletion {
        completion,
        is_loading,
        error,
        options: options_signal,
    }
}

/// Send a non-streaming request
async fn send_request(options: &ChatOptions, messages: Vec<ChatMessage>) -> Result<String> {
    let provider: Provider = options
        .provider
        .parse()
        .map_err(|_| LeptosAiError::InvalidProvider(options.provider.clone()))?;

    // Convert ChatMessage to llm-client Message
    let mut llm_messages: Vec<Message> = Vec::new();

    if let Some(ref system) = options.system_prompt {
        llm_messages.push(Message::system(system));
    }

    for msg in &messages {
        match msg.role.as_str() {
            "user" => llm_messages.push(Message::user(&msg.content)),
            "assistant" => llm_messages.push(Message::assistant(&msg.content)),
            "system" => llm_messages.push(Message::system(&msg.content)),
            _ => {}
        }
    }

    // Build request using llm-client
    let http_request = RequestBuilder::new(provider)
        .model(&options.model)
        .api_key(&options.api_key)
        .messages(&llm_messages)
        .temperature(options.temperature)
        .max_tokens(options.max_tokens)
        .stream(false)
        .build()?;

    // Send via web-sys fetch
    let response = fetch(&http_request.url, &http_request.headers, &http_request.body).await?;

    // Parse response using llm-client
    let llm_response = ResponseParser::parse(provider, &response)?;

    Ok(llm_response.content)
}

/// Send a streaming request
async fn send_streaming_request(
    options: &ChatOptions,
    messages: Vec<ChatMessage>,
    streaming_content: RwSignal<String>,
    should_stop: RwSignal<bool>,
) -> Result<String> {
    let provider: Provider = options
        .provider
        .parse()
        .map_err(|_| LeptosAiError::InvalidProvider(options.provider.clone()))?;

    // Convert ChatMessage to llm-client Message
    let mut llm_messages: Vec<Message> = Vec::new();

    if let Some(ref system) = options.system_prompt {
        llm_messages.push(Message::system(system));
    }

    for msg in &messages {
        match msg.role.as_str() {
            "user" => llm_messages.push(Message::user(&msg.content)),
            "assistant" => llm_messages.push(Message::assistant(&msg.content)),
            "system" => llm_messages.push(Message::system(&msg.content)),
            _ => {}
        }
    }

    // Build request using llm-client
    let http_request = RequestBuilder::new(provider)
        .model(&options.model)
        .api_key(&options.api_key)
        .messages(&llm_messages)
        .temperature(options.temperature)
        .max_tokens(options.max_tokens)
        .stream(true)
        .build()?;

    // Send via web-sys fetch and get streaming response
    let response =
        fetch_stream(&http_request.url, &http_request.headers, &http_request.body).await?;

    // Process SSE stream
    let mut full_content = String::new();
    let reader = response
        .body()
        .ok_or_else(|| LeptosAiError::StreamError("No response body".to_string()))?
        .get_reader();

    let reader: web_sys::ReadableStreamDefaultReader = reader.unchecked_into();

    loop {
        if should_stop.get_untracked() {
            break;
        }

        let result = JsFuture::from(reader.read()).await;
        let result = result.map_err(|e| LeptosAiError::StreamError(format!("{:?}", e)))?;

        let done = js_sys::Reflect::get(&result, &JsValue::from_str("done"))
            .map_err(|e| LeptosAiError::StreamError(format!("{:?}", e)))?
            .as_bool()
            .unwrap_or(true);

        if done {
            break;
        }

        let value = js_sys::Reflect::get(&result, &JsValue::from_str("value"))
            .map_err(|e| LeptosAiError::StreamError(format!("{:?}", e)))?;

        let array = js_sys::Uint8Array::new(&value);
        let bytes = array.to_vec();
        let text = String::from_utf8_lossy(&bytes);

        // Parse SSE lines
        for line in text.lines() {
            if let Ok(Some(chunk)) = ResponseParser::parse_stream_line(provider, line) {
                if let Some(content) = chunk.content {
                    full_content.push_str(&content);
                    streaming_content.set(full_content.clone());
                }
                if chunk.done {
                    break;
                }
            }
        }
    }

    Ok(full_content)
}

/// Fetch helper using web-sys
async fn fetch(url: &str, headers: &[(String, String)], body: &str) -> Result<String> {
    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::Cors);
    opts.set_body(&JsValue::from_str(body));

    let js_headers =
        web_sys::Headers::new().map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;

    for (key, value) in headers {
        js_headers
            .set(key, value)
            .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;
    }
    opts.set_headers(&js_headers);

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;

    let window =
        web_sys::window().ok_or_else(|| LeptosAiError::RequestFailed("No window".to_string()))?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;

    let resp: Response = resp_value
        .dyn_into()
        .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;

    if !resp.ok() {
        let status = resp.status();
        let text = JsFuture::from(
            resp.text()
                .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?,
        )
        .await
        .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?
        .as_string()
        .unwrap_or_default();
        return Err(LeptosAiError::ApiError(format!(
            "HTTP {}: {}",
            status, text
        )));
    }

    let text = JsFuture::from(
        resp.text()
            .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?,
    )
    .await
    .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?
    .as_string()
    .unwrap_or_default();

    Ok(text)
}

/// Fetch helper for streaming responses
async fn fetch_stream(url: &str, headers: &[(String, String)], body: &str) -> Result<Response> {
    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::Cors);
    opts.set_body(&JsValue::from_str(body));

    let js_headers =
        web_sys::Headers::new().map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;

    for (key, value) in headers {
        js_headers
            .set(key, value)
            .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;
    }
    opts.set_headers(&js_headers);

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;

    let window =
        web_sys::window().ok_or_else(|| LeptosAiError::RequestFailed("No window".to_string()))?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;

    let resp: Response = resp_value
        .dyn_into()
        .map_err(|e| LeptosAiError::RequestFailed(format!("{:?}", e)))?;

    if !resp.ok() {
        let status = resp.status();
        return Err(LeptosAiError::ApiError(format!("HTTP {}", status)));
    }

    Ok(resp)
}
