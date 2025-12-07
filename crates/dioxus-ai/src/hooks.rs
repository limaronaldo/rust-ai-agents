//! Reactive hooks for Dioxus AI

use dioxus::prelude::*;
use rust_ai_agents_llm_client::{Message, Provider, RequestBuilder, ResponseParser};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

use crate::{ChatMessage, ChatOptions, CompletionOptions, DioxusAiError, Result};

/// State for the chat hook
#[derive(Clone)]
pub struct UseChatState {
    messages: Signal<Vec<ChatMessage>>,
    is_loading: Signal<bool>,
    error: Signal<Option<String>>,
    streaming_content: Signal<String>,
    should_stop: Signal<bool>,
    options: ChatOptions,
}

impl UseChatState {
    /// Get current messages
    pub fn messages(&self) -> Vec<ChatMessage> {
        (self.messages)()
    }

    /// Check if loading
    pub fn is_loading(&self) -> bool {
        (self.is_loading)()
    }

    /// Get current error
    pub fn error(&self) -> Option<String> {
        (self.error)()
    }

    /// Get streaming content
    pub fn streaming_content(&self) -> String {
        (self.streaming_content)()
    }

    /// Send a message
    pub fn send(&mut self, message: &str) {
        let user_msg = ChatMessage::user(message);
        let mut messages = self.messages;
        let mut is_loading = self.is_loading;
        let mut error = self.error;
        let mut streaming_content = self.streaming_content;
        let mut should_stop = self.should_stop;
        let options = self.options.clone();

        // Add user message
        messages.write().push(user_msg);

        // Reset state
        is_loading.set(true);
        error.set(None);
        streaming_content.set(String::new());
        should_stop.set(false);

        let current_messages = messages();

        // Spawn async request
        spawn(async move {
            let result = if options.stream {
                send_streaming_request(&options, current_messages, streaming_content, should_stop).await
            } else {
                send_request(&options, current_messages).await
            };

            match result {
                Ok(content) => {
                    let assistant_msg = ChatMessage::assistant(content);
                    messages.write().push(assistant_msg);
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
    pub fn clear(&mut self) {
        self.messages.write().clear();
        self.error.set(None);
        self.streaming_content.set(String::new());
    }

    /// Stop generation
    pub fn stop(&mut self) {
        self.should_stop.set(true);
    }
}

/// Create a reactive chat interface
///
/// # Example
///
/// ```rust,ignore
/// let mut chat = use_chat(ChatOptions {
///     provider: "openai".to_string(),
///     api_key: "sk-...".to_string(),
///     model: "gpt-4o-mini".to_string(),
///     ..Default::default()
/// });
///
/// chat.send("Hello!");
/// ```
pub fn use_chat(options: ChatOptions) -> UseChatState {
    let messages = use_signal(|| options.initial_messages.clone());
    let is_loading = use_signal(|| false);
    let error = use_signal(|| None::<String>);
    let streaming_content = use_signal(String::new);
    let should_stop = use_signal(|| false);

    UseChatState {
        messages,
        is_loading,
        error,
        streaming_content,
        should_stop,
        options,
    }
}

/// State for the completion hook
#[derive(Clone)]
pub struct UseCompletionState {
    completion: Signal<Option<String>>,
    is_loading: Signal<bool>,
    error: Signal<Option<String>>,
    options: CompletionOptions,
}

impl UseCompletionState {
    /// Get completion result
    pub fn completion(&self) -> Option<String> {
        (self.completion)()
    }

    /// Check if loading
    pub fn is_loading(&self) -> bool {
        (self.is_loading)()
    }

    /// Get current error
    pub fn error(&self) -> Option<String> {
        (self.error)()
    }

    /// Request a completion
    pub fn complete(&mut self, prompt: &str) {
        let mut completion = self.completion;
        let mut is_loading = self.is_loading;
        let mut error = self.error;
        let options = self.options.clone();
        let prompt = prompt.to_string();

        is_loading.set(true);
        error.set(None);
        completion.set(None);

        spawn(async move {
            let messages = vec![ChatMessage::user(&prompt)];

            let chat_opts = ChatOptions {
                provider: options.provider,
                api_key: options.api_key,
                model: options.model,
                system_prompt: options.system_prompt,
                temperature: options.temperature,
                max_tokens: options.max_tokens,
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

/// Create a completion interface
///
/// # Example
///
/// ```rust,ignore
/// let mut completion = use_completion(CompletionOptions {
///     provider: "openai".to_string(),
///     api_key: "sk-...".to_string(),
///     model: "gpt-4o-mini".to_string(),
///     ..Default::default()
/// });
///
/// completion.complete("Translate 'hello' to Spanish");
/// ```
pub fn use_completion(options: CompletionOptions) -> UseCompletionState {
    let completion = use_signal(|| None::<String>);
    let is_loading = use_signal(|| false);
    let error = use_signal(|| None::<String>);

    UseCompletionState {
        completion,
        is_loading,
        error,
        options,
    }
}

/// Send a non-streaming request
async fn send_request(options: &ChatOptions, messages: Vec<ChatMessage>) -> Result<String> {
    let provider: Provider = options
        .provider
        .parse()
        .map_err(|_| DioxusAiError::InvalidProvider(options.provider.clone()))?;

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

    let http_request = RequestBuilder::new(provider)
        .model(&options.model)
        .api_key(&options.api_key)
        .messages(&llm_messages)
        .temperature(options.temperature)
        .max_tokens(options.max_tokens)
        .stream(false)
        .build()?;

    let response = fetch(&http_request.url, &http_request.headers, &http_request.body).await?;
    let llm_response = ResponseParser::parse(provider, &response)?;

    Ok(llm_response.content)
}

/// Send a streaming request
async fn send_streaming_request(
    options: &ChatOptions,
    messages: Vec<ChatMessage>,
    mut streaming_content: Signal<String>,
    should_stop: Signal<bool>,
) -> Result<String> {
    let provider: Provider = options
        .provider
        .parse()
        .map_err(|_| DioxusAiError::InvalidProvider(options.provider.clone()))?;

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

    let http_request = RequestBuilder::new(provider)
        .model(&options.model)
        .api_key(&options.api_key)
        .messages(&llm_messages)
        .temperature(options.temperature)
        .max_tokens(options.max_tokens)
        .stream(true)
        .build()?;

    let response = fetch_stream(&http_request.url, &http_request.headers, &http_request.body).await?;

    let mut full_content = String::new();
    let reader = response
        .body()
        .ok_or_else(|| DioxusAiError::StreamError("No response body".to_string()))?
        .get_reader();

    let reader: web_sys::ReadableStreamDefaultReader = reader.unchecked_into();

    loop {
        if should_stop() {
            break;
        }

        let result = JsFuture::from(reader.read()).await;
        let result = result.map_err(|e| DioxusAiError::StreamError(format!("{:?}", e)))?;

        let done = js_sys::Reflect::get(&result, &JsValue::from_str("done"))
            .map_err(|e| DioxusAiError::StreamError(format!("{:?}", e)))?
            .as_bool()
            .unwrap_or(true);

        if done {
            break;
        }

        let value = js_sys::Reflect::get(&result, &JsValue::from_str("value"))
            .map_err(|e| DioxusAiError::StreamError(format!("{:?}", e)))?;

        let array = js_sys::Uint8Array::new(&value);
        let bytes = array.to_vec();
        let text = String::from_utf8_lossy(&bytes);

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

async fn fetch(url: &str, headers: &[(String, String)], body: &str) -> Result<String> {
    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::Cors);
    opts.set_body(&JsValue::from_str(body));

    let js_headers = web_sys::Headers::new()
        .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;

    for (key, value) in headers {
        js_headers
            .set(key, value)
            .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;
    }
    opts.set_headers(&js_headers);

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;

    let window = web_sys::window().ok_or_else(|| DioxusAiError::RequestFailed("No window".to_string()))?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;

    let resp: Response = resp_value
        .dyn_into()
        .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;

    if !resp.ok() {
        let status = resp.status();
        let text = JsFuture::from(resp.text().map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?)
            .await
            .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?
            .as_string()
            .unwrap_or_default();
        return Err(DioxusAiError::ApiError(format!("HTTP {}: {}", status, text)));
    }

    let text = JsFuture::from(resp.text().map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?)
        .await
        .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?
        .as_string()
        .unwrap_or_default();

    Ok(text)
}

async fn fetch_stream(url: &str, headers: &[(String, String)], body: &str) -> Result<Response> {
    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::Cors);
    opts.set_body(&JsValue::from_str(body));

    let js_headers = web_sys::Headers::new()
        .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;

    for (key, value) in headers {
        js_headers
            .set(key, value)
            .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;
    }
    opts.set_headers(&js_headers);

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;

    let window = web_sys::window().ok_or_else(|| DioxusAiError::RequestFailed("No window".to_string()))?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;

    let resp: Response = resp_value
        .dyn_into()
        .map_err(|e| DioxusAiError::RequestFailed(format!("{:?}", e)))?;

    if !resp.ok() {
        let status = resp.status();
        return Err(DioxusAiError::ApiError(format!("HTTP {}", status)));
    }

    Ok(resp)
}
