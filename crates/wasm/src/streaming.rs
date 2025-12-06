//! SSE streaming helpers for WASM

use js_sys::{Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{ReadableStream, ReadableStreamDefaultReader};

/// Parse SSE events from a text chunk
pub fn parse_sse_events(chunk: &str) -> Vec<String> {
    chunk
        .split("\n\n")
        .filter(|s| !s.is_empty())
        .filter_map(|event| {
            event
                .lines()
                .find(|line| line.starts_with("data: "))
                .map(|line| line.strip_prefix("data: ").unwrap_or("").to_string())
        })
        .filter(|data| data != "[DONE]")
        .collect()
}

/// Extract text delta from OpenAI streaming response
pub fn extract_openai_delta(json: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| {
            v["choices"][0]["delta"]["content"]
                .as_str()
                .map(|s| s.to_string())
        })
}

/// Extract text delta from Anthropic streaming response
pub fn extract_anthropic_delta(json: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| {
            // Anthropic uses different event types
            let event_type = v["type"].as_str()?;

            match event_type {
                "content_block_delta" => v["delta"]["text"].as_str().map(|s| s.to_string()),
                _ => None,
            }
        })
}

/// Helper to create an async iterator from a ReadableStream for JavaScript
#[wasm_bindgen]
pub struct StreamReader {
    reader: ReadableStreamDefaultReader,
    decoder: web_sys::TextDecoder,
    buffer: String,
    provider: String,
}

#[wasm_bindgen]
impl StreamReader {
    /// Create a new stream reader
    #[wasm_bindgen(constructor)]
    pub fn new(stream: ReadableStream, provider: String) -> Result<StreamReader, JsValue> {
        let reader = stream
            .get_reader()
            .dyn_into::<ReadableStreamDefaultReader>()?;
        let decoder = web_sys::TextDecoder::new()?;

        Ok(Self {
            reader,
            decoder,
            buffer: String::new(),
            provider,
        })
    }

    /// Read the next chunk of text from the stream
    #[wasm_bindgen(js_name = readNext)]
    pub async fn read_next(&mut self) -> Result<JsValue, JsValue> {
        let result = wasm_bindgen_futures::JsFuture::from(self.reader.read()).await?;

        let done = Reflect::get(&result, &JsValue::from_str("done"))?
            .as_bool()
            .unwrap_or(true);

        if done {
            return Ok(JsValue::NULL);
        }

        let value = Reflect::get(&result, &JsValue::from_str("value"))?;
        let array = Uint8Array::new(&value);
        let decoded = self.decoder.decode_with_buffer_source(&array)?;

        self.buffer.push_str(&decoded);

        // Process complete SSE events
        let mut texts = Vec::new();

        while let Some(pos) = self.buffer.find("\n\n") {
            let event = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();

            if let Some(data) = event.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }

                let text = match self.provider.as_str() {
                    "anthropic" => extract_anthropic_delta(data),
                    _ => extract_openai_delta(data),
                };

                if let Some(t) = text {
                    texts.push(t);
                }
            }
        }

        if texts.is_empty() {
            // Return empty string if no complete events yet
            Ok(JsValue::from_str(""))
        } else {
            Ok(JsValue::from_str(&texts.join("")))
        }
    }

    /// Cancel the stream
    #[wasm_bindgen]
    pub fn cancel(&self) -> Result<(), JsValue> {
        let _ = self.reader.cancel();
        Ok(())
    }
}
