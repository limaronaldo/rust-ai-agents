//! Shared types for WASM bindings

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmTool {
    #[wasm_bindgen(skip)]
    pub name: String,
    #[wasm_bindgen(skip)]
    pub description: String,
    #[wasm_bindgen(skip)]
    pub parameters: serde_json::Value,
}

#[wasm_bindgen]
impl WasmTool {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    /// Set parameters from JSON string
    #[wasm_bindgen(js_name = setParameters)]
    pub fn set_parameters(&mut self, json: &str) -> Result<(), JsValue> {
        self.parameters = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        self.description.clone()
    }
}

/// Streaming chunk from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmStreamChunk {
    /// Text content delta
    #[wasm_bindgen(skip)]
    pub text: Option<String>,

    /// Tool call delta
    #[wasm_bindgen(skip)]
    pub tool_call: Option<WasmToolCallDelta>,

    /// Whether this is the final chunk
    #[wasm_bindgen(skip)]
    pub done: bool,
}

#[wasm_bindgen]
impl WasmStreamChunk {
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> Option<String> {
        self.text.clone()
    }

    #[wasm_bindgen(getter, js_name = isDone)]
    pub fn is_done(&self) -> bool {
        self.done
    }
}

/// Tool call delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmToolCallDelta {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments_delta: Option<String>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmModelInfo {
    #[wasm_bindgen(skip)]
    pub provider: String,
    #[wasm_bindgen(skip)]
    pub model: String,
    #[wasm_bindgen(skip)]
    pub max_tokens: u32,
    #[wasm_bindgen(skip)]
    pub supports_streaming: bool,
    #[wasm_bindgen(skip)]
    pub supports_tools: bool,
}

#[wasm_bindgen]
impl WasmModelInfo {
    #[wasm_bindgen(getter)]
    pub fn provider(&self) -> String {
        self.provider.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn model(&self) -> String {
        self.model.clone()
    }

    #[wasm_bindgen(getter, js_name = maxTokens)]
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    #[wasm_bindgen(getter, js_name = supportsStreaming)]
    pub fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }

    #[wasm_bindgen(getter, js_name = supportsTools)]
    pub fn supports_tools(&self) -> bool {
        self.supports_tools
    }
}
