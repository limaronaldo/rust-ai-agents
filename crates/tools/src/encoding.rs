//! Encoding and data transformation tools

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use rust_ai_agents_core::{errors::ToolError, ExecutionContext, Tool, ToolSchema};
use serde_json::{json, Value};
use sha2::{Digest, Sha256, Sha512};

/// JSON manipulation tool
pub struct JsonTool;

impl Default for JsonTool {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for JsonTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("json", "Parse, format, and manipulate JSON data")
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["parse", "stringify", "format", "minify", "get", "set", "merge", "validate"],
                        "description": "Operation to perform"
                    },
                    "data": {
                        "description": "JSON data (string or object)"
                    },
                    "path": {
                        "type": "string",
                        "description": "JSON path for get/set operations (e.g., 'user.name', 'items[0].id')"
                    },
                    "value": {
                        "description": "Value to set (for 'set' operation)"
                    },
                    "merge_with": {
                        "description": "Object to merge with (for 'merge' operation)"
                    }
                },
                "required": ["operation", "data"]
            }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let operation = arguments["operation"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'operation'".to_string()))?;

        match operation {
            "parse" => {
                let data_str = arguments["data"].as_str().ok_or_else(|| {
                    ToolError::InvalidArguments("'data' must be a string for parse".to_string())
                })?;

                let parsed: Value = serde_json::from_str(data_str)
                    .map_err(|e| ToolError::ExecutionFailed(format!("JSON parse error: {}", e)))?;

                Ok(json!({
                    "success": true,
                    "result": parsed
                }))
            }
            "stringify" | "format" => {
                let data = &arguments["data"];
                let formatted = serde_json::to_string_pretty(data)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

                Ok(json!({
                    "success": true,
                    "result": formatted
                }))
            }
            "minify" => {
                let data = &arguments["data"];
                let minified = serde_json::to_string(data)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

                Ok(json!({
                    "success": true,
                    "result": minified
                }))
            }
            "get" => {
                let path = arguments["path"]
                    .as_str()
                    .ok_or_else(|| ToolError::InvalidArguments("Missing 'path'".to_string()))?;

                let value = get_json_path(&arguments["data"], path)?;

                Ok(json!({
                    "success": true,
                    "path": path,
                    "result": value
                }))
            }
            "set" => {
                let path = arguments["path"]
                    .as_str()
                    .ok_or_else(|| ToolError::InvalidArguments("Missing 'path'".to_string()))?;
                let value = &arguments["value"];

                let mut data = arguments["data"].clone();
                set_json_path(&mut data, path, value.clone())?;

                Ok(json!({
                    "success": true,
                    "path": path,
                    "result": data
                }))
            }
            "merge" => {
                let merge_with = &arguments["merge_with"];
                let mut result = arguments["data"].clone();

                merge_json(&mut result, merge_with);

                Ok(json!({
                    "success": true,
                    "result": result
                }))
            }
            "validate" => {
                let data_str = arguments["data"].as_str();
                let (valid, error) = match data_str {
                    Some(s) => match serde_json::from_str::<Value>(s) {
                        Ok(_) => (true, None),
                        Err(e) => (false, Some(e.to_string())),
                    },
                    None => (true, None), // Already parsed as JSON
                };

                Ok(json!({
                    "valid": valid,
                    "error": error
                }))
            }
            _ => Err(ToolError::InvalidArguments(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }
}

fn get_json_path(data: &Value, path: &str) -> Result<Value, ToolError> {
    let mut current = data;

    for part in path.split('.') {
        // Handle array indexing like "items[0]"
        if let Some(bracket_pos) = part.find('[') {
            let key = &part[..bracket_pos];
            let index_str = &part[bracket_pos + 1..part.len() - 1];
            let index: usize = index_str.parse().map_err(|_| {
                ToolError::InvalidArguments(format!("Invalid array index: {}", index_str))
            })?;

            if !key.is_empty() {
                current = current
                    .get(key)
                    .ok_or_else(|| ToolError::ExecutionFailed(format!("Key not found: {}", key)))?;
            }

            current = current.get(index).ok_or_else(|| {
                ToolError::ExecutionFailed(format!("Index {} out of bounds", index))
            })?;
        } else if !part.is_empty() {
            current = current
                .get(part)
                .ok_or_else(|| ToolError::ExecutionFailed(format!("Key not found: {}", part)))?;
        }
    }

    Ok(current.clone())
}

fn set_json_path(data: &mut Value, path: &str, value: Value) -> Result<(), ToolError> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = data;

    for (i, part) in parts.iter().enumerate() {
        let is_last = i == parts.len() - 1;

        if let Some(bracket_pos) = part.find('[') {
            let key = &part[..bracket_pos];
            let index_str = &part[bracket_pos + 1..part.len() - 1];
            let index: usize = index_str.parse().map_err(|_| {
                ToolError::InvalidArguments(format!("Invalid array index: {}", index_str))
            })?;

            if !key.is_empty() {
                current = current
                    .get_mut(key)
                    .ok_or_else(|| ToolError::ExecutionFailed(format!("Key not found: {}", key)))?;
            }

            if is_last {
                if let Some(arr) = current.as_array_mut() {
                    if index < arr.len() {
                        arr[index] = value;
                        return Ok(());
                    }
                }
                return Err(ToolError::ExecutionFailed(format!(
                    "Index {} out of bounds",
                    index
                )));
            }

            current = current.get_mut(index).ok_or_else(|| {
                ToolError::ExecutionFailed(format!("Index {} out of bounds", index))
            })?;
        } else if is_last {
            if let Some(obj) = current.as_object_mut() {
                obj.insert(part.to_string(), value);
                return Ok(());
            }
            return Err(ToolError::ExecutionFailed(
                "Cannot set property on non-object".to_string(),
            ));
        } else {
            current = current
                .get_mut(*part)
                .ok_or_else(|| ToolError::ExecutionFailed(format!("Key not found: {}", part)))?;
        }
    }

    Ok(())
}

fn merge_json(target: &mut Value, source: &Value) {
    match (target, source) {
        (Value::Object(target_map), Value::Object(source_map)) => {
            for (key, value) in source_map {
                if let Some(target_value) = target_map.get_mut(key) {
                    merge_json(target_value, value);
                } else {
                    target_map.insert(key.clone(), value.clone());
                }
            }
        }
        (target, source) => {
            *target = source.clone();
        }
    }
}

/// Base64 encoding/decoding tool
pub struct Base64Tool;

impl Default for Base64Tool {
    fn default() -> Self {
        Self::new()
    }
}

impl Base64Tool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for Base64Tool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("base64", "Encode or decode Base64 strings").with_parameters(json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["encode", "decode"],
                    "description": "Whether to encode or decode"
                },
                "data": {
                    "type": "string",
                    "description": "Data to encode or decode"
                }
            },
            "required": ["operation", "data"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let operation = arguments["operation"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'operation'".to_string()))?;
        let data = arguments["data"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'data'".to_string()))?;

        match operation {
            "encode" => {
                let encoded = BASE64.encode(data.as_bytes());
                Ok(json!({
                    "operation": "encode",
                    "input": data,
                    "result": encoded
                }))
            }
            "decode" => {
                let decoded_bytes = BASE64.decode(data).map_err(|e| {
                    ToolError::ExecutionFailed(format!("Base64 decode error: {}", e))
                })?;
                let decoded = String::from_utf8(decoded_bytes).map_err(|e| {
                    ToolError::ExecutionFailed(format!("UTF-8 decode error: {}", e))
                })?;
                Ok(json!({
                    "operation": "decode",
                    "input": data,
                    "result": decoded
                }))
            }
            _ => Err(ToolError::InvalidArguments(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }
}

/// Hash generation tool
pub struct HashTool;

impl Default for HashTool {
    fn default() -> Self {
        Self::new()
    }
}

impl HashTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for HashTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("hash", "Generate cryptographic hashes").with_parameters(json!({
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data to hash"
                },
                "algorithm": {
                    "type": "string",
                    "enum": ["sha256", "sha512"],
                    "description": "Hash algorithm (default: sha256)"
                },
                "output": {
                    "type": "string",
                    "enum": ["hex", "base64"],
                    "description": "Output format (default: hex)"
                }
            },
            "required": ["data"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let data = arguments["data"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'data'".to_string()))?;
        let algorithm = arguments["algorithm"].as_str().unwrap_or("sha256");
        let output_format = arguments["output"].as_str().unwrap_or("hex");

        let hash_bytes: Vec<u8> = match algorithm {
            "sha256" => {
                let mut hasher = Sha256::new();
                hasher.update(data.as_bytes());
                hasher.finalize().to_vec()
            }
            "sha512" => {
                let mut hasher = Sha512::new();
                hasher.update(data.as_bytes());
                hasher.finalize().to_vec()
            }
            _ => {
                return Err(ToolError::InvalidArguments(format!(
                    "Unknown algorithm: {}",
                    algorithm
                )))
            }
        };

        let result = match output_format {
            "hex" => hex::encode(&hash_bytes),
            "base64" => BASE64.encode(&hash_bytes),
            _ => {
                return Err(ToolError::InvalidArguments(format!(
                    "Unknown output format: {}",
                    output_format
                )))
            }
        };

        Ok(json!({
            "data": data,
            "algorithm": algorithm,
            "output_format": output_format,
            "hash": result,
            "length": hash_bytes.len() * 8
        }))
    }
}

/// URL encoding/decoding tool
pub struct UrlEncodeTool;

impl Default for UrlEncodeTool {
    fn default() -> Self {
        Self::new()
    }
}

impl UrlEncodeTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for UrlEncodeTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("url_encode", "Encode or decode URL strings").with_parameters(json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["encode", "decode"],
                    "description": "Whether to encode or decode"
                },
                "data": {
                    "type": "string",
                    "description": "String to encode or decode"
                }
            },
            "required": ["operation", "data"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let operation = arguments["operation"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'operation'".to_string()))?;
        let data = arguments["data"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'data'".to_string()))?;

        match operation {
            "encode" => {
                let encoded = urlencoding::encode(data);
                Ok(json!({
                    "operation": "encode",
                    "input": data,
                    "result": encoded.to_string()
                }))
            }
            "decode" => {
                let decoded = urlencoding::decode(data)
                    .map_err(|e| ToolError::ExecutionFailed(format!("URL decode error: {}", e)))?;
                Ok(json!({
                    "operation": "decode",
                    "input": data,
                    "result": decoded.to_string()
                }))
            }
            _ => Err(ToolError::InvalidArguments(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_ai_agents_core::types::AgentId;

    fn test_ctx() -> ExecutionContext {
        ExecutionContext::new(AgentId::new("test-agent"))
    }

    #[tokio::test]
    async fn test_json_parse() {
        let tool = JsonTool::new();
        let ctx = test_ctx();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "operation": "parse",
                    "data": r#"{"name": "test", "value": 42}"#
                }),
            )
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert_eq!(result["result"]["name"], "test");
        assert_eq!(result["result"]["value"], 42);
    }

    #[tokio::test]
    async fn test_json_get_path() {
        let tool = JsonTool::new();
        let ctx = test_ctx();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "operation": "get",
                    "data": {"user": {"name": "John", "age": 30}},
                    "path": "user.name"
                }),
            )
            .await
            .unwrap();

        assert_eq!(result["result"], "John");
    }

    #[tokio::test]
    async fn test_json_merge() {
        let tool = JsonTool::new();
        let ctx = test_ctx();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "operation": "merge",
                    "data": {"a": 1, "b": 2},
                    "merge_with": {"b": 3, "c": 4}
                }),
            )
            .await
            .unwrap();

        assert_eq!(result["result"]["a"], 1);
        assert_eq!(result["result"]["b"], 3);
        assert_eq!(result["result"]["c"], 4);
    }

    #[tokio::test]
    async fn test_base64() {
        let tool = Base64Tool::new();
        let ctx = test_ctx();

        let encoded = tool
            .execute(
                &ctx,
                json!({
                    "operation": "encode",
                    "data": "Hello, World!"
                }),
            )
            .await
            .unwrap();

        assert_eq!(encoded["result"], "SGVsbG8sIFdvcmxkIQ==");

        let decoded = tool
            .execute(
                &ctx,
                json!({
                    "operation": "decode",
                    "data": "SGVsbG8sIFdvcmxkIQ=="
                }),
            )
            .await
            .unwrap();

        assert_eq!(decoded["result"], "Hello, World!");
    }

    #[tokio::test]
    async fn test_hash() {
        let tool = HashTool::new();
        let ctx = test_ctx();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "data": "hello",
                    "algorithm": "sha256"
                }),
            )
            .await
            .unwrap();

        assert_eq!(
            result["hash"],
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }
}
