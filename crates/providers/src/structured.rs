//! Structured Output Support
//!
//! Provides JSON schema validation for LLM responses with auto-retry on failures.
//!
//! ## Features
//!
//! - JSON Schema validation using draft-07
//! - Auto-retry with schema error feedback
//! - Type-safe extraction with serde
//! - Custom validation rules
//! - Configurable retry strategies
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents_providers::structured::{StructuredOutput, OutputSchema};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Serialize, Deserialize, OutputSchema)]
//! struct PersonInfo {
//!     name: String,
//!     age: u32,
//!     email: Option<String>,
//! }
//!
//! let structured = StructuredOutput::new(backend)
//!     .with_schema::<PersonInfo>()
//!     .with_max_retries(3);
//!
//! let person: PersonInfo = structured
//!     .generate("Extract person info from: John Doe, 30 years old")
//!     .await?;
//! ```

use crate::{LLMBackend, TokenUsage};
use rust_ai_agents_core::{errors::LLMError, LLMMessage};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// JSON Schema for structured output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Schema type (usually "object")
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Object properties
    #[serde(default)]
    pub properties: HashMap<String, PropertySchema>,
    /// Required fields
    #[serde(default)]
    pub required: Vec<String>,
    /// Additional properties allowed
    #[serde(default = "default_true")]
    pub additional_properties: bool,
    /// Schema description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Schema title
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

fn default_true() -> bool {
    true
}

impl JsonSchema {
    /// Create a new object schema
    pub fn object() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
            additional_properties: true,
            description: None,
            title: None,
        }
    }

    /// Add a property
    pub fn property(mut self, name: impl Into<String>, schema: PropertySchema) -> Self {
        self.properties.insert(name.into(), schema);
        self
    }

    /// Add a required property
    pub fn required_property(mut self, name: impl Into<String>, schema: PropertySchema) -> Self {
        let name = name.into();
        self.properties.insert(name.clone(), schema);
        self.required.push(name);
        self
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Disallow additional properties
    pub fn strict(mut self) -> Self {
        self.additional_properties = false;
        self
    }

    /// Convert to JSON Value
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

/// Property schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    /// Property type
    #[serde(rename = "type")]
    pub property_type: PropertyType,
    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Enum values (for string enums)
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Default value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// Minimum value (for numbers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value (for numbers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    /// Minimum length (for strings/arrays)
    #[serde(rename = "minLength", skip_serializing_if = "Option::is_none")]
    pub min_length: Option<usize>,
    /// Maximum length (for strings/arrays)
    #[serde(rename = "maxLength", skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
    /// Pattern (for strings)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// Array item schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<PropertySchema>>,
    /// Nested object properties
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, PropertySchema>>,
    /// Required nested properties
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

/// Property type enum
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PropertyType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

impl PropertySchema {
    /// Create a string property
    pub fn string() -> Self {
        Self {
            property_type: PropertyType::String,
            description: None,
            enum_values: None,
            default: None,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            pattern: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create a number property
    pub fn number() -> Self {
        Self {
            property_type: PropertyType::Number,
            ..Self::string()
        }
    }

    /// Create an integer property
    pub fn integer() -> Self {
        Self {
            property_type: PropertyType::Integer,
            ..Self::string()
        }
    }

    /// Create a boolean property
    pub fn boolean() -> Self {
        Self {
            property_type: PropertyType::Boolean,
            ..Self::string()
        }
    }

    /// Create an array property
    pub fn array(items: PropertySchema) -> Self {
        Self {
            property_type: PropertyType::Array,
            items: Some(Box::new(items)),
            ..Self::string()
        }
    }

    /// Create an object property
    pub fn object() -> Self {
        Self {
            property_type: PropertyType::Object,
            properties: Some(HashMap::new()),
            ..Self::string()
        }
    }

    /// Create an enum property
    pub fn enum_string(values: Vec<String>) -> Self {
        Self {
            property_type: PropertyType::String,
            enum_values: Some(values),
            ..Self::string()
        }
    }

    /// Add description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set minimum value
    pub fn min(mut self, min: f64) -> Self {
        self.minimum = Some(min);
        self
    }

    /// Set maximum value
    pub fn max(mut self, max: f64) -> Self {
        self.maximum = Some(max);
        self
    }

    /// Set min length
    pub fn min_length(mut self, len: usize) -> Self {
        self.min_length = Some(len);
        self
    }

    /// Set max length
    pub fn max_length(mut self, len: usize) -> Self {
        self.max_length = Some(len);
        self
    }

    /// Set pattern
    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = Some(pattern.into());
        self
    }

    /// Set default value
    pub fn default(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }

    /// Add nested property
    pub fn property(mut self, name: impl Into<String>, schema: PropertySchema) -> Self {
        if self.properties.is_none() {
            self.properties = Some(HashMap::new());
        }
        if let Some(props) = &mut self.properties {
            props.insert(name.into(), schema);
        }
        self
    }
}

/// Validation error details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Path to the error (e.g., "user.email")
    pub path: String,
    /// Error message
    pub message: String,
    /// Expected value/type
    pub expected: Option<String>,
    /// Actual value/type
    pub actual: Option<String>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Validation errors (if any)
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    /// Create a successful result
    pub fn ok() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
        }
    }

    /// Create a failed result with errors
    pub fn failed(errors: Vec<ValidationError>) -> Self {
        Self {
            valid: false,
            errors,
        }
    }

    /// Add an error
    pub fn add_error(&mut self, error: ValidationError) {
        self.valid = false;
        self.errors.push(error);
    }

    /// Get error summary for retry prompt
    pub fn error_summary(&self) -> String {
        self.errors
            .iter()
            .map(|e| format!("- {}", e))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Schema validator
pub struct SchemaValidator {
    schema: JsonSchema,
}

impl SchemaValidator {
    /// Create a new validator
    pub fn new(schema: JsonSchema) -> Self {
        Self { schema }
    }

    /// Validate a JSON value against the schema
    pub fn validate(&self, value: &Value) -> ValidationResult {
        let mut result = ValidationResult::ok();
        self.validate_value(value, &self.schema, "", &mut result);
        result
    }

    fn validate_value(
        &self,
        value: &Value,
        schema: &JsonSchema,
        path: &str,
        result: &mut ValidationResult,
    ) {
        // Check type
        if schema.schema_type.as_str() == "object" {
            if !value.is_object() {
                result.add_error(ValidationError {
                    path: path.to_string(),
                    message: "Expected object".to_string(),
                    expected: Some("object".to_string()),
                    actual: Some(self.type_name(value)),
                });
                return;
            }

            let obj = value.as_object().unwrap();

            // Check required fields
            for required in &schema.required {
                if !obj.contains_key(required) {
                    result.add_error(ValidationError {
                        path: if path.is_empty() {
                            required.clone()
                        } else {
                            format!("{}.{}", path, required)
                        },
                        message: format!("Required field '{}' is missing", required),
                        expected: Some("present".to_string()),
                        actual: Some("missing".to_string()),
                    });
                }
            }

            // Validate properties
            for (key, prop_schema) in &schema.properties {
                if let Some(prop_value) = obj.get(key) {
                    let prop_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path, key)
                    };
                    self.validate_property(prop_value, prop_schema, &prop_path, result);
                }
            }

            // Check for additional properties
            if !schema.additional_properties {
                for key in obj.keys() {
                    if !schema.properties.contains_key(key) {
                        result.add_error(ValidationError {
                            path: if path.is_empty() {
                                key.clone()
                            } else {
                                format!("{}.{}", path, key)
                            },
                            message: format!("Additional property '{}' not allowed", key),
                            expected: None,
                            actual: None,
                        });
                    }
                }
            }
        }
    }

    fn validate_property(
        &self,
        value: &Value,
        schema: &PropertySchema,
        path: &str,
        result: &mut ValidationResult,
    ) {
        // Type check
        let valid_type = match (&schema.property_type, value) {
            (PropertyType::String, Value::String(_)) => true,
            (PropertyType::Number, Value::Number(_)) => true,
            (PropertyType::Integer, Value::Number(n)) => n.is_i64() || n.is_u64(),
            (PropertyType::Boolean, Value::Bool(_)) => true,
            (PropertyType::Array, Value::Array(_)) => true,
            (PropertyType::Object, Value::Object(_)) => true,
            (PropertyType::Null, Value::Null) => true,
            _ => false,
        };

        if !valid_type {
            result.add_error(ValidationError {
                path: path.to_string(),
                message: "Type mismatch".to_string(),
                expected: Some(format!("{:?}", schema.property_type).to_lowercase()),
                actual: Some(self.type_name(value)),
            });
            return;
        }

        // Enum check
        if let Some(enum_values) = &schema.enum_values {
            if let Some(s) = value.as_str() {
                if !enum_values.contains(&s.to_string()) {
                    result.add_error(ValidationError {
                        path: path.to_string(),
                        message: "Value not in enum".to_string(),
                        expected: Some(format!("one of: {}", enum_values.join(", "))),
                        actual: Some(s.to_string()),
                    });
                }
            }
        }

        // Number range
        if let Some(num) = value.as_f64() {
            if let Some(min) = schema.minimum {
                if num < min {
                    result.add_error(ValidationError {
                        path: path.to_string(),
                        message: "Value below minimum".to_string(),
                        expected: Some(format!(">= {}", min)),
                        actual: Some(num.to_string()),
                    });
                }
            }
            if let Some(max) = schema.maximum {
                if num > max {
                    result.add_error(ValidationError {
                        path: path.to_string(),
                        message: "Value above maximum".to_string(),
                        expected: Some(format!("<= {}", max)),
                        actual: Some(num.to_string()),
                    });
                }
            }
        }

        // String length
        if let Some(s) = value.as_str() {
            if let Some(min) = schema.min_length {
                if s.len() < min {
                    result.add_error(ValidationError {
                        path: path.to_string(),
                        message: "String too short".to_string(),
                        expected: Some(format!("min length {}", min)),
                        actual: Some(format!("length {}", s.len())),
                    });
                }
            }
            if let Some(max) = schema.max_length {
                if s.len() > max {
                    result.add_error(ValidationError {
                        path: path.to_string(),
                        message: "String too long".to_string(),
                        expected: Some(format!("max length {}", max)),
                        actual: Some(format!("length {}", s.len())),
                    });
                }
            }
            // Pattern check
            if let Some(pattern) = &schema.pattern {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if !re.is_match(s) {
                        result.add_error(ValidationError {
                            path: path.to_string(),
                            message: "String doesn't match pattern".to_string(),
                            expected: Some(format!("pattern: {}", pattern)),
                            actual: Some(s.to_string()),
                        });
                    }
                }
            }
        }

        // Array items
        if let Some(arr) = value.as_array() {
            if let Some(items_schema) = &schema.items {
                for (i, item) in arr.iter().enumerate() {
                    let item_path = format!("{}[{}]", path, i);
                    self.validate_property(item, items_schema, &item_path, result);
                }
            }
        }

        // Nested object
        if let Some(obj) = value.as_object() {
            if let Some(nested_props) = &schema.properties {
                if let Some(required) = &schema.required {
                    for req_field in required {
                        if !obj.contains_key(req_field) {
                            result.add_error(ValidationError {
                                path: format!("{}.{}", path, req_field),
                                message: format!("Required field '{}' is missing", req_field),
                                expected: Some("present".to_string()),
                                actual: Some("missing".to_string()),
                            });
                        }
                    }
                }
                for (key, prop_schema) in nested_props {
                    if let Some(prop_value) = obj.get(key) {
                        let prop_path = format!("{}.{}", path, key);
                        self.validate_property(prop_value, prop_schema, &prop_path, result);
                    }
                }
            }
        }
    }

    fn type_name(&self, value: &Value) -> String {
        match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
        .to_string()
    }
}

/// Configuration for structured output
#[derive(Debug, Clone)]
pub struct StructuredConfig {
    /// Maximum retries on validation failure
    pub max_retries: u32,
    /// Temperature to use
    pub temperature: f32,
    /// Whether to include schema in system prompt
    pub include_schema_in_prompt: bool,
    /// Custom system prompt prefix
    pub system_prompt_prefix: Option<String>,
    /// Retry temperature adjustment (increase on retry)
    pub retry_temperature_delta: f32,
}

impl Default for StructuredConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            temperature: 0.0, // Low temperature for structured output
            include_schema_in_prompt: true,
            system_prompt_prefix: None,
            retry_temperature_delta: 0.1,
        }
    }
}

/// Structured output wrapper for LLM backends
pub struct StructuredOutput<B: LLMBackend> {
    backend: Arc<B>,
    schema: JsonSchema,
    validator: SchemaValidator,
    config: StructuredConfig,
}

impl<B: LLMBackend> StructuredOutput<B> {
    /// Create a new structured output wrapper
    pub fn new(backend: Arc<B>, schema: JsonSchema) -> Self {
        let validator = SchemaValidator::new(schema.clone());
        Self {
            backend,
            schema,
            validator,
            config: StructuredConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: StructuredConfig) -> Self {
        self.config = config;
        self
    }

    /// Set max retries
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    /// Generate structured output from a prompt
    pub async fn generate(&self, prompt: &str) -> Result<StructuredResult, LLMError> {
        let schema_json = serde_json::to_string_pretty(&self.schema)
            .map_err(|e| LLMError::SerializationError(e.to_string()))?;

        let system_prompt = format!(
            "{}You must respond with valid JSON that matches this schema:\n\n```json\n{}\n```\n\nRespond ONLY with the JSON object, no other text or explanation.",
            self.config.system_prompt_prefix.as_deref().unwrap_or(""),
            schema_json
        );

        let mut messages = vec![
            LLMMessage::system(system_prompt),
            LLMMessage::user(prompt.to_string()),
        ];

        let mut attempt = 0;
        let mut last_errors = Vec::new();
        let mut total_usage = TokenUsage::new(0, 0);

        while attempt <= self.config.max_retries {
            let temperature =
                self.config.temperature + (attempt as f32 * self.config.retry_temperature_delta);

            let output = self
                .backend
                .infer(&messages, &[], temperature.min(1.0))
                .await?;

            total_usage.prompt_tokens += output.token_usage.prompt_tokens;
            total_usage.completion_tokens += output.token_usage.completion_tokens;
            total_usage.total_tokens += output.token_usage.total_tokens;

            // Try to parse JSON from response
            let content = output.content.trim();
            let json_content = self.extract_json(content);

            match serde_json::from_str::<Value>(&json_content) {
                Ok(value) => {
                    let validation = self.validator.validate(&value);
                    if validation.valid {
                        return Ok(StructuredResult {
                            value,
                            raw_response: output.content,
                            attempts: attempt + 1,
                            token_usage: total_usage,
                        });
                    }

                    // Validation failed - retry with error feedback
                    last_errors = validation.errors.clone();

                    if attempt < self.config.max_retries {
                        let error_msg = format!(
                            "Your previous response had validation errors:\n{}\n\nPlease fix these issues and respond with valid JSON matching the schema.",
                            validation.error_summary()
                        );
                        messages.push(LLMMessage::assistant(output.content));
                        messages.push(LLMMessage::user(error_msg));
                    }
                }
                Err(parse_error) => {
                    // JSON parse error - retry with feedback
                    last_errors = vec![ValidationError {
                        path: "".to_string(),
                        message: format!("Invalid JSON: {}", parse_error),
                        expected: Some("valid JSON".to_string()),
                        actual: Some(content.chars().take(100).collect()),
                    }];

                    if attempt < self.config.max_retries {
                        let error_msg = format!(
                            "Your response was not valid JSON. Error: {}\n\nPlease respond with ONLY a valid JSON object matching the schema, no other text.",
                            parse_error
                        );
                        messages.push(LLMMessage::assistant(output.content));
                        messages.push(LLMMessage::user(error_msg));
                    }
                }
            }

            attempt += 1;
        }

        // All retries exhausted
        Err(LLMError::ValidationError(format!(
            "Failed to generate valid structured output after {} attempts. Last errors:\n{}",
            self.config.max_retries + 1,
            last_errors
                .iter()
                .map(|e| format!("- {}", e))
                .collect::<Vec<_>>()
                .join("\n")
        )))
    }

    /// Generate and deserialize to a specific type
    pub async fn generate_typed<T: DeserializeOwned>(&self, prompt: &str) -> Result<T, LLMError> {
        let result = self.generate(prompt).await?;
        serde_json::from_value(result.value)
            .map_err(|e| LLMError::SerializationError(e.to_string()))
    }

    /// Extract JSON from response (handles markdown code blocks)
    fn extract_json(&self, content: &str) -> String {
        // Try to extract from markdown code block
        if let Some(start) = content.find("```json") {
            if let Some(end) = content[start + 7..].find("```") {
                return content[start + 7..start + 7 + end].trim().to_string();
            }
        }

        // Try to extract from generic code block
        if let Some(start) = content.find("```") {
            let after_start = start + 3;
            // Skip language identifier if present
            let json_start = content[after_start..]
                .find('\n')
                .map(|i| after_start + i + 1)
                .unwrap_or(after_start);
            if let Some(end) = content[json_start..].find("```") {
                return content[json_start..json_start + end].trim().to_string();
            }
        }

        // Try to find JSON object directly
        if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                return content[start..=end].to_string();
            }
        }

        // Return as-is
        content.to_string()
    }
}

/// Result of structured generation
#[derive(Debug, Clone)]
pub struct StructuredResult {
    /// Parsed and validated JSON value
    pub value: Value,
    /// Raw LLM response
    pub raw_response: String,
    /// Number of attempts taken
    pub attempts: u32,
    /// Total token usage across all attempts
    pub token_usage: TokenUsage,
}

impl StructuredResult {
    /// Deserialize to a specific type
    pub fn into_typed<T: DeserializeOwned>(self) -> Result<T, LLMError> {
        serde_json::from_value(self.value).map_err(|e| LLMError::SerializationError(e.to_string()))
    }
}

/// Builder for creating schemas from Rust types
pub struct SchemaBuilder {
    schema: JsonSchema,
}

impl SchemaBuilder {
    /// Create a new schema builder
    pub fn new() -> Self {
        Self {
            schema: JsonSchema::object(),
        }
    }

    /// Set title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.schema.title = Some(title.into());
        self
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.schema.description = Some(desc.into());
        self
    }

    /// Add a string field
    pub fn string_field(self, name: impl Into<String>, required: bool) -> Self {
        self.field(name, PropertySchema::string(), required)
    }

    /// Add a number field
    pub fn number_field(self, name: impl Into<String>, required: bool) -> Self {
        self.field(name, PropertySchema::number(), required)
    }

    /// Add an integer field
    pub fn integer_field(self, name: impl Into<String>, required: bool) -> Self {
        self.field(name, PropertySchema::integer(), required)
    }

    /// Add a boolean field
    pub fn boolean_field(self, name: impl Into<String>, required: bool) -> Self {
        self.field(name, PropertySchema::boolean(), required)
    }

    /// Add a field with custom schema
    pub fn field(
        mut self,
        name: impl Into<String>,
        schema: PropertySchema,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.schema.properties.insert(name.clone(), schema);
        if required {
            self.schema.required.push(name);
        }
        self
    }

    /// Make schema strict (no additional properties)
    pub fn strict(mut self) -> Self {
        self.schema.additional_properties = false;
        self
    }

    /// Build the schema
    pub fn build(self) -> JsonSchema {
        self.schema
    }
}

impl Default for SchemaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceOutput;
    use rust_ai_agents_core::ToolSchema;

    #[test]
    fn test_schema_builder() {
        let schema = SchemaBuilder::new()
            .title("Person")
            .description("A person's information")
            .string_field("name", true)
            .integer_field("age", true)
            .string_field("email", false)
            .strict()
            .build();

        assert_eq!(schema.title, Some("Person".to_string()));
        assert!(schema.required.contains(&"name".to_string()));
        assert!(schema.required.contains(&"age".to_string()));
        assert!(!schema.required.contains(&"email".to_string()));
        assert!(!schema.additional_properties);
    }

    #[test]
    fn test_validation_success() {
        let schema = SchemaBuilder::new()
            .string_field("name", true)
            .integer_field("age", true)
            .build();

        let validator = SchemaValidator::new(schema);
        let value = serde_json::json!({
            "name": "John",
            "age": 30
        });

        let result = validator.validate(&value);
        assert!(result.valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validation_missing_required() {
        let schema = SchemaBuilder::new()
            .string_field("name", true)
            .integer_field("age", true)
            .build();

        let validator = SchemaValidator::new(schema);
        let value = serde_json::json!({
            "name": "John"
        });

        let result = validator.validate(&value);
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].message.contains("age"));
    }

    #[test]
    fn test_validation_type_mismatch() {
        let schema = SchemaBuilder::new().integer_field("age", true).build();

        let validator = SchemaValidator::new(schema);
        let value = serde_json::json!({
            "age": "thirty"
        });

        let result = validator.validate(&value);
        assert!(!result.valid);
        assert!(result.errors[0].message.contains("Type mismatch"));
    }

    #[test]
    fn test_validation_number_range() {
        let schema = JsonSchema::object()
            .required_property("score", PropertySchema::number().min(0.0).max(100.0));

        let validator = SchemaValidator::new(schema);

        let valid = serde_json::json!({"score": 50});
        assert!(validator.validate(&valid).valid);

        let too_low = serde_json::json!({"score": -5});
        assert!(!validator.validate(&too_low).valid);

        let too_high = serde_json::json!({"score": 150});
        assert!(!validator.validate(&too_high).valid);
    }

    #[test]
    fn test_validation_enum() {
        let schema = JsonSchema::object().required_property(
            "status",
            PropertySchema::enum_string(vec![
                "active".to_string(),
                "inactive".to_string(),
                "pending".to_string(),
            ]),
        );

        let validator = SchemaValidator::new(schema);

        let valid = serde_json::json!({"status": "active"});
        assert!(validator.validate(&valid).valid);

        let invalid = serde_json::json!({"status": "unknown"});
        assert!(!validator.validate(&invalid).valid);
    }

    #[test]
    fn test_validation_string_length() {
        let schema = JsonSchema::object().required_property(
            "code",
            PropertySchema::string().min_length(3).max_length(10),
        );

        let validator = SchemaValidator::new(schema);

        let valid = serde_json::json!({"code": "ABC123"});
        assert!(validator.validate(&valid).valid);

        let too_short = serde_json::json!({"code": "AB"});
        assert!(!validator.validate(&too_short).valid);

        let too_long = serde_json::json!({"code": "ABCDEFGHIJK"});
        assert!(!validator.validate(&too_long).valid);
    }

    #[test]
    fn test_validation_array() {
        let schema = JsonSchema::object()
            .required_property("tags", PropertySchema::array(PropertySchema::string()));

        let validator = SchemaValidator::new(schema);

        let valid = serde_json::json!({"tags": ["a", "b", "c"]});
        assert!(validator.validate(&valid).valid);

        let invalid = serde_json::json!({"tags": [1, 2, 3]});
        assert!(!validator.validate(&invalid).valid);
    }

    #[test]
    fn test_extract_json_from_markdown() {
        struct MockBackend;

        #[async_trait::async_trait]
        impl LLMBackend for MockBackend {
            async fn infer(
                &self,
                _messages: &[LLMMessage],
                _tools: &[ToolSchema],
                _temperature: f32,
            ) -> Result<InferenceOutput, LLMError> {
                unimplemented!()
            }

            async fn embed(&self, _text: &str) -> Result<Vec<f32>, LLMError> {
                unimplemented!()
            }

            fn model_info(&self) -> crate::ModelInfo {
                unimplemented!()
            }
        }

        let structured = StructuredOutput::new(Arc::new(MockBackend), JsonSchema::object());

        // Test markdown extraction
        let content = r#"Here's the JSON:
```json
{"name": "test"}
```
"#;
        assert_eq!(structured.extract_json(content), r#"{"name": "test"}"#);

        // Test direct JSON
        let content = r#"{"name": "test"}"#;
        assert_eq!(structured.extract_json(content), r#"{"name": "test"}"#);

        // Test with extra text
        let content = r#"The result is {"name": "test"} as requested."#;
        assert_eq!(structured.extract_json(content), r#"{"name": "test"}"#);
    }

    #[test]
    fn test_strict_schema() {
        let schema = SchemaBuilder::new()
            .string_field("name", true)
            .strict()
            .build();

        let validator = SchemaValidator::new(schema);

        let valid = serde_json::json!({"name": "John"});
        assert!(validator.validate(&valid).valid);

        let with_extra = serde_json::json!({"name": "John", "extra": "field"});
        let result = validator.validate(&with_extra);
        assert!(!result.valid);
        assert!(result.errors[0].message.contains("Additional property"));
    }
}
