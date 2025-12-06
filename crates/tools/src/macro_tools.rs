//! Tools defined using the derive macro
//!
//! This module demonstrates how to create tools using the `#[derive(Tool)]` macro
//! for minimal boilerplate.

use rust_ai_agents_macros::Tool;

/// A simple echo tool that returns the input message
#[derive(Tool, Default)]
#[tool(name = "echo", description = "Echo back the input message")]
pub struct EchoTool {
    #[tool(param, required, description = "The message to echo back")]
    pub message: String,

    #[tool(param, description = "Number of times to repeat the message")]
    pub repeat: Option<u32>,
}

impl EchoTool {
    pub fn new() -> Self {
        Self::default()
    }

    /// The implementation method that the macro calls
    pub async fn run(
        &self,
        message: String,
        repeat: Option<u32>,
    ) -> Result<serde_json::Value, String> {
        let times = repeat.unwrap_or(1);
        let result: Vec<String> = (0..times).map(|_| message.clone()).collect();

        Ok(serde_json::json!({
            "echoed": if times == 1 { serde_json::json!(message) } else { serde_json::json!(result) },
            "repeat_count": times
        }))
    }
}

/// A greeting tool that generates personalized greetings
#[derive(Tool, Default)]
#[tool(
    name = "greet",
    description = "Generate a personalized greeting message"
)]
pub struct GreetTool {
    #[tool(param, required, description = "Name of the person to greet")]
    pub name: String,

    #[tool(param, description = "Language for the greeting (en, pt, es)")]
    pub language: Option<String>,

    #[tool(param, description = "Whether to use formal greeting")]
    pub formal: Option<bool>,
}

impl GreetTool {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn run(
        &self,
        name: String,
        language: Option<String>,
        formal: Option<bool>,
    ) -> Result<serde_json::Value, String> {
        let lang = language.unwrap_or_else(|| "en".to_string());
        let is_formal = formal.unwrap_or(false);

        let greeting = match (lang.as_str(), is_formal) {
            ("pt", true) => format!("Prezado(a) {}, é um prazer cumprimentá-lo(a).", name),
            ("pt", false) => format!("Olá, {}! Tudo bem?", name),
            ("es", true) => format!("Estimado(a) {}, es un placer saludarle.", name),
            ("es", false) => format!("¡Hola, {}! ¿Qué tal?", name),
            (_, true) => format!("Dear {}, it is a pleasure to greet you.", name),
            (_, false) => format!("Hello, {}! How are you?", name),
        };

        Ok(serde_json::json!({
            "greeting": greeting,
            "language": lang,
            "formal": is_formal
        }))
    }
}

/// A text transformation tool
#[derive(Tool, Default)]
#[tool(
    name = "transform_text",
    description = "Transform text with various operations"
)]
pub struct TransformTextTool {
    #[tool(param, required, description = "The text to transform")]
    pub text: String,

    #[tool(
        param,
        required,
        description = "Operation: uppercase, lowercase, reverse, title"
    )]
    pub operation: String,
}

impl TransformTextTool {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn run(&self, text: String, operation: String) -> Result<serde_json::Value, String> {
        let result = match operation.to_lowercase().as_str() {
            "uppercase" => text.to_uppercase(),
            "lowercase" => text.to_lowercase(),
            "reverse" => text.chars().rev().collect(),
            "title" => text
                .split_whitespace()
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => first
                            .to_uppercase()
                            .chain(chars.flat_map(|c| c.to_lowercase()))
                            .collect(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" "),
            _ => return Err(format!("Unknown operation: {}", operation)),
        };

        Ok(serde_json::json!({
            "original": text,
            "transformed": result,
            "operation": operation
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_ai_agents_core::tool::{ExecutionContext, Tool};
    use rust_ai_agents_core::types::AgentId;

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(AgentId::new("test"))
    }

    #[test]
    fn test_echo_tool_schema() {
        let tool = EchoTool::new();
        let schema = tool.schema();

        assert_eq!(schema.name, "echo");
        assert!(schema.description.contains("Echo"));
        assert!(schema.parameters["properties"]["message"].is_object());
        assert!(schema.parameters["required"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("message")));
    }

    #[tokio::test]
    async fn test_echo_tool_execute() {
        let tool = EchoTool::new();

        let result = tool
            .execute(
                &ctx(),
                serde_json::json!({
                    "message": "Hello, World!"
                }),
            )
            .await
            .unwrap();

        assert_eq!(result["echoed"], "Hello, World!");
        assert_eq!(result["repeat_count"], 1);
    }

    #[tokio::test]
    async fn test_echo_tool_repeat() {
        let tool = EchoTool::new();

        let result = tool
            .execute(
                &ctx(),
                serde_json::json!({
                    "message": "Hi",
                    "repeat": 3
                }),
            )
            .await
            .unwrap();

        assert_eq!(result["echoed"].as_array().unwrap().len(), 3);
        assert_eq!(result["repeat_count"], 3);
    }

    #[test]
    fn test_greet_tool_schema() {
        let tool = GreetTool::new();
        let schema = tool.schema();

        assert_eq!(schema.name, "greet");
        assert!(schema.parameters["properties"]["name"].is_object());
        assert!(schema.parameters["properties"]["language"].is_object());
        assert!(schema.parameters["properties"]["formal"].is_object());
    }

    #[tokio::test]
    async fn test_greet_tool_execute() {
        let tool = GreetTool::new();

        let result = tool
            .execute(
                &ctx(),
                serde_json::json!({
                    "name": "Lucas",
                    "language": "pt",
                    "formal": false
                }),
            )
            .await
            .unwrap();

        assert!(result["greeting"].as_str().unwrap().contains("Olá"));
        assert!(result["greeting"].as_str().unwrap().contains("Lucas"));
    }

    #[tokio::test]
    async fn test_transform_tool_uppercase() {
        let tool = TransformTextTool::new();

        let result = tool
            .execute(
                &ctx(),
                serde_json::json!({
                    "text": "hello world",
                    "operation": "uppercase"
                }),
            )
            .await
            .unwrap();

        assert_eq!(result["transformed"], "HELLO WORLD");
    }

    #[tokio::test]
    async fn test_transform_tool_reverse() {
        let tool = TransformTextTool::new();

        let result = tool
            .execute(
                &ctx(),
                serde_json::json!({
                    "text": "abc",
                    "operation": "reverse"
                }),
            )
            .await
            .unwrap();

        assert_eq!(result["transformed"], "cba");
    }

    #[tokio::test]
    async fn test_missing_required_param() {
        let tool = EchoTool::new();

        let result = tool.execute(&ctx(), serde_json::json!({})).await;

        assert!(result.is_err());
    }
}
