//! Math tools

use async_trait::async_trait;
use rust_ai_agents_core::{Tool, ToolSchema, ExecutionContext, errors::ToolError};
use serde_json::json;

/// Calculator tool
pub struct CalculatorTool;

impl CalculatorTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for CalculatorTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "calculator",
            "Perform mathematical calculations"
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let expression = arguments["expression"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'expression' field".to_string()))?;

        // Simple evaluation using meval (you'd need to add this dependency)
        // For now, just handle basic operations
        let result = self.evaluate_simple(expression)?;

        Ok(json!({
            "expression": expression,
            "result": result
        }))
    }
}

impl CalculatorTool {
    fn evaluate_simple(&self, expr: &str) -> Result<f64, ToolError> {
        // Very basic parser - in production, use a proper math parser like meval
        let expr = expr.replace(" ", "");

        if let Some(pos) = expr.find('+') {
            let left: f64 = expr[..pos].parse()
                .map_err(|_| ToolError::InvalidArguments("Invalid number".to_string()))?;
            let right: f64 = expr[pos+1..].parse()
                .map_err(|_| ToolError::InvalidArguments("Invalid number".to_string()))?;
            return Ok(left + right);
        }

        if let Some(pos) = expr.find('-') {
            if pos > 0 { // Not a negative number
                let left: f64 = expr[..pos].parse()
                    .map_err(|_| ToolError::InvalidArguments("Invalid number".to_string()))?;
                let right: f64 = expr[pos+1..].parse()
                    .map_err(|_| ToolError::InvalidArguments("Invalid number".to_string()))?;
                return Ok(left - right);
            }
        }

        if let Some(pos) = expr.find('*') {
            let left: f64 = expr[..pos].parse()
                .map_err(|_| ToolError::InvalidArguments("Invalid number".to_string()))?;
            let right: f64 = expr[pos+1..].parse()
                .map_err(|_| ToolError::InvalidArguments("Invalid number".to_string()))?;
            return Ok(left * right);
        }

        if let Some(pos) = expr.find('/') {
            let left: f64 = expr[..pos].parse()
                .map_err(|_| ToolError::InvalidArguments("Invalid number".to_string()))?;
            let right: f64 = expr[pos+1..].parse()
                .map_err(|_| ToolError::InvalidArguments("Invalid number".to_string()))?;
            if right == 0.0 {
                return Err(ToolError::ExecutionFailed("Division by zero".to_string()));
            }
            return Ok(left / right);
        }

        // Try to parse as single number
        expr.parse()
            .map_err(|_| ToolError::InvalidArguments(
                "Invalid expression - use format like '2 + 2'".to_string()
            ))
    }
}
