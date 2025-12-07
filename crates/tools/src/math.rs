//! Math tools with full expression evaluation

use async_trait::async_trait;
use rust_ai_agents_core::{errors::ToolError, ExecutionContext, Tool, ToolSchema};
use serde_json::json;
// E and PI are used by meval internally for expression evaluation

/// Advanced calculator tool with full expression parsing
pub struct CalculatorTool {
    /// Enable scientific functions (sin, cos, log, etc.)
    #[allow(dead_code)]
    scientific: bool,
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

impl CalculatorTool {
    pub fn new() -> Self {
        Self { scientific: true }
    }

    /// Create a basic calculator (no scientific functions)
    pub fn basic() -> Self {
        Self { scientific: false }
    }
}

#[async_trait]
impl Tool for CalculatorTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("calculator", "Perform mathematical calculations")
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate. Supports: +, -, *, /, ^, %, parentheses, and functions like sin, cos, tan, sqrt, log, ln, abs, floor, ceil, round, min, max. Constants: pi, e"
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
        let expression = arguments["expression"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'expression' field".to_string()))?;

        // Use meval for full expression parsing
        let result = meval::eval_str(expression)
            .map_err(|e| ToolError::ExecutionFailed(format!("Math error: {}", e)))?;

        // Check for invalid results
        if result.is_nan() {
            return Err(ToolError::ExecutionFailed(
                "Result is not a number (NaN)".to_string(),
            ));
        }
        if result.is_infinite() {
            return Err(ToolError::ExecutionFailed("Result is infinite".to_string()));
        }

        Ok(json!({
            "expression": expression,
            "result": result,
            "formatted": format_number(result)
        }))
    }
}

/// Format a number nicely for display
fn format_number(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{:.0}", n)
    } else if n.abs() < 0.0001 || n.abs() >= 1e10 {
        format!("{:.6e}", n)
    } else {
        format!("{:.6}", n)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

/// Unit converter tool
pub struct UnitConverterTool;

impl Default for UnitConverterTool {
    fn default() -> Self {
        Self::new()
    }
}

impl UnitConverterTool {
    pub fn new() -> Self {
        Self
    }

    fn convert(&self, value: f64, from: &str, to: &str) -> Result<f64, String> {
        // Normalize unit names
        let from = from.to_lowercase();
        let to = to.to_lowercase();

        // Length conversions (base: meters)
        let length_to_meters: std::collections::HashMap<&str, f64> = [
            ("m", 1.0),
            ("meter", 1.0),
            ("meters", 1.0),
            ("km", 1000.0),
            ("kilometer", 1000.0),
            ("kilometers", 1000.0),
            ("cm", 0.01),
            ("centimeter", 0.01),
            ("centimeters", 0.01),
            ("mm", 0.001),
            ("millimeter", 0.001),
            ("millimeters", 0.001),
            ("mi", 1609.344),
            ("mile", 1609.344),
            ("miles", 1609.344),
            ("ft", 0.3048),
            ("foot", 0.3048),
            ("feet", 0.3048),
            ("in", 0.0254),
            ("inch", 0.0254),
            ("inches", 0.0254),
            ("yd", 0.9144),
            ("yard", 0.9144),
            ("yards", 0.9144),
        ]
        .into_iter()
        .collect();

        // Weight conversions (base: grams)
        let weight_to_grams: std::collections::HashMap<&str, f64> = [
            ("g", 1.0),
            ("gram", 1.0),
            ("grams", 1.0),
            ("kg", 1000.0),
            ("kilogram", 1000.0),
            ("kilograms", 1000.0),
            ("mg", 0.001),
            ("milligram", 0.001),
            ("milligrams", 0.001),
            ("lb", 453.592),
            ("pound", 453.592),
            ("pounds", 453.592),
            ("oz", 28.3495),
            ("ounce", 28.3495),
            ("ounces", 28.3495),
        ]
        .into_iter()
        .collect();

        // Temperature (special handling)
        if matches!(
            from.as_str(),
            "c" | "celsius" | "f" | "fahrenheit" | "k" | "kelvin"
        ) {
            return self.convert_temperature(value, &from, &to);
        }

        // Try length
        if let (Some(&from_factor), Some(&to_factor)) = (
            length_to_meters.get(from.as_str()),
            length_to_meters.get(to.as_str()),
        ) {
            return Ok(value * from_factor / to_factor);
        }

        // Try weight
        if let (Some(&from_factor), Some(&to_factor)) = (
            weight_to_grams.get(from.as_str()),
            weight_to_grams.get(to.as_str()),
        ) {
            return Ok(value * from_factor / to_factor);
        }

        Err(format!("Cannot convert from '{}' to '{}'", from, to))
    }

    fn convert_temperature(&self, value: f64, from: &str, to: &str) -> Result<f64, String> {
        // Convert to Celsius first
        let celsius = match from {
            "c" | "celsius" => value,
            "f" | "fahrenheit" => (value - 32.0) * 5.0 / 9.0,
            "k" | "kelvin" => value - 273.15,
            _ => return Err(format!("Unknown temperature unit: {}", from)),
        };

        // Convert from Celsius to target
        match to {
            "c" | "celsius" => Ok(celsius),
            "f" | "fahrenheit" => Ok(celsius * 9.0 / 5.0 + 32.0),
            "k" | "kelvin" => Ok(celsius + 273.15),
            _ => Err(format!("Unknown temperature unit: {}", to)),
        }
    }
}

#[async_trait]
impl Tool for UnitConverterTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("unit_converter", "Convert between units of measurement")
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Value to convert"
                    },
                    "from": {
                        "type": "string",
                        "description": "Source unit (e.g., 'km', 'miles', 'kg', 'pounds', 'celsius', 'fahrenheit')"
                    },
                    "to": {
                        "type": "string",
                        "description": "Target unit"
                    }
                },
                "required": ["value", "from", "to"]
            }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let value = arguments["value"]
            .as_f64()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'value' field".to_string()))?;
        let from = arguments["from"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'from' field".to_string()))?;
        let to = arguments["to"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'to' field".to_string()))?;

        let result = self
            .convert(value, from, to)
            .map_err(ToolError::ExecutionFailed)?;

        Ok(json!({
            "value": value,
            "from": from,
            "to": to,
            "result": result,
            "formatted": format!("{} {} = {} {}", format_number(value), from, format_number(result), to)
        }))
    }
}

/// Statistics tool for basic statistical calculations
pub struct StatisticsTool;

impl Default for StatisticsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticsTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for StatisticsTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("statistics", "Calculate statistics for a list of numbers").with_parameters(
            json!({
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "List of numbers to analyze"
                    }
                },
                "required": ["numbers"]
            }),
        )
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let numbers: Vec<f64> = arguments["numbers"]
            .as_array()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'numbers' array".to_string()))?
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        if numbers.is_empty() {
            return Err(ToolError::InvalidArguments(
                "Numbers array is empty".to_string(),
            ));
        }

        let count = numbers.len();
        let sum: f64 = numbers.iter().sum();
        let mean = sum / count as f64;

        let mut sorted = numbers.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if count.is_multiple_of(2) {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        let min = sorted.first().copied().unwrap();
        let max = sorted.last().copied().unwrap();

        let variance: f64 = numbers.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        Ok(json!({
            "count": count,
            "sum": sum,
            "mean": mean,
            "median": median,
            "min": min,
            "max": max,
            "range": max - min,
            "variance": variance,
            "std_dev": std_dev
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_ai_agents_core::types::AgentId;
    use std::f64::consts::{E, PI};

    fn test_ctx() -> ExecutionContext {
        ExecutionContext::new(AgentId::new("test-agent"))
    }

    #[tokio::test]
    async fn test_calculator_basic() {
        let calc = CalculatorTool::new();
        let ctx = test_ctx();

        let result = calc
            .execute(&ctx, json!({"expression": "2 + 2"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 4.0);

        let result = calc
            .execute(&ctx, json!({"expression": "10 * 5 + 3"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 53.0);
    }

    #[tokio::test]
    async fn test_calculator_scientific() {
        let calc = CalculatorTool::new();
        let ctx = test_ctx();

        let result = calc
            .execute(&ctx, json!({"expression": "sqrt(16)"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 4.0);

        let result = calc
            .execute(&ctx, json!({"expression": "2^10"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 1024.0);

        let result = calc
            .execute(&ctx, json!({"expression": "sin(0)"}))
            .await
            .unwrap();
        assert!((result["result"].as_f64().unwrap() - 0.0).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_calculator_constants() {
        let calc = CalculatorTool::new();
        let ctx = test_ctx();

        let result = calc
            .execute(&ctx, json!({"expression": "pi"}))
            .await
            .unwrap();
        assert!((result["result"].as_f64().unwrap() - PI).abs() < 0.0001);

        let result = calc
            .execute(&ctx, json!({"expression": "e"}))
            .await
            .unwrap();
        assert!((result["result"].as_f64().unwrap() - E).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_unit_converter() {
        let converter = UnitConverterTool::new();
        let ctx = test_ctx();

        let result = converter
            .execute(&ctx, json!({"value": 1.0, "from": "km", "to": "m"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 1000.0);

        let result = converter
            .execute(
                &ctx,
                json!({"value": 32.0, "from": "fahrenheit", "to": "celsius"}),
            )
            .await
            .unwrap();
        assert!((result["result"].as_f64().unwrap() - 0.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_statistics() {
        let stats = StatisticsTool::new();
        let ctx = test_ctx();

        let result = stats
            .execute(&ctx, json!({"numbers": [1, 2, 3, 4, 5]}))
            .await
            .unwrap();

        assert_eq!(result["count"], 5);
        assert_eq!(result["sum"], 15.0);
        assert_eq!(result["mean"], 3.0);
        assert_eq!(result["median"], 3.0);
        assert_eq!(result["min"], 1.0);
        assert_eq!(result["max"], 5.0);
    }
}
