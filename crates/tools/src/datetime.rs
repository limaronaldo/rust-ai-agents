//! DateTime tools for time operations

use async_trait::async_trait;
use chrono::{DateTime, Duration, Local, NaiveDate, NaiveDateTime, TimeZone, Utc};
use rust_ai_agents_core::{errors::ToolError, ExecutionContext, Tool, ToolSchema};
use serde_json::json;

/// Get current date and time
pub struct CurrentTimeTool;

impl Default for CurrentTimeTool {
    fn default() -> Self {
        Self::new()
    }
}

impl CurrentTimeTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for CurrentTimeTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("current_time", "Get the current date and time").with_parameters(json!({
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone (e.g., 'UTC', 'local'). Default: 'local'"
                },
                "format": {
                    "type": "string",
                    "description": "Output format. Options: 'iso', 'rfc2822', 'unix', 'custom'. Default: 'iso'"
                },
                "custom_format": {
                    "type": "string",
                    "description": "Custom strftime format string (only used when format='custom')"
                }
            }
        }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let timezone = arguments["timezone"].as_str().unwrap_or("local");
        let format = arguments["format"].as_str().unwrap_or("iso");

        let now_utc = Utc::now();
        let now_local = Local::now();

        let (datetime_str, timestamp) = match timezone.to_lowercase().as_str() {
            "utc" => {
                let formatted =
                    format_datetime(&now_utc, format, arguments["custom_format"].as_str())?;
                (formatted, now_utc.timestamp())
            }
            "local" | _ => {
                let formatted =
                    format_datetime(&now_local, format, arguments["custom_format"].as_str())?;
                (formatted, now_local.timestamp())
            }
        };

        Ok(json!({
            "datetime": datetime_str,
            "timestamp": timestamp,
            "timezone": timezone,
            "utc": now_utc.to_rfc3339(),
            "local": now_local.to_rfc3339(),
            "components": {
                "year": now_local.format("%Y").to_string(),
                "month": now_local.format("%m").to_string(),
                "day": now_local.format("%d").to_string(),
                "hour": now_local.format("%H").to_string(),
                "minute": now_local.format("%M").to_string(),
                "second": now_local.format("%S").to_string(),
                "weekday": now_local.format("%A").to_string(),
                "week_number": now_local.format("%W").to_string()
            }
        }))
    }
}

fn format_datetime<Tz: TimeZone>(
    dt: &DateTime<Tz>,
    format: &str,
    custom_format: Option<&str>,
) -> Result<String, ToolError>
where
    Tz::Offset: std::fmt::Display,
{
    match format {
        "iso" => Ok(dt.to_rfc3339()),
        "rfc2822" => Ok(dt.to_rfc2822()),
        "unix" => Ok(dt.timestamp().to_string()),
        "custom" => {
            let fmt = custom_format.ok_or_else(|| {
                ToolError::InvalidArguments(
                    "custom_format required when format='custom'".to_string(),
                )
            })?;
            Ok(dt.format(fmt).to_string())
        }
        _ => Ok(dt.to_rfc3339()),
    }
}

/// Parse and format dates
pub struct DateParserTool;

impl Default for DateParserTool {
    fn default() -> Self {
        Self::new()
    }
}

impl DateParserTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for DateParserTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("parse_date", "Parse a date string into components")
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date string to parse (supports ISO 8601, RFC 2822, common formats)"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Desired output format (strftime format string)"
                    }
                },
                "required": ["date"]
            }))
    }

    async fn execute(
        &self,
        _context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let date_str = arguments["date"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'date' field".to_string()))?;

        // Try different parsing strategies
        let parsed = parse_flexible_date(date_str)?;

        let output_format = arguments["output_format"].as_str();
        let formatted = if let Some(fmt) = output_format {
            parsed.format(fmt).to_string()
        } else {
            parsed.to_rfc3339()
        };

        Ok(json!({
            "input": date_str,
            "parsed": parsed.to_rfc3339(),
            "formatted": formatted,
            "timestamp": parsed.timestamp(),
            "components": {
                "year": parsed.format("%Y").to_string().parse::<i32>().unwrap_or(0),
                "month": parsed.format("%m").to_string().parse::<u32>().unwrap_or(0),
                "day": parsed.format("%d").to_string().parse::<u32>().unwrap_or(0),
                "hour": parsed.format("%H").to_string().parse::<u32>().unwrap_or(0),
                "minute": parsed.format("%M").to_string().parse::<u32>().unwrap_or(0),
                "second": parsed.format("%S").to_string().parse::<u32>().unwrap_or(0),
                "weekday": parsed.format("%A").to_string(),
                "day_of_year": parsed.format("%j").to_string().parse::<u32>().unwrap_or(0)
            }
        }))
    }
}

fn parse_flexible_date(s: &str) -> Result<DateTime<Utc>, ToolError> {
    // Try RFC 3339 / ISO 8601
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }

    // Try RFC 2822
    if let Ok(dt) = DateTime::parse_from_rfc2822(s) {
        return Ok(dt.with_timezone(&Utc));
    }

    // Try common formats
    let formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%Y%m%d",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ];

    for fmt in &formats {
        if let Ok(naive) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(Utc.from_utc_datetime(&naive));
        }
        if let Ok(naive) = NaiveDate::parse_from_str(s, fmt) {
            return Ok(Utc.from_utc_datetime(&naive.and_hms_opt(0, 0, 0).unwrap()));
        }
    }

    // Try unix timestamp
    if let Ok(ts) = s.parse::<i64>() {
        if let Some(dt) = DateTime::from_timestamp(ts, 0) {
            return Ok(dt);
        }
    }

    Err(ToolError::InvalidArguments(format!(
        "Could not parse date: '{}'",
        s
    )))
}

/// Calculate date differences and offsets
pub struct DateCalculatorTool;

impl Default for DateCalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

impl DateCalculatorTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for DateCalculatorTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "date_calculator",
            "Calculate date differences or add/subtract time",
        )
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["diff", "add", "subtract"],
                    "description": "Operation to perform"
                },
                "date1": {
                    "type": "string",
                    "description": "First date (or base date for add/subtract)"
                },
                "date2": {
                    "type": "string",
                    "description": "Second date (for diff operation)"
                },
                "amount": {
                    "type": "integer",
                    "description": "Amount to add/subtract"
                },
                "unit": {
                    "type": "string",
                    "enum": ["days", "hours", "minutes", "seconds", "weeks"],
                    "description": "Unit for add/subtract operation"
                }
            },
            "required": ["operation", "date1"]
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
        let date1_str = arguments["date1"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'date1'".to_string()))?;

        let date1 = parse_flexible_date(date1_str)?;

        match operation {
            "diff" => {
                let date2_str = arguments["date2"].as_str().ok_or_else(|| {
                    ToolError::InvalidArguments("Missing 'date2' for diff".to_string())
                })?;
                let date2 = parse_flexible_date(date2_str)?;

                let diff = date2.signed_duration_since(date1);

                Ok(json!({
                    "date1": date1.to_rfc3339(),
                    "date2": date2.to_rfc3339(),
                    "difference": {
                        "total_seconds": diff.num_seconds(),
                        "total_minutes": diff.num_minutes(),
                        "total_hours": diff.num_hours(),
                        "total_days": diff.num_days(),
                        "total_weeks": diff.num_weeks(),
                        "human_readable": format_duration(diff)
                    }
                }))
            }
            "add" | "subtract" => {
                let amount = arguments["amount"]
                    .as_i64()
                    .ok_or_else(|| ToolError::InvalidArguments("Missing 'amount'".to_string()))?;
                let unit = arguments["unit"]
                    .as_str()
                    .ok_or_else(|| ToolError::InvalidArguments("Missing 'unit'".to_string()))?;

                let duration = match unit {
                    "seconds" => Duration::seconds(amount),
                    "minutes" => Duration::minutes(amount),
                    "hours" => Duration::hours(amount),
                    "days" => Duration::days(amount),
                    "weeks" => Duration::weeks(amount),
                    _ => {
                        return Err(ToolError::InvalidArguments(format!(
                            "Unknown unit: {}",
                            unit
                        )))
                    }
                };

                let result = if operation == "add" {
                    date1 + duration
                } else {
                    date1 - duration
                };

                Ok(json!({
                    "original": date1.to_rfc3339(),
                    "operation": operation,
                    "amount": amount,
                    "unit": unit,
                    "result": result.to_rfc3339(),
                    "timestamp": result.timestamp()
                }))
            }
            _ => Err(ToolError::InvalidArguments(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }
}

fn format_duration(dur: Duration) -> String {
    let total_seconds = dur.num_seconds().abs();
    let days = total_seconds / 86400;
    let hours = (total_seconds % 86400) / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    let sign = if dur.num_seconds() < 0 { "-" } else { "" };

    if days > 0 {
        format!("{}{}d {}h {}m {}s", sign, days, hours, minutes, seconds)
    } else if hours > 0 {
        format!("{}{}h {}m {}s", sign, hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}{}m {}s", sign, minutes, seconds)
    } else {
        format!("{}{}s", sign, seconds)
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
    async fn test_current_time() {
        let tool = CurrentTimeTool::new();
        let ctx = test_ctx();

        let result = tool.execute(&ctx, json!({})).await.unwrap();
        assert!(result["datetime"].as_str().is_some());
        assert!(result["timestamp"].as_i64().is_some());
    }

    #[tokio::test]
    async fn test_parse_date_iso() {
        let tool = DateParserTool::new();
        let ctx = test_ctx();

        let result = tool
            .execute(&ctx, json!({"date": "2024-01-15T10:30:00Z"}))
            .await
            .unwrap();

        assert_eq!(result["components"]["year"], 2024);
        assert_eq!(result["components"]["month"], 1);
        assert_eq!(result["components"]["day"], 15);
    }

    #[tokio::test]
    async fn test_parse_date_formats() {
        let tool = DateParserTool::new();
        let ctx = test_ctx();

        // YYYY-MM-DD
        let result = tool
            .execute(&ctx, json!({"date": "2024-06-20"}))
            .await
            .unwrap();
        assert_eq!(result["components"]["year"], 2024);
        assert_eq!(result["components"]["month"], 6);

        // Unix timestamp
        let result = tool
            .execute(&ctx, json!({"date": "1700000000"}))
            .await
            .unwrap();
        assert!(result["components"]["year"].as_i64().unwrap() >= 2023);
    }

    #[tokio::test]
    async fn test_date_calculator_diff() {
        let tool = DateCalculatorTool::new();
        let ctx = test_ctx();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "operation": "diff",
                    "date1": "2024-01-01",
                    "date2": "2024-01-08"
                }),
            )
            .await
            .unwrap();

        assert_eq!(result["difference"]["total_days"], 7);
        assert_eq!(result["difference"]["total_weeks"], 1);
    }

    #[tokio::test]
    async fn test_date_calculator_add() {
        let tool = DateCalculatorTool::new();
        let ctx = test_ctx();

        let result = tool
            .execute(
                &ctx,
                json!({
                    "operation": "add",
                    "date1": "2024-01-01T00:00:00Z",
                    "amount": 7,
                    "unit": "days"
                }),
            )
            .await
            .unwrap();

        assert!(result["result"].as_str().unwrap().contains("2024-01-08"));
    }
}
