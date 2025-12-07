//! Planning module for agent task execution
//!
//! This module provides planning capabilities for agents, allowing them to:
//! - Generate execution plans before acting
//! - Execute plans step by step
//! - Re-plan adaptively based on results
//!
//! # Planning Modes
//!
//! - `Disabled`: No planning, direct execution (default)
//! - `BeforeTask`: Generate a plan before each task
//! - `FullPlan`: Generate complete plan upfront, then execute all steps
//! - `Adaptive`: Re-plan after each step based on results

use rust_ai_agents_core::tool::ToolSchema;
use rust_ai_agents_core::types::{ExecutionPlan, PlanStep};
use rust_ai_agents_core::LLMMessage;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Prompt template for plan generation
const PLANNING_PROMPT: &str = r#"You are a planning agent. Your task is to create a detailed execution plan for the following goal.

GOAL: {goal}

AVAILABLE TOOLS:
{tools}

Create a step-by-step plan to achieve this goal. For each step:
1. Describe what action to take
2. Specify which tool(s) to use (if any)
3. Describe the expected result

Respond ONLY with a JSON object in this exact format:
{
  "reasoning": "Brief explanation of your approach",
  "steps": [
    {
      "step_number": 1,
      "description": "What to do in this step",
      "expected_result": "What we expect to get from this step",
      "tools": ["tool_name_1", "tool_name_2"]
    }
  ]
}

Keep the plan concise but complete. Aim for 3-7 steps maximum."#;

/// Prompt template for adaptive re-planning
const REPLAN_PROMPT: &str = r#"You are a planning agent. Review the current plan progress and decide if re-planning is needed.

ORIGINAL GOAL: {goal}

CURRENT PLAN:
{current_plan}

COMPLETED STEPS:
{completed_steps}

LAST RESULT: {last_result}

REMAINING STEPS:
{remaining_steps}

Based on the results so far, should we:
1. Continue with the current plan
2. Modify the remaining steps
3. Add new steps

Respond ONLY with a JSON object:
{
  "action": "continue" | "modify" | "add",
  "reasoning": "Why this decision",
  "modified_steps": [
    // Only if action is "modify" or "add"
    {
      "step_number": N,
      "description": "...",
      "expected_result": "...",
      "tools": []
    }
  ]
}"#;

/// Plan generator for creating execution plans from LLM responses
pub struct PlanGenerator;

impl PlanGenerator {
    /// Generate a planning prompt for the given goal and tools
    pub fn create_planning_prompt(goal: &str, tools: &[ToolSchema]) -> String {
        let tools_desc = tools
            .iter()
            .map(|t| format!("- {}: {}", t.name, t.description))
            .collect::<Vec<_>>()
            .join("\n");

        PLANNING_PROMPT
            .replace("{goal}", goal)
            .replace("{tools}", &tools_desc)
    }

    /// Create an LLM message for plan generation
    pub fn create_planning_message(goal: &str, tools: &[ToolSchema]) -> LLMMessage {
        LLMMessage::user(Self::create_planning_prompt(goal, tools))
    }

    /// Parse a plan from LLM JSON response
    pub fn parse_plan(goal: &str, response: &str) -> Result<ExecutionPlan, PlanParseError> {
        // Try to extract JSON from the response
        let json_str = extract_json(response)?;

        // Parse the JSON
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PlanParseError::InvalidJson(e.to_string()))?;

        // Extract reasoning
        let reasoning = parsed
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Extract steps
        let steps_value = parsed
            .get("steps")
            .ok_or_else(|| PlanParseError::MissingField("steps".to_string()))?;

        let steps_array = steps_value
            .as_array()
            .ok_or_else(|| PlanParseError::InvalidField("steps must be an array".to_string()))?;

        let mut steps = Vec::with_capacity(steps_array.len());

        for (idx, step_value) in steps_array.iter().enumerate() {
            let step_number = step_value
                .get("step_number")
                .and_then(|v| v.as_u64())
                .unwrap_or((idx + 1) as u64) as usize;

            let description = step_value
                .get("description")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    PlanParseError::InvalidField(format!("step {} missing description", idx + 1))
                })?
                .to_string();

            let expected_result = step_value
                .get("expected_result")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let tools = step_value
                .get("tools")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            steps.push(
                PlanStep::new(step_number, description)
                    .with_expected_result(expected_result)
                    .with_tools(tools),
            );
        }

        if steps.is_empty() {
            return Err(PlanParseError::EmptyPlan);
        }

        Ok(ExecutionPlan::new(goal)
            .with_steps(steps)
            .with_reasoning(reasoning))
    }

    /// Create a re-planning prompt
    pub fn create_replan_prompt(plan: &ExecutionPlan, last_result: &str) -> String {
        let current_plan = format!(
            "Goal: {}\nTotal steps: {}\nProgress: {:.0}%",
            plan.goal,
            plan.steps.len(),
            plan.progress() * 100.0
        );

        let completed_steps = plan
            .steps
            .iter()
            .filter(|s| s.completed)
            .map(|s| {
                format!(
                    "Step {}: {} -> {}",
                    s.step_number,
                    s.description,
                    s.actual_result.as_deref().unwrap_or("(no result)")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let remaining_steps = plan
            .steps
            .iter()
            .filter(|s| !s.completed)
            .map(|s| format!("Step {}: {}", s.step_number, s.description))
            .collect::<Vec<_>>()
            .join("\n");

        REPLAN_PROMPT
            .replace("{goal}", &plan.goal)
            .replace("{current_plan}", &current_plan)
            .replace("{completed_steps}", &completed_steps)
            .replace("{last_result}", last_result)
            .replace("{remaining_steps}", &remaining_steps)
    }

    /// Parse re-planning response and update plan if needed
    pub fn apply_replan(plan: &mut ExecutionPlan, response: &str) -> Result<bool, PlanParseError> {
        let json_str = extract_json(response)?;
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PlanParseError::InvalidJson(e.to_string()))?;

        let action = parsed
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("continue");

        match action {
            "continue" => {
                debug!("Plan continues unchanged");
                Ok(false)
            }
            "modify" | "add" => {
                if let Some(modified_steps) =
                    parsed.get("modified_steps").and_then(|v| v.as_array())
                {
                    // Remove incomplete steps
                    plan.steps.retain(|s| s.completed);

                    // Add modified/new steps
                    let base_number = plan.steps.len();
                    for (idx, step_value) in modified_steps.iter().enumerate() {
                        let description = step_value
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown step")
                            .to_string();

                        let expected_result = step_value
                            .get("expected_result")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        let tools = step_value
                            .get("tools")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();

                        plan.steps.push(
                            PlanStep::new(base_number + idx + 1, description)
                                .with_expected_result(expected_result)
                                .with_tools(tools),
                        );
                    }

                    info!("Plan modified: now {} steps", plan.steps.len());
                    Ok(true)
                } else {
                    warn!("Replan response missing modified_steps");
                    Ok(false)
                }
            }
            _ => {
                warn!("Unknown replan action: {}", action);
                Ok(false)
            }
        }
    }
}

/// Errors that can occur during plan parsing
#[derive(Debug, Error)]
pub enum PlanParseError {
    #[error("No JSON found in response")]
    NoJson,
    #[error("Invalid JSON: {0}")]
    InvalidJson(String),
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Invalid field: {0}")]
    InvalidField(String),
    #[error("Plan has no steps")]
    EmptyPlan,
}

/// Extract JSON from a response that might contain markdown code blocks or other text
fn extract_json(response: &str) -> Result<String, PlanParseError> {
    // First, try to find JSON in code blocks
    if let Some(start) = response.find("```json") {
        let content_start = start + 7;
        if let Some(end) = response[content_start..].find("```") {
            return Ok(response[content_start..content_start + end]
                .trim()
                .to_string());
        }
    }

    // Try generic code block
    if let Some(start) = response.find("```") {
        let content_start = start + 3;
        // Skip language identifier if present
        let content_start = response[content_start..]
            .find('\n')
            .map(|n| content_start + n + 1)
            .unwrap_or(content_start);

        if let Some(end) = response[content_start..].find("```") {
            return Ok(response[content_start..content_start + end]
                .trim()
                .to_string());
        }
    }

    // Try to find raw JSON object
    if let Some(start) = response.find('{') {
        // Find matching closing brace
        let mut depth = 0;
        let mut end = start;
        for (i, c) in response[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 && end > start {
            return Ok(response[start..end].to_string());
        }
    }

    Err(PlanParseError::NoJson)
}

/// Step execution context for tracking progress
#[derive(Debug, Clone)]
pub struct StepExecutionContext {
    /// Current step being executed
    pub step: PlanStep,
    /// Prompt to send to the agent for this step
    pub prompt: String,
}

impl StepExecutionContext {
    /// Create execution context for a plan step
    pub fn from_step(step: &PlanStep, plan: &ExecutionPlan) -> Self {
        let mut prompt = format!(
            "Execute step {} of the plan to: {}\n\n",
            step.step_number, plan.goal
        );

        prompt.push_str(&format!("CURRENT STEP: {}\n", step.description));

        if !step.expected_result.is_empty() {
            prompt.push_str(&format!("EXPECTED RESULT: {}\n", step.expected_result));
        }

        if !step.tools.is_empty() {
            prompt.push_str(&format!("SUGGESTED TOOLS: {}\n", step.tools.join(", ")));
        }

        // Add context from previous steps
        let completed: Vec<_> = plan.steps.iter().filter(|s| s.completed).collect();
        if !completed.is_empty() {
            prompt.push_str("\nPREVIOUS RESULTS:\n");
            for prev in completed {
                if let Some(result) = &prev.actual_result {
                    prompt.push_str(&format!("- Step {}: {}\n", prev.step_number, result));
                }
            }
        }

        prompt.push_str("\nExecute this step and provide the result.");

        Self {
            step: step.clone(),
            prompt,
        }
    }
}

/// Check if any stop words are present in text
pub fn check_stop_words(text: &str, stop_words: &[String]) -> Option<String> {
    let text_lower = text.to_lowercase();
    for word in stop_words {
        if text_lower.contains(&word.to_lowercase()) {
            return Some(word.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_code_block() {
        let response = r#"Here's the plan:
```json
{"steps": [{"step_number": 1, "description": "Test"}]}
```
"#;
        let json = extract_json(response).unwrap();
        assert!(json.contains("steps"));
    }

    #[test]
    fn test_extract_json_raw() {
        let response = r#"{"steps": [{"step_number": 1, "description": "Test"}]}"#;
        let json = extract_json(response).unwrap();
        assert!(json.contains("steps"));
    }

    #[test]
    fn test_parse_plan() {
        let response = r#"{"reasoning": "Simple test", "steps": [
            {"step_number": 1, "description": "First step", "expected_result": "Done", "tools": ["tool1"]}
        ]}"#;

        let plan = PlanGenerator::parse_plan("Test goal", response).unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].description, "First step");
        assert_eq!(plan.steps[0].tools, vec!["tool1"]);
    }

    #[test]
    fn test_check_stop_words() {
        let stop_words = vec!["DONE".to_string(), "FINISHED".to_string()];

        assert!(check_stop_words("Task is DONE", &stop_words).is_some());
        assert!(check_stop_words("Task finished successfully", &stop_words).is_some());
        assert!(check_stop_words("Still working", &stop_words).is_none());
    }
}
