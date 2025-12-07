//! YAML workflow parser with validation

use std::collections::{HashMap, HashSet};
use std::path::Path;
use tracing::{debug, info};

use crate::error::WorkflowError;
use crate::schema::WorkflowDefinition;

/// Parser for YAML workflow files
pub struct WorkflowParser;

impl WorkflowParser {
    /// Parse workflow from YAML string
    pub fn parse(yaml: &str) -> Result<WorkflowDefinition, WorkflowError> {
        let workflow: WorkflowDefinition = serde_yaml::from_str(yaml)?;
        Self::validate(&workflow)?;
        Ok(workflow)
    }

    /// Parse workflow from file
    pub fn parse_file(path: impl AsRef<Path>) -> Result<WorkflowDefinition, WorkflowError> {
        let content = std::fs::read_to_string(path)?;
        Self::parse(&content)
    }

    /// Validate workflow definition
    pub fn validate(workflow: &WorkflowDefinition) -> Result<(), WorkflowError> {
        info!(name = %workflow.name, "Validating workflow");

        // Check workflow has a name
        if workflow.name.is_empty() {
            return Err(WorkflowError::ValidationError(
                "Workflow name is required".to_string(),
            ));
        }

        // Check agent IDs are unique
        let mut agent_ids = HashSet::new();
        for agent in &workflow.agents {
            if !agent_ids.insert(&agent.id) {
                return Err(WorkflowError::ValidationError(format!(
                    "Duplicate agent ID: {}",
                    agent.id
                )));
            }
        }

        // Check task IDs are unique
        let mut task_ids = HashSet::new();
        for task in &workflow.tasks {
            if !task_ids.insert(&task.id) {
                return Err(WorkflowError::ValidationError(format!(
                    "Duplicate task ID: {}",
                    task.id
                )));
            }
        }

        // Check task agents exist
        for task in &workflow.tasks {
            if !agent_ids.contains(&task.agent) {
                return Err(WorkflowError::AgentNotFound(format!(
                    "Task '{}' references unknown agent '{}'",
                    task.id, task.agent
                )));
            }
        }

        // Check task dependencies exist
        for task in &workflow.tasks {
            for dep in &task.depends_on {
                if !task_ids.contains(dep) {
                    return Err(WorkflowError::TaskNotFound(format!(
                        "Task '{}' depends on unknown task '{}'",
                        task.id, dep
                    )));
                }
            }
        }

        // Check for circular dependencies
        Self::check_circular_dependencies(workflow)?;

        debug!("Workflow validation passed");
        Ok(())
    }

    /// Check for circular dependencies in tasks
    fn check_circular_dependencies(workflow: &WorkflowDefinition) -> Result<(), WorkflowError> {
        // Build dependency graph
        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
        for task in &workflow.tasks {
            graph.insert(
                &task.id,
                task.depends_on.iter().map(|s| s.as_str()).collect(),
            );
        }

        // DFS to detect cycles
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for task in &workflow.tasks {
            if Self::has_cycle(&task.id, &graph, &mut visited, &mut rec_stack) {
                return Err(WorkflowError::CircularDependency(format!(
                    "Circular dependency detected involving task '{}'",
                    task.id
                )));
            }
        }

        Ok(())
    }

    fn has_cycle<'a>(
        node: &'a str,
        graph: &HashMap<&'a str, Vec<&'a str>>,
        visited: &mut HashSet<&'a str>,
        rec_stack: &mut HashSet<&'a str>,
    ) -> bool {
        if rec_stack.contains(node) {
            return true;
        }
        if visited.contains(node) {
            return false;
        }

        visited.insert(node);
        rec_stack.insert(node);

        if let Some(deps) = graph.get(node) {
            for dep in deps {
                if Self::has_cycle(dep, graph, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }

    /// Get execution order for tasks (topological sort)
    pub fn get_execution_order(
        workflow: &WorkflowDefinition,
    ) -> Result<Vec<String>, WorkflowError> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();

        // Initialize
        for task in &workflow.tasks {
            in_degree.entry(&task.id).or_insert(0);
            graph.entry(&task.id).or_default();
        }

        // Build graph (reverse direction: dependency -> dependent)
        for task in &workflow.tasks {
            for dep in &task.depends_on {
                graph
                    .entry(dep.as_str())
                    .or_default()
                    .push(&task.id);
                *in_degree.entry(&task.id).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node.to_string());

            if let Some(dependents) = graph.get(node) {
                for &dep in dependents {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(dep);
                        }
                    }
                }
            }
        }

        if result.len() != workflow.tasks.len() {
            return Err(WorkflowError::CircularDependency(
                "Could not determine execution order due to circular dependencies".to_string(),
            ));
        }

        Ok(result)
    }

    /// Interpolate variables in a string
    /// Replaces {{variable}} with values from context
    pub fn interpolate(
        template: &str,
        context: &HashMap<String, String>,
    ) -> Result<String, WorkflowError> {
        let mut result = template.to_string();

        // Find all {{variable}} patterns
        let re = regex_lite::Regex::new(r"\{\{([^}]+)\}\}").unwrap();

        for cap in re.captures_iter(template) {
            let full_match = cap.get(0).unwrap().as_str();
            let var_name = cap.get(1).unwrap().as_str().trim();

            // Handle nested access like task.output
            let value = if var_name.contains('.') {
                let parts: Vec<&str> = var_name.split('.').collect();
                let key = format!("{}.{}", parts[0], parts[1]);
                context.get(&key).cloned()
            } else {
                context.get(var_name).cloned()
            };

            match value {
                Some(v) => {
                    result = result.replace(full_match, &v);
                }
                None => {
                    return Err(WorkflowError::VariableNotFound(var_name.to_string()));
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_workflow() {
        let yaml = r#"
name: test
agents:
  - id: a1
    name: Agent 1
tasks:
  - id: t1
    agent: a1
    description: Task 1
"#;
        let result = WorkflowParser::parse(yaml);
        assert!(result.is_ok());
    }

    #[test]
    fn test_duplicate_agent_id() {
        let yaml = r#"
name: test
agents:
  - id: a1
    name: Agent 1
  - id: a1
    name: Agent 2
tasks: []
"#;
        let result = WorkflowParser::parse(yaml);
        assert!(matches!(result, Err(WorkflowError::ValidationError(_))));
    }

    #[test]
    fn test_unknown_agent() {
        let yaml = r#"
name: test
agents:
  - id: a1
    name: Agent 1
tasks:
  - id: t1
    agent: unknown
    description: Task 1
"#;
        let result = WorkflowParser::parse(yaml);
        assert!(matches!(result, Err(WorkflowError::AgentNotFound(_))));
    }

    #[test]
    fn test_circular_dependency() {
        let yaml = r#"
name: test
agents:
  - id: a1
    name: Agent 1
tasks:
  - id: t1
    agent: a1
    description: Task 1
    depends_on: [t2]
  - id: t2
    agent: a1
    description: Task 2
    depends_on: [t1]
"#;
        let result = WorkflowParser::parse(yaml);
        assert!(matches!(result, Err(WorkflowError::CircularDependency(_))));
    }

    #[test]
    fn test_execution_order() {
        let yaml = r#"
name: test
agents:
  - id: a1
    name: Agent 1
tasks:
  - id: t3
    agent: a1
    description: Task 3
    depends_on: [t1, t2]
  - id: t1
    agent: a1
    description: Task 1
  - id: t2
    agent: a1
    description: Task 2
    depends_on: [t1]
"#;
        let workflow = WorkflowParser::parse(yaml).unwrap();
        let order = WorkflowParser::get_execution_order(&workflow).unwrap();

        // t1 must come before t2 and t3
        let t1_pos = order.iter().position(|x| x == "t1").unwrap();
        let t2_pos = order.iter().position(|x| x == "t2").unwrap();
        let t3_pos = order.iter().position(|x| x == "t3").unwrap();

        assert!(t1_pos < t2_pos);
        assert!(t1_pos < t3_pos);
        assert!(t2_pos < t3_pos);
    }

    #[test]
    fn test_interpolate() {
        let mut context = HashMap::new();
        context.insert("name".to_string(), "World".to_string());
        context.insert("research.output".to_string(), "Some data".to_string());

        let result = WorkflowParser::interpolate("Hello {{name}}!", &context).unwrap();
        assert_eq!(result, "Hello World!");

        let result = WorkflowParser::interpolate("Data: {{research.output}}", &context).unwrap();
        assert_eq!(result, "Data: Some data");
    }

    #[test]
    fn test_interpolate_missing_variable() {
        let context = HashMap::new();
        let result = WorkflowParser::interpolate("Hello {{name}}!", &context);
        assert!(matches!(result, Err(WorkflowError::VariableNotFound(_))));
    }
}
