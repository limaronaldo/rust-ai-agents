//! Task management and dependency resolution

use futures::future::join_all;
use std::collections::{HashMap, HashSet, VecDeque};

use rust_ai_agents_core::{errors::CrewError, Task, TaskResult};

/// Task manager with dependency resolution
pub struct TaskManager {
    tasks: HashMap<String, Task>,
    max_concurrency: usize,
}

impl TaskManager {
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            tasks: HashMap::new(),
            max_concurrency,
        }
    }

    /// Add a task
    pub fn add_task(&mut self, task: Task) -> Result<(), CrewError> {
        // Check for circular dependencies
        if self.has_circular_dependency(&task)? {
            return Err(CrewError::CircularDependency);
        }

        self.tasks.insert(task.id.clone(), task);
        Ok(())
    }

    /// Get all tasks
    pub fn get_all_tasks(&self) -> Vec<Task> {
        self.tasks.values().cloned().collect()
    }

    /// Get task count
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Execute tasks respecting dependencies
    pub async fn execute_with_dependencies<F, Fut>(
        &self,
        executor: F,
    ) -> Result<Vec<TaskResult>, CrewError>
    where
        F: Fn(Task) -> Fut + Clone + Send + 'static,
        Fut: std::future::Future<Output = Result<TaskResult, CrewError>> + Send,
    {
        let mut results = Vec::new();
        let mut completed = HashSet::new();
        let mut in_progress = HashSet::new();

        // Build dependency graph
        let dep_graph = self.build_dependency_graph();

        // Find tasks with no dependencies
        let mut ready_queue: VecDeque<String> = self
            .tasks
            .values()
            .filter(|task| task.dependencies.is_empty())
            .map(|task| task.id.clone())
            .collect();

        while !ready_queue.is_empty() || !in_progress.is_empty() {
            // Execute ready tasks in parallel (up to max_concurrency)
            let mut batch = Vec::new();

            while batch.len() < self.max_concurrency && !ready_queue.is_empty() {
                if let Some(task_id) = ready_queue.pop_front() {
                    if let Some(task) = self.tasks.get(&task_id) {
                        batch.push(task.clone());
                        in_progress.insert(task_id);
                    }
                }
            }

            if batch.is_empty() && !in_progress.is_empty() {
                // Wait a bit for in-progress tasks
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            }

            // Execute batch
            let executor_clone = executor.clone();
            let futures: Vec<_> = batch
                .iter()
                .map(|task| executor_clone(task.clone()))
                .collect();

            let batch_results = join_all(futures).await;

            // Process results
            for (idx, result) in batch_results.into_iter().enumerate() {
                let task = &batch[idx];

                match result {
                    Ok(task_result) => {
                        results.push(task_result);
                        completed.insert(task.id.clone());
                        in_progress.remove(&task.id);

                        // Check if any dependent tasks are now ready
                        for (dep_task_id, deps) in &dep_graph {
                            if deps.iter().all(|d| completed.contains(d))
                                && !completed.contains(dep_task_id)
                                && !in_progress.contains(dep_task_id)
                                && !ready_queue.contains(dep_task_id)
                            {
                                ready_queue.push_back(dep_task_id.clone());
                            }
                        }
                    }
                    Err(e) => {
                        in_progress.remove(&task.id);
                        return Err(e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Build dependency graph
    fn build_dependency_graph(&self) -> HashMap<String, Vec<String>> {
        self.tasks
            .values()
            .filter(|task| !task.dependencies.is_empty())
            .map(|task| (task.id.clone(), task.dependencies.clone()))
            .collect()
    }

    /// Check for circular dependencies
    fn has_circular_dependency(&self, new_task: &Task) -> Result<bool, CrewError> {
        let mut visited = HashSet::new();
        let mut stack = vec![new_task.id.clone()];

        while let Some(task_id) = stack.pop() {
            if visited.contains(&task_id) {
                return Ok(true); // Circular dependency detected
            }

            visited.insert(task_id.clone());

            // Get dependencies
            let deps = if task_id == new_task.id {
                &new_task.dependencies
            } else if let Some(task) = self.tasks.get(&task_id) {
                &task.dependencies
            } else {
                continue;
            };

            for dep_id in deps {
                if dep_id == &new_task.id {
                    return Ok(true); // Points back to new task
                }
                stack.push(dep_id.clone());
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_dependency_detection() {
        let mut manager = TaskManager::new(4);

        let mut task1 = Task::new("Task 1").with_dependencies(vec!["task2".to_string()]);
        task1.id = "task1".to_string();

        let mut task2 = Task::new("Task 2").with_dependencies(vec!["task1".to_string()]);
        task2.id = "task2".to_string();

        manager.add_task(task1).unwrap();
        let result = manager.add_task(task2);

        assert!(result.is_err());
    }
}
