//! Tool/function definitions for agent capabilities

use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::types::AgentId;
use crate::errors::ToolError;

/// Tool/function schema for LLM function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name
    pub name: String,

    /// Tool description
    pub description: String,

    /// Input parameters schema (JSON Schema)
    pub parameters: serde_json::Value,

    /// Whether tool is dangerous and requires confirmation
    pub dangerous: bool,

    /// Tool metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ToolSchema {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            dangerous: false,
            metadata: HashMap::new(),
        }
    }

    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn with_dangerous(mut self, dangerous: bool) -> Self {
        self.dangerous = dangerous;
        self
    }

    pub fn add_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Execution context for tool calls
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Agent making the tool call
    pub agent_id: AgentId,

    /// Additional context data
    pub data: HashMap<String, serde_json::Value>,
}

impl ExecutionContext {
    pub fn new(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            data: HashMap::new(),
        }
    }

    pub fn with_data(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.data.insert(key.into(), value);
        self
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }
}

/// Tool/function trait
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get tool schema
    fn schema(&self) -> ToolSchema;

    /// Execute the tool
    async fn execute(
        &self,
        context: &ExecutionContext,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError>;

    /// Validate arguments before execution
    fn validate(&self, _arguments: &serde_json::Value) -> Result<(), ToolError> {
        Ok(())
    }
}

/// Tool registry
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Arc<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: Arc::new(HashMap::new()),
        }
    }

    /// Register a tool
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let schema = tool.schema();
        Arc::get_mut(&mut self.tools)
            .expect("Cannot register tools after cloning")
            .insert(schema.name.clone(), tool);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// List all tool schemas
    pub fn list_schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|tool| tool.schema()).collect()
    }

    /// Check if tool exists
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get tool count
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro to easily define tools
#[macro_export]
macro_rules! define_tool {
    (
        $name:ident,
        schema: $schema:expr,
        execute: |$ctx:ident, $args:ident| $body:expr
    ) => {
        pub struct $name {
            schema: $crate::tool::ToolSchema,
        }

        impl $name {
            pub fn new() -> Self {
                Self { schema: $schema }
            }
        }

        #[async_trait::async_trait]
        impl $crate::tool::Tool for $name {
            fn schema(&self) -> $crate::tool::ToolSchema {
                self.schema.clone()
            }

            async fn execute(
                &self,
                $ctx: &$crate::tool::ExecutionContext,
                $args: serde_json::Value,
            ) -> Result<serde_json::Value, $crate::errors::ToolError> {
                $body.await
            }
        }
    };
}
