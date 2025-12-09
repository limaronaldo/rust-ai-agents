//! Unit tests for tool definitions

#[cfg(test)]
mod tests {
    use crate::tool::*;
    use crate::types::AgentId;
    use serde_json::json;

    #[test]
    fn test_tool_schema_creation() {
        let schema = ToolSchema::new("calculator", "Performs calculations");

        assert_eq!(schema.name, "calculator");
        assert_eq!(schema.description, "Performs calculations");
        assert_eq!(schema.dangerous, false);
        assert!(schema.metadata.is_empty());
    }

    #[test]
    fn test_tool_schema_with_parameters() {
        let params = json!({
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        });

        let schema = ToolSchema::new("add", "Add two numbers").with_parameters(params.clone());

        assert_eq!(schema.parameters, params);
    }

    #[test]
    fn test_tool_schema_dangerous_flag() {
        let schema = ToolSchema::new("delete_file", "Deletes a file").with_dangerous(true);

        assert_eq!(schema.dangerous, true);
    }

    #[test]
    fn test_tool_schema_metadata() {
        let schema = ToolSchema::new("tool", "description")
            .add_metadata("version", json!("1.0"))
            .add_metadata("author", json!("test"));

        assert_eq!(schema.metadata.len(), 2);
        assert_eq!(schema.metadata.get("version").unwrap(), &json!("1.0"));
        assert_eq!(schema.metadata.get("author").unwrap(), &json!("test"));
    }

    #[test]
    fn test_tool_schema_builder_pattern() {
        let schema = ToolSchema::new("search", "Search the web")
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }))
            .with_dangerous(false)
            .add_metadata("category", json!("search"));

        assert_eq!(schema.name, "search");
        assert_eq!(schema.dangerous, false);
        assert!(!schema.metadata.is_empty());
    }

    #[test]
    fn test_execution_context_creation() {
        let agent_id = AgentId::new("test-agent");
        let context = ExecutionContext::new(agent_id.clone());

        assert_eq!(context.agent_id, agent_id);
        assert!(context.data.is_empty());
    }

    #[test]
    fn test_execution_context_with_data() {
        let context = ExecutionContext::new(AgentId::new("agent"))
            .with_data("key1", json!("value1"))
            .with_data("key2", json!(42));

        assert_eq!(context.data.len(), 2);
        assert_eq!(context.data.get("key1").unwrap(), &json!("value1"));
        assert_eq!(context.data.get("key2").unwrap(), &json!(42));
    }

    #[test]
    fn test_execution_context_builder_data() {
        let context = ExecutionContext::new(AgentId::new("agent"))
            .with_data("runtime", json!("test"))
            .with_data("timeout", json!(30));

        assert_eq!(context.data.len(), 2);
        assert_eq!(context.data.get("runtime").unwrap(), &json!("test"));
    }

    #[test]
    fn test_tool_schema_serialization() {
        let schema = ToolSchema::new("test_tool", "A test tool")
            .with_parameters(json!({"type": "object"}))
            .with_dangerous(true);

        let json = serde_json::to_string(&schema).unwrap();
        let deserialized: ToolSchema = serde_json::from_str(&json).unwrap();

        assert_eq!(schema.name, deserialized.name);
        assert_eq!(schema.description, deserialized.description);
        assert_eq!(schema.dangerous, deserialized.dangerous);
    }

    #[test]
    fn test_tool_schema_default_parameters() {
        let schema = ToolSchema::new("simple", "Simple tool");

        // Should have default JSON Schema object structure
        assert!(schema.parameters.is_object());
        assert!(schema.parameters["type"] == "object");
    }

    #[test]
    fn test_execution_context_get_data() {
        let context = ExecutionContext::new(AgentId::new("agent"))
            .with_data("config", json!({"enabled": true}));

        let config = context.data.get("config");
        assert!(config.is_some());
        assert_eq!(config.unwrap()["enabled"], true);
    }

    #[test]
    fn test_multiple_metadata_additions() {
        let mut schema = ToolSchema::new("tool", "description");

        schema = schema
            .add_metadata("meta1", json!("value1"))
            .add_metadata("meta2", json!("value2"))
            .add_metadata("meta3", json!("value3"));

        assert_eq!(schema.metadata.len(), 3);
    }

    #[test]
    fn test_execution_context_get() {
        let context = ExecutionContext::new(AgentId::new("agent")).with_data("key", json!("value"));

        let value = context.get("key");
        assert!(value.is_some());
        assert_eq!(value.unwrap(), &json!("value"));

        let missing = context.get("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_tool_schema_clone() {
        let schema1 = ToolSchema::new("tool", "description").with_dangerous(true);

        let schema2 = schema1.clone();

        assert_eq!(schema1.name, schema2.name);
        assert_eq!(schema1.dangerous, schema2.dangerous);
    }
}
