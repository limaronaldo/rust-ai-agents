//! Unit tests for message types

#[cfg(test)]
mod tests {
    use crate::message::*;
    use crate::types::AgentId;
    use serde_json::json;

    #[test]
    fn test_content_text() {
        let content = Content::Text("Hello world".to_string());
        assert!(matches!(content, Content::Text(_)));
    }

    #[test]
    fn test_content_structured_data() {
        let data = json!({"key": "value", "count": 42});
        let content = Content::StructuredData(data.clone());

        if let Content::StructuredData(d) = content {
            assert_eq!(d["key"], "value");
            assert_eq!(d["count"], 42);
        } else {
            panic!("Expected StructuredData");
        }
    }

    #[test]
    fn test_content_image() {
        let content = Content::Image {
            data: "base64data".to_string(),
            mime_type: "image/png".to_string(),
        };

        if let Content::Image { data, mime_type } = content {
            assert_eq!(data, "base64data");
            assert_eq!(mime_type, "image/png");
        } else {
            panic!("Expected Image");
        }
    }

    #[test]
    fn test_content_error() {
        let content = Content::Error {
            code: "ERR001".to_string(),
            message: "Something went wrong".to_string(),
        };

        if let Content::Error { code, message } = content {
            assert_eq!(code, "ERR001");
            assert_eq!(message, "Something went wrong");
        } else {
            panic!("Expected Error");
        }
    }

    #[test]
    fn test_tool_call_creation() {
        let call = ToolCall::new("search", json!({"query": "rust"}));

        assert_eq!(call.name, "search");
        assert_eq!(call.arguments["query"], "rust");
        assert!(!call.id.is_empty());
        assert!(uuid::Uuid::parse_str(&call.id).is_ok());
    }

    #[test]
    fn test_tool_call_unique_ids() {
        let call1 = ToolCall::new("tool1", json!({}));
        let call2 = ToolCall::new("tool2", json!({}));

        assert_ne!(call1.id, call2.id);
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("call-123".to_string(), json!({"result": "ok"}));

        assert_eq!(result.call_id, "call-123");
        assert_eq!(result.success, true);
        assert_eq!(result.data["result"], "ok");
        assert!(result.error.is_none());
    }

    #[test]
    fn test_tool_result_failure() {
        let result = ToolResult::failure("call-456".to_string(), "Failed to execute".to_string());

        assert_eq!(result.call_id, "call-456");
        assert_eq!(result.success, false);
        assert_eq!(result.error, Some("Failed to execute".to_string()));
    }

    #[test]
    fn test_message_creation() {
        let from = AgentId::new("agent-1");
        let to = AgentId::new("agent-2");
        let content = Content::Text("Hello".to_string());

        let message = Message::new(from.clone(), to.clone(), content);

        assert_eq!(message.from, from);
        assert_eq!(message.to, to);
        assert!(matches!(message.content, Content::Text(_)));
        assert!(!message.id.is_empty());
    }

    #[test]
    fn test_message_metadata() {
        let from = AgentId::new("agent-1");
        let to = AgentId::new("agent-2");
        let content = Content::Text("Test".to_string());

        let message = Message::new(from, to, content)
            .with_metadata("priority", json!("high"))
            .with_metadata("category", json!("urgent"));

        assert_eq!(message.metadata.get("priority").unwrap(), &json!("high"));
        assert_eq!(message.metadata.get("category").unwrap(), &json!("urgent"));
        assert_eq!(message.metadata.len(), 2);
    }

    #[test]
    fn test_content_serialization() {
        let contents = vec![
            Content::Text("Hello".to_string()),
            Content::StructuredData(json!({"key": "value"})),
            Content::Error {
                code: "E001".to_string(),
                message: "Error".to_string(),
            },
        ];

        for content in contents {
            let json = serde_json::to_string(&content).unwrap();
            let deserialized: Content = serde_json::from_str(&json).unwrap();

            // Basic check that deserialization works
            match content {
                Content::Text(_) => assert!(matches!(deserialized, Content::Text(_))),
                Content::StructuredData(_) => {
                    assert!(matches!(deserialized, Content::StructuredData(_)))
                }
                Content::Error { .. } => assert!(matches!(deserialized, Content::Error { .. })),
                _ => {}
            }
        }
    }

    #[test]
    fn test_tool_call_serialization() {
        let call = ToolCall::new("calculator", json!({"op": "add", "a": 1, "b": 2}));
        let json = serde_json::to_string(&call).unwrap();
        let deserialized: ToolCall = serde_json::from_str(&json).unwrap();

        assert_eq!(call.name, deserialized.name);
        assert_eq!(call.id, deserialized.id);
        assert_eq!(call.arguments, deserialized.arguments);
    }

    #[test]
    fn test_tool_result_serialization() {
        let result = ToolResult::success("call-1".to_string(), json!({"sum": 3}));
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ToolResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.call_id, deserialized.call_id);
        assert_eq!(result.success, deserialized.success);
        assert_eq!(result.data, deserialized.data);
    }

    #[test]
    fn test_message_serialization() {
        let message = Message::new(
            AgentId::new("sender"),
            AgentId::new("receiver"),
            Content::Text("Test message".to_string()),
        );

        let json = serde_json::to_string(&message).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();

        assert_eq!(message.from, deserialized.from);
        assert_eq!(message.to, deserialized.to);
        assert_eq!(message.id, deserialized.id);
    }

    #[test]
    fn test_content_tool_calls() {
        let calls = vec![
            ToolCall::new("tool1", json!({"arg1": "value1"})),
            ToolCall::new("tool2", json!({"arg2": "value2"})),
        ];

        let content = Content::ToolCall(calls.clone());

        if let Content::ToolCall(tool_calls) = content {
            assert_eq!(tool_calls.len(), 2);
            assert_eq!(tool_calls[0].name, "tool1");
            assert_eq!(tool_calls[1].name, "tool2");
        } else {
            panic!("Expected ToolCall content");
        }
    }

    #[test]
    fn test_content_tool_results() {
        let results = vec![
            ToolResult::success("call-1".to_string(), json!({"result": 1})),
            ToolResult::failure("call-2".to_string(), "Error".to_string()),
        ];

        let content = Content::ToolResult(results.clone());

        if let Content::ToolResult(tool_results) = content {
            assert_eq!(tool_results.len(), 2);
            assert_eq!(tool_results[0].success, true);
            assert_eq!(tool_results[1].success, false);
        } else {
            panic!("Expected ToolResult content");
        }
    }
}
