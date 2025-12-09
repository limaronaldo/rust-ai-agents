//! Integration tests for dashboard state and types

use chrono::Utc;
use rust_ai_agents_dashboard::{AgentStatus, Session, SessionStatus, TraceEntry, TraceEntryType};
use serde_json::json;

#[test]
fn test_agent_status_creation() {
    let agent_status = AgentStatus {
        id: "agent-1".to_string(),
        name: "Test Agent".to_string(),
        role: "Executor".to_string(),
        status: "active".to_string(),
        messages_processed: 10,
        last_activity: Some(Utc::now()),
        current_task: Some("Processing data".to_string()),
    };

    assert_eq!(agent_status.id, "agent-1");
    assert_eq!(agent_status.name, "Test Agent");
    assert_eq!(agent_status.messages_processed, 10);
}

#[test]
fn test_agent_status_serialization() {
    let agent_status = AgentStatus {
        id: "agent-1".to_string(),
        name: "Test Agent".to_string(),
        role: "Executor".to_string(),
        status: "active".to_string(),
        messages_processed: 5,
        last_activity: None,
        current_task: None,
    };

    let json = serde_json::to_string(&agent_status).unwrap();
    let deserialized: AgentStatus = serde_json::from_str(&json).unwrap();

    assert_eq!(agent_status.id, deserialized.id);
    assert_eq!(agent_status.name, deserialized.name);
    assert_eq!(
        agent_status.messages_processed,
        deserialized.messages_processed
    );
}

#[test]
fn test_session_creation() {
    let session = Session {
        id: "session-1".to_string(),
        name: Some("Test Session".to_string()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        message_count: 3,
        status: SessionStatus::Active,
        agent_id: Some("agent-1".to_string()),
        metadata: Some(json!({"key": "value"})),
    };

    assert_eq!(session.id, "session-1");
    assert_eq!(session.name, Some("Test Session".to_string()));
    assert_eq!(session.message_count, 3);
}

#[test]
fn test_session_status_variants() {
    let statuses = vec![
        SessionStatus::Active,
        SessionStatus::Completed,
        SessionStatus::Failed,
        SessionStatus::Archived,
    ];

    assert_eq!(statuses.len(), 4);
}

#[test]
fn test_session_serialization() {
    let session = Session {
        id: "session-1".to_string(),
        name: Some("Test".to_string()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        message_count: 0,
        status: SessionStatus::Active,
        agent_id: None,
        metadata: None,
    };

    let json = serde_json::to_string(&session).unwrap();
    let deserialized: Session = serde_json::from_str(&json).unwrap();

    assert_eq!(session.id, deserialized.id);
    assert_eq!(session.message_count, deserialized.message_count);
}

#[test]
fn test_trace_entry_agent_thought() {
    let trace = TraceEntry {
        id: "trace-1".to_string(),
        session_id: "session-1".to_string(),
        timestamp: Utc::now(),
        entry_type: TraceEntryType::AgentThought {
            thought: "Analyzing the input data".to_string(),
        },
        duration_ms: Some(100.0),
        metadata: None,
    };

    assert_eq!(trace.id, "trace-1");
    assert_eq!(trace.session_id, "session-1");
    assert!(matches!(
        trace.entry_type,
        TraceEntryType::AgentThought { .. }
    ));
}

#[test]
fn test_trace_entry_llm_request() {
    let trace = TraceEntry {
        id: "trace-2".to_string(),
        session_id: "session-1".to_string(),
        timestamp: Utc::now(),
        entry_type: TraceEntryType::LlmRequest {
            model: "gpt-4".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            cost: 0.015,
        },
        duration_ms: Some(500.0),
        metadata: Some(json!({"provider": "openai"})),
    };

    if let TraceEntryType::LlmRequest {
        model,
        prompt_tokens,
        completion_tokens,
        cost,
    } = trace.entry_type
    {
        assert_eq!(model, "gpt-4");
        assert_eq!(prompt_tokens, 100);
        assert_eq!(completion_tokens, 50);
        assert_eq!(cost, 0.015);
    } else {
        panic!("Expected LlmRequest trace type");
    }
}

#[test]
fn test_trace_entry_tool_call() {
    let trace = TraceEntry {
        id: "trace-3".to_string(),
        session_id: "session-1".to_string(),
        timestamp: Utc::now(),
        entry_type: TraceEntryType::ToolCall {
            tool_name: "calculator".to_string(),
            arguments: json!({"operation": "add", "a": 2, "b": 3}),
        },
        duration_ms: Some(50.0),
        metadata: None,
    };

    if let TraceEntryType::ToolCall {
        tool_name,
        arguments,
    } = trace.entry_type
    {
        assert_eq!(tool_name, "calculator");
        assert_eq!(arguments["operation"], "add");
        assert_eq!(arguments["a"], 2);
    } else {
        panic!("Expected ToolCall trace type");
    }
}

#[test]
fn test_trace_entry_tool_result() {
    let trace = TraceEntry {
        id: "trace-4".to_string(),
        session_id: "session-1".to_string(),
        timestamp: Utc::now(),
        entry_type: TraceEntryType::ToolResult {
            tool_name: "calculator".to_string(),
            result: json!({"sum": 5}),
            success: true,
        },
        duration_ms: Some(25.0),
        metadata: None,
    };

    if let TraceEntryType::ToolResult {
        tool_name,
        result,
        success,
    } = trace.entry_type
    {
        assert_eq!(tool_name, "calculator");
        assert_eq!(result["sum"], 5);
        assert_eq!(success, true);
    } else {
        panic!("Expected ToolResult trace type");
    }
}

#[test]
fn test_trace_entry_error() {
    let trace = TraceEntry {
        id: "trace-5".to_string(),
        session_id: "session-1".to_string(),
        timestamp: Utc::now(),
        entry_type: TraceEntryType::Error {
            message: "Failed to execute tool".to_string(),
            error_type: "ToolExecutionError".to_string(),
        },
        duration_ms: None,
        metadata: None,
    };

    if let TraceEntryType::Error {
        message,
        error_type,
    } = trace.entry_type
    {
        assert_eq!(message, "Failed to execute tool");
        assert_eq!(error_type, "ToolExecutionError");
    } else {
        panic!("Expected Error trace type");
    }
}

#[test]
fn test_trace_entry_serialization() {
    let trace = TraceEntry {
        id: "trace-1".to_string(),
        session_id: "session-1".to_string(),
        timestamp: Utc::now(),
        entry_type: TraceEntryType::AgentThought {
            thought: "Test thought".to_string(),
        },
        duration_ms: Some(100.0),
        metadata: None,
    };

    let json = serde_json::to_string(&trace).unwrap();
    let deserialized: TraceEntry = serde_json::from_str(&json).unwrap();

    assert_eq!(trace.id, deserialized.id);
    assert_eq!(trace.session_id, deserialized.session_id);
}
