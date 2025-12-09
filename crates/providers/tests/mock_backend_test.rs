//! Tests for MockBackend LLM provider

use rust_ai_agents_core::{LLMMessage, ToolSchema};
use rust_ai_agents_providers::{LLMBackend, MockBackend, MockConfig, MockResponse};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn test_mock_backend_simple_text() {
    let backend = MockBackend::new()
        .with_response(MockResponse::text("Hello, world!"));

    let messages = vec![LLMMessage::user("Hi")];

    let result = backend
        .infer(&messages, &[], 0.7)
        .await
        .unwrap();

    assert_eq!(result.content, "Hello, world!");
    assert!(result.tool_calls.is_none());
}

#[tokio::test]
async fn test_mock_backend_tool_call() {
    let backend = MockBackend::new()
        .with_response(MockResponse::tool_call(
            "calculator",
            json!({"operation": "add", "a": 2, "b": 3})
        ));

    let messages = vec![LLMMessage::user("What is 2 + 3?")];

    let result = backend
        .infer(&messages, &[], 0.7)
        .await
        .unwrap();

    assert!(result.tool_calls.is_some());
    let tool_calls = result.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "calculator");
    assert_eq!(tool_calls[0].arguments["operation"], "add");
}

#[tokio::test]
async fn test_mock_backend_multiple_responses() {
    let backend = MockBackend::new()
        .with_response(MockResponse::text("First response"))
        .with_response(MockResponse::text("Second response"))
        .with_response(MockResponse::text("Third response"));

    let messages = vec![LLMMessage::user("Test")];

    // First call
    let result1 = backend.infer(&messages, &[], 0.7).await.unwrap();
    assert_eq!(result1.content, "First response");

    // Second call
    let result2 = backend.infer(&messages, &[], 0.7).await.unwrap();
    assert_eq!(result2.content, "Second response");

    // Third call
    let result3 = backend.infer(&messages, &[], 0.7).await.unwrap();
    assert_eq!(result3.content, "Third response");
}

#[tokio::test]
async fn test_mock_backend_echo_mode() {
    let backend = MockBackend::echo();

    let messages = vec![LLMMessage::user("Echo this message")];

    let result = backend
        .infer(&messages, &[], 0.7)
        .await
        .unwrap();

    assert!(result.content.contains("Echo this message"));
}

#[tokio::test]
async fn test_mock_backend_with_reasoning() {
    let backend = MockBackend::new()
        .with_response(
            MockResponse::text("Answer")
                .with_reasoning("First, I analyzed the question...")
        );

    let messages = vec![LLMMessage::user("Question")];

    let result = backend
        .infer(&messages, &[], 0.7)
        .await
        .unwrap();

    assert_eq!(result.content, "Answer");
    assert!(result.reasoning.is_some());
    assert_eq!(result.reasoning.unwrap(), "First, I analyzed the question...");
}

#[tokio::test]
async fn test_mock_backend_with_confidence() {
    let backend = MockBackend::new()
        .with_response(
            MockResponse::text("High confidence answer")
                .with_confidence(0.95)
        );

    let messages = vec![LLMMessage::user("Question")];

    let result = backend
        .infer(&messages, &[], 0.7)
        .await
        .unwrap();

    assert_eq!(result.confidence, 0.95);
}

#[tokio::test]
async fn test_mock_backend_error_response() {
    let backend = MockBackend::new()
        .with_response(MockResponse::error("Something went wrong"));

    let messages = vec![LLMMessage::user("Test")];

    let result = backend.infer(&messages, &[], 0.7).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_mock_backend_latency_simulation() {
    use std::time::Instant;

    let backend = MockBackend::new()
        .with_response(
            MockResponse::text("Delayed response")
                .with_latency(100) // 100ms delay
        );

    let messages = vec![LLMMessage::user("Test")];

    let start = Instant::now();
    let _result = backend.infer(&messages, &[], 0.7).await.unwrap();
    let duration = start.elapsed();

    // Should take at least 100ms due to simulated latency
    assert!(duration.as_millis() >= 90); // Allow some tolerance
}

#[tokio::test]
async fn test_mock_backend_recorded_calls() {
    let config = MockConfig {
        record_calls: true,
        ..Default::default()
    };

    let backend = MockBackend::with_config(config)
        .with_response(MockResponse::text("Response 1"))
        .with_response(MockResponse::text("Response 2"));

    let backend = Arc::new(backend);

    let messages1 = vec![LLMMessage::user("First question")];
    let messages2 = vec![LLMMessage::user("Second question")];

    backend.infer(&messages1, &[], 0.7).await.unwrap();
    backend.infer(&messages2, &[], 0.7).await.unwrap();

    let calls = backend.get_recorded_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].messages[0].content, "First question");
    assert_eq!(calls[1].messages[0].content, "Second question");
}

#[tokio::test]
async fn test_mock_backend_multiple_tool_calls() {
    let backend = MockBackend::new()
        .with_response(MockResponse::tool_calls(vec![
            ("tool1", json!({"arg1": "value1"})),
            ("tool2", json!({"arg2": "value2"})),
        ]));

    let messages = vec![LLMMessage::user("Use multiple tools")];

    let result = backend
        .infer(&messages, &[], 0.7)
        .await
        .unwrap();

    assert!(result.tool_calls.is_some());
    let tool_calls = result.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0].name, "tool1");
    assert_eq!(tool_calls[1].name, "tool2");
}

#[tokio::test]
async fn test_mock_backend_model_info() {
    let backend = MockBackend::new();
    let info = backend.model_info();

    assert_eq!(info.provider, "mock");
    assert_eq!(info.model, "mock-model");
}

#[tokio::test]
async fn test_mock_backend_with_content_and_tool_call() {
    let backend = MockBackend::new()
        .with_response(
            MockResponse::tool_call("search", json!({"query": "test"}))
                .with_content("I'll search for that")
        );

    let messages = vec![LLMMessage::user("Search for test")];

    let result = backend
        .infer(&messages, &[], 0.7)
        .await
        .unwrap();

    assert_eq!(result.content, "I'll search for that");
    assert!(result.tool_calls.is_some());
}

#[tokio::test]
async fn test_mock_response_builder() {
    let response = MockResponse::text("Test")
        .with_reasoning("Because...")
        .with_confidence(0.8)
        .with_latency(50);

    assert_eq!(response.content, "Test");
    assert_eq!(response.reasoning, Some("Because...".to_string()));
    assert_eq!(response.confidence, 0.8);
    assert_eq!(response.latency_ms, 50);
}

#[tokio::test]
async fn test_mock_backend_with_tools() {
    let backend = MockBackend::new()
        .with_response(MockResponse::tool_call("search", json!({"q": "test"})));

    let tools = vec![
        ToolSchema::new("search", "Search for information"),
        ToolSchema::new("calculator", "Perform calculations"),
    ];

    let messages = vec![LLMMessage::user("Search for test")];

    let result = backend.infer(&messages, &tools, 0.7).await.unwrap();

    assert!(result.tool_calls.is_some());
}

#[tokio::test]
async fn test_mock_backend_token_usage() {
    let backend = MockBackend::new()
        .with_response(MockResponse::text("Response"));

    let messages = vec![LLMMessage::user("Test")];

    let result = backend.infer(&messages, &[], 0.7).await.unwrap();

    // MockBackend should return reasonable token counts
    assert!(result.token_usage.prompt_tokens > 0);
    assert!(result.token_usage.completion_tokens > 0);
    assert_eq!(
        result.token_usage.total_tokens,
        result.token_usage.prompt_tokens + result.token_usage.completion_tokens
    );
}

