# Automated Testing Documentation

## Overview

This document describes the automated testing infrastructure for the Rust AI Agents project.

## Test Coverage Summary

The project now includes comprehensive test coverage across multiple layers:

### Unit Tests (Core Library)
- **Location**: `crates/core/src/*_tests.rs`
- **Count**: 48 tests
- **Coverage**:
  - Core types (AgentId, Message, ToolCall, etc.)
  - Tool schema and execution context
  - Message types and serialization
  - Planning mode and configuration
  - Memory management types

### Integration Tests (Dashboard)
- **Location**: `crates/dashboard/tests/handlers_test.rs`
- **Count**: 11 tests
- **Coverage**:
  - Agent status types
  - Session management
  - Trace entries and types
  - Data serialization

### Mock Tests (LLM Providers)
- **Location**: `crates/providers/tests/mock_backend_test.rs`
- **Count**: 15 tests
- **Coverage**:
  - MockBackend functionality
  - Response builders
  - Tool call mocking
  - Token usage tracking
  - Call recording
  - Latency simulation
  - Error handling

### E2E Tests (Agent Engine)
- **Location**: `crates/agents/src/engine_integration_tests.rs`
- **Count**: 12 tests (existing)
- **Coverage**:
  - Agent lifecycle (spawn, execute, shutdown)
  - Multi-agent communication
  - Tool execution flow
  - ReACT loop with MockBackend
  - Stop word detection
  - Max iterations handling

## Total Test Count

**656 tests** passing across the entire workspace

## Running Tests

### Run All Tests
```bash
cargo test --workspace
```

### Run Tests for Specific Crate
```bash
# Core library tests
cargo test -p rust-ai-agents-core

# Dashboard tests
cargo test -p rust-ai-agents-dashboard

# Provider tests (including MockBackend)
cargo test -p rust-ai-agents-providers

# Agent engine tests
cargo test -p rust-ai-agents-agents
```

### Run Specific Test File
```bash
# Integration tests only
cargo test --test handlers_test

# Mock backend tests
cargo test --test mock_backend_test
```

### Run Tests with Output
```bash
cargo test -- --nocapture
```

### Run Tests in Parallel
```bash
cargo test --workspace -- --test-threads=4
```

## CI/CD Integration

The project uses GitHub Actions for continuous integration. The workflow is defined in `.github/workflows/ci.yml`.

### CI Pipeline Steps

1. **Code Formatting Check**
   ```bash
   cargo fmt --all -- --check
   ```

2. **Linting with Clippy**
   ```bash
   cargo clippy --workspace --all-targets -- -W clippy::all
   ```

3. **Run All Tests**
   ```bash
   cargo test --workspace
   ```

4. **Build Release Binaries**
   ```bash
   cargo build --workspace --release
   ```

## Test Organization

### Unit Tests
Unit tests are located in the same crate as the code they test, typically in separate test modules:
- `src/*_tests.rs` - Test modules for specific source files
- Imported via `#[cfg(test)] mod tests;` in the main module

### Integration Tests
Integration tests are located in the `tests/` directory of each crate:
- `crates/dashboard/tests/` - Dashboard integration tests
- `crates/providers/tests/` - Provider integration tests

### Test Utilities
- **MockBackend**: A deterministic mock LLM backend for testing agent behavior
- **Test Helpers**: Utility functions for creating test data

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Determinism**: Use MockBackend for deterministic LLM responses
3. **Coverage**: Aim for comprehensive coverage of critical paths
4. **Speed**: Keep tests fast by avoiding unnecessary I/O or network calls
5. **Clarity**: Use descriptive test names that explain what is being tested

## Writing New Tests

### Unit Test Example
```rust
#[test]
fn test_agent_id_creation() {
    let id = AgentId::new("test-agent");
    assert_eq!(id.0, "test-agent");
}
```

### Async Integration Test Example
```rust
#[tokio::test]
async fn test_mock_backend_simple_text() {
    let backend = MockBackend::new()
        .with_response(MockResponse::text("Hello!"));
    
    let messages = vec![LLMMessage::user("Hi")];
    let result = backend.infer(&messages, &[], 0.7).await.unwrap();
    
    assert_eq!(result.content, "Hello!");
}
```

### E2E Test Example
```rust
#[tokio::test]
async fn test_agent_tool_execution() {
    let backend = Arc::new(MockBackend::new()
        .with_response(MockResponse::tool_call("calculator", json!({"op": "add"})))
        .with_response(MockResponse::text("The result is 5")));
    
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool));
    
    let engine = AgentEngine::new();
    let config = create_test_config("agent-1");
    let agent_id = engine.spawn_agent(config, Arc::new(registry), backend).await.unwrap();
    
    // Send message and verify execution
    // ...
}
```

## Test Coverage Goals

- ✅ Unit tests for core functions
- ✅ Integration tests for API handlers
- ✅ Mock tests for LLM calls
- ✅ E2E tests for critical flows
- ✅ CI/CD pipeline with automated testing

## Future Improvements

1. **Code Coverage Reporting**: Add coverage metrics with `cargo-tarpaulin`
2. **Performance Testing**: Add benchmarks for critical paths
3. **Stress Testing**: Test system behavior under load
4. **Property-based Testing**: Use `proptest` for more comprehensive testing
5. **Mutation Testing**: Verify test quality with mutation testing

## Troubleshooting

### Test Failures

1. **Prometheus Recorder Error**: Some tests may fail if Prometheus recorder is initialized multiple times. Run tests with `--test-threads=1` if needed.

2. **Async Runtime Issues**: Ensure async tests use `#[tokio::test]` attribute.

3. **Timeout Issues**: Increase timeout for slow tests or CI environments.

### Debugging Tests

```bash
# Run with debug output
RUST_LOG=debug cargo test -- --nocapture

# Run specific test
cargo test test_name -- --exact --nocapture

# Show test output even on success
cargo test -- --show-output
```

## Contributing

When adding new features:
1. Write tests first (TDD approach recommended)
2. Ensure all existing tests pass
3. Add integration tests for new components
4. Update this documentation with new test categories

## Resources

- [Rust Testing Documentation](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Tokio Testing Guide](https://tokio.rs/tokio/topics/testing)
- [MockBackend Documentation](../crates/providers/src/mock.rs)
