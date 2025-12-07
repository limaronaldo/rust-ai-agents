# Contributing to HyperAgent

Thank you for your interest in contributing to HyperAgent!

## Branch Strategy

We use **GitHub Flow**:

```
main                 → stable, always deployable
  └── feat/*         → new features (feat/sse-transport)
  └── fix/*          → bug fixes (fix/cache-expiry)
  └── docs/*         → documentation updates
  └── refactor/*     → code refactoring
  └── test/*         → test improvements
```

## Development Workflow

### 1. Create a Branch

```bash
# For features
git checkout -b feat/your-feature-name

# For bug fixes
git checkout -b fix/issue-number-description

# For docs
git checkout -b docs/what-youre-documenting
```

### 2. Make Changes

- Write tests for new functionality
- Ensure all tests pass: `cargo test --workspace`
- Format code: `cargo fmt --all`
- Check lints: `cargo clippy --workspace`

### 3. Commit

Use conventional commits:

```
feat: add SSE transport for MCP server
fix: resolve cache expiry timing issue
docs: add approvals guide
refactor: simplify executor logic
test: add integration tests for engine
```

### 4. Open a Pull Request

- Push your branch: `git push origin feat/your-feature`
- Open PR against `main`
- Fill out the PR template
- Wait for CI to pass

## Code Standards

### Formatting

```bash
cargo fmt --all
```

### Linting

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Testing

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p rust-ai-agents-mcp

# Run with output
cargo test --workspace -- --nocapture
```

## Project Structure

```
crates/
├── agents/      # Core agent functionality
├── core/        # Shared types and traits
├── providers/   # LLM backends (OpenAI, Anthropic, etc.)
├── mcp/         # Model Context Protocol
├── cache/       # Caching layer
├── resilience/  # Circuit breaker, retry, etc.
├── workflow/    # YAML workflow DSL
└── ...
```

## Priority Areas

See [ROADMAP.md](./ROADMAP.md) for current priorities:

1. **Testing** - Integration tests with MockBackend
2. **MCP** - SSE transport, Agent-as-MCP-tool
3. **Documentation** - Guides for backends, approvals

## Questions?

Open an issue or start a discussion!
