## Description

Audit `crates/agents/src/engine.rs` to verify it properly uses the `LLMBackend` trait from `crates/providers` with no hardcoded provider logic.

## Context

The engine should already accept `Arc<dyn LLMBackend>`. This issue is to **verify** the current state and document any findings, not necessarily to refactor.

## Analysis Checklist

- [ ] Check if `engine.rs` accepts `Arc<dyn LLMBackend>` in constructor
- [ ] Verify no direct imports from `openai.rs`, `anthropic.rs`, etc.
- [ ] Check that examples properly inject backends
- [ ] Verify tests can use `MockBackend` once implemented

## Expected Outcome

One of:
1. **Already correct** - Document findings, close issue
2. **Minor fixes needed** - Create follow-up issue with specific changes
3. **Major refactor needed** - Unlikely, but create detailed plan if so

## Deliverables

- [ ] Comment on this issue with audit findings
- [ ] If changes needed, create follow-up issues
- [ ] Update any documentation if interface is unclear
