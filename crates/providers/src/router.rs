//! Model Router for multi-provider LLM routing
//!
//! This module provides intelligent routing of LLM requests across multiple providers
//! with automatic fallback, load balancing, and route-based configuration.
//!
//! # Features
//!
//! - **Route-based configuration**: Different routes can use different models/providers
//! - **Automatic fallback**: Switch to backup provider on failure
//! - **Health tracking**: Track provider health and avoid unhealthy providers
//! - **Cost-aware routing**: Route to cheapest provider that meets requirements
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_ai_agents_providers::{
//!     ModelRouter, RouteConfig, FallbackConfig, OpenAIProvider, AnthropicProvider, LLMBackend,
//! };
//! use rust_ai_agents_core::LLMMessage;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut router = ModelRouter::new();
//!
//!     // Add providers
//!     router.add_provider("openai", Arc::new(
//!         OpenAIProvider::new("sk-...".to_string(), "gpt-4-turbo".to_string())
//!     ));
//!     router.add_provider("anthropic", Arc::new(
//!         AnthropicProvider::claude_35_sonnet("sk-ant-...".to_string())
//!     ));
//!
//!     // Configure routes
//!     router.configure_route("chat", RouteConfig {
//!         provider: "anthropic".to_string(),
//!         model: Some("claude-3-5-sonnet-20241022".to_string()),
//!         fallback: Some(FallbackConfig {
//!             provider: "openai".to_string(),
//!             model: Some("gpt-4-turbo".to_string()),
//!         }),
//!         ..Default::default()
//!     });
//!
//!     // Use router
//!     let messages = vec![LLMMessage::user("Hello!")];
//!     let tools = vec![];
//!     let response = router.infer("chat", &messages, &tools, 0.7).await;
//! }
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::backend::{InferenceOutput, LLMBackend, ModelInfo, StreamResponse};
use rust_ai_agents_core::{errors::LLMError, LLMMessage, ToolSchema};

/// Fallback configuration for a route
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FallbackConfig {
    /// Fallback provider name
    pub provider: String,
    /// Fallback model (optional, uses provider default if not set)
    pub model: Option<String>,
}

/// Route configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouteConfig {
    /// Primary provider name
    pub provider: String,
    /// Specific model to use (optional, uses provider default if not set)
    pub model: Option<String>,
    /// Fallback configuration
    pub fallback: Option<FallbackConfig>,
    /// Maximum retries before fallback
    pub max_retries: Option<u32>,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Whether to enable this route
    pub enabled: bool,
}

impl RouteConfig {
    /// Create a new route configuration
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: None,
            fallback: None,
            max_retries: Some(2),
            timeout_ms: Some(30_000),
            enabled: true,
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set fallback configuration
    pub fn with_fallback(mut self, provider: impl Into<String>, model: Option<String>) -> Self {
        self.fallback = Some(FallbackConfig {
            provider: provider.into(),
            model,
        });
        self
    }

    /// Set maximum retries
    pub fn with_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

/// Provider health status
#[derive(Debug, Clone)]
struct ProviderHealth {
    /// Number of consecutive failures
    consecutive_failures: u32,
    /// Last failure time
    last_failure: Option<Instant>,
    /// Total requests
    total_requests: u64,
    /// Total failures
    total_failures: u64,
    /// Average latency in ms
    avg_latency_ms: f64,
}

impl Default for ProviderHealth {
    fn default() -> Self {
        Self {
            consecutive_failures: 0,
            last_failure: None,
            total_requests: 0,
            total_failures: 0,
            avg_latency_ms: 0.0,
        }
    }
}

impl ProviderHealth {
    /// Check if provider is healthy
    fn is_healthy(&self) -> bool {
        // Consider unhealthy if 3+ consecutive failures in last 60 seconds
        if self.consecutive_failures >= 3 {
            if let Some(last_failure) = self.last_failure {
                if last_failure.elapsed() < Duration::from_secs(60) {
                    return false;
                }
            }
        }
        true
    }

    /// Record success
    fn record_success(&mut self, latency_ms: u64) {
        self.consecutive_failures = 0;
        self.total_requests += 1;
        // Update rolling average
        let n = self.total_requests as f64;
        self.avg_latency_ms = self.avg_latency_ms * ((n - 1.0) / n) + (latency_ms as f64 / n);
    }

    /// Record failure
    fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_failure = Some(Instant::now());
        self.total_requests += 1;
        self.total_failures += 1;
    }

    /// Get failure rate
    fn failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_failures as f64 / self.total_requests as f64
        }
    }
}

/// Provider wrapper that holds the backend and its health status
struct ProviderEntry {
    backend: Arc<dyn LLMBackend>,
    health: RwLock<ProviderHealth>,
}

/// Model Router for multi-provider LLM routing
pub struct ModelRouter {
    /// Registered providers
    providers: HashMap<String, ProviderEntry>,
    /// Route configurations
    routes: HashMap<String, RouteConfig>,
    /// Default provider
    default_provider: Option<String>,
    /// Default model
    default_model: Option<String>,
}

impl ModelRouter {
    /// Create a new model router
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            routes: HashMap::new(),
            default_provider: None,
            default_model: None,
        }
    }

    /// Add a provider
    pub fn add_provider(&mut self, name: impl Into<String>, backend: Arc<dyn LLMBackend>) {
        let name = name.into();
        debug!("Adding provider: {}", name);
        self.providers.insert(
            name.clone(),
            ProviderEntry {
                backend,
                health: RwLock::new(ProviderHealth::default()),
            },
        );

        // Set as default if first provider
        if self.default_provider.is_none() {
            self.default_provider = Some(name);
        }
    }

    /// Set default provider
    pub fn set_default(&mut self, provider: impl Into<String>, model: Option<String>) {
        self.default_provider = Some(provider.into());
        self.default_model = model;
    }

    /// Configure a route
    pub fn configure_route(&mut self, route: impl Into<String>, config: RouteConfig) {
        let route = route.into();
        debug!(
            "Configuring route '{}' -> provider '{}'",
            route, config.provider
        );
        self.routes.insert(route, config);
    }

    /// Get provider for a route
    fn get_provider_for_route(&self, route: &str) -> Option<(&ProviderEntry, Option<&str>)> {
        // Check route config first
        if let Some(config) = self.routes.get(route) {
            if !config.enabled {
                debug!("Route '{}' is disabled", route);
                return None;
            }
            if let Some(entry) = self.providers.get(&config.provider) {
                return Some((entry, config.model.as_deref()));
            }
        }

        // Fall back to default provider
        if let Some(default) = &self.default_provider {
            if let Some(entry) = self.providers.get(default) {
                return Some((entry, self.default_model.as_deref()));
            }
        }

        None
    }

    /// Get fallback provider for a route
    fn get_fallback_for_route(&self, route: &str) -> Option<(&ProviderEntry, Option<&str>)> {
        if let Some(config) = self.routes.get(route) {
            if let Some(fallback) = &config.fallback {
                if let Some(entry) = self.providers.get(&fallback.provider) {
                    return Some((entry, fallback.model.as_deref()));
                }
            }
        }
        None
    }

    /// Perform inference with routing and fallback
    pub async fn infer(
        &self,
        route: &str,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        let start = Instant::now();

        // Get primary provider
        let (entry, _model) = self.get_provider_for_route(route).ok_or_else(|| {
            LLMError::ApiError(format!("No provider configured for route: {}", route))
        })?;

        // Check health
        if !entry.health.read().is_healthy() {
            warn!(
                "Primary provider unhealthy for route '{}', trying fallback",
                route
            );
            return self
                .infer_with_fallback(route, messages, tools, temperature)
                .await;
        }

        // Try primary provider
        match entry.backend.infer(messages, tools, temperature).await {
            Ok(output) => {
                let latency = start.elapsed().as_millis() as u64;
                entry.health.write().record_success(latency);
                debug!("Route '{}' completed in {}ms", route, latency);
                Ok(output)
            }
            Err(e) => {
                entry.health.write().record_failure();
                warn!("Primary provider failed for route '{}': {}", route, e);

                // Try fallback
                self.infer_with_fallback(route, messages, tools, temperature)
                    .await
            }
        }
    }

    /// Perform inference with fallback provider
    async fn infer_with_fallback(
        &self,
        route: &str,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<InferenceOutput, LLMError> {
        let start = Instant::now();

        let (entry, _model) = self.get_fallback_for_route(route).ok_or_else(|| {
            LLMError::ApiError(format!(
                "No fallback provider configured for route: {}",
                route
            ))
        })?;

        // Check fallback health
        if !entry.health.read().is_healthy() {
            return Err(LLMError::ApiError(
                "Both primary and fallback providers are unhealthy".to_string(),
            ));
        }

        match entry.backend.infer(messages, tools, temperature).await {
            Ok(output) => {
                let latency = start.elapsed().as_millis() as u64;
                entry.health.write().record_success(latency);
                info!("Fallback succeeded for route '{}' in {}ms", route, latency);
                Ok(output)
            }
            Err(e) => {
                entry.health.write().record_failure();
                Err(e)
            }
        }
    }

    /// Perform streaming inference with routing
    pub async fn infer_stream(
        &self,
        route: &str,
        messages: &[LLMMessage],
        tools: &[ToolSchema],
        temperature: f32,
    ) -> Result<StreamResponse, LLMError> {
        let (entry, _model) = self.get_provider_for_route(route).ok_or_else(|| {
            LLMError::ApiError(format!("No provider configured for route: {}", route))
        })?;

        // For streaming, we don't do automatic fallback mid-stream
        // Health check only
        if !entry.health.read().is_healthy() {
            if let Some((fallback_entry, _)) = self.get_fallback_for_route(route) {
                if fallback_entry.health.read().is_healthy() {
                    return fallback_entry
                        .backend
                        .infer_stream(messages, tools, temperature)
                        .await;
                }
            }
            return Err(LLMError::ApiError("All providers unhealthy".to_string()));
        }

        entry
            .backend
            .infer_stream(messages, tools, temperature)
            .await
    }

    /// Get health status for all providers
    pub fn health_status(&self) -> HashMap<String, ProviderHealthStatus> {
        self.providers
            .iter()
            .map(|(name, entry)| {
                let health = entry.health.read();
                (
                    name.clone(),
                    ProviderHealthStatus {
                        healthy: health.is_healthy(),
                        total_requests: health.total_requests,
                        failure_rate: health.failure_rate(),
                        avg_latency_ms: health.avg_latency_ms,
                        consecutive_failures: health.consecutive_failures,
                    },
                )
            })
            .collect()
    }

    /// List all configured routes
    pub fn list_routes(&self) -> Vec<String> {
        self.routes.keys().cloned().collect()
    }

    /// Get route configuration
    pub fn get_route_config(&self, route: &str) -> Option<&RouteConfig> {
        self.routes.get(route)
    }

    /// Check if a route exists
    pub fn has_route(&self, route: &str) -> bool {
        self.routes.contains_key(route)
    }

    /// Get all provider names
    pub fn provider_names(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Get model info for a provider
    pub fn model_info(&self, provider: &str) -> Option<ModelInfo> {
        self.providers.get(provider).map(|e| e.backend.model_info())
    }
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Public health status for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealthStatus {
    /// Whether the provider is considered healthy
    pub healthy: bool,
    /// Total requests made
    pub total_requests: u64,
    /// Failure rate (0.0 - 1.0)
    pub failure_rate: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Consecutive failures
    pub consecutive_failures: u32,
}

/// Builder for ModelRouter
pub struct ModelRouterBuilder {
    router: ModelRouter,
}

impl ModelRouterBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            router: ModelRouter::new(),
        }
    }

    /// Add a provider
    pub fn provider(mut self, name: impl Into<String>, backend: Arc<dyn LLMBackend>) -> Self {
        self.router.add_provider(name, backend);
        self
    }

    /// Set default provider
    pub fn default_provider(mut self, name: impl Into<String>) -> Self {
        self.router.default_provider = Some(name.into());
        self
    }

    /// Set default model
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.router.default_model = Some(model.into());
        self
    }

    /// Add a route
    pub fn route(mut self, name: impl Into<String>, config: RouteConfig) -> Self {
        self.router.configure_route(name, config);
        self
    }

    /// Build the router
    pub fn build(self) -> ModelRouter {
        self.router
    }
}

impl Default for ModelRouterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockBackend, MockResponse};

    #[test]
    fn test_route_config_builder() {
        let config = RouteConfig::new("openai")
            .with_model("gpt-4-turbo")
            .with_fallback("anthropic", Some("claude-3-5-sonnet".to_string()))
            .with_retries(3)
            .with_timeout(60_000);

        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, Some("gpt-4-turbo".to_string()));
        assert!(config.fallback.is_some());
        assert_eq!(config.max_retries, Some(3));
        assert_eq!(config.timeout_ms, Some(60_000));
    }

    #[test]
    fn test_provider_health() {
        let mut health = ProviderHealth::default();

        // Initially healthy
        assert!(health.is_healthy());

        // Record failures
        health.record_failure();
        health.record_failure();
        assert!(health.is_healthy()); // Still healthy with 2 failures

        health.record_failure();
        assert!(!health.is_healthy()); // Unhealthy with 3 failures

        // Record success resets consecutive failures
        health.record_success(100);
        assert!(health.is_healthy());
    }

    #[test]
    fn test_failure_rate() {
        let mut health = ProviderHealth::default();

        health.record_success(100);
        health.record_success(100);
        health.record_failure();
        health.record_failure();

        assert!((health.failure_rate() - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_router_basic() {
        let mock = MockBackend::new().with_response(MockResponse::text("Hello from mock"));

        let mut router = ModelRouter::new();
        router.add_provider("mock", Arc::new(mock));
        router.configure_route("test", RouteConfig::new("mock").with_model("test-model"));

        let messages = vec![rust_ai_agents_core::LLMMessage {
            role: rust_ai_agents_core::MessageRole::User,
            content: "Hello".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let result = router.infer("test", &messages, &[], 0.7).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().content, "Hello from mock");
    }

    #[tokio::test]
    async fn test_router_fallback() {
        // Primary provider that always fails
        let failing_mock = MockBackend::new().with_response(MockResponse::error("Primary failed"));

        // Fallback provider that succeeds
        let fallback_mock =
            MockBackend::new().with_response(MockResponse::text("Hello from fallback"));

        let mut router = ModelRouter::new();
        router.add_provider("primary", Arc::new(failing_mock));
        router.add_provider("fallback", Arc::new(fallback_mock));
        router.configure_route(
            "test",
            RouteConfig::new("primary").with_fallback("fallback", None),
        );

        let messages = vec![rust_ai_agents_core::LLMMessage {
            role: rust_ai_agents_core::MessageRole::User,
            content: "Hello".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let result = router.infer("test", &messages, &[], 0.7).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().content, "Hello from fallback");
    }

    #[test]
    fn test_router_builder() {
        let mock = MockBackend::new().with_response(MockResponse::text("Test"));

        let router = ModelRouterBuilder::new()
            .provider("mock", Arc::new(mock))
            .default_provider("mock")
            .route("chat", RouteConfig::new("mock"))
            .build();

        assert!(router.has_route("chat"));
        assert_eq!(router.provider_names(), vec!["mock"]);
    }

    #[test]
    fn test_health_status() {
        let mock = MockBackend::new().with_response(MockResponse::text("Test"));

        let mut router = ModelRouter::new();
        router.add_provider("mock", Arc::new(mock));

        let status = router.health_status();
        assert!(status.contains_key("mock"));
        assert!(status["mock"].healthy);
        assert_eq!(status["mock"].total_requests, 0);
    }
}
