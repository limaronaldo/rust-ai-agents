//! HTTP client for dashboard API

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

use super::types::*;

/// API client error
#[derive(Debug, Clone, thiserror::Error)]
pub enum ApiError {
    #[error("Request failed: {0}")]
    RequestFailed(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },
}

/// Result type for API operations
pub type ApiResult<T> = Result<T, ApiError>;

/// API client for the dashboard backend
#[derive(Clone)]
pub struct ApiClient {
    base_url: String,
}

impl ApiClient {
    /// Create a new API client
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }

    /// Create client pointing to current origin
    pub fn from_origin() -> Self {
        let window = web_sys::window().expect("no window");
        let origin = window.location().origin().unwrap_or_default();
        Self::new(origin)
    }

    // ==================== Metrics ====================

    /// Get current metrics
    pub async fn get_metrics(&self) -> ApiResult<DashboardMetrics> {
        self.get("/api/metrics").await
    }

    /// Get cost stats
    pub async fn get_stats(&self) -> ApiResult<CostStats> {
        self.get("/api/stats").await
    }

    // ==================== Traces ====================

    /// Get all traces
    pub async fn get_traces(&self) -> ApiResult<Vec<TraceEntry>> {
        self.get("/api/traces").await
    }

    /// Get traces for a session
    pub async fn get_session_traces(&self, session_id: &str) -> ApiResult<Vec<TraceEntry>> {
        self.get(&format!("/api/sessions/{}/traces", session_id))
            .await
    }

    // ==================== Sessions ====================

    /// Get all sessions
    pub async fn get_sessions(&self) -> ApiResult<Vec<Session>> {
        self.get("/api/sessions").await
    }

    /// Get session by ID
    pub async fn get_session(&self, id: &str) -> ApiResult<Session> {
        self.get(&format!("/api/sessions/{}", id)).await
    }

    /// Get messages for a session
    pub async fn get_session_messages(&self, session_id: &str) -> ApiResult<Vec<SessionMessage>> {
        self.get(&format!("/api/sessions/{}/messages", session_id))
            .await
    }

    // ==================== Agents ====================

    /// Get all registered agents
    pub async fn get_agents(&self) -> ApiResult<Vec<AgentStatus>> {
        self.get("/api/agents").await
    }

    /// Get agent by ID
    pub async fn get_agent(&self, id: &str) -> ApiResult<AgentStatus> {
        self.get(&format!("/api/agents/{}", id)).await
    }

    /// Start an agent
    pub async fn start_agent(&self, id: &str) -> ApiResult<()> {
        self.post(&format!("/api/agents/{}/start", id), None).await
    }

    /// Stop an agent
    pub async fn stop_agent(&self, id: &str) -> ApiResult<()> {
        self.post(&format!("/api/agents/{}/stop", id), None).await
    }

    /// Restart an agent
    pub async fn restart_agent(&self, id: &str) -> ApiResult<()> {
        self.post(&format!("/api/agents/{}/restart", id), None)
            .await
    }

    // ==================== HTTP Methods ====================

    /// Generic GET request
    async fn get<T: for<'de> serde::Deserialize<'de>>(&self, path: &str) -> ApiResult<T> {
        let url = format!("{}{}", self.base_url, path);

        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?;

        request
            .headers()
            .set("Accept", "application/json")
            .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?;

        self.execute(request).await
    }

    /// Generic POST request
    async fn post<T: for<'de> serde::Deserialize<'de>>(
        &self,
        path: &str,
        body: Option<&str>,
    ) -> ApiResult<T> {
        let url = format!("{}{}", self.base_url, path);

        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_mode(RequestMode::Cors);

        if let Some(body) = body {
            opts.set_body(&JsValue::from_str(body));
        }

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?;

        request
            .headers()
            .set("Accept", "application/json")
            .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?;

        request
            .headers()
            .set("Content-Type", "application/json")
            .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?;

        self.execute(request).await
    }

    /// Execute a request and parse the response
    async fn execute<T: for<'de> serde::Deserialize<'de>>(&self, request: Request) -> ApiResult<T> {
        let window =
            web_sys::window().ok_or_else(|| ApiError::RequestFailed("No window".to_string()))?;

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?;

        if !resp.ok() {
            let status = resp.status();
            let text = JsFuture::from(
                resp.text()
                    .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?,
            )
            .await
            .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?
            .as_string()
            .unwrap_or_default();
            return Err(ApiError::ApiError {
                status,
                message: text,
            });
        }

        let text = JsFuture::from(
            resp.text()
                .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?,
        )
        .await
        .map_err(|e| ApiError::RequestFailed(format!("{:?}", e)))?
        .as_string()
        .unwrap_or_default();

        // Handle empty response for void endpoints
        if text.is_empty() || text == "null" {
            // Try to parse as the expected type, which works for () via serde
            return serde_json::from_str("null").map_err(|e| ApiError::ParseError(e.to_string()));
        }

        serde_json::from_str(&text).map_err(|e| ApiError::ParseError(e.to_string()))
    }
}

impl Default for ApiClient {
    fn default() -> Self {
        Self::from_origin()
    }
}
