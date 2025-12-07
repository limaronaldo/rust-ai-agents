//! HTTP client for Cloudflare Workers using fetch API

use crate::error::{CloudflareError, Result};
use rust_ai_agents_llm_client::{HttpRequest, LlmResponse, Provider, ResponseParser};
use worker::Fetch;

/// HTTP client that uses Cloudflare Workers' fetch API
pub struct CloudflareHttpClient;

impl CloudflareHttpClient {
    /// Execute an HTTP request using the Workers fetch API
    pub async fn execute(request: HttpRequest) -> Result<LlmResponse> {
        let mut headers = worker::Headers::new();
        for (key, value) in &request.headers {
            headers
                .set(key, value)
                .map_err(|e| CloudflareError::HttpError(e.to_string()))?;
        }

        let mut init = worker::RequestInit::new();
        init.with_method(worker::Method::Post);
        init.with_headers(headers);
        init.with_body(Some(wasm_bindgen::JsValue::from_str(&request.body)));

        let worker_request = worker::Request::new_with_init(&request.url, &init)
            .map_err(|e| CloudflareError::HttpError(e.to_string()))?;

        let mut response = Fetch::Request(worker_request)
            .send()
            .await
            .map_err(|e| CloudflareError::HttpError(e.to_string()))?;

        let status = response.status_code();
        let body = response
            .text()
            .await
            .map_err(|e| CloudflareError::HttpError(e.to_string()))?;

        if status != 200 {
            return Err(CloudflareError::ProviderError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        // Parse the response using the shared parser
        let provider = extract_provider_from_url(&request.url);
        let llm_response = ResponseParser::parse(provider, &body)?;

        Ok(llm_response)
    }

    /// Execute a streaming HTTP request
    pub async fn execute_stream(request: HttpRequest) -> Result<worker::Response> {
        let mut headers = worker::Headers::new();
        for (key, value) in &request.headers {
            headers
                .set(key, value)
                .map_err(|e| CloudflareError::HttpError(e.to_string()))?;
        }

        let mut init = worker::RequestInit::new();
        init.with_method(worker::Method::Post);
        init.with_headers(headers);
        init.with_body(Some(wasm_bindgen::JsValue::from_str(&request.body)));

        let worker_request = worker::Request::new_with_init(&request.url, &init)
            .map_err(|e| CloudflareError::HttpError(e.to_string()))?;

        let response = Fetch::Request(worker_request)
            .send()
            .await
            .map_err(|e| CloudflareError::HttpError(e.to_string()))?;

        let status = response.status_code();
        if status != 200 {
            return Err(CloudflareError::ProviderError(format!(
                "HTTP {} error",
                status
            )));
        }

        Ok(response)
    }
}

/// Extract provider from URL for response parsing
fn extract_provider_from_url(url: &str) -> Provider {
    if url.contains("anthropic") {
        Provider::Anthropic
    } else if url.contains("openrouter") {
        Provider::OpenRouter
    } else {
        Provider::OpenAI
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_provider() {
        assert!(matches!(
            extract_provider_from_url("https://api.anthropic.com/v1/messages"),
            Provider::Anthropic
        ));
        assert!(matches!(
            extract_provider_from_url("https://openrouter.ai/api/v1/chat"),
            Provider::OpenRouter
        ));
        assert!(matches!(
            extract_provider_from_url("https://api.openai.com/v1/chat/completions"),
            Provider::OpenAI
        ));
    }
}
