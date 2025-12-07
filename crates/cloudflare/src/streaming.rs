//! Streaming response support for Cloudflare Workers

use rust_ai_agents_llm_client::Provider;
use serde::{Deserialize, Serialize};

/// A chunk from a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Text content delta
    pub content: String,
    /// Whether this is the final chunk
    pub done: bool,
    /// Token usage (only on final chunk)
    pub usage: Option<StreamUsage>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Iterator over streaming response chunks
pub struct StreamingResponse {
    buffer: String,
    provider: Provider,
    done: bool,
}

impl StreamingResponse {
    /// Create a new streaming response parser
    pub fn new(provider: Provider) -> Self {
        Self {
            buffer: String::new(),
            provider,
            done: false,
        }
    }

    /// Process incoming data and extract chunks
    pub fn process(&mut self, data: &str) -> Vec<StreamChunk> {
        self.buffer.push_str(data);
        let mut chunks = Vec::new();

        // Process complete SSE lines
        while let Some(pos) = self.buffer.find("\n\n") {
            let line = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();

            if let Some(chunk) = self.parse_sse_line(&line) {
                if chunk.done {
                    self.done = true;
                }
                chunks.push(chunk);
            }
        }

        chunks
    }

    /// Check if the stream is complete
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Parse a single SSE line
    fn parse_sse_line(&self, line: &str) -> Option<StreamChunk> {
        // Handle data: prefix
        let data = line.strip_prefix("data: ")?;

        // Handle [DONE] marker
        if data.trim() == "[DONE]" {
            return Some(StreamChunk {
                content: String::new(),
                done: true,
                usage: None,
            });
        }

        // Parse JSON based on provider
        match self.provider {
            Provider::OpenAI | Provider::OpenRouter => self.parse_openai_chunk(data),
            Provider::Anthropic => self.parse_anthropic_chunk(data),
        }
    }

    /// Parse OpenAI/OpenRouter format chunk
    fn parse_openai_chunk(&self, data: &str) -> Option<StreamChunk> {
        #[derive(Deserialize)]
        struct OpenAiChunk {
            choices: Vec<OpenAiChoice>,
            usage: Option<OpenAiUsage>,
        }

        #[derive(Deserialize)]
        struct OpenAiChoice {
            delta: OpenAiDelta,
            finish_reason: Option<String>,
        }

        #[derive(Deserialize)]
        struct OpenAiDelta {
            content: Option<String>,
        }

        #[derive(Deserialize)]
        struct OpenAiUsage {
            prompt_tokens: u32,
            completion_tokens: u32,
            total_tokens: u32,
        }

        let chunk: OpenAiChunk = serde_json::from_str(data).ok()?;
        let choice = chunk.choices.first()?;

        let done = choice.finish_reason.is_some();
        let content = choice.delta.content.clone().unwrap_or_default();
        let usage = chunk.usage.map(|u| StreamUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Some(StreamChunk {
            content,
            done,
            usage,
        })
    }

    /// Parse Anthropic format chunk
    fn parse_anthropic_chunk(&self, data: &str) -> Option<StreamChunk> {
        #[derive(Deserialize)]
        struct AnthropicEvent {
            #[serde(rename = "type")]
            event_type: String,
            delta: Option<AnthropicDelta>,
            usage: Option<AnthropicUsage>,
        }

        #[derive(Deserialize)]
        struct AnthropicDelta {
            #[serde(rename = "type")]
            _delta_type: Option<String>,
            text: Option<String>,
        }

        #[derive(Deserialize)]
        struct AnthropicUsage {
            input_tokens: u32,
            output_tokens: u32,
        }

        let event: AnthropicEvent = serde_json::from_str(data).ok()?;

        match event.event_type.as_str() {
            "content_block_delta" => {
                let text = event.delta.and_then(|d| d.text).unwrap_or_default();
                Some(StreamChunk {
                    content: text,
                    done: false,
                    usage: None,
                })
            }
            "message_stop" => Some(StreamChunk {
                content: String::new(),
                done: true,
                usage: event.usage.map(|u| StreamUsage {
                    prompt_tokens: u.input_tokens,
                    completion_tokens: u.output_tokens,
                    total_tokens: u.input_tokens + u.output_tokens,
                }),
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_chunk_parsing() {
        let mut stream = StreamingResponse::new(Provider::OpenAI);

        let data = r#"data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}

"#;

        let chunks = stream.process(data);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Hello");
        assert!(!chunks[0].done);
    }

    #[test]
    fn test_done_marker() {
        let mut stream = StreamingResponse::new(Provider::OpenAI);

        let data = "data: [DONE]\n\n";
        let chunks = stream.process(data);

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].done);
        assert!(stream.is_done());
    }

    #[test]
    fn test_anthropic_chunk_parsing() {
        let mut stream = StreamingResponse::new(Provider::Anthropic);

        let data = r#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}

"#;

        let chunks = stream.process(data);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Hi");
        assert!(!chunks[0].done);
    }
}
