//! Multi-Agent Orchestra with Parallel Execution and Perspective Synthesis
//!
//! Inspired by make-it-heavy's approach to deploying multiple specialized agents
//! in parallel for comprehensive, multi-perspective analysis.
//!
//! # Features
//! - **Parallel or Sequential Execution**: Choose execution mode based on needs
//! - **Perspective Synthesis**: Merge multiple agent responses intelligently
//! - **Dynamic Question Decomposition**: Break complex queries into sub-questions
//! - **Specialized Agent Roles**: Each agent can have a distinct perspective/expertise
//!
//! # Example
//! ```ignore
//! use rust_ai_agents_crew::orchestra::*;
//!
//! let orchestra = Orchestra::builder()
//!     .add_agent(AgentPerspective::new("analyst", "You are a data analyst..."))
//!     .add_agent(AgentPerspective::new("critic", "You are a critical thinker..."))
//!     .add_agent(AgentPerspective::new("creative", "You are a creative problem solver..."))
//!     .with_synthesizer(SynthesisStrategy::Comprehensive)
//!     .with_execution_mode(ExecutionMode::Parallel)
//!     .build();
//!
//! let result = orchestra.analyze("How can we improve user engagement?").await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rust_ai_agents_core::errors::CrewError;
use rust_ai_agents_providers::LLMBackend;

/// Execution mode for the orchestra
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecutionMode {
    /// Execute agents one after another (default, lower resource usage)
    #[default]
    Sequential,
    /// Execute all agents simultaneously (faster, higher resource usage)
    Parallel,
    /// Execute in waves - groups of agents run in parallel sequentially
    Wave { wave_size: usize },
}

/// Agent perspective/role configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerspective {
    /// Unique identifier for this perspective
    pub id: String,
    /// Display name for the perspective
    pub name: String,
    /// System prompt defining the agent's role and expertise
    pub system_prompt: String,
    /// Optional specific model to use for this agent
    pub model: Option<String>,
    /// Temperature for this agent's responses (creativity level)
    pub temperature: f32,
    /// Priority weight for synthesis (higher = more influence)
    pub weight: f32,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl AgentPerspective {
    pub fn new(id: impl Into<String>, system_prompt: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            system_prompt: system_prompt.into(),
            model: None,
            temperature: 0.7,
            weight: 1.0,
            tags: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Response from a single agent perspective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerspectiveResponse {
    /// Which perspective generated this response
    pub perspective_id: String,
    /// The perspective's name
    pub perspective_name: String,
    /// The actual response content
    pub content: String,
    /// Time taken to generate response
    pub latency_ms: u64,
    /// Confidence score (0.0 - 1.0) if available
    pub confidence: Option<f32>,
    /// Key points extracted from the response
    pub key_points: Vec<String>,
    /// Any metadata from the response
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Synthesized result from multiple perspectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedResponse {
    /// The final synthesized response
    pub synthesis: String,
    /// Individual perspective responses
    pub perspectives: Vec<PerspectiveResponse>,
    /// Sub-questions that were analyzed (if decomposition was used)
    pub sub_questions: Vec<SubQuestion>,
    /// Total time for all processing
    pub total_latency_ms: u64,
    /// Execution mode used
    pub execution_mode: ExecutionMode,
    /// Synthesis strategy used
    pub synthesis_strategy: SynthesisStrategy,
    /// Agreement score between perspectives (0.0 - 1.0)
    pub agreement_score: Option<f32>,
    /// Areas of consensus
    pub consensus_points: Vec<String>,
    /// Areas of disagreement
    pub divergence_points: Vec<String>,
}

/// Strategy for synthesizing multiple perspectives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SynthesisStrategy {
    /// Comprehensive synthesis including all perspectives
    #[default]
    Comprehensive,
    /// Consensus-based - focus on agreed points
    Consensus,
    /// Best-of - select the highest quality response
    BestOf,
    /// Weighted average based on perspective weights
    Weighted,
    /// Debate-style - present contrasting views
    Debate,
    /// No synthesis - return raw perspectives only
    None,
}

/// A decomposed sub-question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQuestion {
    /// The sub-question text
    pub question: String,
    /// Which perspectives should answer this
    pub target_perspectives: Vec<String>,
    /// Responses to this sub-question
    pub responses: Vec<PerspectiveResponse>,
}

/// Trait for custom synthesis implementations
#[async_trait]
pub trait Synthesizer: Send + Sync {
    /// Synthesize multiple perspective responses into a final response
    async fn synthesize(
        &self,
        query: &str,
        responses: Vec<PerspectiveResponse>,
        backend: &dyn LLMBackend,
    ) -> Result<SynthesizedResponse, CrewError>;
}

/// Trait for question decomposition
#[async_trait]
pub trait QuestionDecomposer: Send + Sync {
    /// Decompose a complex query into sub-questions
    async fn decompose(
        &self,
        query: &str,
        perspectives: &[AgentPerspective],
        backend: &dyn LLMBackend,
    ) -> Result<Vec<SubQuestion>, CrewError>;
}

/// Default synthesizer implementation
pub struct DefaultSynthesizer {
    strategy: SynthesisStrategy,
}

impl DefaultSynthesizer {
    pub fn new(strategy: SynthesisStrategy) -> Self {
        Self { strategy }
    }

    fn extract_key_points(content: &str) -> Vec<String> {
        // Simple extraction: split by sentences and take first few
        content
            .split(['.', '\n'])
            .filter(|s| s.trim().len() > 20)
            .take(5)
            .map(|s| s.trim().to_string())
            .collect()
    }

    fn find_consensus(responses: &[PerspectiveResponse]) -> Vec<String> {
        // Simple consensus: find common themes/keywords
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for response in responses {
            let words: std::collections::HashSet<_> = response
                .content
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 4)
                .map(|s| s.to_string())
                .collect();

            for word in words {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Find words that appear in most responses
        let threshold = (responses.len() as f32 * 0.6).ceil() as usize;
        word_counts
            .into_iter()
            .filter(|(_, count)| *count >= threshold)
            .map(|(word, _)| word)
            .take(10)
            .collect()
    }

    fn calculate_agreement(responses: &[PerspectiveResponse]) -> f32 {
        if responses.len() < 2 {
            return 1.0;
        }

        // Simple agreement: based on common key points
        let all_points: Vec<_> = responses.iter().flat_map(|r| &r.key_points).collect();
        let unique_points: std::collections::HashSet<_> = all_points.iter().collect();

        if unique_points.is_empty() {
            return 0.5;
        }

        // More overlap = higher agreement
        1.0 - (unique_points.len() as f32 / all_points.len().max(1) as f32)
    }
}

#[async_trait]
impl Synthesizer for DefaultSynthesizer {
    async fn synthesize(
        &self,
        query: &str,
        responses: Vec<PerspectiveResponse>,
        backend: &dyn LLMBackend,
    ) -> Result<SynthesizedResponse, CrewError> {
        use rust_ai_agents_core::{LLMMessage, MessageRole};

        let start = Instant::now();

        if responses.is_empty() {
            return Err(CrewError::ExecutionFailed(
                "No responses to synthesize".to_string(),
            ));
        }

        let synthesis = match self.strategy {
            SynthesisStrategy::None => {
                // No synthesis, just format responses
                responses
                    .iter()
                    .map(|r| format!("## {}\n{}", r.perspective_name, r.content))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            }
            SynthesisStrategy::BestOf => {
                // Select best response based on length and confidence
                responses
                    .iter()
                    .max_by(|a, b| {
                        let score_a = a.content.len() as f32 * a.confidence.unwrap_or(0.5);
                        let score_b = b.content.len() as f32 * b.confidence.unwrap_or(0.5);
                        score_a.partial_cmp(&score_b).unwrap()
                    })
                    .map(|r| r.content.clone())
                    .unwrap_or_default()
            }
            _ => {
                // Use LLM for sophisticated synthesis
                let synthesis_prompt = self.build_synthesis_prompt(query, &responses);

                let messages = vec![
                    LLMMessage {
                        role: MessageRole::System,
                        content: self.get_synthesis_system_prompt(),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    LLMMessage {
                        role: MessageRole::User,
                        content: synthesis_prompt,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                ];

                let inference = backend
                    .infer(&messages, &[], 0.3)
                    .await
                    .map_err(|e| CrewError::ExecutionFailed(e.to_string()))?;

                inference.content
            }
        };

        let consensus_points = Self::find_consensus(&responses);
        let agreement_score = Self::calculate_agreement(&responses);

        Ok(SynthesizedResponse {
            synthesis,
            perspectives: responses,
            sub_questions: Vec::new(),
            total_latency_ms: start.elapsed().as_millis() as u64,
            execution_mode: ExecutionMode::Sequential, // Will be set by orchestra
            synthesis_strategy: self.strategy,
            agreement_score: Some(agreement_score),
            consensus_points,
            divergence_points: Vec::new(),
        })
    }
}

impl DefaultSynthesizer {
    fn get_synthesis_system_prompt(&self) -> String {
        match self.strategy {
            SynthesisStrategy::Comprehensive => {
                "You are an expert synthesizer. Your task is to combine multiple expert perspectives \
                into a comprehensive, well-structured response. Include insights from all perspectives, \
                highlight areas of agreement, and note any important differences in viewpoint. \
                Provide a balanced, thorough analysis.".to_string()
            }
            SynthesisStrategy::Consensus => {
                "You are a consensus builder. Your task is to identify and articulate the points \
                where multiple experts agree. Focus on shared conclusions and common ground. \
                Minimize discussion of disagreements unless critical.".to_string()
            }
            SynthesisStrategy::Weighted => {
                "You are a weighted synthesizer. Consider the weight/importance of each perspective \
                when combining responses. Give more emphasis to higher-weighted perspectives while \
                still incorporating insights from all sources.".to_string()
            }
            SynthesisStrategy::Debate => {
                "You are a debate moderator. Present the different perspectives as a structured debate, \
                highlighting contrasting viewpoints and the reasoning behind each position. \
                Help the reader understand the trade-offs and considerations from each angle.".to_string()
            }
            _ => "Synthesize the following perspectives into a coherent response.".to_string(),
        }
    }

    fn build_synthesis_prompt(&self, query: &str, responses: &[PerspectiveResponse]) -> String {
        let mut prompt = format!("Original Query: {}\n\n", query);
        prompt.push_str("Expert Perspectives:\n\n");

        for (i, response) in responses.iter().enumerate() {
            prompt.push_str(&format!(
                "=== Perspective {}: {} ===\n{}\n\n",
                i + 1,
                response.perspective_name,
                response.content
            ));
        }

        prompt.push_str(
            "Please synthesize these perspectives into a comprehensive response that addresses \
            the original query while incorporating insights from all experts.",
        );

        prompt
    }
}

/// Default question decomposer
pub struct DefaultDecomposer {
    max_sub_questions: usize,
}

impl DefaultDecomposer {
    pub fn new(max_sub_questions: usize) -> Self {
        Self { max_sub_questions }
    }
}

#[async_trait]
impl QuestionDecomposer for DefaultDecomposer {
    async fn decompose(
        &self,
        query: &str,
        perspectives: &[AgentPerspective],
        backend: &dyn LLMBackend,
    ) -> Result<Vec<SubQuestion>, CrewError> {
        use rust_ai_agents_core::{LLMMessage, MessageRole};

        let perspective_names: Vec<_> = perspectives.iter().map(|p| p.name.as_str()).collect();

        let system_prompt = format!(
            "You are a question decomposition expert. Break down complex queries into simpler, \
            focused sub-questions. Each sub-question should be answerable by one of these specialists: {}. \
            Return a JSON array of objects with 'question' and 'target_perspectives' (array of specialist names) fields. \
            Generate at most {} sub-questions. If the query is already simple enough, return an empty array.",
            perspective_names.join(", "),
            self.max_sub_questions
        );

        let messages = vec![
            LLMMessage {
                role: MessageRole::System,
                content: system_prompt,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            LLMMessage {
                role: MessageRole::User,
                content: format!("Decompose this query: {}", query),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];

        let inference = backend
            .infer(&messages, &[], 0.3)
            .await
            .map_err(|e| CrewError::ExecutionFailed(e.to_string()))?;

        // Parse JSON response
        let sub_questions: Vec<SubQuestion> = serde_json::from_str(&inference.content)
            .unwrap_or_else(|_| {
                // If parsing fails, return empty (no decomposition)
                Vec::new()
            });

        Ok(sub_questions)
    }
}

/// Orchestra configuration
#[derive(Debug, Clone)]
pub struct OrchestraConfig {
    /// Execution mode
    pub execution_mode: ExecutionMode,
    /// Synthesis strategy
    pub synthesis_strategy: SynthesisStrategy,
    /// Whether to decompose questions
    pub enable_decomposition: bool,
    /// Maximum sub-questions for decomposition
    pub max_sub_questions: usize,
    /// Timeout per agent
    pub agent_timeout: Duration,
    /// Whether to continue if some agents fail
    pub continue_on_error: bool,
    /// Minimum number of responses required for synthesis
    pub min_responses: usize,
}

impl Default for OrchestraConfig {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Sequential,
            synthesis_strategy: SynthesisStrategy::Comprehensive,
            enable_decomposition: false,
            max_sub_questions: 5,
            agent_timeout: Duration::from_secs(60),
            continue_on_error: true,
            min_responses: 1,
        }
    }
}

/// The main Orchestra struct for multi-agent coordination
pub struct Orchestra {
    /// Agent perspectives
    perspectives: Vec<AgentPerspective>,
    /// Configuration
    config: OrchestraConfig,
    /// LLM backend for agent inference
    backend: Arc<dyn LLMBackend>,
    /// Custom synthesizer (optional)
    synthesizer: Option<Arc<dyn Synthesizer>>,
    /// Custom decomposer (optional)
    decomposer: Option<Arc<dyn QuestionDecomposer>>,
}

impl Orchestra {
    /// Create a new orchestra builder
    pub fn builder(backend: Arc<dyn LLMBackend>) -> OrchestraBuilder {
        OrchestraBuilder::new(backend)
    }

    /// Analyze a query using all perspectives
    pub async fn analyze(&self, query: &str) -> Result<SynthesizedResponse, CrewError> {
        let start = Instant::now();

        // Optionally decompose the question
        let sub_questions = if self.config.enable_decomposition {
            self.decompose_query(query).await?
        } else {
            Vec::new()
        };

        // Get responses from all perspectives
        let responses = self.get_all_responses(query, &sub_questions).await?;

        if responses.len() < self.config.min_responses {
            return Err(CrewError::ExecutionFailed(format!(
                "Only {} responses received, minimum {} required",
                responses.len(),
                self.config.min_responses
            )));
        }

        // Synthesize responses
        let mut result = self.synthesize_responses(query, responses).await?;
        result.sub_questions = sub_questions;
        result.execution_mode = self.config.execution_mode;
        result.total_latency_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// Get a single perspective's response (useful for sequential/targeted use)
    pub async fn get_perspective_response(
        &self,
        perspective_id: &str,
        query: &str,
    ) -> Result<PerspectiveResponse, CrewError> {
        let perspective = self
            .perspectives
            .iter()
            .find(|p| p.id == perspective_id)
            .ok_or_else(|| {
                CrewError::ExecutionFailed(format!("Perspective not found: {}", perspective_id))
            })?;

        self.query_perspective(perspective, query).await
    }

    /// List available perspectives
    pub fn perspectives(&self) -> &[AgentPerspective] {
        &self.perspectives
    }

    /// Get configuration
    pub fn config(&self) -> &OrchestraConfig {
        &self.config
    }

    // Internal methods

    async fn decompose_query(&self, query: &str) -> Result<Vec<SubQuestion>, CrewError> {
        let default_decomposer = DefaultDecomposer::new(self.config.max_sub_questions);
        let decomposer: &dyn QuestionDecomposer = self
            .decomposer
            .as_ref()
            .map(|d| d.as_ref())
            .unwrap_or(&default_decomposer);

        decomposer
            .decompose(query, &self.perspectives, self.backend.as_ref())
            .await
    }

    async fn get_all_responses(
        &self,
        query: &str,
        _sub_questions: &[SubQuestion],
    ) -> Result<Vec<PerspectiveResponse>, CrewError> {
        match self.config.execution_mode {
            ExecutionMode::Sequential => self.execute_sequential(query).await,
            ExecutionMode::Parallel => self.execute_parallel(query).await,
            ExecutionMode::Wave { wave_size } => self.execute_wave(query, wave_size).await,
        }
    }

    async fn execute_sequential(&self, query: &str) -> Result<Vec<PerspectiveResponse>, CrewError> {
        let mut responses = Vec::new();

        for perspective in &self.perspectives {
            match tokio::time::timeout(
                self.config.agent_timeout,
                self.query_perspective(perspective, query),
            )
            .await
            {
                Ok(Ok(response)) => responses.push(response),
                Ok(Err(e)) => {
                    tracing::warn!(
                        perspective = %perspective.id,
                        error = %e,
                        "Perspective failed"
                    );
                    if !self.config.continue_on_error {
                        return Err(e);
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        perspective = %perspective.id,
                        "Perspective timed out"
                    );
                    if !self.config.continue_on_error {
                        return Err(CrewError::ExecutionFailed(format!(
                            "Perspective {} timed out",
                            perspective.id
                        )));
                    }
                }
            }
        }

        Ok(responses)
    }

    async fn execute_parallel(&self, query: &str) -> Result<Vec<PerspectiveResponse>, CrewError> {
        use futures::future::join_all;

        let futures: Vec<_> = self
            .perspectives
            .iter()
            .map(|p| {
                let query = query.to_string();
                let timeout = self.config.agent_timeout;
                async move {
                    tokio::time::timeout(timeout, self.query_perspective(p, &query)).await
                }
            })
            .collect();

        let results = join_all(futures).await;

        let mut responses = Vec::new();
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(Ok(response)) => responses.push(response),
                Ok(Err(e)) => {
                    tracing::warn!(
                        perspective = %self.perspectives[i].id,
                        error = %e,
                        "Perspective failed"
                    );
                    if !self.config.continue_on_error {
                        return Err(e);
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        perspective = %self.perspectives[i].id,
                        "Perspective timed out"
                    );
                    if !self.config.continue_on_error {
                        return Err(CrewError::ExecutionFailed(format!(
                            "Perspective {} timed out",
                            self.perspectives[i].id
                        )));
                    }
                }
            }
        }

        Ok(responses)
    }

    async fn execute_wave(
        &self,
        query: &str,
        wave_size: usize,
    ) -> Result<Vec<PerspectiveResponse>, CrewError> {
        use futures::future::join_all;

        let mut all_responses = Vec::new();

        for chunk in self.perspectives.chunks(wave_size) {
            let futures: Vec<_> =
                chunk
                    .iter()
                    .map(|p| {
                        let query = query.to_string();
                        let timeout = self.config.agent_timeout;
                        async move {
                            tokio::time::timeout(timeout, self.query_perspective(p, &query)).await
                        }
                    })
                    .collect();

            let results = join_all(futures).await;

            for (i, result) in results.into_iter().enumerate() {
                match result {
                    Ok(Ok(response)) => all_responses.push(response),
                    Ok(Err(e)) => {
                        if !self.config.continue_on_error {
                            return Err(e);
                        }
                        tracing::warn!(error = %e, "Perspective in wave failed");
                    }
                    Err(_) => {
                        if !self.config.continue_on_error {
                            return Err(CrewError::ExecutionFailed(format!(
                                "Perspective {} timed out",
                                chunk[i].id
                            )));
                        }
                        tracing::warn!("Perspective in wave timed out");
                    }
                }
            }
        }

        Ok(all_responses)
    }

    async fn query_perspective(
        &self,
        perspective: &AgentPerspective,
        query: &str,
    ) -> Result<PerspectiveResponse, CrewError> {
        use rust_ai_agents_core::{LLMMessage, MessageRole};

        let start = Instant::now();

        let messages = vec![
            LLMMessage {
                role: MessageRole::System,
                content: perspective.system_prompt.clone(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            LLMMessage {
                role: MessageRole::User,
                content: query.to_string(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];

        let inference = self
            .backend
            .infer(&messages, &[], perspective.temperature)
            .await
            .map_err(|e| CrewError::ExecutionFailed(e.to_string()))?;

        let key_points = DefaultSynthesizer::extract_key_points(&inference.content);

        Ok(PerspectiveResponse {
            perspective_id: perspective.id.clone(),
            perspective_name: perspective.name.clone(),
            content: inference.content,
            latency_ms: start.elapsed().as_millis() as u64,
            confidence: None,
            key_points,
            metadata: HashMap::new(),
        })
    }

    async fn synthesize_responses(
        &self,
        query: &str,
        responses: Vec<PerspectiveResponse>,
    ) -> Result<SynthesizedResponse, CrewError> {
        let default_synthesizer = DefaultSynthesizer::new(self.config.synthesis_strategy);
        let synthesizer: &dyn Synthesizer = self
            .synthesizer
            .as_ref()
            .map(|s| s.as_ref())
            .unwrap_or(&default_synthesizer);

        synthesizer
            .synthesize(query, responses, self.backend.as_ref())
            .await
    }
}

/// Builder for Orchestra
pub struct OrchestraBuilder {
    backend: Arc<dyn LLMBackend>,
    perspectives: Vec<AgentPerspective>,
    config: OrchestraConfig,
    synthesizer: Option<Arc<dyn Synthesizer>>,
    decomposer: Option<Arc<dyn QuestionDecomposer>>,
}

impl OrchestraBuilder {
    pub fn new(backend: Arc<dyn LLMBackend>) -> Self {
        Self {
            backend,
            perspectives: Vec::new(),
            config: OrchestraConfig::default(),
            synthesizer: None,
            decomposer: None,
        }
    }

    /// Add an agent perspective
    pub fn add_perspective(mut self, perspective: AgentPerspective) -> Self {
        self.perspectives.push(perspective);
        self
    }

    /// Add multiple perspectives at once
    pub fn with_perspectives(mut self, perspectives: Vec<AgentPerspective>) -> Self {
        self.perspectives.extend(perspectives);
        self
    }

    /// Set execution mode
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.config.execution_mode = mode;
        self
    }

    /// Set synthesis strategy
    pub fn with_synthesis_strategy(mut self, strategy: SynthesisStrategy) -> Self {
        self.config.synthesis_strategy = strategy;
        self
    }

    /// Enable question decomposition
    pub fn with_decomposition(mut self, enable: bool) -> Self {
        self.config.enable_decomposition = enable;
        self
    }

    /// Set maximum sub-questions
    pub fn with_max_sub_questions(mut self, max: usize) -> Self {
        self.config.max_sub_questions = max;
        self
    }

    /// Set agent timeout
    pub fn with_agent_timeout(mut self, timeout: Duration) -> Self {
        self.config.agent_timeout = timeout;
        self
    }

    /// Set whether to continue on agent errors
    pub fn with_continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.config.continue_on_error = continue_on_error;
        self
    }

    /// Set minimum responses required
    pub fn with_min_responses(mut self, min: usize) -> Self {
        self.config.min_responses = min;
        self
    }

    /// Set custom synthesizer
    pub fn with_synthesizer(mut self, synthesizer: Arc<dyn Synthesizer>) -> Self {
        self.synthesizer = Some(synthesizer);
        self
    }

    /// Set custom decomposer
    pub fn with_decomposer(mut self, decomposer: Arc<dyn QuestionDecomposer>) -> Self {
        self.decomposer = Some(decomposer);
        self
    }

    /// Build the orchestra
    pub fn build(self) -> Result<Orchestra, CrewError> {
        if self.perspectives.is_empty() {
            return Err(CrewError::InvalidConfiguration(
                "At least one perspective is required".to_string(),
            ));
        }

        Ok(Orchestra {
            perspectives: self.perspectives,
            config: self.config,
            backend: self.backend,
            synthesizer: self.synthesizer,
            decomposer: self.decomposer,
        })
    }
}

/// Preset perspective configurations for common use cases
pub mod presets {
    use super::*;

    /// Create a balanced analysis team
    pub fn balanced_analysis() -> Vec<AgentPerspective> {
        vec![
            AgentPerspective::new(
                "analyst",
                "You are a data-driven analyst. Focus on facts, statistics, and empirical evidence. \
                Provide structured, quantitative analysis where possible. Be objective and thorough."
            )
            .with_name("Data Analyst")
            .with_temperature(0.3)
            .with_weight(1.0),

            AgentPerspective::new(
                "critic",
                "You are a critical thinker and devil's advocate. Question assumptions, identify \
                potential flaws, risks, and weaknesses. Consider edge cases and failure modes. \
                Be constructively skeptical."
            )
            .with_name("Critical Thinker")
            .with_temperature(0.5)
            .with_weight(0.8),

            AgentPerspective::new(
                "creative",
                "You are a creative problem solver. Think outside the box, propose innovative \
                solutions, and make unexpected connections. Consider unconventional approaches \
                and possibilities others might miss."
            )
            .with_name("Creative Innovator")
            .with_temperature(0.9)
            .with_weight(0.7),

            AgentPerspective::new(
                "pragmatist",
                "You are a pragmatic implementer. Focus on what's practical, achievable, and \
                cost-effective. Consider real-world constraints, timelines, and resources. \
                Provide actionable recommendations."
            )
            .with_name("Pragmatist")
            .with_temperature(0.4)
            .with_weight(1.0),
        ]
    }

    /// Create a technical review team
    pub fn technical_review() -> Vec<AgentPerspective> {
        vec![
            AgentPerspective::new(
                "architect",
                "You are a software architect. Evaluate system design, scalability, and \
                architectural patterns. Consider maintainability, extensibility, and best practices."
            )
            .with_name("Software Architect")
            .with_temperature(0.3),

            AgentPerspective::new(
                "security",
                "You are a security expert. Identify vulnerabilities, security risks, and \
                potential attack vectors. Recommend security best practices and mitigations."
            )
            .with_name("Security Expert")
            .with_temperature(0.2),

            AgentPerspective::new(
                "performance",
                "You are a performance engineer. Analyze efficiency, identify bottlenecks, \
                and suggest optimizations. Consider resource usage, latency, and throughput."
            )
            .with_name("Performance Engineer")
            .with_temperature(0.3),
        ]
    }

    /// Create a business strategy team
    pub fn business_strategy() -> Vec<AgentPerspective> {
        vec![
            AgentPerspective::new(
                "strategist",
                "You are a business strategist. Analyze market positioning, competitive \
                advantage, and long-term growth opportunities. Think big picture.",
            )
            .with_name("Strategist")
            .with_temperature(0.5),
            AgentPerspective::new(
                "financial",
                "You are a financial analyst. Focus on costs, ROI, revenue potential, and \
                financial viability. Provide quantitative financial analysis.",
            )
            .with_name("Financial Analyst")
            .with_temperature(0.2),
            AgentPerspective::new(
                "customer",
                "You are a customer experience expert. Consider user needs, pain points, \
                and satisfaction. Advocate for the customer perspective.",
            )
            .with_name("Customer Advocate")
            .with_temperature(0.6),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_ai_agents_core::{LLMMessage, ToolSchema};
    use rust_ai_agents_providers::{InferenceOutput, ModelInfo, TokenUsage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Mock backend for testing
    struct MockBackend {
        call_count: AtomicUsize,
        responses: Vec<String>,
    }

    impl MockBackend {
        fn new(responses: Vec<String>) -> Self {
            Self {
                call_count: AtomicUsize::new(0),
                responses,
            }
        }
    }

    #[async_trait]
    impl LLMBackend for MockBackend {
        async fn infer(
            &self,
            _messages: &[LLMMessage],
            _tools: &[ToolSchema],
            _temperature: f32,
        ) -> Result<InferenceOutput, rust_ai_agents_core::errors::LLMError> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let response = self
                .responses
                .get(idx % self.responses.len())
                .cloned()
                .unwrap_or_else(|| format!("Response {}", idx));

            Ok(InferenceOutput {
                content: response,
                tool_calls: None,
                reasoning: None,
                confidence: 0.9,
                token_usage: TokenUsage::new(10, 20),
                metadata: HashMap::new(),
            })
        }

        async fn embed(
            &self,
            _text: &str,
        ) -> Result<Vec<f32>, rust_ai_agents_core::errors::LLMError> {
            Ok(vec![0.1, 0.2, 0.3])
        }

        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                model: "mock-model".to_string(),
                provider: "mock".to_string(),
                max_tokens: 4096,
                input_cost_per_1m: 0.0,
                output_cost_per_1m: 0.0,
                supports_functions: true,
                supports_vision: false,
            }
        }
    }

    #[tokio::test]
    async fn test_orchestra_sequential() {
        let backend = Arc::new(MockBackend::new(vec![
            "Analysis from analyst perspective.".to_string(),
            "Critical review of the topic.".to_string(),
        ]));

        let orchestra = Orchestra::builder(backend)
            .add_perspective(AgentPerspective::new("analyst", "You are an analyst."))
            .add_perspective(AgentPerspective::new("critic", "You are a critic."))
            .with_execution_mode(ExecutionMode::Sequential)
            .with_synthesis_strategy(SynthesisStrategy::None)
            .build()
            .unwrap();

        let result = orchestra.analyze("Test query").await.unwrap();

        assert_eq!(result.perspectives.len(), 2);
        assert_eq!(result.execution_mode, ExecutionMode::Sequential);
    }

    #[tokio::test]
    async fn test_orchestra_parallel() {
        let backend = Arc::new(MockBackend::new(vec![
            "Response 1".to_string(),
            "Response 2".to_string(),
            "Response 3".to_string(),
        ]));

        let orchestra = Orchestra::builder(backend)
            .add_perspective(AgentPerspective::new("p1", "Perspective 1"))
            .add_perspective(AgentPerspective::new("p2", "Perspective 2"))
            .add_perspective(AgentPerspective::new("p3", "Perspective 3"))
            .with_execution_mode(ExecutionMode::Parallel)
            .with_synthesis_strategy(SynthesisStrategy::None)
            .build()
            .unwrap();

        let result = orchestra.analyze("Test").await.unwrap();

        assert_eq!(result.perspectives.len(), 3);
        assert_eq!(result.execution_mode, ExecutionMode::Parallel);
    }

    #[tokio::test]
    async fn test_orchestra_wave() {
        let backend = Arc::new(MockBackend::new(vec!["Wave response".to_string()]));

        let orchestra = Orchestra::builder(backend)
            .add_perspective(AgentPerspective::new("p1", "P1"))
            .add_perspective(AgentPerspective::new("p2", "P2"))
            .add_perspective(AgentPerspective::new("p3", "P3"))
            .add_perspective(AgentPerspective::new("p4", "P4"))
            .with_execution_mode(ExecutionMode::Wave { wave_size: 2 })
            .with_synthesis_strategy(SynthesisStrategy::None)
            .build()
            .unwrap();

        let result = orchestra.analyze("Test").await.unwrap();

        assert_eq!(result.perspectives.len(), 4);
    }

    #[tokio::test]
    async fn test_single_perspective() {
        let backend = Arc::new(MockBackend::new(vec!["Single response".to_string()]));

        let orchestra = Orchestra::builder(backend)
            .add_perspective(
                AgentPerspective::new("analyst", "You are an analyst.").with_name("Data Analyst"),
            )
            .build()
            .unwrap();

        let response = orchestra
            .get_perspective_response("analyst", "Analyze this")
            .await
            .unwrap();

        assert_eq!(response.perspective_id, "analyst");
        assert_eq!(response.perspective_name, "Data Analyst");
        assert!(!response.content.is_empty());
    }

    #[tokio::test]
    async fn test_presets() {
        let perspectives = presets::balanced_analysis();
        assert_eq!(perspectives.len(), 4);

        let perspectives = presets::technical_review();
        assert_eq!(perspectives.len(), 3);

        let perspectives = presets::business_strategy();
        assert_eq!(perspectives.len(), 3);
    }

    #[tokio::test]
    async fn test_builder_validation() {
        let backend = Arc::new(MockBackend::new(vec![]));

        // Should fail without perspectives
        let result = Orchestra::builder(backend).build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_synthesis_best_of() {
        let backend = Arc::new(MockBackend::new(vec![
            "Short".to_string(),
            "This is a much longer and more detailed response that should be selected.".to_string(),
        ]));

        let orchestra = Orchestra::builder(backend)
            .add_perspective(AgentPerspective::new("short", "Be brief"))
            .add_perspective(AgentPerspective::new("detailed", "Be detailed"))
            .with_synthesis_strategy(SynthesisStrategy::BestOf)
            .build()
            .unwrap();

        let result = orchestra.analyze("Test").await.unwrap();

        // Best-of should select the longer response
        assert!(result.synthesis.len() > 10);
    }

    #[tokio::test]
    async fn test_continue_on_error() {
        let backend = Arc::new(MockBackend::new(vec!["Valid response".to_string()]));

        let orchestra = Orchestra::builder(backend)
            .add_perspective(AgentPerspective::new("p1", "P1"))
            .with_continue_on_error(true)
            .with_min_responses(1)
            .build()
            .unwrap();

        let result = orchestra.analyze("Test").await;
        assert!(result.is_ok());
    }
}
