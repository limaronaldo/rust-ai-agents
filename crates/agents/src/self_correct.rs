//! # Self-Correcting Workflows
//!
//! LLM-as-Judge pattern for automatic quality assessment and correction.
//!
//! Inspired by Swarm's self-correcting workflows and AutoGen patterns.
//!
//! ## Features
//!
//! - **Quality Assessment**: Judge output quality using LLM or rules
//! - **Auto-Correction**: Automatically retry with feedback on failure
//! - **Validation Rules**: Custom validation criteria
//! - **Scoring**: Multi-dimensional quality scoring
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_agents::self_correct::{SelfCorrectingWorkflow, Judge, QualityCriteria};
//!
//! let workflow = SelfCorrectingWorkflow::new()
//!     .add_judge(CodeQualityJudge::new())
//!     .add_judge(FactualAccuracyJudge::new())
//!     .max_iterations(3)
//!     .quality_threshold(0.8);
//!
//! let result = workflow.execute(|ctx| async {
//!     // Generate response
//!     generate_response(&ctx.prompt).await
//! }).await?;
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Quality score for a dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    /// Dimension name (e.g., "accuracy", "completeness", "clarity")
    pub dimension: String,
    /// Score from 0.0 to 1.0
    pub score: f32,
    /// Explanation for the score
    pub explanation: String,
    /// Specific feedback for improvement
    pub feedback: Option<String>,
}

impl DimensionScore {
    pub fn new(dimension: impl Into<String>, score: f32) -> Self {
        Self {
            dimension: dimension.into(),
            score: score.clamp(0.0, 1.0),
            explanation: String::new(),
            feedback: None,
        }
    }

    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = explanation.into();
        self
    }

    pub fn with_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.feedback = Some(feedback.into());
        self
    }

    pub fn passed(&self, threshold: f32) -> bool {
        self.score >= threshold
    }
}

/// Overall judgment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Judgment {
    /// Whether the output passed quality checks
    pub passed: bool,
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
    /// Individual dimension scores
    pub dimension_scores: Vec<DimensionScore>,
    /// Summary feedback
    pub summary: String,
    /// Specific corrections needed
    pub corrections: Vec<String>,
    /// Metadata from the judge
    pub metadata: HashMap<String, String>,
}

impl Judgment {
    pub fn passed(score: f32) -> Self {
        Self {
            passed: true,
            overall_score: score.clamp(0.0, 1.0),
            dimension_scores: Vec::new(),
            summary: "Quality check passed".to_string(),
            corrections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn failed(score: f32, summary: impl Into<String>) -> Self {
        Self {
            passed: false,
            overall_score: score.clamp(0.0, 1.0),
            dimension_scores: Vec::new(),
            summary: summary.into(),
            corrections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_dimension(mut self, dimension: DimensionScore) -> Self {
        self.dimension_scores.push(dimension);
        self
    }

    pub fn with_correction(mut self, correction: impl Into<String>) -> Self {
        self.corrections.push(correction.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get feedback string for retry prompt
    pub fn feedback_for_retry(&self) -> String {
        let mut feedback = vec![format!(
            "Previous attempt scored {:.1}%",
            self.overall_score * 100.0
        )];
        feedback.push(format!("Feedback: {}", self.summary));

        if !self.corrections.is_empty() {
            feedback.push("Corrections needed:".to_string());
            for (i, correction) in self.corrections.iter().enumerate() {
                feedback.push(format!("  {}. {}", i + 1, correction));
            }
        }

        for dim in &self.dimension_scores {
            if let Some(ref fb) = dim.feedback {
                feedback.push(format!("- {}: {}", dim.dimension, fb));
            }
        }

        feedback.join("\n")
    }
}

/// Context for judgment
#[derive(Debug, Clone)]
pub struct JudgmentContext {
    /// Original input/prompt
    pub input: String,
    /// Generated output to judge
    pub output: String,
    /// Iteration number (1-based)
    pub iteration: u32,
    /// Previous judgments (for comparison)
    pub previous_judgments: Vec<Judgment>,
    /// Additional context
    pub metadata: HashMap<String, String>,
}

impl JudgmentContext {
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            iteration: 1,
            previous_judgments: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_iteration(mut self, iteration: u32) -> Self {
        self.iteration = iteration;
        self
    }

    pub fn with_previous(mut self, judgments: Vec<Judgment>) -> Self {
        self.previous_judgments = judgments;
        self
    }

    pub fn is_improving(&self) -> bool {
        if self.previous_judgments.len() < 2 {
            return true;
        }
        let last = &self.previous_judgments[self.previous_judgments.len() - 1];
        let prev = &self.previous_judgments[self.previous_judgments.len() - 2];
        last.overall_score > prev.overall_score
    }
}

/// Trait for implementing judges
#[async_trait]
pub trait Judge: Send + Sync {
    /// Unique name of this judge
    fn name(&self) -> &str;

    /// Evaluate the output and return a judgment
    async fn evaluate(&self, context: &JudgmentContext) -> Judgment;

    /// Weight of this judge in overall scoring (default 1.0)
    fn weight(&self) -> f32 {
        1.0
    }

    /// Whether this judge is critical (must pass)
    fn is_critical(&self) -> bool {
        false
    }
}

/// Boxed judge for type erasure
pub type BoxedJudge = Arc<dyn Judge>;

/// Error type for self-correcting workflows
#[derive(Debug, thiserror::Error)]
pub enum SelfCorrectError {
    #[error("Max iterations ({0}) exceeded without passing quality threshold")]
    MaxIterationsExceeded(u32),

    #[error("Critical judge '{0}' failed")]
    CriticalJudgeFailed(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("No improvement after {0} iterations")]
    NoImprovement(u32),
}

/// Configuration for self-correcting workflow
#[derive(Debug, Clone)]
pub struct SelfCorrectConfig {
    /// Maximum iterations before giving up
    pub max_iterations: u32,
    /// Quality threshold to pass (0.0 to 1.0)
    pub quality_threshold: f32,
    /// Stop if no improvement for N iterations
    pub stop_on_plateau: Option<u32>,
    /// Whether to include previous feedback in retry prompt
    pub include_feedback: bool,
}

impl Default for SelfCorrectConfig {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            quality_threshold: 0.8,
            stop_on_plateau: Some(2),
            include_feedback: true,
        }
    }
}

/// Result of a self-correcting workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfCorrectResult {
    /// Final output
    pub output: String,
    /// Whether quality threshold was met
    pub passed: bool,
    /// Final quality score
    pub final_score: f32,
    /// Number of iterations taken
    pub iterations: u32,
    /// History of judgments
    pub judgment_history: Vec<Judgment>,
    /// Total time taken
    pub duration_ms: u64,
}

impl SelfCorrectResult {
    pub fn improvement(&self) -> Option<f32> {
        if self.judgment_history.len() < 2 {
            return None;
        }
        let first = self.judgment_history.first()?.overall_score;
        let last = self.judgment_history.last()?.overall_score;
        Some(last - first)
    }
}

/// Statistics for workflow execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelfCorrectStats {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub total_iterations: u64,
    pub average_iterations: f64,
    pub average_final_score: f64,
}

/// A self-correcting workflow with judges
pub struct SelfCorrectingWorkflow {
    judges: Vec<BoxedJudge>,
    config: SelfCorrectConfig,
    stats: Arc<RwLock<SelfCorrectStats>>,
}

impl SelfCorrectingWorkflow {
    pub fn new() -> Self {
        Self {
            judges: Vec::new(),
            config: SelfCorrectConfig::default(),
            stats: Arc::new(RwLock::new(SelfCorrectStats::default())),
        }
    }

    pub fn with_config(mut self, config: SelfCorrectConfig) -> Self {
        self.config = config;
        self
    }

    pub fn add_judge<J: Judge + 'static>(mut self, judge: J) -> Self {
        self.judges.push(Arc::new(judge));
        self
    }

    pub fn add_judge_boxed(mut self, judge: BoxedJudge) -> Self {
        self.judges.push(judge);
        self
    }

    pub fn max_iterations(mut self, max: u32) -> Self {
        self.config.max_iterations = max;
        self
    }

    pub fn quality_threshold(mut self, threshold: f32) -> Self {
        self.config.quality_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Execute with a generator function
    pub async fn execute<F, Fut>(
        &self,
        input: impl Into<String>,
        generator: F,
    ) -> Result<SelfCorrectResult, SelfCorrectError>
    where
        F: Fn(String) -> Fut,
        Fut: Future<Output = Result<String, String>>,
    {
        let input = input.into();
        let start = Instant::now();
        let mut judgment_history = Vec::new();
        let mut current_prompt = input.clone();
        let mut best_output = String::new();
        let mut best_score = 0.0f32;
        let mut plateau_count = 0u32;

        for iteration in 1..=self.config.max_iterations {
            debug!(iteration, "Starting self-correct iteration");

            // Generate output
            let output = generator(current_prompt.clone())
                .await
                .map_err(|e| SelfCorrectError::ExecutionFailed(e))?;

            // Evaluate with all judges
            let context = JudgmentContext::new(&input, &output)
                .with_iteration(iteration)
                .with_previous(judgment_history.clone());

            let judgment = self.evaluate_all(&context).await;

            // Track best
            if judgment.overall_score > best_score {
                best_score = judgment.overall_score;
                best_output = output.clone();
                plateau_count = 0;
            } else {
                plateau_count += 1;
            }

            judgment_history.push(judgment.clone());

            // Check if passed
            if judgment.passed {
                return Ok(self.create_result(
                    output,
                    true,
                    judgment.overall_score,
                    iteration,
                    judgment_history,
                    start.elapsed().as_millis() as u64,
                ));
            }

            // Check for critical judge failure
            for judge in &self.judges {
                if judge.is_critical() {
                    let judge_result = judge.evaluate(&context).await;
                    if !judge_result.passed {
                        return Err(SelfCorrectError::CriticalJudgeFailed(
                            judge.name().to_string(),
                        ));
                    }
                }
            }

            // Check for plateau
            if let Some(plateau_limit) = self.config.stop_on_plateau {
                if plateau_count >= plateau_limit {
                    warn!(iteration, plateau_count, "No improvement, stopping early");
                    break;
                }
            }

            // Prepare retry prompt with feedback
            if self.config.include_feedback && iteration < self.config.max_iterations {
                current_prompt = format!(
                    "{}\n\n--- Previous Attempt Feedback ---\n{}",
                    input,
                    judgment.feedback_for_retry()
                );
            }
        }

        // Return best result even if didn't pass
        Ok(self.create_result(
            best_output,
            best_score >= self.config.quality_threshold,
            best_score,
            self.config.max_iterations,
            judgment_history,
            start.elapsed().as_millis() as u64,
        ))
    }

    async fn evaluate_all(&self, context: &JudgmentContext) -> Judgment {
        if self.judges.is_empty() {
            return Judgment::passed(1.0);
        }

        let mut total_score = 0.0f32;
        let mut total_weight = 0.0f32;
        let mut all_dimensions = Vec::new();
        let mut all_corrections = Vec::new();
        let mut summaries = Vec::new();

        for judge in &self.judges {
            let result = judge.evaluate(context).await;
            let weight = judge.weight();

            total_score += result.overall_score * weight;
            total_weight += weight;

            all_dimensions.extend(result.dimension_scores);
            all_corrections.extend(result.corrections);

            if !result.summary.is_empty() {
                summaries.push(format!("{}: {}", judge.name(), result.summary));
            }
        }

        let overall_score = if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        };

        let passed = overall_score >= self.config.quality_threshold;

        Judgment {
            passed,
            overall_score,
            dimension_scores: all_dimensions,
            summary: summaries.join("; "),
            corrections: all_corrections,
            metadata: HashMap::new(),
        }
    }

    fn create_result(
        &self,
        output: String,
        passed: bool,
        score: f32,
        iterations: u32,
        history: Vec<Judgment>,
        duration_ms: u64,
    ) -> SelfCorrectResult {
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_executions += 1;
            if passed {
                stats.successful_executions += 1;
            } else {
                stats.failed_executions += 1;
            }
            stats.total_iterations += iterations as u64;
            let total = stats.total_executions as f64;
            stats.average_iterations =
                (stats.average_iterations * (total - 1.0) + iterations as f64) / total;
            stats.average_final_score =
                (stats.average_final_score * (total - 1.0) + score as f64) / total;
        }

        SelfCorrectResult {
            output,
            passed,
            final_score: score,
            iterations,
            judgment_history: history,
            duration_ms,
        }
    }

    pub fn stats(&self) -> SelfCorrectStats {
        self.stats.read().clone()
    }
}

impl Default for SelfCorrectingWorkflow {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Built-in Judges
// ============================================================================

/// Simple length-based judge
pub struct LengthJudge {
    min_length: Option<usize>,
    max_length: Option<usize>,
}

impl LengthJudge {
    pub fn new() -> Self {
        Self {
            min_length: None,
            max_length: None,
        }
    }

    pub fn min(mut self, len: usize) -> Self {
        self.min_length = Some(len);
        self
    }

    pub fn max(mut self, len: usize) -> Self {
        self.max_length = Some(len);
        self
    }

    pub fn range(mut self, min: usize, max: usize) -> Self {
        self.min_length = Some(min);
        self.max_length = Some(max);
        self
    }
}

impl Default for LengthJudge {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Judge for LengthJudge {
    fn name(&self) -> &str {
        "length_judge"
    }

    async fn evaluate(&self, context: &JudgmentContext) -> Judgment {
        let len = context.output.len();
        let mut score = 1.0f32;
        let mut feedback = Vec::new();

        if let Some(min) = self.min_length {
            if len < min {
                score *= len as f32 / min as f32;
                feedback.push(format!("Output too short ({} chars, minimum {})", len, min));
            }
        }

        if let Some(max) = self.max_length {
            if len > max {
                score *= max as f32 / len as f32;
                feedback.push(format!("Output too long ({} chars, maximum {})", len, max));
            }
        }

        if feedback.is_empty() {
            Judgment::passed(score)
        } else {
            Judgment::failed(score, feedback.join("; "))
        }
    }
}

/// Keyword presence judge
pub struct KeywordJudge {
    required: Vec<String>,
    forbidden: Vec<String>,
}

impl KeywordJudge {
    pub fn new() -> Self {
        Self {
            required: Vec::new(),
            forbidden: Vec::new(),
        }
    }

    pub fn require(mut self, keyword: impl Into<String>) -> Self {
        self.required.push(keyword.into());
        self
    }

    pub fn forbid(mut self, keyword: impl Into<String>) -> Self {
        self.forbidden.push(keyword.into());
        self
    }
}

impl Default for KeywordJudge {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Judge for KeywordJudge {
    fn name(&self) -> &str {
        "keyword_judge"
    }

    async fn evaluate(&self, context: &JudgmentContext) -> Judgment {
        let output_lower = context.output.to_lowercase();
        let mut missing = Vec::new();
        let mut found_forbidden = Vec::new();

        for keyword in &self.required {
            if !output_lower.contains(&keyword.to_lowercase()) {
                missing.push(keyword.clone());
            }
        }

        for keyword in &self.forbidden {
            if output_lower.contains(&keyword.to_lowercase()) {
                found_forbidden.push(keyword.clone());
            }
        }

        let required_score = if self.required.is_empty() {
            1.0
        } else {
            (self.required.len() - missing.len()) as f32 / self.required.len() as f32
        };

        let forbidden_score = if self.forbidden.is_empty() {
            1.0
        } else if found_forbidden.is_empty() {
            1.0
        } else {
            0.0
        };

        let score = required_score * 0.7 + forbidden_score * 0.3;

        let mut judgment = if missing.is_empty() && found_forbidden.is_empty() {
            Judgment::passed(score)
        } else {
            let mut summary = Vec::new();
            if !missing.is_empty() {
                summary.push(format!("Missing keywords: {}", missing.join(", ")));
            }
            if !found_forbidden.is_empty() {
                summary.push(format!(
                    "Forbidden keywords found: {}",
                    found_forbidden.join(", ")
                ));
            }
            Judgment::failed(score, summary.join("; "))
        };

        for keyword in &missing {
            judgment = judgment.with_correction(format!("Include '{}' in the response", keyword));
        }

        for keyword in &found_forbidden {
            judgment = judgment.with_correction(format!("Remove '{}' from the response", keyword));
        }

        judgment
    }

    fn is_critical(&self) -> bool {
        !self.forbidden.is_empty() // Critical if we have forbidden keywords
    }
}

/// Regex pattern judge
pub struct PatternJudge {
    name: String,
    required_patterns: Vec<(regex::Regex, String)>,
    forbidden_patterns: Vec<(regex::Regex, String)>,
}

impl PatternJudge {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            required_patterns: Vec::new(),
            forbidden_patterns: Vec::new(),
        }
    }

    pub fn require(
        mut self,
        pattern: &str,
        description: impl Into<String>,
    ) -> Result<Self, regex::Error> {
        self.required_patterns
            .push((regex::Regex::new(pattern)?, description.into()));
        Ok(self)
    }

    pub fn forbid(
        mut self,
        pattern: &str,
        description: impl Into<String>,
    ) -> Result<Self, regex::Error> {
        self.forbidden_patterns
            .push((regex::Regex::new(pattern)?, description.into()));
        Ok(self)
    }
}

#[async_trait]
impl Judge for PatternJudge {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, context: &JudgmentContext) -> Judgment {
        let mut missing = Vec::new();
        let mut found_forbidden = Vec::new();

        for (pattern, desc) in &self.required_patterns {
            if !pattern.is_match(&context.output) {
                missing.push(desc.clone());
            }
        }

        for (pattern, desc) in &self.forbidden_patterns {
            if pattern.is_match(&context.output) {
                found_forbidden.push(desc.clone());
            }
        }

        let total = self.required_patterns.len() + self.forbidden_patterns.len();
        let failed = missing.len() + found_forbidden.len();
        let score = if total == 0 {
            1.0
        } else {
            (total - failed) as f32 / total as f32
        };

        if missing.is_empty() && found_forbidden.is_empty() {
            Judgment::passed(score)
        } else {
            let mut summary = Vec::new();
            if !missing.is_empty() {
                summary.push(format!("Missing: {}", missing.join(", ")));
            }
            if !found_forbidden.is_empty() {
                summary.push(format!("Found forbidden: {}", found_forbidden.join(", ")));
            }
            Judgment::failed(score, summary.join("; "))
        }
    }
}

/// Function-based judge
pub struct FnJudge<F>
where
    F: Fn(&JudgmentContext) -> Pin<Box<dyn Future<Output = Judgment> + Send>> + Send + Sync,
{
    name: String,
    evaluate_fn: F,
    weight: f32,
    critical: bool,
}

impl<F> FnJudge<F>
where
    F: Fn(&JudgmentContext) -> Pin<Box<dyn Future<Output = Judgment> + Send>> + Send + Sync,
{
    pub fn new(name: impl Into<String>, evaluate_fn: F) -> Self {
        Self {
            name: name.into(),
            evaluate_fn,
            weight: 1.0,
            critical: false,
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn critical(mut self) -> Self {
        self.critical = true;
        self
    }
}

#[async_trait]
impl<F> Judge for FnJudge<F>
where
    F: Fn(&JudgmentContext) -> Pin<Box<dyn Future<Output = Judgment> + Send>> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, context: &JudgmentContext) -> Judgment {
        (self.evaluate_fn)(context).await
    }

    fn weight(&self) -> f32 {
        self.weight
    }

    fn is_critical(&self) -> bool {
        self.critical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_length_judge() {
        let judge = LengthJudge::new().range(10, 100);

        // Too short
        let ctx = JudgmentContext::new("input", "short");
        let result = judge.evaluate(&ctx).await;
        assert!(!result.passed);

        // Just right
        let ctx = JudgmentContext::new("input", "This is a properly sized response.");
        let result = judge.evaluate(&ctx).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_keyword_judge() {
        let judge = KeywordJudge::new()
            .require("rust")
            .require("programming")
            .forbid("python");

        // Has required, no forbidden
        let ctx = JudgmentContext::new("input", "Rust is a great programming language.");
        let result = judge.evaluate(&ctx).await;
        assert!(result.passed);

        // Missing required
        let ctx = JudgmentContext::new("input", "Rust is great.");
        let result = judge.evaluate(&ctx).await;
        assert!(!result.passed);
        assert!(result.summary.contains("programming"));

        // Has forbidden
        let ctx = JudgmentContext::new("input", "Rust is better than Python for programming.");
        let result = judge.evaluate(&ctx).await;
        assert!(!result.passed);
    }

    #[tokio::test]
    async fn test_self_correcting_workflow() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let workflow = SelfCorrectingWorkflow::new()
            .add_judge(LengthJudge::new().min(20))
            .quality_threshold(0.8);

        // This generator improves each iteration
        let attempt = AtomicU32::new(0);
        let result = workflow
            .execute("Write something", |_prompt| {
                let current = attempt.fetch_add(1, Ordering::SeqCst);
                async move {
                    if current == 0 {
                        Ok("Too short".to_string())
                    } else {
                        Ok(
                            "This is a much longer response that should pass the length check."
                                .to_string(),
                        )
                    }
                }
            })
            .await
            .unwrap();

        assert!(result.passed);
        assert!(result.iterations <= 2);
    }

    #[tokio::test]
    async fn test_workflow_max_iterations() {
        let workflow = SelfCorrectingWorkflow::new()
            .add_judge(LengthJudge::new().min(1000)) // Impossible to satisfy
            .max_iterations(2)
            .quality_threshold(0.9);

        let result = workflow
            .execute("input", |_| async { Ok("short".to_string()) })
            .await
            .unwrap();

        assert!(!result.passed);
        assert_eq!(result.iterations, 2);
    }

    #[tokio::test]
    async fn test_judgment_feedback() {
        let judgment = Judgment::failed(0.5, "Quality issues found")
            .with_correction("Fix the formatting")
            .with_correction("Add more details");

        let feedback = judgment.feedback_for_retry();
        assert!(feedback.contains("50.0%"));
        assert!(feedback.contains("Fix the formatting"));
        assert!(feedback.contains("Add more details"));
    }

    #[tokio::test]
    async fn test_dimension_score() {
        let dim = DimensionScore::new("accuracy", 0.8)
            .with_explanation("Good accuracy overall")
            .with_feedback("Could improve citation quality");

        assert!(dim.passed(0.7));
        assert!(!dim.passed(0.9));
        assert_eq!(
            dim.feedback.as_deref(),
            Some("Could improve citation quality")
        );
    }

    #[tokio::test]
    async fn test_pattern_judge() {
        let judge = PatternJudge::new("format_check")
            .require(r"\d{4}-\d{2}-\d{2}", "date format YYYY-MM-DD")
            .unwrap();

        let ctx = JudgmentContext::new("input", "The date is 2024-01-15");
        let result = judge.evaluate(&ctx).await;
        assert!(result.passed);

        let ctx = JudgmentContext::new("input", "The date is January 15");
        let result = judge.evaluate(&ctx).await;
        assert!(!result.passed);
    }

    #[tokio::test]
    async fn test_fn_judge() {
        let judge = FnJudge::new("custom", |ctx| {
            let has_greeting = ctx.output.to_lowercase().contains("hello");
            Box::pin(async move {
                if has_greeting {
                    Judgment::passed(1.0)
                } else {
                    Judgment::failed(0.0, "Missing greeting")
                }
            })
        });

        let ctx = JudgmentContext::new("input", "Hello, world!");
        let result = judge.evaluate(&ctx).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_workflow_stats() {
        let workflow = SelfCorrectingWorkflow::new()
            .add_judge(LengthJudge::new().min(5))
            .quality_threshold(0.8);

        workflow
            .execute("test", |_| async { Ok("Hello World".to_string()) })
            .await
            .unwrap();

        workflow
            .execute("test", |_| async {
                Ok("Another test response".to_string())
            })
            .await
            .unwrap();

        let stats = workflow.stats();
        assert_eq!(stats.total_executions, 2);
        assert_eq!(stats.successful_executions, 2);
    }

    #[tokio::test]
    async fn test_judgment_context_improving() {
        let j1 = Judgment::failed(0.3, "Poor");
        let j2 = Judgment::failed(0.5, "Better");
        let j3 = Judgment::passed(0.8);

        let ctx = JudgmentContext::new("input", "output").with_previous(vec![j1, j2, j3]);

        assert!(ctx.is_improving());
    }

    #[tokio::test]
    async fn test_result_improvement() {
        let result = SelfCorrectResult {
            output: "final".to_string(),
            passed: true,
            final_score: 0.9,
            iterations: 3,
            judgment_history: vec![
                Judgment::failed(0.3, ""),
                Judgment::failed(0.6, ""),
                Judgment::passed(0.9),
            ],
            duration_ms: 100,
        };

        let improvement = result.improvement().unwrap();
        assert!((improvement - 0.6).abs() < 0.01);
    }
}
