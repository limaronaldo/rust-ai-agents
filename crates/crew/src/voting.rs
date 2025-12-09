//! # Agent Voting System
//!
//! MassGen-inspired voting system for evaluating and ranking agent answers.
//! Enables agents to vote on each other's responses for quality assessment.
//!
//! ## Features
//!
//! - **Vote Types**: Approve, reject, abstain, or score-based voting
//! - **Weighted Voting**: Agent weights influence final scores
//! - **Quorum Requirements**: Configurable minimum votes for validity
//! - **Veto Power**: Optional veto capability for strategic agents
//! - **Vote Aggregation**: Multiple strategies for combining votes
//!
//! ## Example
//!
//! ```ignore
//! use rust_ai_agents_crew::voting::*;
//!
//! let mut ballot = VotingBallot::new("query-123", "What is the property value?");
//! ballot.add_answer(AgentAnswer::new("appraiser", "The value is $500,000"));
//! ballot.add_answer(AgentAnswer::new("analyst", "Based on comparables, $480,000-$520,000"));
//!
//! let mut session = VotingSession::new(ballot, VotingConfig::default());
//! session.cast_vote(Vote::approve("critic", "appraiser", Some("Well-reasoned")));
//! session.cast_vote(Vote::score("validator", "analyst", 0.9, Some("Good range estimate")));
//!
//! let result = session.tally()?;
//! println!("Winner: {:?}", result.winner());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Vote type representing different voting options
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VoteType {
    /// Approve the answer
    Approve,
    /// Reject the answer
    Reject,
    /// Abstain from voting (counted for quorum but not for/against)
    Abstain,
    /// Score-based vote (0.0 to 1.0)
    Score(f32),
    /// Veto - blocks the answer regardless of other votes (requires veto power)
    Veto,
}

impl VoteType {
    /// Convert vote to numeric score for aggregation
    pub fn to_score(&self) -> f32 {
        match self {
            VoteType::Approve => 1.0,
            VoteType::Reject => 0.0,
            VoteType::Abstain => 0.5,
            VoteType::Score(s) => s.clamp(0.0, 1.0),
            VoteType::Veto => 0.0,
        }
    }

    /// Check if this is a veto vote
    pub fn is_veto(&self) -> bool {
        matches!(self, VoteType::Veto)
    }
}

/// A single vote cast by an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// ID of the agent casting the vote
    pub voter_id: String,
    /// ID of the answer being voted on
    pub answer_id: String,
    /// The vote type
    pub vote_type: VoteType,
    /// Optional reasoning for the vote
    pub reasoning: Option<String>,
    /// Confidence in the vote (0.0 to 1.0)
    pub confidence: f32,
    /// Timestamp when vote was cast
    pub timestamp: u64,
}

impl Vote {
    /// Create an approval vote
    pub fn approve(
        voter: impl Into<String>,
        answer: impl Into<String>,
        reason: Option<&str>,
    ) -> Self {
        Self {
            voter_id: voter.into(),
            answer_id: answer.into(),
            vote_type: VoteType::Approve,
            reasoning: reason.map(String::from),
            confidence: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Create a rejection vote
    pub fn reject(
        voter: impl Into<String>,
        answer: impl Into<String>,
        reason: Option<&str>,
    ) -> Self {
        Self {
            voter_id: voter.into(),
            answer_id: answer.into(),
            vote_type: VoteType::Reject,
            reasoning: reason.map(String::from),
            confidence: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Create an abstain vote
    pub fn abstain(
        voter: impl Into<String>,
        answer: impl Into<String>,
        reason: Option<&str>,
    ) -> Self {
        Self {
            voter_id: voter.into(),
            answer_id: answer.into(),
            vote_type: VoteType::Abstain,
            reasoning: reason.map(String::from),
            confidence: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Create a score-based vote
    pub fn score(
        voter: impl Into<String>,
        answer: impl Into<String>,
        score: f32,
        reason: Option<&str>,
    ) -> Self {
        Self {
            voter_id: voter.into(),
            answer_id: answer.into(),
            vote_type: VoteType::Score(score.clamp(0.0, 1.0)),
            reasoning: reason.map(String::from),
            confidence: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Create a veto vote
    pub fn veto(voter: impl Into<String>, answer: impl Into<String>, reason: &str) -> Self {
        Self {
            voter_id: voter.into(),
            answer_id: answer.into(),
            vote_type: VoteType::Veto,
            reasoning: Some(reason.to_string()),
            confidence: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Set confidence level for this vote
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// An answer submitted by an agent for voting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAnswer {
    /// Unique identifier for this answer
    pub id: String,
    /// ID of the agent that provided this answer
    pub agent_id: String,
    /// The answer content
    pub content: String,
    /// Confidence score from the answering agent
    pub confidence: f32,
    /// Optional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp when answer was submitted
    pub submitted_at: u64,
}

impl AgentAnswer {
    /// Create a new agent answer
    pub fn new(agent_id: impl Into<String>, content: impl Into<String>) -> Self {
        let agent = agent_id.into();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: format!("{}_{}", agent, now),
            agent_id: agent,
            content: content.into(),
            confidence: 1.0,
            metadata: HashMap::new(),
            submitted_at: now,
        }
    }

    /// Set confidence for this answer
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add metadata to this answer
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A ballot containing answers to vote on
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingBallot {
    /// Unique identifier for this ballot
    pub id: String,
    /// The original query/question
    pub query: String,
    /// Answers to vote on
    pub answers: Vec<AgentAnswer>,
    /// When the ballot was created
    pub created_at: u64,
}

impl VotingBallot {
    /// Create a new voting ballot
    pub fn new(id: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            query: query.into(),
            answers: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Add an answer to the ballot
    pub fn add_answer(&mut self, answer: AgentAnswer) {
        self.answers.push(answer);
    }

    /// Get answer by ID
    pub fn get_answer(&self, id: &str) -> Option<&AgentAnswer> {
        self.answers.iter().find(|a| a.id == id)
    }

    /// Get answer by agent ID
    pub fn get_answer_by_agent(&self, agent_id: &str) -> Option<&AgentAnswer> {
        self.answers.iter().find(|a| a.agent_id == agent_id)
    }
}

/// Strategy for aggregating votes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AggregationStrategy {
    /// Simple majority wins
    #[default]
    Majority,
    /// Weighted average based on voter weights
    WeightedAverage,
    /// Highest score wins (for score-based voting)
    HighestScore,
    /// Consensus required (all must agree)
    Consensus,
    /// Ranked choice voting
    RankedChoice,
    /// Borda count
    BordaCount,
}

/// Configuration for voting sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingConfig {
    /// Minimum votes required for a valid result
    pub quorum: usize,
    /// Timeout for voting session
    pub timeout: Duration,
    /// Vote aggregation strategy
    pub strategy: AggregationStrategy,
    /// Whether self-voting is allowed (voting on own answer)
    pub allow_self_vote: bool,
    /// Whether veto power is enabled
    pub enable_veto: bool,
    /// Agents with veto power
    pub veto_agents: Vec<String>,
    /// Agent weights for weighted voting
    pub agent_weights: HashMap<String, f32>,
    /// Minimum score threshold to be considered valid
    pub min_score_threshold: f32,
    /// Whether to require reasoning for rejection votes
    pub require_rejection_reasoning: bool,
}

impl Default for VotingConfig {
    fn default() -> Self {
        Self {
            quorum: 2,
            timeout: Duration::from_secs(30),
            strategy: AggregationStrategy::WeightedAverage,
            allow_self_vote: false,
            enable_veto: true,
            veto_agents: vec!["strategic_director".to_string()],
            agent_weights: HashMap::new(),
            min_score_threshold: 0.5,
            require_rejection_reasoning: true,
        }
    }
}

impl VotingConfig {
    /// Create a new voting config builder
    pub fn builder() -> VotingConfigBuilder {
        VotingConfigBuilder::default()
    }

    /// Get weight for an agent (default 1.0)
    pub fn get_weight(&self, agent_id: &str) -> f32 {
        self.agent_weights.get(agent_id).copied().unwrap_or(1.0)
    }

    /// Check if agent has veto power
    pub fn has_veto_power(&self, agent_id: &str) -> bool {
        self.enable_veto && self.veto_agents.iter().any(|a| a == agent_id)
    }
}

/// Builder for VotingConfig
#[derive(Debug, Default)]
pub struct VotingConfigBuilder {
    config: VotingConfig,
}

impl VotingConfigBuilder {
    pub fn quorum(mut self, quorum: usize) -> Self {
        self.config.quorum = quorum;
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn strategy(mut self, strategy: AggregationStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn allow_self_vote(mut self, allow: bool) -> Self {
        self.config.allow_self_vote = allow;
        self
    }

    pub fn enable_veto(mut self, enable: bool) -> Self {
        self.config.enable_veto = enable;
        self
    }

    pub fn add_veto_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.config.veto_agents.push(agent_id.into());
        self
    }

    pub fn set_weight(mut self, agent_id: impl Into<String>, weight: f32) -> Self {
        self.config.agent_weights.insert(agent_id.into(), weight);
        self
    }

    pub fn min_score_threshold(mut self, threshold: f32) -> Self {
        self.config.min_score_threshold = threshold;
        self
    }

    pub fn require_rejection_reasoning(mut self, require: bool) -> Self {
        self.config.require_rejection_reasoning = require;
        self
    }

    pub fn build(self) -> VotingConfig {
        self.config
    }
}

/// Result for a single answer after tallying votes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerResult {
    /// The answer ID
    pub answer_id: String,
    /// The agent that provided this answer
    pub agent_id: String,
    /// Final aggregated score (0.0 to 1.0)
    pub score: f32,
    /// Number of approval votes
    pub approvals: usize,
    /// Number of rejection votes
    pub rejections: usize,
    /// Number of abstentions
    pub abstentions: usize,
    /// Whether this answer was vetoed
    pub vetoed: bool,
    /// Veto reason if applicable
    pub veto_reason: Option<String>,
    /// All votes for this answer
    pub votes: Vec<Vote>,
    /// Rank among all answers (1 = best)
    pub rank: usize,
}

impl AnswerResult {
    /// Check if this answer passed the voting
    pub fn passed(&self, threshold: f32) -> bool {
        !self.vetoed && self.score >= threshold
    }

    /// Get net approval (approvals - rejections)
    pub fn net_approval(&self) -> i32 {
        self.approvals as i32 - self.rejections as i32
    }
}

/// Overall voting session result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingResult {
    /// Ballot ID
    pub ballot_id: String,
    /// Results for each answer, sorted by score descending
    pub answer_results: Vec<AnswerResult>,
    /// Whether quorum was reached
    pub quorum_reached: bool,
    /// Total votes cast
    pub total_votes: usize,
    /// Total unique voters
    pub unique_voters: usize,
    /// Session duration in milliseconds
    pub duration_ms: u64,
    /// Aggregation strategy used
    pub strategy: AggregationStrategy,
    /// Whether any answer was vetoed
    pub has_vetoes: bool,
}

impl VotingResult {
    /// Get the winning answer (highest score, not vetoed)
    pub fn winner(&self) -> Option<&AnswerResult> {
        self.answer_results
            .iter()
            .find(|r| !r.vetoed && r.rank == 1)
    }

    /// Get all passing answers above threshold
    pub fn passing_answers(&self, threshold: f32) -> Vec<&AnswerResult> {
        self.answer_results
            .iter()
            .filter(|r| r.passed(threshold))
            .collect()
    }

    /// Get all vetoed answers
    pub fn vetoed_answers(&self) -> Vec<&AnswerResult> {
        self.answer_results.iter().filter(|r| r.vetoed).collect()
    }

    /// Check if voting was successful (quorum reached and has winner)
    pub fn is_valid(&self) -> bool {
        self.quorum_reached && self.winner().is_some()
    }
}

/// Voting session error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VotingError {
    /// Quorum not reached
    QuorumNotReached { required: usize, received: usize },
    /// Session timed out
    Timeout,
    /// Invalid vote (e.g., self-vote when not allowed)
    InvalidVote(String),
    /// Answer not found
    AnswerNotFound(String),
    /// No answers to vote on
    NoAnswers,
    /// Voting already complete
    VotingComplete,
    /// Veto without proper authority
    UnauthorizedVeto(String),
}

impl std::fmt::Display for VotingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VotingError::QuorumNotReached { required, received } => {
                write!(f, "Quorum not reached: {}/{} votes", received, required)
            }
            VotingError::Timeout => write!(f, "Voting session timed out"),
            VotingError::InvalidVote(msg) => write!(f, "Invalid vote: {}", msg),
            VotingError::AnswerNotFound(id) => write!(f, "Answer not found: {}", id),
            VotingError::NoAnswers => write!(f, "No answers to vote on"),
            VotingError::VotingComplete => write!(f, "Voting session already complete"),
            VotingError::UnauthorizedVeto(agent) => {
                write!(f, "Agent {} does not have veto authority", agent)
            }
        }
    }
}

impl std::error::Error for VotingError {}

/// Active voting session
#[derive(Debug, Clone)]
pub struct VotingSession {
    /// The ballot being voted on
    ballot: VotingBallot,
    /// Configuration for this session
    config: VotingConfig,
    /// Votes cast so far
    votes: Vec<Vote>,
    /// Session start time
    started_at: Instant,
    /// Whether voting is complete
    complete: bool,
}

impl VotingSession {
    /// Create a new voting session
    pub fn new(ballot: VotingBallot, config: VotingConfig) -> Self {
        Self {
            ballot,
            config,
            votes: Vec::new(),
            started_at: Instant::now(),
            complete: false,
        }
    }

    /// Cast a vote
    pub fn cast_vote(&mut self, vote: Vote) -> Result<(), VotingError> {
        if self.complete {
            return Err(VotingError::VotingComplete);
        }

        if self.started_at.elapsed() > self.config.timeout {
            self.complete = true;
            return Err(VotingError::Timeout);
        }

        // Validate answer exists
        if self.ballot.get_answer(&vote.answer_id).is_none()
            && self.ballot.get_answer_by_agent(&vote.answer_id).is_none()
        {
            return Err(VotingError::AnswerNotFound(vote.answer_id.clone()));
        }

        // Check self-vote
        if !self.config.allow_self_vote {
            if let Some(answer) = self.ballot.get_answer(&vote.answer_id) {
                if answer.agent_id == vote.voter_id {
                    return Err(VotingError::InvalidVote(
                        "Self-voting is not allowed".to_string(),
                    ));
                }
            }
        }

        // Check veto authority
        if vote.vote_type.is_veto() && !self.config.has_veto_power(&vote.voter_id) {
            return Err(VotingError::UnauthorizedVeto(vote.voter_id.clone()));
        }

        // Check rejection reasoning requirement
        if self.config.require_rejection_reasoning
            && vote.vote_type == VoteType::Reject
            && vote.reasoning.is_none()
        {
            return Err(VotingError::InvalidVote(
                "Rejection votes require reasoning".to_string(),
            ));
        }

        self.votes.push(vote);
        Ok(())
    }

    /// Check if quorum is reached
    pub fn quorum_reached(&self) -> bool {
        let unique_voters: std::collections::HashSet<_> =
            self.votes.iter().map(|v| &v.voter_id).collect();
        unique_voters.len() >= self.config.quorum
    }

    /// Get current vote count
    pub fn vote_count(&self) -> usize {
        self.votes.len()
    }

    /// Get unique voter count
    pub fn voter_count(&self) -> usize {
        let unique_voters: std::collections::HashSet<_> =
            self.votes.iter().map(|v| &v.voter_id).collect();
        unique_voters.len()
    }

    /// Check if voting is complete
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Check if session has timed out
    pub fn is_timed_out(&self) -> bool {
        self.started_at.elapsed() > self.config.timeout
    }

    /// Get remaining time
    pub fn remaining_time(&self) -> Duration {
        self.config
            .timeout
            .saturating_sub(self.started_at.elapsed())
    }

    /// Tally votes and get result
    pub fn tally(&mut self) -> Result<VotingResult, VotingError> {
        if self.ballot.answers.is_empty() {
            return Err(VotingError::NoAnswers);
        }

        let duration_ms = self.started_at.elapsed().as_millis() as u64;
        let unique_voters: std::collections::HashSet<_> =
            self.votes.iter().map(|v| &v.voter_id).collect();
        let quorum_reached = unique_voters.len() >= self.config.quorum;

        // Group votes by answer
        let mut votes_by_answer: HashMap<String, Vec<Vote>> = HashMap::new();
        for vote in &self.votes {
            votes_by_answer
                .entry(vote.answer_id.clone())
                .or_default()
                .push(vote.clone());
        }

        // Also check by agent_id for backwards compatibility
        for answer in &self.ballot.answers {
            if !votes_by_answer.contains_key(&answer.id) {
                if let Some(votes) = votes_by_answer.remove(&answer.agent_id) {
                    votes_by_answer.insert(answer.id.clone(), votes);
                }
            }
        }

        // Calculate results for each answer
        let mut answer_results: Vec<AnswerResult> = self
            .ballot
            .answers
            .iter()
            .map(|answer| {
                let votes = votes_by_answer.get(&answer.id).cloned().unwrap_or_default();
                self.calculate_answer_result(answer, votes)
            })
            .collect();

        // Sort by score descending and assign ranks
        answer_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        for (i, result) in answer_results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        let has_vetoes = answer_results.iter().any(|r| r.vetoed);

        self.complete = true;

        Ok(VotingResult {
            ballot_id: self.ballot.id.clone(),
            answer_results,
            quorum_reached,
            total_votes: self.votes.len(),
            unique_voters: unique_voters.len(),
            duration_ms,
            strategy: self.config.strategy,
            has_vetoes,
        })
    }

    /// Calculate result for a single answer
    fn calculate_answer_result(&self, answer: &AgentAnswer, votes: Vec<Vote>) -> AnswerResult {
        let mut approvals = 0;
        let mut rejections = 0;
        let mut abstentions = 0;
        let mut vetoed = false;
        let mut veto_reason = None;

        for vote in &votes {
            match &vote.vote_type {
                VoteType::Approve => approvals += 1,
                VoteType::Reject => rejections += 1,
                VoteType::Abstain => abstentions += 1,
                VoteType::Score(_) => {} // Handled separately
                VoteType::Veto => {
                    vetoed = true;
                    veto_reason = vote.reasoning.clone();
                }
            }
        }

        let score = if vetoed {
            0.0
        } else {
            self.aggregate_score(&votes)
        };

        AnswerResult {
            answer_id: answer.id.clone(),
            agent_id: answer.agent_id.clone(),
            score,
            approvals,
            rejections,
            abstentions,
            vetoed,
            veto_reason,
            votes,
            rank: 0, // Will be set after sorting
        }
    }

    /// Aggregate scores based on strategy
    fn aggregate_score(&self, votes: &[Vote]) -> f32 {
        if votes.is_empty() {
            return 0.5; // Neutral score if no votes
        }

        match self.config.strategy {
            AggregationStrategy::Majority => {
                let approvals = votes
                    .iter()
                    .filter(|v| v.vote_type == VoteType::Approve)
                    .count();
                let total_votes = votes
                    .iter()
                    .filter(|v| v.vote_type != VoteType::Abstain)
                    .count();
                if total_votes == 0 {
                    0.5
                } else {
                    approvals as f32 / total_votes as f32
                }
            }
            AggregationStrategy::WeightedAverage => {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;
                for vote in votes {
                    let weight = self.config.get_weight(&vote.voter_id) * vote.confidence;
                    weighted_sum += vote.vote_type.to_score() * weight;
                    total_weight += weight;
                }
                if total_weight == 0.0 {
                    0.5
                } else {
                    weighted_sum / total_weight
                }
            }
            AggregationStrategy::HighestScore => votes
                .iter()
                .map(|v| v.vote_type.to_score())
                .fold(0.0f32, |a, b| a.max(b)),
            AggregationStrategy::Consensus => {
                let all_approve = votes.iter().all(|v| {
                    v.vote_type == VoteType::Approve
                        || v.vote_type == VoteType::Abstain
                        || matches!(v.vote_type, VoteType::Score(s) if s >= 0.8)
                });
                if all_approve {
                    1.0
                } else {
                    0.0
                }
            }
            AggregationStrategy::RankedChoice | AggregationStrategy::BordaCount => {
                // For these strategies, we need rankings which aren't in simple votes
                // Fall back to weighted average
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;
                for vote in votes {
                    let weight = self.config.get_weight(&vote.voter_id);
                    weighted_sum += vote.vote_type.to_score() * weight;
                    total_weight += weight;
                }
                if total_weight == 0.0 {
                    0.5
                } else {
                    weighted_sum / total_weight
                }
            }
        }
    }

    /// Get the ballot
    pub fn ballot(&self) -> &VotingBallot {
        &self.ballot
    }

    /// Get current votes
    pub fn votes(&self) -> &[Vote] {
        &self.votes
    }

    /// Get config
    pub fn config(&self) -> &VotingConfig {
        &self.config
    }
}

/// Quick voting helper for simple use cases
pub struct QuickVote;

impl QuickVote {
    /// Quick majority vote on multiple answers
    pub fn majority_vote(
        answers: Vec<AgentAnswer>,
        votes: Vec<Vote>,
        quorum: usize,
    ) -> Result<VotingResult, VotingError> {
        let ballot = VotingBallot {
            id: format!("quick_{}", chrono::Utc::now().timestamp()),
            query: "Quick vote".to_string(),
            answers,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let config = VotingConfig::builder()
            .quorum(quorum)
            .strategy(AggregationStrategy::Majority)
            .build();

        let mut session = VotingSession::new(ballot, config);
        for vote in votes {
            session.cast_vote(vote)?;
        }

        session.tally()
    }

    /// Score-based evaluation of answers
    pub fn score_answers(
        answers: Vec<AgentAnswer>,
        scores: HashMap<String, f32>,
    ) -> Result<VotingResult, VotingError> {
        let ballot = VotingBallot {
            id: format!("score_{}", chrono::Utc::now().timestamp()),
            query: "Score evaluation".to_string(),
            answers,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let config = VotingConfig::builder()
            .quorum(1)
            .strategy(AggregationStrategy::HighestScore)
            .allow_self_vote(true)
            .build();

        let mut session = VotingSession::new(ballot, config);
        for (answer_id, score) in scores {
            session.cast_vote(Vote::score("evaluator", answer_id, score, None))?;
        }

        session.tally()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vote_types() {
        assert_eq!(VoteType::Approve.to_score(), 1.0);
        assert_eq!(VoteType::Reject.to_score(), 0.0);
        assert_eq!(VoteType::Abstain.to_score(), 0.5);
        assert_eq!(VoteType::Score(0.7).to_score(), 0.7);
        assert_eq!(VoteType::Veto.to_score(), 0.0);

        assert!(!VoteType::Approve.is_veto());
        assert!(VoteType::Veto.is_veto());
    }

    #[test]
    fn test_create_votes() {
        let vote = Vote::approve("agent1", "answer1", Some("Good answer"));
        assert_eq!(vote.voter_id, "agent1");
        assert_eq!(vote.answer_id, "answer1");
        assert_eq!(vote.vote_type, VoteType::Approve);
        assert_eq!(vote.reasoning, Some("Good answer".to_string()));

        let vote = Vote::score("agent2", "answer2", 0.85, None);
        assert!(matches!(vote.vote_type, VoteType::Score(s) if (s - 0.85).abs() < 0.001));
    }

    #[test]
    fn test_voting_session_basic() {
        let mut ballot = VotingBallot::new("test-1", "What is 2+2?");
        ballot.add_answer(AgentAnswer::new("math_agent", "4"));
        ballot.add_answer(AgentAnswer::new("random_agent", "5"));

        let config = VotingConfig::builder()
            .quorum(2)
            .strategy(AggregationStrategy::Majority)
            .build();

        let mut session = VotingSession::new(ballot, config);

        // Cast votes
        session
            .cast_vote(Vote::approve("voter1", "math_agent", Some("Correct")))
            .unwrap();
        session
            .cast_vote(Vote::approve("voter2", "math_agent", None))
            .unwrap();
        session
            .cast_vote(Vote::reject("voter1", "random_agent", Some("Wrong answer")))
            .unwrap();
        session
            .cast_vote(Vote::reject("voter2", "random_agent", Some("Incorrect")))
            .unwrap();

        assert!(session.quorum_reached());

        let result = session.tally().unwrap();
        assert!(result.quorum_reached);
        assert!(result.is_valid());

        let winner = result.winner().unwrap();
        assert_eq!(winner.agent_id, "math_agent");
        assert_eq!(winner.approvals, 2);
        assert_eq!(winner.rejections, 0);
    }

    #[test]
    fn test_weighted_voting() {
        let mut ballot = VotingBallot::new("test-2", "Complex question");
        ballot.add_answer(AgentAnswer::new("expert", "Expert answer"));
        ballot.add_answer(AgentAnswer::new("novice", "Novice answer"));

        let config = VotingConfig::builder()
            .quorum(2)
            .strategy(AggregationStrategy::WeightedAverage)
            .set_weight("expert_voter", 2.0)
            .set_weight("novice_voter", 0.5)
            .build();

        let mut session = VotingSession::new(ballot, config);

        // Expert voter prefers expert answer, novice prefers novice
        session
            .cast_vote(Vote::score("expert_voter", "expert", 0.9, None))
            .unwrap();
        session
            .cast_vote(Vote::score("novice_voter", "novice", 0.9, None))
            .unwrap();
        session
            .cast_vote(Vote::score("expert_voter", "novice", 0.3, None))
            .unwrap();
        session
            .cast_vote(Vote::score("novice_voter", "expert", 0.5, None))
            .unwrap();

        let result = session.tally().unwrap();
        let winner = result.winner().unwrap();

        // Expert answer should win due to higher weight
        assert_eq!(winner.agent_id, "expert");
    }

    #[test]
    fn test_veto() {
        let mut ballot = VotingBallot::new("test-3", "Sensitive question");
        ballot.add_answer(AgentAnswer::new("agent1", "Potentially wrong answer"));

        let config = VotingConfig::builder()
            .quorum(1)
            .enable_veto(true)
            .add_veto_agent("strategic_director")
            .build();

        let mut session = VotingSession::new(ballot, config);

        // Strategic director vetoes
        session
            .cast_vote(Vote::veto(
                "strategic_director",
                "agent1",
                "Answer contains errors",
            ))
            .unwrap();

        let result = session.tally().unwrap();
        assert!(result.has_vetoes);

        let vetoed = result.vetoed_answers();
        assert_eq!(vetoed.len(), 1);
        assert!(vetoed[0].vetoed);
        assert_eq!(
            vetoed[0].veto_reason,
            Some("Answer contains errors".to_string())
        );
    }

    #[test]
    fn test_unauthorized_veto() {
        let mut ballot = VotingBallot::new("test-4", "Question");
        ballot.add_answer(AgentAnswer::new("agent1", "Answer"));

        let config = VotingConfig::builder()
            .enable_veto(true)
            .add_veto_agent("strategic_director")
            .build();

        let mut session = VotingSession::new(ballot, config);

        // Regular agent tries to veto
        let result = session.cast_vote(Vote::veto("regular_agent", "agent1", "I disagree"));

        assert!(matches!(result, Err(VotingError::UnauthorizedVeto(_))));
    }

    #[test]
    fn test_self_vote_prevention() {
        let mut ballot = VotingBallot::new("test-5", "Question");
        let answer = AgentAnswer::new("agent1", "My answer");
        let answer_id = answer.id.clone();
        ballot.add_answer(answer);

        let config = VotingConfig::builder().allow_self_vote(false).build();

        let mut session = VotingSession::new(ballot, config);

        let result = session.cast_vote(Vote::approve("agent1", answer_id, None));
        assert!(matches!(result, Err(VotingError::InvalidVote(_))));
    }

    #[test]
    fn test_quorum_not_reached() {
        let mut ballot = VotingBallot::new("test-6", "Question");
        ballot.add_answer(AgentAnswer::new("agent1", "Answer"));

        let config = VotingConfig::builder().quorum(3).build();

        let mut session = VotingSession::new(ballot, config);
        session
            .cast_vote(Vote::approve("voter1", "agent1", None))
            .unwrap();

        let result = session.tally().unwrap();
        assert!(!result.quorum_reached);
    }

    #[test]
    fn test_consensus_strategy() {
        let mut ballot = VotingBallot::new("test-7", "Question");
        ballot.add_answer(AgentAnswer::new("agent1", "Answer"));

        let config = VotingConfig::builder()
            .quorum(3)
            .strategy(AggregationStrategy::Consensus)
            .build();

        let mut session = VotingSession::new(ballot, config);
        session
            .cast_vote(Vote::approve("voter1", "agent1", None))
            .unwrap();
        session
            .cast_vote(Vote::approve("voter2", "agent1", None))
            .unwrap();
        session
            .cast_vote(Vote::approve("voter3", "agent1", None))
            .unwrap();

        let result = session.tally().unwrap();
        assert!(result.is_valid());
        assert_eq!(result.winner().unwrap().score, 1.0);
    }

    #[test]
    fn test_quick_vote() {
        let answers = vec![
            AgentAnswer::new("agent1", "Answer 1"),
            AgentAnswer::new("agent2", "Answer 2"),
        ];

        let votes = vec![
            Vote::approve("voter1", "agent1", None),
            Vote::approve("voter2", "agent1", None),
            Vote::reject("voter1", "agent2", Some("Wrong")),
        ];

        let result = QuickVote::majority_vote(answers, votes, 2).unwrap();
        assert!(result.is_valid());
        assert_eq!(result.winner().unwrap().agent_id, "agent1");
    }

    #[test]
    fn test_rejection_requires_reasoning() {
        let mut ballot = VotingBallot::new("test-8", "Question");
        ballot.add_answer(AgentAnswer::new("agent1", "Answer"));

        let config = VotingConfig::builder()
            .require_rejection_reasoning(true)
            .build();

        let mut session = VotingSession::new(ballot, config);

        // Rejection without reasoning should fail
        let result = session.cast_vote(Vote::reject("voter1", "agent1", None));
        assert!(matches!(result, Err(VotingError::InvalidVote(_))));

        // With reasoning should work
        let result = session.cast_vote(Vote::reject("voter1", "agent1", Some("Invalid")));
        assert!(result.is_ok());
    }
}
