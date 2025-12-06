//! Name matching with Brazilian conventions
//!
//! Fuzzy matching that handles:
//! - Prepositions (de, da, do, das, dos)
//! - Abbreviations (M. for Melo)
//! - Accents and special characters
//! - Case insensitivity

use strsim::{jaro_winkler, levenshtein};
use tracing::debug;
use unicode_normalization::UnicodeNormalization;

/// Portuguese prepositions commonly found in names
const PREPOSITIONS: &[&str] = &["de", "da", "do", "das", "dos", "e", "&"];

/// Name matcher with Brazilian conventions
#[derive(Debug, Clone)]
pub struct NameMatcher {
    /// Minimum similarity threshold for a match
    pub threshold: f64,
    /// Weight for first name matching
    pub first_name_weight: f64,
    /// Weight for last name matching
    pub last_name_weight: f64,
}

impl Default for NameMatcher {
    fn default() -> Self {
        Self {
            threshold: 0.85,
            first_name_weight: 0.4,
            last_name_weight: 0.6,
        }
    }
}

impl NameMatcher {
    /// Create a new name matcher
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum similarity threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Normalize a name for comparison
    ///
    /// - Converts to lowercase
    /// - Removes accents
    /// - Removes prepositions
    /// - Normalizes whitespace
    pub fn normalize(&self, name: &str) -> String {
        // Normalize unicode and convert to lowercase
        let normalized: String = name
            .nfkd()
            .filter(|c| !c.is_ascii() || c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .to_lowercase();

        // Remove accents (keep only ASCII)
        let ascii: String = normalized
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || c.is_whitespace())
            .collect();

        // Split into words and filter prepositions
        let words: Vec<&str> = ascii
            .split_whitespace()
            .filter(|word| !PREPOSITIONS.contains(word))
            .collect();

        words.join(" ")
    }

    /// Extract name parts (first name, middle names, last name)
    pub fn extract_parts(&self, name: &str) -> NameParts {
        let normalized = self.normalize(name);
        let words: Vec<&str> = normalized.split_whitespace().collect();

        match words.len() {
            0 => NameParts {
                first: String::new(),
                middle: Vec::new(),
                last: String::new(),
                full_normalized: normalized,
            },
            1 => NameParts {
                first: words[0].to_string(),
                middle: Vec::new(),
                last: String::new(),
                full_normalized: normalized,
            },
            2 => NameParts {
                first: words[0].to_string(),
                middle: Vec::new(),
                last: words[1].to_string(),
                full_normalized: normalized,
            },
            _ => NameParts {
                first: words[0].to_string(),
                middle: words[1..words.len() - 1]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                last: words[words.len() - 1].to_string(),
                full_normalized: normalized,
            },
        }
    }

    /// Calculate similarity between two names
    ///
    /// Returns a score between 0.0 and 1.0
    pub fn similarity(&self, name1: &str, name2: &str) -> f64 {
        let parts1 = self.extract_parts(name1);
        let parts2 = self.extract_parts(name2);

        // If either name is empty, no match
        if parts1.full_normalized.is_empty() || parts2.full_normalized.is_empty() {
            return 0.0;
        }

        // Calculate component scores
        let first_score = self.compare_parts(&parts1.first, &parts2.first);
        let last_score = self.compare_parts(&parts1.last, &parts2.last);

        // Full name comparison using Jaro-Winkler
        let full_score = jaro_winkler(&parts1.full_normalized, &parts2.full_normalized);

        // Check for abbreviation matches
        let abbrev_bonus = self.abbreviation_bonus(&parts1, &parts2);

        // Weighted combination
        let base_score = if parts1.last.is_empty() || parts2.last.is_empty() {
            // Only first name available
            first_score * 0.6 + full_score * 0.4
        } else {
            first_score * self.first_name_weight + last_score * self.last_name_weight
        };

        // Combine with full name score and abbreviation bonus
        let combined = (base_score * 0.7 + full_score * 0.3 + abbrev_bonus).min(1.0);

        debug!(
            name1 = name1,
            name2 = name2,
            first_score,
            last_score,
            full_score,
            abbrev_bonus,
            combined,
            "Name similarity calculated"
        );

        combined
    }

    /// Compare two name parts
    fn compare_parts(&self, part1: &str, part2: &str) -> f64 {
        if part1.is_empty() || part2.is_empty() {
            return 0.0;
        }

        // Check for abbreviation
        if self.is_abbreviation(part1, part2) || self.is_abbreviation(part2, part1) {
            return 0.95;
        }

        // Use Jaro-Winkler for fuzzy matching
        jaro_winkler(part1, part2)
    }

    /// Check if one string is an abbreviation of another
    fn is_abbreviation(&self, short: &str, long: &str) -> bool {
        if short.len() > long.len() {
            return false;
        }

        // Check if short is just the first letter(s) of long
        let short_clean = short.trim_end_matches('.');
        if short_clean.len() <= 2 && long.starts_with(short_clean) {
            return true;
        }

        false
    }

    /// Calculate bonus for abbreviation matches
    fn abbreviation_bonus(&self, parts1: &NameParts, parts2: &NameParts) -> f64 {
        let mut bonus: f64 = 0.0;

        // Check middle name abbreviations
        for m1 in &parts1.middle {
            for m2 in &parts2.middle {
                if self.is_abbreviation(m1, m2) || self.is_abbreviation(m2, m1) {
                    bonus += 0.05;
                }
            }
        }

        // Check if middle name in one matches full in other
        for m1 in &parts1.middle {
            if self.is_abbreviation(m1, &parts2.first) || self.is_abbreviation(m1, &parts2.last) {
                bonus += 0.03;
            }
        }

        bonus.min(0.15)
    }

    /// Check if two names match above threshold
    pub fn matches(&self, name1: &str, name2: &str) -> bool {
        self.similarity(name1, name2) >= self.threshold
    }

    /// Generate a normalized entity ID from a name
    pub fn to_entity_id(&self, name: &str) -> String {
        self.normalize(name).replace(' ', "_")
    }

    /// Calculate Levenshtein distance between names
    pub fn distance(&self, name1: &str, name2: &str) -> usize {
        let n1 = self.normalize(name1);
        let n2 = self.normalize(name2);
        levenshtein(&n1, &n2)
    }
}

/// Extracted name parts
#[derive(Debug, Clone)]
pub struct NameParts {
    /// First name
    pub first: String,
    /// Middle names
    pub middle: Vec<String>,
    /// Last name
    pub last: String,
    /// Full normalized name
    pub full_normalized: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matcher = NameMatcher::new();

        assert_eq!(
            matcher.normalize("Lucas Melo de Oliveira"),
            "lucas melo oliveira"
        );

        assert_eq!(matcher.normalize("ANA CLARA DA SILVA"), "ana clara silva");

        assert_eq!(
            matcher.normalize("João dos Santos & Filhos"),
            "joao santos filhos"
        );

        assert_eq!(matcher.normalize("JOSÉ MARÍA"), "jose maria");
    }

    #[test]
    fn test_extract_parts() {
        let matcher = NameMatcher::new();

        let parts = matcher.extract_parts("Lucas Melo de Oliveira");
        assert_eq!(parts.first, "lucas");
        assert_eq!(parts.middle, vec!["melo"]);
        assert_eq!(parts.last, "oliveira");

        let parts2 = matcher.extract_parts("Ana Silva");
        assert_eq!(parts2.first, "ana");
        assert!(parts2.middle.is_empty());
        assert_eq!(parts2.last, "silva");
    }

    #[test]
    fn test_high_similarity() {
        let matcher = NameMatcher::new();

        // Very similar names
        let sim1 = matcher.similarity("Lucas Melo Oliveira", "Lucas M. Oliveira");
        assert!(sim1 >= 0.85, "Expected >= 0.85, got {}", sim1);

        let sim2 = matcher.similarity("Ana Clara Silva", "Ana C Silva");
        assert!(sim2 >= 0.85, "Expected >= 0.85, got {}", sim2);

        // Exact match
        let sim3 = matcher.similarity("João Santos", "João Santos");
        assert!(sim3 >= 0.99, "Expected >= 0.99, got {}", sim3);
    }

    #[test]
    fn test_low_similarity() {
        let matcher = NameMatcher::new();

        // Very different names should have low similarity
        let sim = matcher.similarity("Lucas Oliveira", "Maria Souza");
        assert!(sim < 0.6, "Expected < 0.6, got {}", sim);

        let sim2 = matcher.similarity("Pedro Silva", "Ana Santos");
        assert!(sim2 < 0.6, "Expected < 0.6, got {}", sim2);

        // Completely unrelated names
        let sim3 = matcher.similarity("Xyz Abc", "Qrs Tuv");
        assert!(sim3 < 0.5, "Expected < 0.5, got {}", sim3);
    }

    #[test]
    fn test_abbreviation_handling() {
        let matcher = NameMatcher::new();

        // M. should match Melo
        let sim = matcher.similarity("Lucas M. Oliveira", "Lucas Melo Oliveira");
        assert!(sim >= 0.90, "Abbreviation should score high: {}", sim);

        // C. should match Clara
        let sim2 = matcher.similarity("Ana C. Silva", "Ana Clara Silva");
        assert!(sim2 >= 0.88, "Abbreviation should score high: {}", sim2);
    }

    #[test]
    fn test_matches() {
        let matcher = NameMatcher::new().with_threshold(0.85);

        assert!(matcher.matches("Lucas Oliveira", "Lucas M. Oliveira"));
        assert!(!matcher.matches("Lucas Oliveira", "Pedro Santos"));
    }

    #[test]
    fn test_to_entity_id() {
        let matcher = NameMatcher::new();

        assert_eq!(
            matcher.to_entity_id("Lucas Melo de Oliveira"),
            "lucas_melo_oliveira"
        );
    }

    #[test]
    fn test_distance() {
        let matcher = NameMatcher::new();

        assert_eq!(matcher.distance("Lucas", "Lucas"), 0);
        assert!(matcher.distance("Lucas", "Lucaz") <= 2);
    }
}
