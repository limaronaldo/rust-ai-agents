//! CPF (Cadastro de Pessoa Física) matching and validation
//!
//! Brazilian individual taxpayer registry number validation and normalization.

use tracing::debug;

/// CPF matcher with normalization and validation
#[derive(Debug, Clone, Default)]
pub struct CpfMatcher {
    /// Whether to validate CPF check digits
    validate_digits: bool,
}

impl CpfMatcher {
    /// Create a new CPF matcher
    pub fn new() -> Self {
        Self {
            validate_digits: true,
        }
    }

    /// Create a matcher that skips digit validation (for testing)
    pub fn without_validation() -> Self {
        Self {
            validate_digits: false,
        }
    }

    /// Normalize a CPF string to 11 digits
    ///
    /// Handles various input formats:
    /// - "123.456.789-00"
    /// - "12345678900"
    /// - " 123 456 789 00 "
    /// - "123-456-789-00"
    pub fn normalize(&self, cpf: &str) -> Option<String> {
        // Extract only digits
        let digits: String = cpf.chars().filter(|c| c.is_ascii_digit()).collect();

        // Must have exactly 11 digits
        if digits.len() != 11 {
            debug!(
                input = cpf,
                digits_found = digits.len(),
                "Invalid CPF length"
            );
            return None;
        }

        Some(digits)
    }

    /// Format a normalized CPF with standard punctuation
    pub fn format(&self, cpf: &str) -> Option<String> {
        let normalized = self.normalize(cpf)?;
        Some(format!(
            "{}.{}.{}-{}",
            &normalized[0..3],
            &normalized[3..6],
            &normalized[6..9],
            &normalized[9..11]
        ))
    }

    /// Check if a CPF is valid (proper check digits)
    pub fn is_valid(&self, cpf: &str) -> bool {
        let Some(digits) = self.normalize(cpf) else {
            return false;
        };

        if !self.validate_digits {
            return true;
        }

        // Check for all same digits (invalid)
        let chars: Vec<char> = digits.chars().collect();
        if chars.iter().all(|&c| c == chars[0]) {
            debug!(cpf = cpf, "CPF has all same digits");
            return false;
        }

        // Validate first check digit
        let digits_vec: Vec<u32> = chars.iter().filter_map(|c| c.to_digit(10)).collect();

        let sum1: u32 = digits_vec[0..9]
            .iter()
            .enumerate()
            .map(|(i, &d)| d * (10 - i as u32))
            .sum();

        let remainder1 = (sum1 * 10) % 11;
        let check1 = if remainder1 == 10 { 0 } else { remainder1 };

        if check1 != digits_vec[9] {
            debug!(
                cpf = cpf,
                expected = check1,
                got = digits_vec[9],
                "First check digit invalid"
            );
            return false;
        }

        // Validate second check digit
        let sum2: u32 = digits_vec[0..10]
            .iter()
            .enumerate()
            .map(|(i, &d)| d * (11 - i as u32))
            .sum();

        let remainder2 = (sum2 * 10) % 11;
        let check2 = if remainder2 == 10 { 0 } else { remainder2 };

        if check2 != digits_vec[10] {
            debug!(
                cpf = cpf,
                expected = check2,
                got = digits_vec[10],
                "Second check digit invalid"
            );
            return false;
        }

        true
    }

    /// Compare two CPFs for equality (normalizing both)
    pub fn matches(&self, cpf1: &str, cpf2: &str) -> bool {
        match (self.normalize(cpf1), self.normalize(cpf2)) {
            (Some(n1), Some(n2)) => n1 == n2,
            _ => false,
        }
    }

    /// Calculate match score between two CPFs
    ///
    /// Returns 1.0 for exact match, 0.0 for no match
    pub fn score(&self, cpf1: &str, cpf2: &str) -> f64 {
        if self.matches(cpf1, cpf2) {
            1.0
        } else {
            0.0
        }
    }

    /// Generate a valid CPF for testing (with proper check digits)
    #[cfg(test)]
    pub fn generate_valid() -> String {
        // Base digits
        let base = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        // Calculate first check digit
        let sum1: u32 = base
            .iter()
            .enumerate()
            .map(|(i, &d)| d * (10 - i as u32))
            .sum();
        let check1 = {
            let r = (sum1 * 10) % 11;
            if r == 10 {
                0
            } else {
                r
            }
        };

        // Calculate second check digit
        let mut extended = base.to_vec();
        extended.push(check1);
        let sum2: u32 = extended
            .iter()
            .enumerate()
            .map(|(i, &d)| d * (11 - i as u32))
            .sum();
        let check2 = {
            let r = (sum2 * 10) % 11;
            if r == 10 {
                0
            } else {
                r
            }
        };

        format!(
            "{}{}{}.{}{}{}.{}{}{}-{}{}",
            base[0],
            base[1],
            base[2],
            base[3],
            base[4],
            base[5],
            base[6],
            base[7],
            base[8],
            check1,
            check2
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_various_formats() {
        let matcher = CpfMatcher::new();

        // Standard format
        assert_eq!(
            matcher.normalize("123.456.789-00"),
            Some("12345678900".to_string())
        );

        // Digits only
        assert_eq!(
            matcher.normalize("12345678900"),
            Some("12345678900".to_string())
        );

        // With spaces
        assert_eq!(
            matcher.normalize(" 123 456 789 00 "),
            Some("12345678900".to_string())
        );

        // Alternative separators
        assert_eq!(
            matcher.normalize("123-456-789-00"),
            Some("12345678900".to_string())
        );
    }

    #[test]
    fn test_normalize_invalid() {
        let matcher = CpfMatcher::new();

        // Too short
        assert_eq!(matcher.normalize("123456789"), None);

        // Too long
        assert_eq!(matcher.normalize("1234567890012"), None);

        // Letters
        assert_eq!(matcher.normalize("abc"), None);
    }

    #[test]
    fn test_format() {
        let matcher = CpfMatcher::new();

        assert_eq!(
            matcher.format("12345678900"),
            Some("123.456.789-00".to_string())
        );

        assert_eq!(
            matcher.format("123.456.789-00"),
            Some("123.456.789-00".to_string())
        );
    }

    #[test]
    fn test_is_valid_rejects_same_digits() {
        let matcher = CpfMatcher::new();

        assert!(!matcher.is_valid("111.111.111-11"));
        assert!(!matcher.is_valid("000.000.000-00"));
        assert!(!matcher.is_valid("999.999.999-99"));
    }

    #[test]
    fn test_is_valid_with_generated() {
        let matcher = CpfMatcher::new();
        let valid_cpf = CpfMatcher::generate_valid();

        assert!(
            matcher.is_valid(&valid_cpf),
            "Generated CPF should be valid: {}",
            valid_cpf
        );
    }

    #[test]
    fn test_matches() {
        let matcher = CpfMatcher::new();

        assert!(matcher.matches("123.456.789-00", "12345678900"));
        assert!(matcher.matches("12345678900", " 123 456 789 00 "));
        assert!(!matcher.matches("123.456.789-00", "123.456.789-01"));
    }

    #[test]
    fn test_score() {
        let matcher = CpfMatcher::new();

        assert_eq!(matcher.score("123.456.789-00", "12345678900"), 1.0);
        assert_eq!(matcher.score("123.456.789-00", "987.654.321-00"), 0.0);
    }
}
