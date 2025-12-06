//! CNPJ (Cadastro Nacional da Pessoa Jurídica) matcher
//!
//! Validation and normalization for Brazilian company tax IDs.

use tracing::debug;

/// CNPJ matcher with validation and normalization
#[derive(Debug, Clone, Default)]
pub struct CnpjMatcher {
    /// Whether to validate check digits
    validate_digits: bool,
}

impl CnpjMatcher {
    /// Create a new CNPJ matcher with validation enabled
    pub fn new() -> Self {
        Self {
            validate_digits: true,
        }
    }

    /// Create a matcher without check digit validation (faster)
    pub fn without_validation() -> Self {
        Self {
            validate_digits: false,
        }
    }

    /// Normalize a CNPJ to 14 digits only
    pub fn normalize(&self, cnpj: &str) -> Option<String> {
        let digits: String = cnpj.chars().filter(|c| c.is_ascii_digit()).collect();

        if digits.len() != 14 {
            debug!(
                input = cnpj,
                digits_len = digits.len(),
                "Invalid CNPJ length"
            );
            return None;
        }

        Some(digits)
    }

    /// Validate a CNPJ including check digits
    pub fn is_valid(&self, cnpj: &str) -> bool {
        let Some(normalized) = self.normalize(cnpj) else {
            return false;
        };

        // Check for all same digits (invalid)
        if normalized
            .chars()
            .all(|c| c == normalized.chars().next().unwrap())
        {
            debug!(cnpj = %normalized, "CNPJ with all same digits");
            return false;
        }

        if self.validate_digits {
            self.validate_check_digits(&normalized)
        } else {
            true
        }
    }

    /// Validate check digits using the official algorithm
    fn validate_check_digits(&self, cnpj: &str) -> bool {
        let digits: Vec<u32> = cnpj.chars().filter_map(|c| c.to_digit(10)).collect();

        if digits.len() != 14 {
            return false;
        }

        // First check digit
        let weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2];
        let sum1: u32 = digits[..12]
            .iter()
            .zip(weights1.iter())
            .map(|(d, w)| d * w)
            .sum();
        let remainder1 = sum1 % 11;
        let check1 = if remainder1 < 2 { 0 } else { 11 - remainder1 };

        if digits[12] != check1 {
            return false;
        }

        // Second check digit
        let weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2];
        let sum2: u32 = digits[..13]
            .iter()
            .zip(weights2.iter())
            .map(|(d, w)| d * w)
            .sum();
        let remainder2 = sum2 % 11;
        let check2 = if remainder2 < 2 { 0 } else { 11 - remainder2 };

        digits[13] == check2
    }

    /// Format a CNPJ with standard punctuation: XX.XXX.XXX/XXXX-XX
    pub fn format(&self, cnpj: &str) -> Option<String> {
        let normalized = self.normalize(cnpj)?;
        Some(format!(
            "{}.{}.{}/{}-{}",
            &normalized[0..2],
            &normalized[2..5],
            &normalized[5..8],
            &normalized[8..12],
            &normalized[12..14]
        ))
    }

    /// Check if two CNPJs match (normalized comparison)
    pub fn matches(&self, cnpj1: &str, cnpj2: &str) -> bool {
        match (self.normalize(cnpj1), self.normalize(cnpj2)) {
            (Some(n1), Some(n2)) => n1 == n2,
            _ => false,
        }
    }

    /// Calculate similarity score between two CNPJs
    pub fn score(&self, cnpj1: &str, cnpj2: &str) -> f64 {
        if self.matches(cnpj1, cnpj2) {
            1.0
        } else {
            0.0
        }
    }

    /// Extract the root CNPJ (first 8 digits - company identifier)
    pub fn root(&self, cnpj: &str) -> Option<String> {
        self.normalize(cnpj).map(|n| n[0..8].to_string())
    }

    /// Extract the branch number (4 digits after root)
    pub fn branch(&self, cnpj: &str) -> Option<String> {
        self.normalize(cnpj).map(|n| n[8..12].to_string())
    }

    /// Check if two CNPJs belong to the same company (same root)
    pub fn same_company(&self, cnpj1: &str, cnpj2: &str) -> bool {
        match (self.root(cnpj1), self.root(cnpj2)) {
            (Some(r1), Some(r2)) => r1 == r2,
            _ => false,
        }
    }

    /// Check if this is a headquarters CNPJ (branch = 0001)
    pub fn is_headquarters(&self, cnpj: &str) -> bool {
        self.branch(cnpj).map(|b| b == "0001").unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_various_formats() {
        let matcher = CnpjMatcher::new();

        assert_eq!(
            matcher.normalize("11.222.333/0001-81"),
            Some("11222333000181".to_string())
        );
        assert_eq!(
            matcher.normalize("11222333000181"),
            Some("11222333000181".to_string())
        );
        assert_eq!(
            matcher.normalize(" 11 222 333 0001 81 "),
            Some("11222333000181".to_string())
        );
    }

    #[test]
    fn test_normalize_invalid() {
        let matcher = CnpjMatcher::new();

        assert_eq!(matcher.normalize(""), None);
        assert_eq!(matcher.normalize("1234567890"), None);
        assert_eq!(matcher.normalize("123456789012345"), None);
    }

    #[test]
    fn test_is_valid_rejects_same_digits() {
        let matcher = CnpjMatcher::new();

        assert!(!matcher.is_valid("00.000.000/0000-00"));
        assert!(!matcher.is_valid("11.111.111/1111-11"));
        assert!(!matcher.is_valid("99999999999999"));
    }

    #[test]
    fn test_format() {
        let matcher = CnpjMatcher::new();

        assert_eq!(
            matcher.format("11222333000181"),
            Some("11.222.333/0001-81".to_string())
        );
        assert_eq!(
            matcher.format("11.222.333/0001-81"),
            Some("11.222.333/0001-81".to_string())
        );
    }

    #[test]
    fn test_matches() {
        let matcher = CnpjMatcher::new();

        assert!(matcher.matches("11.222.333/0001-81", "11222333000181"));
        assert!(matcher.matches("11 222 333 0001 81", "11.222.333/0001-81"));
        assert!(!matcher.matches("11.222.333/0001-81", "11.222.333/0002-62"));
    }

    #[test]
    fn test_root_and_branch() {
        let matcher = CnpjMatcher::new();

        assert_eq!(
            matcher.root("11.222.333/0001-81"),
            Some("11222333".to_string())
        );
        assert_eq!(
            matcher.branch("11.222.333/0001-81"),
            Some("0001".to_string())
        );
        assert_eq!(
            matcher.branch("11.222.333/0002-62"),
            Some("0002".to_string())
        );
    }

    #[test]
    fn test_same_company() {
        let matcher = CnpjMatcher::new();

        assert!(matcher.same_company("11.222.333/0001-81", "11.222.333/0002-62"));
        assert!(!matcher.same_company("11.222.333/0001-81", "99.888.777/0001-00"));
    }

    #[test]
    fn test_is_headquarters() {
        let matcher = CnpjMatcher::new();

        assert!(matcher.is_headquarters("11.222.333/0001-81"));
        assert!(!matcher.is_headquarters("11.222.333/0002-62"));
    }
}
