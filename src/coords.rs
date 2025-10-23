/// Type-safe coordinate system to prevent off-by-one errors.
/// 
/// DESIGN PRINCIPLE: Coordinate conversion happens ONLY at data interfaces:
/// - parse_user_region(): Convert user input (1-based) -> internal (0-based)
/// - format_for_user(): Convert internal (0-based) -> user output (1-based)
/// 
/// Everything else uses 0-based half-open intervals [start, end).

/// Parse user input region string and convert to 0-based half-open coordinates.
/// 
/// User input: "chr1:100-200" means 1-based inclusive [100, 200]
/// Returns: (chr, start=99, end=200) as 0-based half-open [99, 200)
/// 
/// This is the ONLY place we convert from 1-based to 0-based.
pub fn parse_user_region(region_str: &str) -> Option<(String, usize, usize)> {
    // e.g. "grch38#chr1:120616922-120626943"
    let (chr_part, rng_part) = region_str.split_once(':')?;
    let (s, e) = rng_part.split_once('-')?;
    let start_one_based = s.parse::<usize>().ok()?;
    let end_one_based = e.parse::<usize>().ok()?;
    
    // Validate
    if start_one_based == 0 || end_one_based == 0 || start_one_based > end_one_based {
        return None;
    }
    
    // Convert to 0-based half-open: [start-1, end)
    let start_zero = start_one_based - 1;
    let end_zero = end_one_based; // end becomes exclusive
    
    Some((chr_part.to_string(), start_zero, end_zero))
}

/// Format internal 0-based coordinates for user output.
/// 
/// Internal: [99, 200) means 0-based half-open
/// Returns: "100-200" as 1-based inclusive
/// 
/// This is the ONLY place we convert from 0-based to 1-based.
pub fn format_for_user(start_zero: usize, end_zero: usize) -> String {
    let start_one = start_zero + 1;
    let end_one = end_zero; // end is already exclusive, becomes inclusive
    format!("{}-{}", start_one, end_one)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_user_region() {
        // User: chr1:1-1000 (1-based inclusive)
        let (chr, start, end) = parse_user_region("chr1:1-1000").unwrap();
        assert_eq!(chr, "chr1");
        assert_eq!(start, 0); // 0-based
        assert_eq!(end, 1000); // exclusive
        assert_eq!(end - start, 1000); // length
    }

    #[test]
    fn test_full_chromosome() {
        // chr20: 1-64444167 (1-based)
        let (chr, start, end) = parse_user_region("grch38#chr20:1-64444167").unwrap();
        assert_eq!(chr, "grch38#chr20");
        assert_eq!(start, 0);
        assert_eq!(end, 64444167);
        assert_eq!(end - start, 64444167);
    }

    #[test]
    fn test_format_for_user() {
        // Internal: [0, 1000)
        let formatted = format_for_user(0, 1000);
        assert_eq!(formatted, "1-1000");
    }

    #[test]
    fn test_roundtrip() {
        let input = "chr1:100-200";
        let (chr, start, end) = parse_user_region(input).unwrap();
        let formatted = format!("{}:{}", chr, format_for_user(start, end));
        assert_eq!(formatted, input);
    }

    #[test]
    fn test_reject_zero() {
        assert!(parse_user_region("chr1:0-100").is_none());
        assert!(parse_user_region("chr1:1-0").is_none());
    }

    #[test]
    fn test_reject_invalid_range() {
        assert!(parse_user_region("chr1:200-100").is_none());
    }
}
