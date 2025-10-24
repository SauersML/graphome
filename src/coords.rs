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
/// When the start coordinate is `0`, the caller is using 0-based inclusive
/// intervals (common when interoperating with sampling tools). In that case we
/// interpret the range as [0, end] (inclusive) and convert it to [0, end+1)
/// internally.
///
/// This is the ONLY place we convert from user coordinates to 0-based.
pub fn parse_user_region(region_str: &str) -> Option<(String, usize, usize)> {
    // e.g. "grch38#chr1:120616922-120626943"
    let (chr_part, rng_part) = region_str.split_once(':')?;
    let (s, e) = rng_part.split_once('-')?;
    let start_raw = s.parse::<usize>().ok()?;
    let end_raw = e.parse::<usize>().ok()?;

    // Validate
    if end_raw < start_raw {
        return None;
    }

    // Convert to 0-based half-open intervals.
    let (start_zero, end_zero) = if start_raw == 0 {
        let end_exclusive = end_raw.checked_add(1)?;
        (0, end_exclusive)
    } else {
        (start_raw - 1, end_raw)
    };

    Some((chr_part.to_string(), start_zero, end_zero))
}

/// Extract the raw start and end coordinates provided by the user.
///
/// This preserves the original numbering scheme (0-based or 1-based) so that
/// we can mirror the input in logs, filenames, and FASTA headers.
pub fn user_region_bounds(region_str: &str) -> Option<(usize, usize)> {
    let (_, rng_part) = region_str.split_once(':')?;
    let (s, e) = rng_part.split_once('-')?;
    let start = s.parse::<usize>().ok()?;
    let end = e.parse::<usize>().ok()?;
    Some((start, end))
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
    fn test_zero_based_input() {
        let (chr, start, end) = parse_user_region("chr10:0-548").unwrap();
        assert_eq!(chr, "chr10");
        assert_eq!(start, 0);
        assert_eq!(end, 549);

        let (user_start, user_end) = user_region_bounds("chr10:0-548").unwrap();
        assert_eq!(user_start, 0);
        assert_eq!(user_end, 548);
    }

    #[test]
    fn test_reject_invalid_range() {
        assert!(parse_user_region("chr1:200-100").is_none());
    }
}
