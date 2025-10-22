use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

const BAR_CHARS: &str = "█▓░";

/// Create a byte-based progress bar with a consistent style. If `total_bytes`
/// is `None`, the progress bar falls back to a spinner that still reports the
/// number of processed bytes and throughput.
pub fn byte_progress_bar(label: impl Into<String>, total_bytes: Option<u64>) -> ProgressBar {
    let label = label.into();
    match total_bytes {
        Some(total) => {
            let pb = ProgressBar::new(total);
            pb.set_style(
                ProgressStyle::with_template(
                    "{prefix:.bold.dim} {spinner:.green} [{elapsed_precise}] {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({binary_bytes_per_sec}) {msg}",
                )
                .unwrap()
                .progress_chars(BAR_CHARS),
            );
            pb.set_prefix(label);
            pb.enable_steady_tick(Duration::from_millis(75));
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template(
                    "{prefix:.bold.dim} {spinner:.green} [{elapsed_precise}] {bytes} downloaded ({binary_bytes_per_sec}) {msg}",
                )
                .unwrap(),
            );
            pb.set_prefix(label);
            pb.enable_steady_tick(Duration::from_millis(75));
            pb
        }
    }
}

/// Create a progress bar that tracks how many logical items (lines, segments,
/// records, …) have been processed. When the total is unknown the bar falls
/// back to a spinner that still tracks the item count.
pub fn count_progress_bar(
    label: impl Into<String>,
    unit_label: &str,
    total_items: Option<u64>,
) -> ProgressBar {
    let label = label.into();
    match total_items {
        Some(total) => {
            let pb = ProgressBar::new(total);
            let template = format!(
                "{{prefix:.bold.dim}} {{spinner:.green}} [{{elapsed_precise}}] {{wide_bar:.cyan/blue}} {{pos}}/{{len}} {unit_label} ({{eta}} @ {{per_sec}} {unit_label}/s) {{msg}}",
            );
            pb.set_style(
                ProgressStyle::with_template(&template)
                    .unwrap()
                    .progress_chars(BAR_CHARS),
            );
            pb.set_prefix(label);
            pb.enable_steady_tick(Duration::from_millis(75));
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            let template = format!(
                "{{prefix:.bold.dim}} {{spinner:.green}} [{{elapsed_precise}}] {{pos}} {unit_label} processed ({{per_sec}} {unit_label}/s) {{msg}}",
            );
            pb.set_style(ProgressStyle::with_template(&template).unwrap());
            pb.set_prefix(label);
            pb.enable_steady_tick(Duration::from_millis(75));
            pb
        }
    }
}

/// Create a spinner-style progress bar with a uniform appearance.
pub fn spinner_progress(label: impl Into<String>, message: impl Into<String>) -> ProgressBar {
    let label = label.into();
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:.bold.dim} {spinner:.green} {msg} [{elapsed_precise}]",
        )
        .unwrap(),
    );
    pb.set_prefix(label);
    pb.set_message(message.into());
    pb.enable_steady_tick(Duration::from_millis(75));
    pb
}
