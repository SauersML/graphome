use clap::{Parser, Subcommand};
use std::io;
use std::path::PathBuf;
use graphome::{convert, extract, eigen, dsbevd, window, entropy};

/// Graphome: GFA to Adjacency Matrix Converter and Analyzer
#[derive(Parser)]
#[command(
    name = "graphome",
    about = "Convert GFA file to adjacency matrix, extract submatrices, and perform analysis"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert GFA file to adjacency matrix in edge list format
    Convert {
        /// Path to the input GFA file
        #[arg(short, long)]
        input: String,
        /// Path to the output adjacency matrix file
        #[arg(short, long, default_value = "adjacency_matrix.bin")]
        output: String,
    },
    /// Extract adjacency submatrix for a node range and perform eigenanalysis
    Extract {
        /// Path to the adjacency matrix file
        #[arg(short, long)]
        input: String,
        /// Start node index (inclusive)
        #[arg(long)]
        start_node: usize,
        /// End node index (inclusive)
        #[arg(long)]
        end_node: usize,
        /// Output .gam file
        #[arg(short, long)]
        output: String,
    },
    /// Extract adjacency and Laplacian matrices and save as .npy files
    ExtractMatrices {
        /// Path to the adjacency matrix file
        #[arg(short, long)]
        input: String,
        /// Start node index (inclusive)
        #[arg(long)]
        start_node: usize,
        /// End node index (inclusive)
        #[arg(long)]
        end_node: usize,
        /// Output directory for .npy files
        #[arg(short, long)]
        output: String,
    },
    /// Extract overlapping windows of Laplacian matrices in parallel
    ExtractWindows {
        /// Path to the adjacency matrix file
        #[arg(short, long)]
        input: String,
        /// Start node index (inclusive)
        #[arg(long)]
        start_node: usize,
        /// End node index (inclusive)
        #[arg(long)]
        end_node: usize,
        /// Size of each window
        #[arg(long)]
        window_size: usize,
        /// Size of overlap between windows
        #[arg(long)]
        overlap: usize,
        /// Output directory for .npy files
        #[arg(short, long)]
        output: String,
    },
    /// Analyze eigenvalues from extracted windows and compute NGEC
    AnalyzeWindows {
        /// Directory containing window_* subdirectories with eigenvalues.npy files
        #[arg(short, long)]
        input: PathBuf,
    },
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Convert { input, output } => {
            convert::convert_gfa_to_edge_list(input, output)?;
        }
        Commands::Extract {
            input,
            start_node,
            end_node,
            output,
        } => {
            extract::extract_and_analyze_submatrix(input, *start_node, *end_node)?;
        }
        Commands::ExtractMatrices {
            input,
            start_node,
            end_node,
            output,
        } => {
            extract::extract_and_save_matrices(input, *start_node, *end_node, output)?;
        }
        Commands::ExtractWindows {
            input,
            start_node,
            end_node,
            window_size,
            overlap,
            output,
        } => {
            let config = window::WindowConfig::new(
                *start_node,
                *end_node,
                *window_size,
                *overlap,
            );
            window::parallel_extract_windows(input, output, config)?;
        }
        Commands::AnalyzeWindows { input } => {
            entropy::analyze_windows(input)?;
        }
    }
    Ok(())
}
