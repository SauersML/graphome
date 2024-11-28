use clap::{Parser, Subcommand};
use std::io;
use graphome::{convert, extract, eigen, dsbevd};

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
    }
    Ok(())
}
