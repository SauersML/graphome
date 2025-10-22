use clap::{Parser, Subcommand};
use std::io;
use std::path::PathBuf;
use graphome::{convert, extract, window, entropy, map, viz, embed, video, make_sequence, gfa2gbz};

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

    /// Map (with two sub-subcommands: node2coord, coord2node)
    Map {
        /// The path to the GFA file
        #[arg(long)]
        gfa: String,
        /// The path to the PAF file
        #[arg(long)]
        paf: String,
        #[command(subcommand)]
        map_command: MapCommand,
    },
    /// Visualize a range of nodes as a colored TGA
    Viz {
        /// Path to the GFA file
        #[arg(long)]
        gfa: String,

        /// Lowest node ID (string comparison)
        #[arg(long)]
        start_node: String,

        /// Highest node ID (string comparison)
        #[arg(long)]
        end_node: String,

        /// Path to output TGA file
        #[arg(long)]
        output_tga: String,

        /// Whether to run force-directed refinement after spectral
        #[arg(long, default_value_t = false)]
        force_directed: bool,
    },
    /// Embed a submatrix in 3D and visualize with rotation
    Embed {
        /// Path to the adjacency matrix file
        #[arg(short, long)]
        input: String,
        /// Start node index (inclusive)
        #[arg(long)]
        start_node: usize,
        /// End node index (inclusive)
        #[arg(long)]
        end_node: usize,
    },
    /// Extract sequence based on coordinates
    MakeSequence {
        /// The path to the GFA file
        #[arg(long)]
        gfa: String,
        /// The path to the PAF file
        #[arg(long)]
        paf: String,
        /// The region in hg38, e.g. grch38#chr1:100000-200000
        #[arg(long)]
        region: String,
        /// Sample name to use in the output file
        #[arg(long)]
        sample: String,
        /// Path to output FASTA file (will append _samplename_path_start-end.fa)
        #[arg(long)]
        output: String,
    },
    /// Convert GFA file to GBZ format.
    Gfa2gbz {
        /// Path to the input GFA file.
        #[arg(short, long)]
        input: String,
    },
}

#[derive(Subcommand)]
enum MapCommand {
    /// Node->Coordinate
    Node2coord {
        /// The node ID in the GFA
        node_id: String,
    },
    /// Coordinate->Node
    Coord2node {
        /// The region in hg38, e.g. grch38#chr1:100000-200000
        region: String,
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
            output: _,
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

        // The "Map" command has sub-subcommands: node2coord, coord2node
        Commands::Map { gfa, paf, map_command } => {
            match map_command {
                MapCommand::Node2coord { node_id } => {
                    map::run_node2coord(gfa, paf, node_id);
                },
                MapCommand::Coord2node { region } => {
                    map::run_coord2node(gfa, paf, region);
                },
            }
        }
        Commands::Viz { gfa, start_node, end_node, output_tga, force_directed } => {
            if let Err(err) = viz::run_viz(gfa, start_node, end_node, output_tga, *force_directed) {
                eprintln!("[viz error] {}", err);
                std::process::exit(1);
            }
        }
        Commands::Embed {
            input,
            start_node,
            end_node,
        } => {
            let points = embed::embed(*start_node, *end_node, input)?;
            video::make_video(&points).map_err(std::io::Error::other)?;
        }
        Commands::MakeSequence { gfa, paf, region, sample, output } => {
            make_sequence::run_make_sequence(gfa, paf, region, sample, output);
        }
        Commands::Gfa2gbz { input } => {
            gfa2gbz::run_gfa2gbz(input);
        }
    }
    Ok(())
}
