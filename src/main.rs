use clap::{Args, Parser, Subcommand};
use graphome::{
    convert, eigen_region, embed, entropy, extract, gfa2gbz, make_sequence, map, video, viz, window,
};
use std::io;
use std::path::{Path, PathBuf};

/// Graphome: GFA to Adjacency Matrix Converter and Analyzer
#[derive(Parser, Debug)]
#[command(
    name = "graphome",
    about = "Convert GFA or GBZ graphs into adjacency data, run analyses, and visualise results",
    version,
    propagate_version = true,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convert a graph (GFA or GBZ) to an adjacency matrix in edge list format
    Convert(ConvertArgs),
    /// Extract adjacency submatrix for a node range and perform eigenanalysis
    Extract(ExtractArgs),
    /// Extract adjacency and Laplacian matrices and save as .npy files
    ExtractMatrices(ExtractMatricesArgs),
    /// Extract overlapping windows of Laplacian matrices in parallel
    ExtractWindows(ExtractWindowsArgs),
    /// Analyse eigenvalues from extracted windows and compute NGEC
    AnalyzeWindows(AnalyzeWindowsArgs),
    /// Map (with two subcommands: node2coord, coord2node)
    Map(MapArgs),
    /// Visualise a range of nodes as a coloured TGA
    Viz(VizArgs),
    /// Embed a submatrix in 3D and visualise with rotation
    Embed(EmbedArgs),
    /// Extract sequence based on coordinates
    MakeSequence(MakeSequenceArgs),
    /// Convert GFA file to GBZ format.
    Gfa2gbz(Gfa2gbzArgs),
    /// Perform eigendecomposition on a genomic region
    EigenRegion(EigenRegionArgs),
}

#[derive(Args, Debug, Clone)]
struct NodeRange {
    /// Start node index (inclusive)
    #[arg(long, value_name = "START")]
    start_node: usize,
    /// End node index (inclusive)
    #[arg(long, value_name = "END")]
    end_node: usize,
}

impl NodeRange {
    fn bounds(&self) -> (usize, usize) {
        (self.start_node, self.end_node)
    }
}

#[derive(Args, Debug)]
struct ConvertArgs {
    /// Path to the input graph file (GFA or GBZ)
    #[arg(short, long, value_name = "GRAPH")]
    input: PathBuf,
    /// Path to the output adjacency matrix file
    #[arg(
        short,
        long,
        value_name = "EDGE_LIST",
        default_value = "adjacency_matrix.bin"
    )]
    output: PathBuf,
}

#[derive(Args, Debug)]
struct ExtractArgs {
    /// Path to the adjacency matrix file
    #[arg(short, long, value_name = "EDGE_LIST")]
    input: PathBuf,
    #[command(flatten)]
    range: NodeRange,
}

#[derive(Args, Debug)]
struct ExtractMatricesArgs {
    /// Path to the adjacency matrix file
    #[arg(short, long, value_name = "EDGE_LIST")]
    input: PathBuf,
    #[command(flatten)]
    range: NodeRange,
    /// Output directory for .npy files
    #[arg(short, long, value_name = "DIR")]
    output: PathBuf,
}

#[derive(Args, Debug)]
struct ExtractWindowsArgs {
    /// Path to the adjacency matrix file
    #[arg(short, long, value_name = "EDGE_LIST")]
    input: PathBuf,
    #[command(flatten)]
    range: NodeRange,
    /// Size of each window
    #[arg(long, value_name = "SIZE")]
    window_size: usize,
    /// Size of overlap between windows
    #[arg(long, value_name = "OVERLAP")]
    overlap: usize,
    /// Output directory for .npy files
    #[arg(short, long, value_name = "DIR")]
    output: PathBuf,
}

#[derive(Args, Debug)]
struct AnalyzeWindowsArgs {
    /// Directory containing window_* subdirectories with eigenvalues.npy files
    #[arg(short, long, value_name = "DIR")]
    input: PathBuf,
}

#[derive(Args, Debug)]
struct MapArgs {
    /// The path to the graph file (GFA or GBZ)
    #[arg(long, value_name = "GRAPH")]
    gfa: PathBuf,
    /// The path to the PAF file (optional, not needed for GBZ files)
    #[arg(long, value_name = "PAF")]
    paf: Option<PathBuf>,
    #[command(subcommand)]
    map_command: MapCommand,
}

#[derive(Subcommand, Debug)]
enum MapCommand {
    /// Node->Coordinate
    Node2coord {
        /// The node ID in the graph
        #[arg(value_name = "NODE_ID")]
        node_id: String,
    },
    /// Coordinate->Node
    Coord2node {
        /// The region in hg38, e.g. grch38#chr1:100000-200000
        #[arg(value_name = "REGION")]
        region: String,
    },
}

#[derive(Args, Debug)]
struct VizArgs {
    /// Path to the graph file (GFA or GBZ)
    #[arg(long, value_name = "GRAPH")]
    gfa: PathBuf,
    /// Lowest node ID (string comparison)
    #[arg(long, value_name = "NODE_ID")]
    start_node: String,
    /// Highest node ID (string comparison)
    #[arg(long, value_name = "NODE_ID")]
    end_node: String,
    /// Path to output TGA file
    #[arg(long, value_name = "FILE")]
    output_tga: PathBuf,
    /// Whether to run force-directed refinement after spectral
    #[arg(long, default_value_t = false)]
    force_directed: bool,
}

#[derive(Args, Debug)]
struct EmbedArgs {
    /// Path to the adjacency matrix file
    #[arg(short, long, value_name = "EDGE_LIST")]
    input: PathBuf,
    #[command(flatten)]
    range: NodeRange,
}

#[derive(Args, Debug)]
struct MakeSequenceArgs {
    /// The path to the graph file (GFA or GBZ)
    #[arg(long, value_name = "GRAPH")]
    gfa: PathBuf,
    /// The path to the PAF file
    #[arg(long, value_name = "PAF")]
    paf: PathBuf,
    /// The region in hg38, e.g. grch38#chr1:100000-200000
    #[arg(long, value_name = "REGION")]
    region: String,
    /// Sample name to use in the output file
    #[arg(long, value_name = "SAMPLE")]
    sample: String,
    /// Path to output FASTA file (will append _samplename_path_start-end.fa)
    #[arg(long, value_name = "PATH")]
    output: PathBuf,
}

#[derive(Args, Debug)]
struct Gfa2gbzArgs {
    /// Path to the input GFA file.
    #[arg(short, long, value_name = "GFA")]
    input: PathBuf,
}

#[derive(Args, Debug)]
struct EigenRegionArgs {
    /// Path to GBZ file (or S3 URL)
    #[arg(long, value_name = "GRAPH")]
    gfa: PathBuf,
    /// Genomic region (chr:start-end)
    #[arg(long, value_name = "REGION")]
    region: String,
    /// Show terminal visualisation
    #[arg(long, default_value_t = false)]
    viz: bool,
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Convert(args) => {
            let ConvertArgs { input, output } = args;
            convert::convert_graph_to_edge_list(&input, &output)?;
        }
        Command::Extract(args) => {
            let ExtractArgs { input, range } = args;
            let (start_node, end_node) = range.bounds();
            extract::extract_and_analyze_submatrix(&input, start_node, end_node)?;
        }
        Command::ExtractMatrices(args) => {
            let ExtractMatricesArgs {
                input,
                range,
                output,
            } = args;
            let (start_node, end_node) = range.bounds();
            extract::extract_and_save_matrices(&input, start_node, end_node, &output)?;
        }
        Command::ExtractWindows(args) => {
            let ExtractWindowsArgs {
                input,
                range,
                window_size,
                overlap,
                output,
            } = args;
            let (start_node, end_node) = range.bounds();
            let config = window::WindowConfig::new(start_node, end_node, window_size, overlap);
            window::parallel_extract_windows(&input, &output, config)?;
        }
        Command::AnalyzeWindows(args) => {
            let AnalyzeWindowsArgs { input } = args;
            entropy::analyze_windows(input.as_path())?;
        }
        Command::Map(args) => {
            let MapArgs {
                gfa,
                paf,
                map_command,
            } = args;
            let gfa_path = path_to_string(&gfa);
            let paf_path = paf.map(|p| path_to_string(&p));
            match map_command {
                MapCommand::Node2coord { node_id } => {
                    map::run_node2coord(&gfa_path, paf_path.as_deref().unwrap_or(""), &node_id);
                }
                MapCommand::Coord2node { region } => {
                    map::run_coord2node(&gfa_path, paf_path.as_deref().unwrap_or(""), &region);
                }
            }
        }
        Command::Viz(args) => {
            let VizArgs {
                gfa,
                start_node,
                end_node,
                output_tga,
                force_directed,
            } = args;
            let gfa_path = path_to_string(&gfa);
            let output_path = path_to_string(&output_tga);
            if let Err(err) = viz::run_viz(
                &gfa_path,
                &start_node,
                &end_node,
                &output_path,
                force_directed,
            ) {
                eprintln!("[viz error] {}", err);
                std::process::exit(1);
            }
        }
        Command::Embed(args) => {
            let EmbedArgs { input, range } = args;
            let (start_node, end_node) = range.bounds();
            let input_path = path_to_string(&input);
            let points = embed::embed(start_node, end_node, &input_path)?;
            video::make_video(&points).map_err(std::io::Error::other)?;
        }
        Command::MakeSequence(args) => {
            let MakeSequenceArgs {
                gfa,
                paf,
                region,
                sample,
                output,
            } = args;
            let gfa_path = path_to_string(&gfa);
            let paf_path = path_to_string(&paf);
            let output_path = path_to_string(&output);
            make_sequence::run_make_sequence(&gfa_path, &paf_path, &region, &sample, &output_path);
        }
        Command::Gfa2gbz(args) => {
            let Gfa2gbzArgs { input } = args;
            let input_path = path_to_string(&input);
            gfa2gbz::run_gfa2gbz(&input_path);
        }
        Command::EigenRegion(args) => {
            let EigenRegionArgs { gfa, region, viz } = args;
            let gfa_path = path_to_string(&gfa);
            if let Err(err) = eigen_region::run_eigen_region(&gfa_path, &region, viz) {
                eprintln!("[eigen-region error] {}", err);
                std::process::exit(1);
            }
        }
    }

    Ok(())
}
