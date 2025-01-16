use clap::{Parser, Subcommand};
use std::io;
use std::path::PathBuf;
use graphome::{convert, extract, eigen_print, dsbevd, window, entropy, map};


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

        Commands::Map { gfa, paf, map_command } => {
            match map_command {
                MapCommand::Node2coord { node_id } => {
                    // This calls the node->coord function from map.rs
                    {
                    use map::{GlobalData, parse_gfa_memmap, parse_paf_parallel, build_ref_trees, node_to_coords};
                    let mut global = GlobalData {
                        node_map: Default::default(),
                        path_map: Default::default(),
                        node_to_paths: Default::default(),
                        alignment_by_path: Default::default(),
                        ref_trees: Default::default(),
                    };
                    parse_gfa_memmap(gfa, &mut global);
                    parse_paf_parallel(paf, &mut global);
                    build_ref_trees(&mut global);
                
                    let results = node_to_coords(&global, node_id);
                    if results.is_empty() {
                        println!("No reference coords found for node {}", node_id);
                    } else {
                        for (chr, st, en) in results {
                            println!("{}:{}-{}", chr, st, en);
                        }
                    }
                    Ok(())
                }
                    
                },
                MapCommand::Coord2node { region } => {
                    // This calls the coord->node function from map.rs
                    map::coord2node(gfa, paf, region)?;

                    {
                    use map::{GlobalData, parse_gfa_memmap, parse_paf_parallel, build_ref_trees, coord_to_nodes};
                    let mut global = GlobalData {
                        node_map: Default::default(),
                        path_map: Default::default(),
                        node_to_paths: Default::default(),
                        alignment_by_path: Default::default(),
                        ref_trees: Default::default(),
                    };
                    parse_gfa_memmap(gfa, &mut global);
                    parse_paf_parallel(paf, &mut global);
                    build_ref_trees(&mut global);
                
                    let results = coord_to_nodes(&global, region);
                    // `coord_to_nodes` wants (GlobalData, &str, start, end).
                    // So parse region here or do a separate parse:
                    if let Some((chr, st, en)) = map::parse_region(region) {
                        let nodes = coord_to_nodes(&global, &chr, st, en);
                        if nodes.is_empty() {
                            println!("No nodes found for region {}:{}-{}", chr, st, en);
                        } else {
                            for r in nodes {
                                println!("path={} node={}({}) offsets=[{}..{}]",
                                         r.path_name, r.node_id,
                                         if r.node_orient {'+'} else {'-'},
                                         r.path_off_start, r.path_off_end);
                            }
                        }
                    } else {
                        eprintln!("Could not parse region format: {}", region);
                    }
                    Ok(())
                },
            }
        }
    }
    Ok(())
}
