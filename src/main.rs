use clap::{Args, Parser, Subcommand};
use graphome::{
    convert, eigen_region, entropy, extract, gfa2gbz, make_sequence, map,
    pangenome_catalog::{catalog_from_runtime, FeatureCatalogManifest},
    pangenome_runtime::{build_runtime_from_gbz, load_topology_tsv, HaplotypeStep, HaplotypeWalk},
    viz, window,
};
use gbz::{Orientation, GBZ};
use simple_sds::serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
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
    /// Extract sequence based on coordinates
    MakeSequence(MakeSequenceArgs),
    /// Convert GFA file to GBZ format.
    Gfa2gbz(Gfa2gbzArgs),
    /// Perform eigendecomposition on a genomic region
    EigenRegion(EigenRegionArgs),
    /// Pangenome feature catalog + encoding utilities
    Pangenome(PangenomeArgs),
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
struct MakeSequenceArgs {
    /// The path to the graph file (GFA or GBZ)
    #[arg(long, value_name = "GRAPH")]
    gfa: PathBuf,
    /// The path to the PAF file (optional for GBZ files, required for GFA files)
    #[arg(long, value_name = "PAF")]
    paf: Option<PathBuf>,
    /// Reference assembly for coordinates (e.g., grch38, chm13)
    #[arg(long, value_name = "ASSEMBLY")]
    assembly: String,
    /// The genomic region, e.g. chr1:100000-200000
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

#[derive(Args, Debug)]
struct PangenomeArgs {
    #[command(subcommand)]
    command: PangenomeCommand,
}

#[derive(Subcommand, Debug)]
enum PangenomeCommand {
    /// Build a feature catalog from GBZ + topology TSV
    Catalog(PangenomeCatalogArgs),
    /// Encode one haplotype walk from GBZ using topology TSV
    Encode(PangenomeEncodeArgs),
    /// List GBZ walks as sample#phase#contig identifiers
    ListWalks(PangenomeListWalksArgs),
}

#[derive(Args, Debug)]
struct PangenomeCatalogArgs {
    /// Path to GBZ file
    #[arg(long, value_name = "GBZ")]
    gbz: PathBuf,
    /// Path to topology TSV
    #[arg(long, value_name = "TSV")]
    topology: PathBuf,
    /// Output directory for features.bin/traversals.bin/manifest.tsv
    #[arg(short, long, value_name = "DIR")]
    output: PathBuf,
    /// Manifest graph_build_id
    #[arg(long, default_value = "unknown")]
    graph_build_id: String,
    /// Manifest graph_construction_pipeline
    #[arg(long, default_value = "unknown")]
    graph_pipeline: String,
    /// Manifest reference_coordinates
    #[arg(long, default_value = "unknown")]
    reference_coordinates: String,
    /// Manifest hprc_release
    #[arg(long, default_value = "unknown")]
    hprc_release: String,
    /// Manifest snarl_decomposition_tool
    #[arg(long, default_value = "graphome")]
    snarl_tool: String,
}

#[derive(Args, Debug)]
struct PangenomeEncodeArgs {
    /// Path to GBZ file
    #[arg(long, value_name = "GBZ")]
    gbz: PathBuf,
    /// Path to topology TSV
    #[arg(long, value_name = "TSV")]
    topology: PathBuf,
    /// Haplotype walk ID: sample#phase#contig (use `list-walks`)
    #[arg(long, value_name = "WALK_ID")]
    walk_id: String,
    /// Output TSV path for encoded features (feature_index<TAB>value)
    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,
}

#[derive(Args, Debug)]
struct PangenomeListWalksArgs {
    /// Path to GBZ file
    #[arg(long, value_name = "GBZ")]
    gbz: PathBuf,
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn walk_id_from_metadata(metadata: &gbz::Metadata, path_id: usize) -> Option<String> {
    let path_name = metadata.path(path_id)?;
    let sample_name = metadata.sample_name(path_name.sample());
    let phase = path_name.phase();
    let contig_name = metadata.contig_name(path_name.contig());
    Some(format!("{}#{}#{}", sample_name, phase, contig_name))
}

fn load_walk_by_id(gbz: &GBZ, walk_id: &str) -> io::Result<HaplotypeWalk> {
    let metadata = gbz.metadata().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "GBZ metadata unavailable; cannot enumerate haplotype walks",
        )
    })?;

    for path_id in 0..metadata.paths() {
        let Some(id) = walk_id_from_metadata(metadata, path_id) else {
            continue;
        };
        if id != walk_id {
            continue;
        }

        let mut steps = Vec::new();
        if let Some(path_iter) = gbz.path(path_id, Orientation::Forward) {
            for (node_id, orientation) in path_iter {
                steps.push(HaplotypeStep {
                    node_id,
                    orientation,
                });
            }
        }
        return Ok(HaplotypeWalk { id, steps });
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("walk '{}' not found in GBZ", walk_id),
    ))
}

fn write_encoded_tsv(path: &Path, encoded: &[Option<f64>]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "feature_index\tvalue")?;
    for (idx, value) in encoded.iter().enumerate() {
        match value {
            Some(v) => writeln!(writer, "{}\t{}", idx, v)?,
            None => writeln!(writer, "{}\tNA", idx)?,
        }
    }
    writer.flush()
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
        Command::MakeSequence(args) => {
            let MakeSequenceArgs {
                gfa,
                paf,
                assembly,
                region,
                sample,
                output,
            } = args;
            let gfa_path = path_to_string(&gfa);
            let paf_path = paf.as_ref().map(|p| path_to_string(p)).unwrap_or_default();
            let output_path = path_to_string(&output);
            make_sequence::run_make_sequence(
                &gfa_path,
                &paf_path,
                &assembly,
                &region,
                &sample,
                &output_path,
            );
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
        Command::Pangenome(args) => match args.command {
            PangenomeCommand::Catalog(cargs) => {
                let gbz: GBZ =
                    serialize::load_from(&cargs.gbz).map_err(|e| io::Error::other(e.to_string()))?;
                let topology = load_topology_tsv(&path_to_string(&cargs.topology))?;
                let runtime = build_runtime_from_gbz(&topology, &gbz, HashMap::new());
                let catalog = catalog_from_runtime(&runtime);

                let haplotype_count = gbz
                    .metadata()
                    .map(|m| m.paths() as u32)
                    .unwrap_or_default();

                let manifest = FeatureCatalogManifest {
                    graph_build_id: cargs.graph_build_id,
                    graph_construction_pipeline: cargs.graph_pipeline,
                    reference_coordinates: cargs.reference_coordinates,
                    hprc_release: cargs.hprc_release,
                    haplotype_count,
                    snarl_decomposition_tool: cargs.snarl_tool,
                };
                catalog.write_dir_with_manifest(&cargs.output, &manifest)?;
                eprintln!(
                    "[INFO] Wrote pangenome catalog with {} features to {}",
                    catalog.features.len(),
                    cargs.output.display()
                );
            }
            PangenomeCommand::Encode(eargs) => {
                let gbz: GBZ =
                    serialize::load_from(&eargs.gbz).map_err(|e| io::Error::other(e.to_string()))?;
                let topology = load_topology_tsv(&path_to_string(&eargs.topology))?;
                let runtime = build_runtime_from_gbz(&topology, &gbz, HashMap::new());
                let walk = load_walk_by_id(&gbz, &eargs.walk_id)?;
                let encoded = runtime.encode_haplotype(&walk);
                write_encoded_tsv(&eargs.output, &encoded)?;
                eprintln!(
                    "[INFO] Encoded {} features for '{}' -> {}",
                    encoded.len(),
                    eargs.walk_id,
                    eargs.output.display()
                );
            }
            PangenomeCommand::ListWalks(largs) => {
                let gbz: GBZ =
                    serialize::load_from(&largs.gbz).map_err(|e| io::Error::other(e.to_string()))?;
                let metadata = gbz.metadata().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "GBZ metadata unavailable")
                })?;
                for path_id in 0..metadata.paths() {
                    if let Some(id) = walk_id_from_metadata(metadata, path_id) {
                        println!("{}", id);
                    }
                }
            }
        },
    }

    Ok(())
}
