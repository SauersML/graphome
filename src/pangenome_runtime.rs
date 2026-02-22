use crate::pangenome_features::{
    encode_haploid_acyclic_with_reference, encode_haploid_cyclic, sum_diploid_site, FeatureBuilder,
    FeatureKind, FeatureSchema, SiteClass, Snarl, SnarlType, TraversalCondition,
};
use gbwt::{Orientation, GBZ};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HaplotypeStep {
    pub node_id: usize,
    pub orientation: Orientation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HaplotypeWalk {
    pub id: String,
    pub steps: Vec<HaplotypeStep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnarlTopology {
    pub id: String,
    pub snarl_type: SnarlType,
    pub entry_node: usize,
    pub exit_node: usize,
    pub genomic_region: Option<String>,
    pub children: Vec<SnarlTopology>,
}

impl SnarlTopology {
    pub fn acyclic(
        id: impl Into<String>,
        entry_node: usize,
        exit_node: usize,
        children: Vec<SnarlTopology>,
    ) -> Self {
        Self {
            id: id.into(),
            snarl_type: SnarlType::Acyclic,
            entry_node,
            exit_node,
            genomic_region: None,
            children,
        }
    }

    pub fn cyclic(id: impl Into<String>, entry_node: usize, exit_node: usize) -> Self {
        Self {
            id: id.into(),
            snarl_type: SnarlType::Cyclic,
            entry_node,
            exit_node,
            genomic_region: None,
            children: Vec::new(),
        }
    }

    pub fn with_genomic_region(mut self, genomic_region: impl Into<String>) -> Self {
        self.genomic_region = Some(genomic_region.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct InferredPanel {
    pub roots: Vec<Snarl>,
    pub lookup: HashMap<String, SnarlLookup>,
}

#[derive(Debug, Clone)]
pub struct SnarlLookup {
    pub snarl_id: String,
    pub snarl_type: SnarlType,
    pub entry_node: usize,
    pub exit_node: usize,
    pub child_ids: Vec<String>,
    pub flat_alleles: Vec<String>,
    pub skeleton_alleles: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FeatureRuntime {
    pub schema: FeatureSchema,
    pub panel: InferredPanel,
}

pub fn load_topology_tsv(path: &str) -> io::Result<Vec<SnarlTopology>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut rows = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split('\t').collect();
        if fields.len() < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {}: expected at least 5 tab-separated fields", lineno + 1),
            ));
        }
        let id = fields[0].to_string();
        let snarl_type = parse_snarl_type(fields[1], lineno + 1)?;
        let entry_node = parse_usize_field(fields[2], "entry_node", lineno + 1)?;
        let exit_node = parse_usize_field(fields[3], "exit_node", lineno + 1)?;
        let parent_id = parse_optional_string(fields[4]);
        let genomic_region = if fields.len() >= 6 {
            parse_optional_string(fields[5])
        } else {
            None
        };
        rows.push(TopologyRow {
            id,
            snarl_type,
            entry_node,
            exit_node,
            parent_id,
            genomic_region,
        });
    }

    let mut child_map: HashMap<String, Vec<String>> = HashMap::new();
    let mut row_map: HashMap<String, TopologyRow> = HashMap::new();
    let mut root_ids = Vec::new();
    for row in rows {
        if let Some(parent_id) = &row.parent_id {
            child_map
                .entry(parent_id.clone())
                .or_default()
                .push(row.id.clone());
        } else {
            root_ids.push(row.id.clone());
        }
        if row_map.insert(row.id.clone(), row).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "duplicate snarl id in topology TSV",
            ));
        }
    }

    let mut built = HashMap::new();
    let mut roots = Vec::new();
    for root_id in root_ids {
        roots.push(build_topology_node(&root_id, &row_map, &child_map, &mut built)?);
    }
    Ok(roots)
}

pub fn extract_haplotype_walks_from_gbz(gbz: &GBZ) -> Vec<HaplotypeWalk> {
    let Some(metadata) = gbz.metadata() else {
        return Vec::new();
    };

    let mut walks = Vec::new();
    for (path_id, path_name) in metadata.path_iter().enumerate() {
        let sample_name = metadata.sample_name(path_name.sample());
        let phase = path_name.phase();
        let contig_name = metadata.contig_name(path_name.contig());
        let id = format!("{}#{}#{}", sample_name, phase, contig_name);
        let mut steps = Vec::new();
        if let Some(path_iter) = gbz.path(path_id, Orientation::Forward) {
            for (node_id, orientation) in path_iter {
                steps.push(HaplotypeStep {
                    node_id,
                    orientation,
                });
            }
        }
        walks.push(HaplotypeWalk { id, steps });
    }
    walks
}

fn parse_snarl_type(raw: &str, line: usize) -> io::Result<SnarlType> {
    match raw {
        "acyclic" | "Acyclic" => Ok(SnarlType::Acyclic),
        "cyclic" | "Cyclic" => Ok(SnarlType::Cyclic),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("line {}: invalid snarl type '{}'", line, raw),
        )),
    }
}

fn parse_usize_field(raw: &str, field: &str, line: usize) -> io::Result<usize> {
    raw.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("line {}: invalid {} '{}'", line, field, raw),
        )
    })
}

fn parse_optional_string(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "." || trimmed == "NA" {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub fn infer_snarl_panel(topology_roots: &[SnarlTopology], panel_walks: &[HaplotypeWalk]) -> InferredPanel {
    let mut lookup = HashMap::new();
    let roots = topology_roots
        .iter()
        .map(|root| infer_snarl_recursive(root, panel_walks, &mut lookup))
        .collect();
    InferredPanel { roots, lookup }
}

pub fn build_runtime_from_walks(
    topology_roots: &[SnarlTopology],
    panel_walks: &[HaplotypeWalk],
    node_counts: HashMap<String, usize>,
) -> FeatureRuntime {
    let panel = infer_snarl_panel(topology_roots, panel_walks);
    let schema = FeatureBuilder::with_node_counts(node_counts).optimize(&panel.roots);
    FeatureRuntime { schema, panel }
}

pub fn build_runtime_from_gbz(
    topology_roots: &[SnarlTopology],
    gbz: &GBZ,
    node_counts: HashMap<String, usize>,
) -> FeatureRuntime {
    let panel_walks = extract_haplotype_walks_from_gbz(gbz);
    build_runtime_from_walks(topology_roots, &panel_walks, node_counts)
}

impl FeatureRuntime {
    pub fn encode_haplotype(&self, walk: &HaplotypeWalk) -> Vec<Option<f64>> {
        let mut out = vec![None; self.schema.total_features];
        let mut skeleton_cache: HashMap<String, Option<usize>> = HashMap::new();

        for site in &self.schema.sites {
            let lookup = self
                .panel
                .lookup
                .get(&site.snarl_id)
                .unwrap_or_else(|| panic!("missing snarl lookup for site {}", site.snarl_id));

            let traverses = conditions_met(self, walk, &site.conditional_on, &mut skeleton_cache);
            if !traverses {
                continue;
            }

            match site.class {
                SiteClass::Cyclic => {
                    let repeat = repeat_count(walk, lookup);
                    let encoded = encode_haploid_cyclic(repeat);
                    out[site.feature_start] = encoded[0];
                }
                SiteClass::Biallelic | SiteClass::Multiallelic => {
                    let allele_index = match site.kind {
                        FeatureKind::Flat | FeatureKind::Leaf => flat_allele_index(walk, lookup),
                        FeatureKind::Skeleton => {
                            skeleton_allele_index(self, walk, lookup, &mut skeleton_cache)
                        }
                        FeatureKind::Cyclic => None,
                    };
                    let reference = site.reference_allele_index().unwrap_or_else(|| {
                        panic!(
                            "site {} missing valid allele frequencies; cannot choose reference allele by most-common rule",
                            site.snarl_id
                        )
                    });
                    let encoded = encode_haploid_acyclic_with_reference(
                        allele_index,
                        site.allele_count,
                        reference,
                    );
                    out[site.feature_start..site.feature_end].copy_from_slice(&encoded);
                }
            }
        }

        out
    }

    pub fn encode_diploid(&self, left: &HaplotypeWalk, right: &HaplotypeWalk) -> Vec<Option<f64>> {
        let left_encoded = self.encode_haplotype(left);
        let right_encoded = self.encode_haplotype(right);
        sum_diploid_site(&left_encoded, &right_encoded)
    }
}

fn build_topology_node(
    id: &str,
    rows: &HashMap<String, TopologyRow>,
    child_map: &HashMap<String, Vec<String>>,
    cache: &mut HashMap<String, SnarlTopology>,
) -> io::Result<SnarlTopology> {
    if let Some(cached) = cache.get(id) {
        return Ok(cached.clone());
    }
    let Some(row) = rows.get(id) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing row for snarl id {}", id),
        ));
    };
    let row = row.clone();
    let mut children = Vec::new();
    if let Some(child_ids) = child_map.get(&row.id) {
        for child_id in child_ids {
            children.push(build_topology_node(child_id, rows, child_map, cache)?);
        }
    }
    let topology = SnarlTopology {
        id: row.id.clone(),
        snarl_type: row.snarl_type,
        entry_node: row.entry_node,
        exit_node: row.exit_node,
        genomic_region: row.genomic_region.clone(),
        children,
    };
    cache.insert(id.to_string(), topology.clone());
    Ok(topology)
}

fn infer_snarl_recursive(
    topology: &SnarlTopology,
    panel_walks: &[HaplotypeWalk],
    lookup: &mut HashMap<String, SnarlLookup>,
) -> Snarl {
    let traversals = collect_traversals(topology, panel_walks);
    match topology.snarl_type {
        SnarlType::Cyclic => {
            let snarl = Snarl::cyclic(topology.id.clone()).with_genomic_region_opt(topology.genomic_region.clone());
            lookup.insert(
                topology.id.clone(),
                SnarlLookup {
                    snarl_id: topology.id.clone(),
                    snarl_type: topology.snarl_type,
                    entry_node: topology.entry_node,
                    exit_node: topology.exit_node,
                    child_ids: Vec::new(),
                    flat_alleles: Vec::new(),
                    skeleton_alleles: Vec::new(),
                },
            );
            snarl
        }
        SnarlType::Acyclic => {
            let (flat_alleles, flat_freqs) = frequencies_and_labels(
                traversals
                    .iter()
                    .map(|tr| tr.flat_signature.clone())
                    .collect::<Vec<_>>(),
            );
            let mut children = Vec::new();
            for child in &topology.children {
                children.push(infer_snarl_recursive(child, panel_walks, lookup));
            }

            if topology.children.is_empty() {
                let mut snarl =
                    Snarl::leaf(topology.id.clone(), flat_alleles.len()).with_allele_frequencies(flat_freqs);
                if let Some(region) = &topology.genomic_region {
                    snarl = snarl.with_genomic_region(region.clone());
                }
                lookup.insert(
                    topology.id.clone(),
                    SnarlLookup {
                        snarl_id: topology.id.clone(),
                        snarl_type: topology.snarl_type,
                        entry_node: topology.entry_node,
                        exit_node: topology.exit_node,
                        child_ids: Vec::new(),
                        flat_alleles,
                        skeleton_alleles: Vec::new(),
                    },
                );
                return snarl;
            }

            let mut skeleton_signatures = Vec::new();
            let mut child_to_skeleton_sigs: HashMap<String, Vec<String>> = HashMap::new();
            for traversal in &traversals {
                let (skeleton_sig, present_children) = skeleton_signature(topology, traversal, lookup);
                skeleton_signatures.push(skeleton_sig.clone());
                for child_id in present_children {
                    child_to_skeleton_sigs
                        .entry(child_id)
                        .or_default()
                        .push(skeleton_sig.clone());
                }
            }
            let (skeleton_alleles, skeleton_freqs) = frequencies_and_labels(skeleton_signatures);
            let skeleton_index: HashMap<String, usize> = skeleton_alleles
                .iter()
                .enumerate()
                .map(|(idx, label)| (label.clone(), idx))
                .collect();

            for child in &mut children {
                let mut allowed = BTreeSet::new();
                if let Some(sigs) = child_to_skeleton_sigs.get(&child.id) {
                    for sig in sigs {
                        if let Some(idx) = skeleton_index.get(sig) {
                            allowed.insert(*idx);
                        }
                    }
                }
                if skeleton_alleles.len() > 1 {
                    child.parent_skeleton_alleles = Some(allowed.into_iter().collect());
                }
            }

            let mut snarl = Snarl::compound(
                topology.id.clone(),
                flat_alleles.len(),
                skeleton_alleles.len(),
                children,
            )
            .with_allele_frequencies(flat_freqs)
            .with_skeleton_allele_frequencies(skeleton_freqs);
            if let Some(region) = &topology.genomic_region {
                snarl = snarl.with_genomic_region(region.clone());
            }

            lookup.insert(
                topology.id.clone(),
                SnarlLookup {
                    snarl_id: topology.id.clone(),
                    snarl_type: topology.snarl_type,
                    entry_node: topology.entry_node,
                    exit_node: topology.exit_node,
                    child_ids: topology.children.iter().map(|child| child.id.clone()).collect(),
                    flat_alleles,
                    skeleton_alleles,
                },
            );
            snarl
        }
    }
}

#[derive(Clone)]
struct TopologyRow {
    id: String,
    snarl_type: SnarlType,
    entry_node: usize,
    exit_node: usize,
    parent_id: Option<String>,
    genomic_region: Option<String>,
}

fn collect_traversals<'a>(topology: &SnarlTopology, panel_walks: &'a [HaplotypeWalk]) -> Vec<Traversal<'a>> {
    let mut traversals = Vec::new();
    for walk in panel_walks {
        if let Some((start, end)) = find_span(&walk.steps, topology.entry_node, topology.exit_node) {
            let steps = &walk.steps[start..=end];
            traversals.push(Traversal {
                walk,
                start,
                end,
                flat_signature: step_signature(steps),
            });
        }
    }
    traversals
}

fn frequencies_and_labels(signatures: Vec<String>) -> (Vec<String>, Vec<f64>) {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for signature in signatures {
        *counts.entry(signature).or_insert(0) += 1;
    }
    let total: usize = counts.values().sum();
    if total == 0 {
        return (vec!["MISSING".to_string()], vec![1.0]);
    }

    let mut entries = counts.into_iter().collect::<Vec<_>>();
    entries.sort_by(|(sig_a, count_a), (sig_b, count_b)| count_b.cmp(count_a).then(sig_a.cmp(sig_b)));
    let labels = entries.iter().map(|(sig, _)| sig.clone()).collect::<Vec<_>>();
    let freqs = entries
        .iter()
        .map(|(_, count)| (*count as f64) / (total as f64))
        .collect::<Vec<_>>();
    (labels, freqs)
}

fn conditions_met(
    runtime: &FeatureRuntime,
    walk: &HaplotypeWalk,
    conditions: &[TraversalCondition],
    skeleton_cache: &mut HashMap<String, Option<usize>>,
) -> bool {
    for condition in conditions {
        let Some(condition_lookup) = runtime.panel.lookup.get(&condition.snarl_id) else {
            return false;
        };
        let skeleton_idx = skeleton_allele_index(runtime, walk, condition_lookup, skeleton_cache);
        let Some(skeleton_idx) = skeleton_idx else {
            return false;
        };
        if !condition
            .allowed_parent_skeleton_alleles
            .iter()
            .any(|allowed| *allowed == skeleton_idx)
        {
            return false;
        }
    }
    true
}

fn flat_allele_index(walk: &HaplotypeWalk, lookup: &SnarlLookup) -> Option<usize> {
    let (start, end) = find_span(&walk.steps, lookup.entry_node, lookup.exit_node)?;
    let signature = step_signature(&walk.steps[start..=end]);
    lookup
        .flat_alleles
        .iter()
        .position(|candidate| candidate == &signature)
}

fn skeleton_allele_index(
    runtime: &FeatureRuntime,
    walk: &HaplotypeWalk,
    lookup: &SnarlLookup,
    skeleton_cache: &mut HashMap<String, Option<usize>>,
) -> Option<usize> {
    if let Some(cached) = skeleton_cache.get(&lookup.snarl_id) {
        return *cached;
    }
    let (start, end) = find_span(&walk.steps, lookup.entry_node, lookup.exit_node)?;
    let parent_steps = &walk.steps[start..=end];

    let mut child_spans = Vec::new();
    for child_id in &lookup.child_ids {
        let Some(child_lookup) = runtime.panel.lookup.get(child_id) else {
            continue;
        };
        if let Some((child_start, child_end)) =
            find_span(parent_steps, child_lookup.entry_node, child_lookup.exit_node)
        {
            child_spans.push((child_start, child_end, child_id.clone()));
        }
    }
    child_spans.sort_by_key(|(start_idx, _, _)| *start_idx);
    let mut signature_parts = Vec::new();
    let mut cursor = 0usize;
    for (child_start, child_end, child_id) in child_spans {
        if child_start > cursor {
            signature_parts.push(step_signature(&parent_steps[cursor..child_start]));
        }
        signature_parts.push(format!("{{{}}}", child_id));
        cursor = child_end + 1;
    }
    if cursor < parent_steps.len() {
        signature_parts.push(step_signature(&parent_steps[cursor..]));
    }
    let signature = signature_parts.join("|");
    let index = lookup
        .skeleton_alleles
        .iter()
        .position(|candidate| candidate == &signature);
    skeleton_cache.insert(lookup.snarl_id.clone(), index);
    index
}

fn repeat_count(walk: &HaplotypeWalk, lookup: &SnarlLookup) -> Option<u32> {
    let (start, end) = find_span(&walk.steps, lookup.entry_node, lookup.exit_node)?;
    let segment = &walk.steps[start..=end];
    if segment.is_empty() {
        return None;
    }
    let entry_hits = segment
        .iter()
        .filter(|step| step.node_id == lookup.entry_node)
        .count();
    let copies = entry_hits.max(1);
    Some(copies as u32)
}

fn skeleton_signature(
    topology: &SnarlTopology,
    traversal: &Traversal<'_>,
    lookup: &HashMap<String, SnarlLookup>,
) -> (String, Vec<String>) {
    let parent_steps = &traversal.walk.steps[traversal.start..=traversal.end];
    let mut child_spans = Vec::new();
    let mut present_children = Vec::new();

    for child in &topology.children {
        let Some(child_lookup) = lookup.get(&child.id) else {
            continue;
        };
        if let Some((child_start, child_end)) =
            find_span(parent_steps, child_lookup.entry_node, child_lookup.exit_node)
        {
            child_spans.push((child_start, child_end, child.id.clone()));
            present_children.push(child.id.clone());
        }
    }
    child_spans.sort_by_key(|(start_idx, _, _)| *start_idx);

    let mut parts = Vec::new();
    let mut cursor = 0usize;
    for (child_start, child_end, child_id) in child_spans {
        if child_start > cursor {
            parts.push(step_signature(&parent_steps[cursor..child_start]));
        }
        parts.push(format!("{{{}}}", child_id));
        cursor = child_end + 1;
    }
    if cursor < parent_steps.len() {
        parts.push(step_signature(&parent_steps[cursor..]));
    }
    (parts.join("|"), present_children)
}

fn step_signature(steps: &[HaplotypeStep]) -> String {
    steps
        .iter()
        .map(|step| {
            let orient = if step.orientation == Orientation::Forward {
                '+'
            } else {
                '-'
            };
            format!("{}{}", step.node_id, orient)
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn find_span(steps: &[HaplotypeStep], entry_node: usize, exit_node: usize) -> Option<(usize, usize)> {
    let mut best_span: Option<(usize, usize)> = None;
    for start_idx in 0..steps.len() {
        if steps[start_idx].node_id != entry_node {
            continue;
        }
        let mut found_end = None;
        for (offset, step) in steps[start_idx..].iter().enumerate() {
            if step.node_id == exit_node {
                found_end = Some(start_idx + offset);
                break;
            }
        }
        let Some(end_idx) = found_end else {
            continue;
        };
        match best_span {
            None => best_span = Some((start_idx, end_idx)),
            Some((best_start, best_end)) => {
                let best_len = best_end.saturating_sub(best_start);
                let curr_len = end_idx.saturating_sub(start_idx);
                if curr_len < best_len {
                    best_span = Some((start_idx, end_idx));
                }
            }
        }
    }
    best_span
}

#[derive(Debug)]
struct Traversal<'a> {
    walk: &'a HaplotypeWalk,
    start: usize,
    end: usize,
    flat_signature: String,
}

trait OptionalRegionExt {
    fn with_genomic_region_opt(self, region: Option<String>) -> Self;
}

impl OptionalRegionExt for Snarl {
    fn with_genomic_region_opt(mut self, region: Option<String>) -> Self {
        if let Some(region) = region {
            self = self.with_genomic_region(region);
        }
        self
    }
}
