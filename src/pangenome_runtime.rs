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

#[derive(Debug, Clone, PartialEq)]
pub struct NodeTraversal {
    pub node_id: usize,
    pub value: f64,
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
                format!(
                    "line {}: expected at least 5 tab-separated fields",
                    lineno + 1
                ),
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
    let mut visiting = BTreeSet::new();
    let mut roots = Vec::new();
    for root_id in root_ids {
        roots.push(build_topology_node(
            &root_id,
            &row_map,
            &child_map,
            &mut built,
            &mut visiting,
        )?);
    }
    Ok(roots)
}

pub fn extract_haplotype_walks_from_gbz(gbz: &GBZ) -> Vec<HaplotypeWalk> {
    let Some(metadata) = gbz.metadata() else {
        return Vec::new();
    };

    let mut walks = Vec::new();
    for path_id in 0..metadata.paths() {
        let Some(path_name) = metadata.path(path_id) else {
            continue;
        };
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

pub fn infer_snarl_panel(
    topology_roots: &[SnarlTopology],
    panel_walks: &[HaplotypeWalk],
) -> InferredPanel {
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

    pub fn encode_haplotype_node_values(&self, traversals: &[NodeTraversal]) -> Vec<Option<f64>> {
        let mut values = HashMap::new();
        for traversal in traversals {
            if traversal.value.is_finite() {
                values.insert(traversal.node_id, traversal.value.clamp(0.0, 1.0));
            }
        }
        self.encode_haplotype_node_value_map(&values)
    }

    pub fn encode_haplotype_node_value_map(
        &self,
        node_values: &HashMap<usize, f64>,
    ) -> Vec<Option<f64>> {
        let mut out = vec![None; self.schema.total_features];
        let mut skeleton_cache: HashMap<String, Option<usize>> = HashMap::new();

        for site in &self.schema.sites {
            let lookup = self
                .panel
                .lookup
                .get(&site.snarl_id)
                .unwrap_or_else(|| panic!("missing snarl lookup for site {}", site.snarl_id));

            let traverses = conditions_met_node_values(
                self,
                node_values,
                &site.conditional_on,
                &mut skeleton_cache,
            );
            if !traverses
                || !node_map_traverses_snarl(node_values, lookup.entry_node, lookup.exit_node)
            {
                continue;
            }

            match site.class {
                SiteClass::Cyclic => {
                    let repeat = repeat_count_node_values(node_values, lookup);
                    out[site.feature_start] = repeat;
                }
                SiteClass::Biallelic | SiteClass::Multiallelic => {
                    let allele_index = match site.kind {
                        FeatureKind::Flat | FeatureKind::Leaf => {
                            flat_allele_index_node_values(node_values, lookup)
                        }
                        FeatureKind::Skeleton => skeleton_allele_index_node_values(
                            self,
                            node_values,
                            lookup,
                            &mut skeleton_cache,
                        ),
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

    pub fn encode_diploid_node_values(
        &self,
        left: &[NodeTraversal],
        right: &[NodeTraversal],
    ) -> Vec<Option<f64>> {
        let left_encoded = self.encode_haplotype_node_values(left);
        let right_encoded = self.encode_haplotype_node_values(right);
        sum_diploid_site(&left_encoded, &right_encoded)
    }
}

fn build_topology_node(
    id: &str,
    rows: &HashMap<String, TopologyRow>,
    child_map: &HashMap<String, Vec<String>>,
    cache: &mut HashMap<String, SnarlTopology>,
    visiting: &mut BTreeSet<String>,
) -> io::Result<SnarlTopology> {
    if let Some(cached) = cache.get(id) {
        return Ok(cached.clone());
    }
    if !visiting.insert(id.to_string()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cycle detected in topology at snarl {}", id),
        ));
    }
    let Some(row) = rows.get(id) else {
        visiting.remove(id);
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing row for snarl id {}", id),
        ));
    };
    let row = row.clone();
    let mut children = Vec::new();
    if let Some(child_ids) = child_map.get(&row.id) {
        for child_id in child_ids {
            children.push(build_topology_node(
                child_id, rows, child_map, cache, visiting,
            )?);
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
    visiting.remove(id);
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
            let snarl = Snarl::cyclic(topology.id.clone())
                .with_genomic_region_opt(topology.genomic_region.clone());
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
                let mut snarl = Snarl::leaf(topology.id.clone(), flat_alleles.len())
                    .with_allele_frequencies(flat_freqs);
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
                let (skeleton_sig, present_children) =
                    skeleton_signature(topology, traversal, lookup);
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
                    child_ids: topology
                        .children
                        .iter()
                        .map(|child| child.id.clone())
                        .collect(),
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

fn collect_traversals<'a>(
    topology: &SnarlTopology,
    panel_walks: &'a [HaplotypeWalk],
) -> Vec<Traversal<'a>> {
    let mut traversals = Vec::new();
    for walk in panel_walks {
        if let Some((start, end)) = find_span(&walk.steps, topology.entry_node, topology.exit_node)
        {
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

    let mut non_singletons = Vec::new();
    let mut singleton_total = 0usize;
    for (sig, count) in counts {
        if count >= 2 {
            non_singletons.push((sig, count));
        } else {
            singleton_total += count;
        }
    }
    non_singletons
        .sort_by(|(sig_a, count_a), (sig_b, count_b)| count_b.cmp(count_a).then(sig_a.cmp(sig_b)));

    let mut labels = non_singletons
        .iter()
        .map(|(sig, _)| sig.clone())
        .collect::<Vec<_>>();
    let mut freqs = non_singletons
        .iter()
        .map(|(_, count)| (*count as f64) / (total as f64))
        .collect::<Vec<_>>();
    if singleton_total > 0 {
        labels.push("OTHER".to_string());
        freqs.push((singleton_total as f64) / (total as f64));
    }

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

fn conditions_met_node_values(
    runtime: &FeatureRuntime,
    node_values: &HashMap<usize, f64>,
    conditions: &[TraversalCondition],
    skeleton_cache: &mut HashMap<String, Option<usize>>,
) -> bool {
    for condition in conditions {
        let Some(condition_lookup) = runtime.panel.lookup.get(&condition.snarl_id) else {
            return false;
        };
        let skeleton_idx = skeleton_allele_index_node_values(
            runtime,
            node_values,
            condition_lookup,
            skeleton_cache,
        );
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
        .or_else(|| {
            lookup
                .flat_alleles
                .iter()
                .position(|candidate| candidate == "OTHER")
        })
}

fn flat_allele_index_node_values(
    node_values: &HashMap<usize, f64>,
    lookup: &SnarlLookup,
) -> Option<usize> {
    best_signature_match(node_values, &lookup.flat_alleles)
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
        if let Some((child_start, child_end)) = find_span(
            parent_steps,
            child_lookup.entry_node,
            child_lookup.exit_node,
        ) {
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
    let resolved = index.or_else(|| {
        lookup
            .skeleton_alleles
            .iter()
            .position(|candidate| candidate == "OTHER")
    });
    skeleton_cache.insert(lookup.snarl_id.clone(), resolved);
    resolved
}

fn skeleton_allele_index_node_values(
    runtime: &FeatureRuntime,
    node_values: &HashMap<usize, f64>,
    lookup: &SnarlLookup,
    skeleton_cache: &mut HashMap<String, Option<usize>>,
) -> Option<usize> {
    if let Some(cached) = skeleton_cache.get(&lookup.snarl_id) {
        return *cached;
    }

    if !node_map_traverses_snarl(node_values, lookup.entry_node, lookup.exit_node) {
        skeleton_cache.insert(lookup.snarl_id.clone(), None);
        return None;
    }

    let mut child_present = HashMap::new();
    for child_id in &lookup.child_ids {
        let present = runtime
            .panel
            .lookup
            .get(child_id)
            .map(|child_lookup| {
                node_map_traverses_snarl(
                    node_values,
                    child_lookup.entry_node,
                    child_lookup.exit_node,
                )
            })
            .unwrap_or(false);
        child_present.insert(child_id.as_str(), present);
    }
    let index =
        best_skeleton_signature_match(node_values, &lookup.skeleton_alleles, &child_present);
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

fn repeat_count_node_values(
    node_values: &HashMap<usize, f64>,
    lookup: &SnarlLookup,
) -> Option<f64> {
    let entry = node_values.get(&lookup.entry_node).copied().unwrap_or(0.0);
    let exit = node_values.get(&lookup.exit_node).copied().unwrap_or(0.0);
    let support = entry.max(exit);
    if support <= 0.0 {
        None
    } else {
        Some(support)
    }
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
        if let Some((child_start, child_end)) = find_span(
            parent_steps,
            child_lookup.entry_node,
            child_lookup.exit_node,
        ) {
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

fn node_map_traverses_snarl(
    node_values: &HashMap<usize, f64>,
    entry_node: usize,
    exit_node: usize,
) -> bool {
    node_values.get(&entry_node).copied().unwrap_or(0.0) > 0.0
        && node_values.get(&exit_node).copied().unwrap_or(0.0) > 0.0
}

fn best_signature_match(node_values: &HashMap<usize, f64>, signatures: &[String]) -> Option<usize> {
    if signatures.is_empty() {
        return None;
    }
    let mut best_idx = None;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, signature) in signatures.iter().enumerate() {
        if signature == "MISSING" {
            continue;
        }
        if signature == "OTHER" {
            // Prefer concrete alleles when available; OTHER is a fallback.
            continue;
        }
        let node_ids = signature_node_ids(signature);
        if node_ids.is_empty() {
            continue;
        }
        let score = node_ids
            .iter()
            .map(|node| node_values.get(node).copied().unwrap_or(0.0))
            .sum::<f64>()
            / (node_ids.len() as f64);
        if best_idx.is_none() || score > best_score {
            best_idx = Some(idx);
            best_score = score;
        }
    }
    if best_idx.is_some() {
        return best_idx;
    }
    signatures.iter().position(|candidate| candidate == "OTHER")
}

fn best_skeleton_signature_match(
    node_values: &HashMap<usize, f64>,
    signatures: &[String],
    child_present: &HashMap<&str, bool>,
) -> Option<usize> {
    if signatures.is_empty() {
        return None;
    }
    let mut best_idx = None;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, signature) in signatures.iter().enumerate() {
        if signature == "MISSING" || signature == "OTHER" {
            continue;
        }
        let score = score_skeleton_signature(node_values, signature, child_present);
        if best_idx.is_none() || score > best_score {
            best_idx = Some(idx);
            best_score = score;
        }
    }
    if best_idx.is_some() {
        return best_idx;
    }
    signatures.iter().position(|candidate| candidate == "OTHER")
}

fn score_skeleton_signature(
    node_values: &HashMap<usize, f64>,
    signature: &str,
    child_present: &HashMap<&str, bool>,
) -> f64 {
    let mut score_sum = 0.0f64;
    let mut score_n = 0usize;
    for segment in signature.split('|') {
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('{') && trimmed.ends_with('}') && trimmed.len() > 2 {
            let child_id = &trimmed[1..trimmed.len() - 1];
            let present = child_present.get(child_id).copied().unwrap_or(false);
            score_sum += if present { 1.0 } else { 0.0 };
            score_n += 1;
            continue;
        }

        for node in signature_node_ids(trimmed) {
            score_sum += node_values.get(&node).copied().unwrap_or(0.0);
            score_n += 1;
        }
    }
    if score_n == 0 {
        0.0
    } else {
        score_sum / (score_n as f64)
    }
}

fn signature_node_ids(signature: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for token in signature.split(['|', ',']) {
        let trimmed = token.trim();
        if trimmed.is_empty() || trimmed.starts_with('{') {
            continue;
        }
        let numeric = trimmed.trim_end_matches('+').trim_end_matches('-');
        if numeric.is_empty() {
            continue;
        }
        if let Ok(node_id) = numeric.parse::<usize>() {
            out.push(node_id);
        }
    }
    out
}

fn find_span(
    steps: &[HaplotypeStep],
    entry_node: usize,
    exit_node: usize,
) -> Option<(usize, usize)> {
    let forward = find_ordered_span(steps, entry_node, exit_node);
    let reverse = find_ordered_span(steps, exit_node, entry_node);

    match (forward, reverse) {
        (Some(fwd), Some(rev)) => {
            let fwd_len = fwd.1.saturating_sub(fwd.0);
            let rev_len = rev.1.saturating_sub(rev.0);
            if fwd_len <= rev_len {
                Some(fwd)
            } else {
                Some(rev)
            }
        }
        (Some(fwd), None) => Some(fwd),
        (None, Some(rev)) => Some(rev),
        (None, None) => None,
    }
}

fn find_ordered_span(
    steps: &[HaplotypeStep],
    start_node: usize,
    end_node: usize,
) -> Option<(usize, usize)> {
    let mut best_span: Option<(usize, usize)> = None;
    for start_idx in 0..steps.len() {
        if steps[start_idx].node_id != start_node {
            continue;
        }
        let mut found_end = None;
        for (offset, step) in steps[start_idx..].iter().enumerate() {
            if step.node_id == end_node {
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
