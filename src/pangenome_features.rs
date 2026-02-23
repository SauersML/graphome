use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnarlType {
    Acyclic,
    Cyclic,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Snarl {
    pub id: String,
    pub snarl_type: SnarlType,
    pub allele_count: usize,
    pub allele_frequencies: Vec<f64>,
    pub genomic_region: Option<String>,
    pub skeleton_allele_count: Option<usize>,
    pub skeleton_allele_frequencies: Option<Vec<f64>>,
    pub parent_skeleton_alleles: Option<Vec<usize>>,
    pub children: Vec<Snarl>,
}

impl Snarl {
    pub fn leaf(id: impl Into<String>, allele_count: usize) -> Self {
        Self {
            id: id.into(),
            snarl_type: SnarlType::Acyclic,
            allele_count,
            allele_frequencies: Vec::new(),
            genomic_region: None,
            skeleton_allele_count: None,
            skeleton_allele_frequencies: None,
            parent_skeleton_alleles: None,
            children: Vec::new(),
        }
    }

    pub fn cyclic(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            snarl_type: SnarlType::Cyclic,
            allele_count: 0,
            allele_frequencies: Vec::new(),
            genomic_region: None,
            skeleton_allele_count: None,
            skeleton_allele_frequencies: None,
            parent_skeleton_alleles: None,
            children: Vec::new(),
        }
    }

    pub fn compound(
        id: impl Into<String>,
        allele_count: usize,
        skeleton_allele_count: usize,
        children: Vec<Snarl>,
    ) -> Self {
        Self {
            id: id.into(),
            snarl_type: SnarlType::Acyclic,
            allele_count,
            allele_frequencies: Vec::new(),
            genomic_region: None,
            skeleton_allele_count: Some(skeleton_allele_count),
            skeleton_allele_frequencies: None,
            parent_skeleton_alleles: None,
            children,
        }
    }

    pub fn with_allele_frequencies(mut self, allele_frequencies: Vec<f64>) -> Self {
        self.allele_frequencies = allele_frequencies;
        self
    }

    pub fn with_genomic_region(mut self, genomic_region: impl Into<String>) -> Self {
        self.genomic_region = Some(genomic_region.into());
        self
    }

    pub fn with_parent_skeleton_alleles(mut self, parent_skeleton_alleles: Vec<usize>) -> Self {
        self.parent_skeleton_alleles = Some(parent_skeleton_alleles);
        self
    }

    pub fn with_skeleton_allele_frequencies(
        mut self,
        skeleton_allele_frequencies: Vec<f64>,
    ) -> Self {
        self.skeleton_allele_frequencies = Some(skeleton_allele_frequencies);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    Flat,
    Skeleton,
    Leaf,
    Cyclic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteClass {
    Biallelic,
    Multiallelic,
    Cyclic,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSite {
    pub snarl_id: String,
    pub kind: FeatureKind,
    pub class: SiteClass,
    pub allele_count: usize,
    pub allele_frequencies: Vec<f64>,
    pub genomic_region: Option<String>,
    pub depth: usize,
    pub parent_snarl_id: Option<String>,
    pub conditional_on: Vec<TraversalCondition>,
    pub node_count: Option<usize>,
    pub feature_start: usize,
    pub feature_end: usize,
}

impl FeatureSite {
    pub fn column_count(&self) -> usize {
        self.feature_end.saturating_sub(self.feature_start)
    }

    pub fn reference_allele_index(&self) -> Option<usize> {
        reference_allele_index(&self.allele_frequencies)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSchema {
    pub sites: Vec<FeatureSite>,
    pub total_features: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct SnarlOptResult {
    cost: usize,
    selected: Vec<SelectedSite>,
}

#[derive(Debug, Clone, PartialEq)]
struct SelectedSite {
    snarl_id: String,
    kind: FeatureKind,
    allele_count: usize,
    allele_frequencies: Vec<f64>,
    genomic_region: Option<String>,
    class: SiteClass,
    depth: usize,
    parent_snarl_id: Option<String>,
    conditional_on: Vec<TraversalCondition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraversalCondition {
    pub snarl_id: String,
    pub allowed_parent_skeleton_alleles: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct FeatureBuilder {
    node_counts: HashMap<String, usize>,
}

impl FeatureBuilder {
    pub fn new() -> Self {
        Self {
            node_counts: HashMap::new(),
        }
    }

    pub fn with_node_counts(node_counts: HashMap<String, usize>) -> Self {
        Self { node_counts }
    }

    pub fn optimize(&self, roots: &[Snarl]) -> FeatureSchema {
        for snarl in roots {
            self.validate_snarl(snarl);
        }

        let mut selected = Vec::new();
        for snarl in roots {
            let result = self.optimize_snarl(snarl, 0, None, &[]);
            selected.extend(result.selected);
        }

        let mut sites = Vec::with_capacity(selected.len());
        let mut next_col = 0usize;
        for site in selected {
            let cols = match site.class {
                SiteClass::Cyclic => 1,
                SiteClass::Biallelic | SiteClass::Multiallelic => {
                    site.allele_count.saturating_sub(1)
                }
            };
            let feature_start = next_col;
            let feature_end = next_col + cols;
            next_col = feature_end;

            sites.push(FeatureSite {
                snarl_id: site.snarl_id.clone(),
                kind: site.kind,
                class: site.class,
                allele_count: site.allele_count,
                allele_frequencies: site.allele_frequencies.clone(),
                genomic_region: site.genomic_region.clone(),
                depth: site.depth,
                parent_snarl_id: site.parent_snarl_id.clone(),
                conditional_on: site.conditional_on.clone(),
                node_count: self.node_counts.get(&site.snarl_id).copied(),
                feature_start,
                feature_end,
            });
        }

        FeatureSchema {
            total_features: next_col,
            sites,
        }
    }

    fn optimize_snarl(
        &self,
        snarl: &Snarl,
        depth: usize,
        parent_id: Option<&str>,
        inherited_conditions: &[TraversalCondition],
    ) -> SnarlOptResult {
        match snarl.snarl_type {
            SnarlType::Cyclic => {
                assert!(
                    snarl.children.is_empty(),
                    "cyclic snarl {} has children; nested cyclic snarls are not supported",
                    snarl.id
                );
                let selected = vec![SelectedSite {
                    snarl_id: snarl.id.clone(),
                    kind: FeatureKind::Cyclic,
                    allele_count: 0,
                    allele_frequencies: snarl.allele_frequencies.clone(),
                    genomic_region: snarl.genomic_region.clone(),
                    class: SiteClass::Cyclic,
                    depth,
                    parent_snarl_id: parent_id.map(str::to_owned),
                    conditional_on: inherited_conditions.to_vec(),
                }];
                SnarlOptResult { cost: 1, selected }
            }
            SnarlType::Acyclic if snarl.children.is_empty() => {
                let cost = snarl
                    .allele_count
                    .checked_sub(1)
                    .expect("validated leaf allele_count must be > 0");
                let selected = if cost == 0 {
                    Vec::new()
                } else {
                    vec![SelectedSite {
                        snarl_id: snarl.id.clone(),
                        kind: FeatureKind::Leaf,
                        allele_count: snarl.allele_count,
                        allele_frequencies: snarl.allele_frequencies.clone(),
                        genomic_region: snarl.genomic_region.clone(),
                        class: if snarl.allele_count <= 2 {
                            SiteClass::Biallelic
                        } else {
                            SiteClass::Multiallelic
                        },
                        depth,
                        parent_snarl_id: parent_id.map(str::to_owned),
                        conditional_on: inherited_conditions.to_vec(),
                    }]
                };
                SnarlOptResult { cost, selected }
            }
            SnarlType::Acyclic => {
                let flat_cost = snarl
                    .allele_count
                    .checked_sub(1)
                    .expect("validated compound allele_count must be > 0");
                let k_skel = snarl.skeleton_allele_count.unwrap_or(1);
                let skeleton_cost = k_skel
                    .checked_sub(1)
                    .expect("validated skeleton_allele_count must be > 0");

                let mut child_cost = 0usize;
                for child in &snarl.children {
                    child_cost += self.snarl_cost(child);
                }

                let decomp_cost = skeleton_cost + child_cost;

                if flat_cost < decomp_cost {
                    let selected = if flat_cost == 0 {
                        Vec::new()
                    } else {
                        vec![SelectedSite {
                            snarl_id: snarl.id.clone(),
                            kind: FeatureKind::Flat,
                            allele_count: snarl.allele_count,
                            allele_frequencies: snarl.allele_frequencies.clone(),
                            genomic_region: snarl.genomic_region.clone(),
                            class: if snarl.allele_count <= 2 {
                                SiteClass::Biallelic
                            } else {
                                SiteClass::Multiallelic
                            },
                            depth,
                            parent_snarl_id: parent_id.map(str::to_owned),
                            conditional_on: inherited_conditions.to_vec(),
                        }]
                    };
                    SnarlOptResult {
                        cost: flat_cost,
                        selected,
                    }
                } else {
                    // Tie-breaker follows the plan: prefer decomposition when costs are equal.
                    let mut selected = Vec::new();
                    if k_skel > 1 {
                        let skeleton_freq = snarl
                            .skeleton_allele_frequencies
                            .clone()
                            .unwrap_or_else(|| {
                                panic!(
                                    "compound snarl {} requires skeleton_allele_frequencies when k_skel > 1",
                                    snarl.id
                                )
                            });
                        assert_eq!(
                            skeleton_freq.len(),
                            k_skel,
                            "snarl {} skeleton_allele_frequencies length {} does not match k_skel {}",
                            snarl.id,
                            skeleton_freq.len(),
                            k_skel
                        );
                        selected.push(SelectedSite {
                            snarl_id: snarl.id.clone(),
                            kind: FeatureKind::Skeleton,
                            allele_count: k_skel,
                            allele_frequencies: skeleton_freq,
                            genomic_region: snarl.genomic_region.clone(),
                            class: if k_skel <= 2 {
                                SiteClass::Biallelic
                            } else {
                                SiteClass::Multiallelic
                            },
                            depth,
                            parent_snarl_id: parent_id.map(str::to_owned),
                            conditional_on: inherited_conditions.to_vec(),
                        });
                    }
                    for child in &snarl.children {
                        let child_conditions =
                            self.child_conditions(&snarl.id, k_skel, child, inherited_conditions);
                        let child_result = self.optimize_snarl(
                            child,
                            depth + 1,
                            Some(&snarl.id),
                            &child_conditions,
                        );
                        selected.extend(child_result.selected);
                    }

                    SnarlOptResult {
                        cost: decomp_cost,
                        selected,
                    }
                }
            }
        }
    }

    fn snarl_cost(&self, snarl: &Snarl) -> usize {
        match snarl.snarl_type {
            SnarlType::Cyclic => 1,
            SnarlType::Acyclic if snarl.children.is_empty() => snarl
                .allele_count
                .checked_sub(1)
                .expect("validated leaf allele_count must be > 0"),
            SnarlType::Acyclic => {
                let flat_cost = snarl
                    .allele_count
                    .checked_sub(1)
                    .expect("validated compound allele_count must be > 0");
                let skeleton_cost = snarl
                    .skeleton_allele_count
                    .unwrap_or(1)
                    .checked_sub(1)
                    .expect("validated skeleton_allele_count must be > 0");
                let child_cost: usize = snarl
                    .children
                    .iter()
                    .map(|child| self.snarl_cost(child))
                    .sum();
                let decomp_cost = skeleton_cost + child_cost;
                if flat_cost < decomp_cost {
                    flat_cost
                } else {
                    decomp_cost
                }
            }
        }
    }

    fn child_conditions(
        &self,
        parent_snarl_id: &str,
        parent_skeleton_allele_count: usize,
        child: &Snarl,
        inherited_conditions: &[TraversalCondition],
    ) -> Vec<TraversalCondition> {
        let mut conditions = inherited_conditions.to_vec();
        if parent_skeleton_allele_count > 1 {
            let allowed = child.parent_skeleton_alleles.clone().unwrap_or_else(|| {
                panic!(
                    "child snarl {} of {} is missing parent_skeleton_alleles",
                    child.id, parent_snarl_id
                )
            });
            for &allele in &allowed {
                assert!(
                    allele < parent_skeleton_allele_count,
                    "child snarl {} has out-of-range parent skeleton allele {} for parent {} (k_skel={})",
                    child.id,
                    allele,
                    parent_snarl_id,
                    parent_skeleton_allele_count
                );
            }
            conditions.push(TraversalCondition {
                snarl_id: parent_snarl_id.to_owned(),
                allowed_parent_skeleton_alleles: allowed,
            });
        } else {
            // Even without parent skeleton branching, child traversal is still conditional
            // on traversing the parent span.
            conditions.push(TraversalCondition {
                snarl_id: parent_snarl_id.to_owned(),
                allowed_parent_skeleton_alleles: vec![0],
            });
        }
        conditions
    }

    fn validate_snarl(&self, snarl: &Snarl) {
        self.validate_snarl_inner(snarl, None);
    }

    fn validate_snarl_inner(&self, snarl: &Snarl, parent: Option<(&str, usize)>) {
        if let Some((parent_id, parent_k_skel)) = parent {
            if parent_k_skel > 1 {
                let allowed = snarl.parent_skeleton_alleles.as_ref().unwrap_or_else(|| {
                    panic!(
                        "child snarl {} of {} must define parent_skeleton_alleles when parent k_skel > 1",
                        snarl.id, parent_id
                    )
                });
                for &allele in allowed {
                    assert!(
                        allele < parent_k_skel,
                        "child snarl {} has invalid parent_skeleton_allele {} for parent {} (k_skel={})",
                        snarl.id,
                        allele,
                        parent_id,
                        parent_k_skel
                    );
                }
            }
        }

        match snarl.snarl_type {
            SnarlType::Cyclic => {
                assert!(
                    snarl.allele_count == 0,
                    "cyclic snarl {} must use allele_count=0",
                    snarl.id
                );
                assert!(
                    snarl.skeleton_allele_count.is_none(),
                    "cyclic snarl {} must not define skeleton_allele_count",
                    snarl.id
                );
                assert!(
                    snarl.skeleton_allele_frequencies.is_none(),
                    "cyclic snarl {} must not define skeleton_allele_frequencies",
                    snarl.id
                );
                assert!(
                    snarl.parent_skeleton_alleles.is_none(),
                    "cyclic snarl {} must not define parent_skeleton_alleles",
                    snarl.id
                );
            }
            SnarlType::Acyclic => {
                assert!(
                    snarl.allele_count > 0,
                    "acyclic snarl {} must have allele_count > 0",
                    snarl.id
                );
                assert_eq!(
                    snarl.allele_frequencies.len(),
                    snarl.allele_count,
                    "snarl {} must provide allele_frequencies for all alleles (got {}, expected {})",
                    snarl.id,
                    snarl.allele_frequencies.len(),
                    snarl.allele_count
                );
                validate_frequency_vector(
                    &snarl.id,
                    "allele_frequencies",
                    &snarl.allele_frequencies,
                );
                if snarl.children.is_empty() {
                    assert!(
                        snarl.skeleton_allele_count.is_none(),
                        "leaf snarl {} must not define skeleton_allele_count",
                        snarl.id
                    );
                    assert!(
                        snarl.skeleton_allele_frequencies.is_none(),
                        "leaf snarl {} must not define skeleton_allele_frequencies",
                        snarl.id
                    );
                } else {
                    let k_skel = snarl.skeleton_allele_count.unwrap_or(1);
                    assert!(
                        k_skel > 0,
                        "compound snarl {} must have skeleton_allele_count > 0",
                        snarl.id
                    );
                    if k_skel > 1 {
                        let freq = snarl.skeleton_allele_frequencies.as_ref().unwrap_or_else(|| {
                            panic!(
                                "compound snarl {} must provide skeleton_allele_frequencies when k_skel > 1",
                                snarl.id
                            )
                        });
                        assert_eq!(
                            freq.len(),
                            k_skel,
                            "snarl {} skeleton_allele_frequencies length {} does not match skeleton_allele_count {}",
                            snarl.id,
                            freq.len(),
                            k_skel
                        );
                        validate_frequency_vector(&snarl.id, "skeleton_allele_frequencies", freq);
                    } else if let Some(freq) = &snarl.skeleton_allele_frequencies {
                        assert_eq!(
                            freq.len(),
                            1,
                            "snarl {} with k_skel=1 may only carry one skeleton frequency",
                            snarl.id
                        );
                        validate_frequency_vector(&snarl.id, "skeleton_allele_frequencies", freq);
                    }
                }
            }
        }

        let parent_k_skel = match snarl.snarl_type {
            SnarlType::Acyclic if !snarl.children.is_empty() => {
                snarl.skeleton_allele_count.unwrap_or(1)
            }
            _ => 1,
        };
        for child in &snarl.children {
            self.validate_snarl_inner(child, Some((&snarl.id, parent_k_skel)));
        }
    }
}

impl Default for FeatureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn encode_haploid_acyclic(
    allele_index: Option<usize>,
    allele_count: usize,
) -> Vec<Option<f64>> {
    encode_haploid_acyclic_with_reference(allele_index, allele_count, 0)
}

pub fn encode_haploid_acyclic_with_reference(
    allele_index: Option<usize>,
    allele_count: usize,
    reference_allele: usize,
) -> Vec<Option<f64>> {
    assert!(
        allele_count > 0,
        "allele_count must be > 0 for acyclic encoding"
    );
    if allele_count == 1 {
        return Vec::new();
    }
    assert!(
        reference_allele < allele_count,
        "reference_allele {} out of bounds for allele_count {}",
        reference_allele,
        allele_count
    );

    let cols = allele_count - 1;
    let Some(allele) = allele_index else {
        return vec![None; cols];
    };
    assert!(
        allele < allele_count,
        "allele_index {} out of bounds for allele_count {}",
        allele,
        allele_count
    );

    if allele_count == 2 {
        let dosage = if allele == reference_allele { 0.0 } else { 1.0 };
        return vec![Some(dosage)];
    }

    let mut out = vec![Some(0.0); cols];
    if allele == reference_allele {
        return out;
    }

    let mut col = 0usize;
    for allele_idx in 0..allele_count {
        if allele_idx == reference_allele {
            continue;
        }
        if allele_idx == allele {
            out[col] = Some(1.0);
            break;
        }
        col += 1;
    }
    out
}

pub fn encode_haploid_acyclic_probabilistic_with_reference(
    allele_probabilities: Option<&[f64]>,
    allele_count: usize,
    reference_allele: usize,
) -> Vec<Option<f64>> {
    assert!(
        allele_count > 0,
        "allele_count must be > 0 for acyclic encoding"
    );
    if allele_count == 1 {
        return Vec::new();
    }
    assert!(
        reference_allele < allele_count,
        "reference_allele {} out of bounds for allele_count {}",
        reference_allele,
        allele_count
    );

    let cols = allele_count - 1;
    let Some(probabilities) = allele_probabilities else {
        return vec![None; cols];
    };
    assert_eq!(
        probabilities.len(),
        allele_count,
        "allele probability length {} does not match allele_count {}",
        probabilities.len(),
        allele_count
    );

    if allele_count == 2 {
        let alt = if reference_allele == 0 { 1 } else { 0 };
        return vec![Some(probabilities[alt])];
    }

    let mut out = Vec::with_capacity(cols);
    for (allele_idx, prob) in probabilities.iter().copied().enumerate() {
        if allele_idx == reference_allele {
            continue;
        }
        out.push(Some(prob));
    }
    out
}

pub fn encode_haploid_cyclic(repeat_count: Option<u32>) -> Vec<Option<f64>> {
    match repeat_count {
        Some(copies) => vec![Some(copies as f64)],
        None => vec![None],
    }
}

pub fn sum_diploid_site(left: &[Option<f64>], right: &[Option<f64>]) -> Vec<Option<f64>> {
    assert_eq!(
        left.len(),
        right.len(),
        "diploid site vectors must have equal length (left={}, right={})",
        left.len(),
        right.len()
    );
    let mut out = Vec::with_capacity(left.len());
    for idx in 0..left.len() {
        let l = left.get(idx).copied().flatten();
        let r = right.get(idx).copied().flatten();
        let value = match (l, r) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        out.push(value);
    }
    out
}

pub fn reference_allele_index(allele_frequencies: &[f64]) -> Option<usize> {
    let mut best_idx = None;
    let mut best_freq = f64::NEG_INFINITY;
    for (idx, freq) in allele_frequencies.iter().copied().enumerate() {
        if !freq.is_finite() {
            continue;
        }
        if best_idx.is_none() || freq > best_freq {
            best_idx = Some(idx);
            best_freq = freq;
        }
    }
    best_idx
}

fn validate_frequency_vector(snarl_id: &str, field: &str, freqs: &[f64]) {
    assert!(
        !freqs.is_empty(),
        "snarl {} {} must not be empty",
        snarl_id,
        field
    );
    let mut sum = 0.0f64;
    for (idx, value) in freqs.iter().copied().enumerate() {
        assert!(
            value.is_finite(),
            "snarl {} {}[{}] must be finite",
            snarl_id,
            field,
            idx
        );
        assert!(
            value >= 0.0,
            "snarl {} {}[{}] must be non-negative",
            snarl_id,
            field,
            idx
        );
        sum += value;
    }
    assert!(
        sum > 0.0,
        "snarl {} {} must have positive total mass",
        snarl_id,
        field
    );
}
