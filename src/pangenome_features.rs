use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnarlType {
    Acyclic,
    Cyclic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snarl {
    pub id: String,
    pub snarl_type: SnarlType,
    pub allele_count: usize,
    pub skeleton_allele_count: Option<usize>,
    pub children: Vec<Snarl>,
}

impl Snarl {
    pub fn leaf(id: impl Into<String>, allele_count: usize) -> Self {
        Self {
            id: id.into(),
            snarl_type: SnarlType::Acyclic,
            allele_count,
            skeleton_allele_count: None,
            children: Vec::new(),
        }
    }

    pub fn cyclic(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            snarl_type: SnarlType::Cyclic,
            allele_count: 0,
            skeleton_allele_count: None,
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
            skeleton_allele_count: Some(skeleton_allele_count),
            children,
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSite {
    pub snarl_id: String,
    pub kind: FeatureKind,
    pub class: SiteClass,
    pub allele_count: usize,
    pub depth: usize,
    pub parent_snarl_id: Option<String>,
    pub conditional_on: Option<String>,
    pub node_count: Option<usize>,
    pub feature_start: usize,
    pub feature_end: usize,
}

impl FeatureSite {
    pub fn column_count(&self) -> usize {
        self.feature_end.saturating_sub(self.feature_start)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSchema {
    pub sites: Vec<FeatureSite>,
    pub total_features: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SnarlOptResult {
    cost: usize,
    selected: Vec<SelectedSite>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedSite {
    snarl_id: String,
    kind: FeatureKind,
    allele_count: usize,
    class: SiteClass,
    depth: usize,
    parent_snarl_id: Option<String>,
    conditional_on: Option<String>,
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
        let mut selected = Vec::new();
        for snarl in roots {
            let result = self.optimize_snarl(snarl, 0, None, None);
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

            let class = match site.class {
                SiteClass::Cyclic => SiteClass::Cyclic,
                _ if site.allele_count <= 2 => SiteClass::Biallelic,
                _ => SiteClass::Multiallelic,
            };

            sites.push(FeatureSite {
                snarl_id: site.snarl_id.clone(),
                kind: site.kind,
                class,
                allele_count: site.allele_count,
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
        conditional_on: Option<String>,
    ) -> SnarlOptResult {
        match snarl.snarl_type {
            SnarlType::Cyclic => {
                let selected = vec![SelectedSite {
                    snarl_id: snarl.id.clone(),
                    kind: FeatureKind::Cyclic,
                    allele_count: 0,
                    class: SiteClass::Cyclic,
                    depth,
                    parent_snarl_id: parent_id.map(str::to_owned),
                    conditional_on,
                }];
                SnarlOptResult { cost: 1, selected }
            }
            SnarlType::Acyclic if snarl.children.is_empty() => {
                let cost = snarl.allele_count.saturating_sub(1);
                let selected = if cost == 0 {
                    Vec::new()
                } else {
                    vec![SelectedSite {
                        snarl_id: snarl.id.clone(),
                        kind: FeatureKind::Leaf,
                        allele_count: snarl.allele_count,
                        class: if snarl.allele_count <= 2 {
                            SiteClass::Biallelic
                        } else {
                            SiteClass::Multiallelic
                        },
                        depth,
                        parent_snarl_id: parent_id.map(str::to_owned),
                        conditional_on,
                    }]
                };
                SnarlOptResult { cost, selected }
            }
            SnarlType::Acyclic => {
                let flat_cost = snarl.allele_count.saturating_sub(1);
                let k_skel = snarl.skeleton_allele_count.unwrap_or(1);
                let skeleton_cost = k_skel.saturating_sub(1);

                let mut child_cost = 0usize;
                let mut child_selected = Vec::new();
                for child in &snarl.children {
                    let child_result = self.optimize_snarl(
                        child,
                        depth + 1,
                        Some(&snarl.id),
                        Some(format!("{}:skeleton", snarl.id)),
                    );
                    child_cost += child_result.cost;
                    child_selected.extend(child_result.selected);
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
                            class: if snarl.allele_count <= 2 {
                                SiteClass::Biallelic
                            } else {
                                SiteClass::Multiallelic
                            },
                            depth,
                            parent_snarl_id: parent_id.map(str::to_owned),
                            conditional_on,
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
                        selected.push(SelectedSite {
                            snarl_id: snarl.id.clone(),
                            kind: FeatureKind::Skeleton,
                            allele_count: k_skel,
                            class: if k_skel <= 2 {
                                SiteClass::Biallelic
                            } else {
                                SiteClass::Multiallelic
                            },
                            depth,
                            parent_snarl_id: parent_id.map(str::to_owned),
                            conditional_on: conditional_on.clone(),
                        });
                    }
                    selected.extend(child_selected);

                    SnarlOptResult {
                        cost: decomp_cost,
                        selected,
                    }
                }
            }
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
    let cols = allele_count.saturating_sub(1);
    let Some(allele) = allele_index else {
        return vec![None; cols];
    };

    if allele_count <= 2 {
        let dosage = if allele == 0 { 0.0 } else { 1.0 };
        return vec![Some(dosage)];
    }

    let mut out = vec![Some(0.0); cols];
    if allele > 0 {
        let idx = allele - 1;
        if idx < cols {
            out[idx] = Some(1.0);
        }
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
    let len = left.len().max(right.len());
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
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
