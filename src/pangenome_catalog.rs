use crate::pangenome_features::{FeatureKind, SiteClass};
use crate::pangenome_runtime::FeatureRuntime;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC: [u8; 4] = *b"SFC\0";
const VERSION: u16 = 1;
const FEATURES_HEADER_SIZE: usize = 24;
const FEATURE_RECORD_SIZE: usize = 32;
const NO_PARENT: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureCatalogManifest {
    pub graph_build_id: String,
    pub graph_construction_pipeline: String,
    pub reference_coordinates: String,
    pub hprc_release: String,
    pub haplotype_count: u32,
    pub snarl_decomposition_tool: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogAlleleTraversal {
    pub node_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogFeature {
    pub feature_id: u32,
    pub boundary_entry: u64,
    pub boundary_exit: u64,
    pub k: u16,
    pub is_cyclic: bool,
    pub parent_feature_id: Option<u32>,
    pub alleles: Vec<CatalogAlleleTraversal>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureCatalog {
    pub features: Vec<CatalogFeature>,
}

impl FeatureCatalog {
    pub fn write_dir(&self, out_dir: &Path) -> io::Result<()> {
        fs::create_dir_all(out_dir)?;
        self.write_bins(
            &out_dir.join("features.bin"),
            &out_dir.join("traversals.bin"),
        )
    }

    pub fn write_dir_with_manifest(
        &self,
        out_dir: &Path,
        manifest: &FeatureCatalogManifest,
    ) -> io::Result<()> {
        self.write_dir(out_dir)?;
        write_manifest(&out_dir.join("manifest.tsv"), manifest)
    }

    pub fn write_bins(&self, features_path: &Path, traversals_path: &Path) -> io::Result<()> {
        let mut traversals_file = File::create(traversals_path)?;
        let mut traversal_offsets = Vec::with_capacity(self.features.len());
        let mut traversal_offset = 0u32;
        let mut n_traversals = 0u32;

        for feature in &self.features {
            traversal_offsets.push(traversal_offset);
            for allele in &feature.alleles {
                let node_count = u16::try_from(allele.node_ids.len()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "allele traversal exceeds uint16 node limit",
                    )
                })?;
                traversals_file.write_all(&node_count.to_le_bytes())?;
                traversal_offset = traversal_offset.checked_add(2).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "traversals.bin overflow")
                })?;
                for node in &allele.node_ids {
                    traversals_file.write_all(&node.to_le_bytes())?;
                    traversal_offset = traversal_offset.checked_add(8).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "traversals.bin overflow")
                    })?;
                }
                n_traversals = n_traversals.checked_add(1).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "too many traversals")
                })?;
            }
        }

        let mut features_file = File::create(features_path)?;
        features_file.write_all(&MAGIC)?;
        features_file.write_all(&VERSION.to_le_bytes())?;
        features_file.write_all(
            &u32::try_from(self.features.len())
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many features"))?
                .to_le_bytes(),
        )?;
        features_file.write_all(&n_traversals.to_le_bytes())?;
        features_file.write_all(&[0u8; 10])?;

        for (idx, feature) in self.features.iter().enumerate() {
            let traversal_offset = traversal_offsets[idx];
            features_file.write_all(&feature.feature_id.to_le_bytes())?;
            features_file.write_all(&feature.boundary_entry.to_le_bytes())?;
            features_file.write_all(&feature.boundary_exit.to_le_bytes())?;
            features_file.write_all(&feature.k.to_le_bytes())?;
            features_file.write_all(&[u8::from(feature.is_cyclic)])?;
            let parent = feature.parent_feature_id.unwrap_or(NO_PARENT);
            features_file.write_all(&parent.to_le_bytes())?;
            features_file.write_all(&traversal_offset.to_le_bytes())?;
            features_file.write_all(&[0u8; 1])?;
        }

        Ok(())
    }

    pub fn read_bins(features_path: &Path, traversals_path: &Path) -> io::Result<Self> {
        let mut features_bytes = Vec::new();
        File::open(features_path)?.read_to_end(&mut features_bytes)?;
        if features_bytes.len() < FEATURES_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "features.bin shorter than header",
            ));
        }
        if features_bytes[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid features.bin magic",
            ));
        }
        let version = u16::from_le_bytes([features_bytes[4], features_bytes[5]]);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported features.bin version {}", version),
            ));
        }
        let n_features = u32::from_le_bytes([
            features_bytes[6],
            features_bytes[7],
            features_bytes[8],
            features_bytes[9],
        ]) as usize;
        let expected_traversal_count = u32::from_le_bytes([
            features_bytes[10],
            features_bytes[11],
            features_bytes[12],
            features_bytes[13],
        ]) as usize;
        let expected_size = FEATURES_HEADER_SIZE + n_features * FEATURE_RECORD_SIZE;
        if features_bytes.len() != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid features.bin size {}, expected {}",
                    features_bytes.len(),
                    expected_size
                ),
            ));
        }

        let mut recs = Vec::with_capacity(n_features);
        for idx in 0..n_features {
            let off = FEATURES_HEADER_SIZE + idx * FEATURE_RECORD_SIZE;
            let feature_id = u32::from_le_bytes([
                features_bytes[off],
                features_bytes[off + 1],
                features_bytes[off + 2],
                features_bytes[off + 3],
            ]);
            let boundary_entry = u64::from_le_bytes([
                features_bytes[off + 4],
                features_bytes[off + 5],
                features_bytes[off + 6],
                features_bytes[off + 7],
                features_bytes[off + 8],
                features_bytes[off + 9],
                features_bytes[off + 10],
                features_bytes[off + 11],
            ]);
            let boundary_exit = u64::from_le_bytes([
                features_bytes[off + 12],
                features_bytes[off + 13],
                features_bytes[off + 14],
                features_bytes[off + 15],
                features_bytes[off + 16],
                features_bytes[off + 17],
                features_bytes[off + 18],
                features_bytes[off + 19],
            ]);
            let k = u16::from_le_bytes([features_bytes[off + 20], features_bytes[off + 21]]);
            let is_cyclic = features_bytes[off + 22] != 0;
            let parent_raw = u32::from_le_bytes([
                features_bytes[off + 23],
                features_bytes[off + 24],
                features_bytes[off + 25],
                features_bytes[off + 26],
            ]);
            let parent_feature_id = if parent_raw == NO_PARENT {
                None
            } else {
                Some(parent_raw)
            };
            let traversal_offset = u32::from_le_bytes([
                features_bytes[off + 27],
                features_bytes[off + 28],
                features_bytes[off + 29],
                features_bytes[off + 30],
            ]);
            recs.push((
                feature_id,
                boundary_entry,
                boundary_exit,
                k,
                is_cyclic,
                parent_feature_id,
                traversal_offset as usize,
            ));
        }

        let mut traversals_bytes = Vec::new();
        File::open(traversals_path)?.read_to_end(&mut traversals_bytes)?;
        let mut features = Vec::with_capacity(n_features);
        let mut observed_traversal_count = 0usize;
        for (idx, rec) in recs.iter().enumerate() {
            let (_, boundary_entry, boundary_exit, k, is_cyclic, parent_feature_id, start_off) =
                *rec;
            let end_off = if idx + 1 < recs.len() {
                recs[idx + 1].6
            } else {
                traversals_bytes.len()
            };
            if start_off > end_off || end_off > traversals_bytes.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid traversal offset range",
                ));
            }
            let mut cursor = start_off;
            let mut alleles = Vec::new();
            while cursor < end_off {
                if cursor + 2 > end_off {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "truncated allele length",
                    ));
                }
                let n_nodes =
                    u16::from_le_bytes([traversals_bytes[cursor], traversals_bytes[cursor + 1]])
                        as usize;
                cursor += 2;
                let bytes_needed = n_nodes.checked_mul(8).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "node byte overflow")
                })?;
                if cursor + bytes_needed > end_off {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "truncated node_ids payload",
                    ));
                }
                let mut node_ids = Vec::with_capacity(n_nodes);
                for _ in 0..n_nodes {
                    node_ids.push(u64::from_le_bytes([
                        traversals_bytes[cursor],
                        traversals_bytes[cursor + 1],
                        traversals_bytes[cursor + 2],
                        traversals_bytes[cursor + 3],
                        traversals_bytes[cursor + 4],
                        traversals_bytes[cursor + 5],
                        traversals_bytes[cursor + 6],
                        traversals_bytes[cursor + 7],
                    ]));
                    cursor += 8;
                }
                alleles.push(CatalogAlleleTraversal { node_ids });
                observed_traversal_count += 1;
            }
            features.push(CatalogFeature {
                feature_id: rec.0,
                boundary_entry,
                boundary_exit,
                k,
                is_cyclic,
                parent_feature_id,
                alleles,
            });
        }
        if observed_traversal_count != expected_traversal_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "traversal count mismatch: header={}, observed={}",
                    expected_traversal_count, observed_traversal_count
                ),
            ));
        }

        Ok(Self { features })
    }

    pub fn read_dir(in_dir: &Path) -> io::Result<Self> {
        Self::read_bins(&in_dir.join("features.bin"), &in_dir.join("traversals.bin"))
    }
}

pub fn catalog_from_runtime(runtime: &FeatureRuntime) -> FeatureCatalog {
    let mut feature_index_by_snarl = HashMap::new();
    for site in &runtime.schema.sites {
        feature_index_by_snarl.insert(
            site.snarl_id.clone(),
            u32::try_from(site.feature_start).unwrap_or_else(|_| {
                panic!("feature_start overflows u32 for snarl {}", site.snarl_id)
            }),
        );
    }

    let mut features = Vec::with_capacity(runtime.schema.sites.len());
    for site in &runtime.schema.sites {
        let lookup = runtime
            .panel
            .lookup
            .get(&site.snarl_id)
            .unwrap_or_else(|| panic!("missing snarl lookup for {}", site.snarl_id));
        let parent_feature_id = site
            .parent_snarl_id
            .as_ref()
            .and_then(|parent_id| feature_index_by_snarl.get(parent_id).copied());
        let is_cyclic = matches!(site.class, SiteClass::Cyclic);
        let k = if is_cyclic {
            0u16
        } else {
            u16::try_from(site.allele_count)
                .unwrap_or_else(|_| panic!("site {} allele_count overflows u16", site.snarl_id))
        };

        let allele_signatures = if is_cyclic {
            Vec::new()
        } else {
            match site.kind {
                FeatureKind::Flat | FeatureKind::Leaf => lookup.flat_alleles.clone(),
                FeatureKind::Skeleton => lookup.skeleton_alleles.clone(),
                FeatureKind::Cyclic => Vec::new(),
            }
        };
        let alleles = allele_signatures
            .iter()
            .map(|signature| CatalogAlleleTraversal {
                node_ids: parse_signature_node_ids(signature),
            })
            .collect::<Vec<_>>();

        let feature_id = u32::try_from(site.feature_start)
            .unwrap_or_else(|_| panic!("feature_start overflows u32 for snarl {}", site.snarl_id));

        features.push(CatalogFeature {
            feature_id,
            boundary_entry: lookup.entry_node as u64,
            boundary_exit: lookup.exit_node as u64,
            k,
            is_cyclic,
            parent_feature_id,
            alleles,
        });
    }
    FeatureCatalog { features }
}

fn parse_signature_node_ids(signature: &str) -> Vec<u64> {
    if signature == "OTHER" || signature == "MISSING" {
        return Vec::new();
    }
    let mut node_ids = Vec::new();
    for token in signature.split(['|', ',']) {
        let trimmed = token.trim();
        if trimmed.is_empty() || trimmed.starts_with('{') {
            continue;
        }
        let numeric = trimmed.trim_end_matches('+').trim_end_matches('-');
        if numeric.is_empty() {
            continue;
        }
        if let Ok(node_id) = numeric.parse::<u64>() {
            node_ids.push(node_id);
        }
    }
    node_ids
}

pub fn write_manifest(path: &Path, manifest: &FeatureCatalogManifest) -> io::Result<()> {
    let mut file = File::create(path)?;
    write_manifest_line(&mut file, "graph_build_id", &manifest.graph_build_id)?;
    write_manifest_line(
        &mut file,
        "graph_construction_pipeline",
        &manifest.graph_construction_pipeline,
    )?;
    write_manifest_line(
        &mut file,
        "reference_coordinates",
        &manifest.reference_coordinates,
    )?;
    write_manifest_line(&mut file, "hprc_release", &manifest.hprc_release)?;
    write_manifest_line(
        &mut file,
        "haplotype_count",
        &manifest.haplotype_count.to_string(),
    )?;
    write_manifest_line(
        &mut file,
        "snarl_decomposition_tool",
        &manifest.snarl_decomposition_tool,
    )?;
    Ok(())
}

pub fn read_manifest(path: &Path) -> io::Result<FeatureCatalogManifest> {
    let content = fs::read_to_string(path)?;
    let mut values = HashMap::new();
    for (lineno, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.splitn(2, '\t');
        let key = parts.next().unwrap_or_default().trim();
        let value = parts.next().unwrap_or_default().trim();
        if key.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("manifest line {} missing key", lineno + 1),
            ));
        }
        values.insert(key.to_string(), value.to_string());
    }

    let haplotype_count = values
        .get("haplotype_count")
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest missing haplotype_count",
            )
        })?
        .parse::<u32>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid haplotype_count"))?;

    Ok(FeatureCatalogManifest {
        graph_build_id: required_manifest_field(&values, "graph_build_id")?,
        graph_construction_pipeline: required_manifest_field(
            &values,
            "graph_construction_pipeline",
        )?,
        reference_coordinates: required_manifest_field(&values, "reference_coordinates")?,
        hprc_release: required_manifest_field(&values, "hprc_release")?,
        haplotype_count,
        snarl_decomposition_tool: required_manifest_field(&values, "snarl_decomposition_tool")?,
    })
}

fn write_manifest_line(file: &mut File, key: &str, value: &str) -> io::Result<()> {
    if value.contains('\n') || value.contains('\r') || value.contains('\t') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "manifest value for {} contains unsupported control characters",
                key
            ),
        ));
    }
    writeln!(file, "{}\t{}", key, value)
}

fn required_manifest_field(values: &HashMap<String, String>, key: &str) -> io::Result<String> {
    values.get(key).cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("manifest missing {}", key),
        )
    })
}
