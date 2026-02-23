use gbwt::Orientation;
use graphome::pangenome_catalog::{
    catalog_from_runtime, read_manifest, write_manifest, CatalogAlleleTraversal, FeatureCatalog,
    FeatureCatalogManifest,
};
use graphome::pangenome_features::{
    encode_haploid_acyclic, encode_haploid_acyclic_with_reference, encode_haploid_cyclic,
    reference_allele_index, sum_diploid_site, FeatureBuilder, FeatureKind, SiteClass, Snarl,
    TraversalCondition,
};
use graphome::pangenome_runtime::{
    build_runtime_from_walks, infer_snarl_panel, load_topology_tsv, HaplotypeStep, HaplotypeWalk,
    NodeTraversal, SnarlTopology,
};
use std::collections::HashMap;
use std::io::Write;
use tempfile::tempdir;
use tempfile::NamedTempFile;

fn leaf_with_freq(id: impl Into<String>, allele_frequencies: Vec<f64>) -> Snarl {
    Snarl::leaf(id, allele_frequencies.len()).with_allele_frequencies(allele_frequencies)
}

#[test]
fn ld_block_prefers_flattening() {
    let children = (0..50)
        .map(|i| {
            leaf_with_freq(format!("leaf_{i}"), vec![0.5, 0.5])
                .with_parent_skeleton_alleles(vec![0, 1])
        })
        .collect();
    let root = Snarl::compound("ld_block", 2, 2, children)
        .with_allele_frequencies(vec![0.5, 0.5])
        .with_skeleton_allele_frequencies(vec![0.5, 0.5]);

    let schema = FeatureBuilder::new().optimize(&[root]);

    assert_eq!(schema.total_features, 1);
    assert_eq!(schema.sites.len(), 1);
    assert_eq!(schema.sites[0].snarl_id, "ld_block");
    assert_eq!(schema.sites[0].kind, FeatureKind::Flat);
    assert_eq!(schema.sites[0].class, SiteClass::Biallelic);
}

#[test]
fn nested_tie_prefers_decomposition() {
    let inner = leaf_with_freq("inner_snp", vec![0.6, 0.4]).with_parent_skeleton_alleles(vec![1]);
    let root = Snarl::compound("outer_sv", 3, 2, vec![inner])
        .with_allele_frequencies(vec![0.4, 0.35, 0.25])
        .with_skeleton_allele_frequencies(vec![0.7, 0.3]);

    let schema = FeatureBuilder::new().optimize(&[root]);

    assert_eq!(schema.total_features, 2);
    assert_eq!(schema.sites.len(), 2);
    assert_eq!(schema.sites[0].snarl_id, "outer_sv");
    assert_eq!(schema.sites[0].kind, FeatureKind::Skeleton);
    assert_eq!(schema.sites[1].snarl_id, "inner_snp");
    assert_eq!(schema.sites[1].kind, FeatureKind::Leaf);
    assert_eq!(schema.sites[1].parent_snarl_id.as_deref(), Some("outer_sv"));
    assert_eq!(
        schema.sites[1].conditional_on,
        vec![TraversalCondition {
            snarl_id: "outer_sv".to_string(),
            allowed_parent_skeleton_alleles: vec![1]
        }]
    );
}

#[test]
fn cyclic_snarl_is_single_feature() {
    let root = Snarl::cyclic("vntr");
    let schema = FeatureBuilder::new().optimize(&[root]);

    assert_eq!(schema.total_features, 1);
    assert_eq!(schema.sites[0].class, SiteClass::Cyclic);
    assert_eq!(schema.sites[0].kind, FeatureKind::Cyclic);
}

#[test]
fn biallelic_haploid_encoding_uses_standard_dosage() {
    assert_eq!(encode_haploid_acyclic(Some(0), 2), vec![Some(0.0)]);
    assert_eq!(encode_haploid_acyclic(Some(1), 2), vec![Some(1.0)]);
}

#[test]
fn multiallelic_haploid_encoding_uses_dummy_coding() {
    assert_eq!(
        encode_haploid_acyclic(Some(0), 4),
        vec![Some(0.0), Some(0.0), Some(0.0)]
    );
    assert_eq!(
        encode_haploid_acyclic(Some(2), 4),
        vec![Some(0.0), Some(1.0), Some(0.0)]
    );
}

#[test]
fn non_traversing_haplotype_is_missing() {
    assert_eq!(encode_haploid_acyclic(None, 2), vec![None]);
    assert_eq!(encode_haploid_acyclic(None, 4), vec![None, None, None]);
    assert_eq!(encode_haploid_cyclic(None), vec![None]);
}

#[test]
fn invariant_site_has_zero_columns() {
    assert_eq!(
        encode_haploid_acyclic(Some(0), 1),
        Vec::<Option<f64>>::new()
    );
    assert_eq!(encode_haploid_acyclic(None, 1), Vec::<Option<f64>>::new());
}

#[test]
#[should_panic(expected = "out of bounds")]
fn out_of_bounds_allele_panics() {
    let _ = encode_haploid_acyclic(Some(3), 3);
}

#[test]
#[should_panic(expected = "must be > 0")]
fn zero_allele_count_panics() {
    let _ = encode_haploid_acyclic(Some(0), 0);
}

#[test]
fn metadata_fields_are_propagated() {
    let leaf = leaf_with_freq("leaf", vec![0.7, 0.3]).with_genomic_region("chr1:10-20");
    let schema = FeatureBuilder::new().optimize(&[leaf]);
    assert_eq!(schema.sites.len(), 1);
    assert_eq!(schema.sites[0].allele_frequencies, vec![0.7, 0.3]);
    assert_eq!(
        schema.sites[0].genomic_region.as_deref(),
        Some("chr1:10-20")
    );
}

#[test]
fn skeleton_allele_frequencies_are_used_for_skeleton_site() {
    let inner = leaf_with_freq("inner", vec![0.8, 0.2]).with_parent_skeleton_alleles(vec![1]);
    let root = Snarl::compound("outer", 4, 2, vec![inner])
        .with_allele_frequencies(vec![0.6, 0.2, 0.15, 0.05])
        .with_skeleton_allele_frequencies(vec![0.9, 0.1]);
    let schema = FeatureBuilder::new().optimize(&[root]);

    assert_eq!(schema.sites[0].snarl_id, "outer");
    assert_eq!(schema.sites[0].kind, FeatureKind::Skeleton);
    assert_eq!(schema.sites[0].allele_frequencies, vec![0.9, 0.1]);
}

#[test]
fn nested_conditions_compose_across_multiple_levels() {
    let grandchild =
        leaf_with_freq("grandchild", vec![0.6, 0.4]).with_parent_skeleton_alleles(vec![0]);
    let child = Snarl::compound("child", 3, 2, vec![grandchild])
        .with_allele_frequencies(vec![0.5, 0.3, 0.2])
        .with_parent_skeleton_alleles(vec![1])
        .with_skeleton_allele_frequencies(vec![0.8, 0.2]);
    let root = Snarl::compound("root", 4, 2, vec![child])
        .with_allele_frequencies(vec![0.25, 0.25, 0.25, 0.25])
        .with_skeleton_allele_frequencies(vec![0.5, 0.5]);
    let schema = FeatureBuilder::new().optimize(&[root]);

    let grandchild_site = schema
        .sites
        .iter()
        .find(|site| site.snarl_id == "grandchild")
        .expect("grandchild site missing");
    assert_eq!(
        grandchild_site.conditional_on,
        vec![
            TraversalCondition {
                snarl_id: "root".to_string(),
                allowed_parent_skeleton_alleles: vec![1],
            },
            TraversalCondition {
                snarl_id: "child".to_string(),
                allowed_parent_skeleton_alleles: vec![0],
            },
        ]
    );
}

#[test]
fn child_not_conditioned_when_parent_has_no_skeleton_choice() {
    let child = leaf_with_freq("child", vec![0.7, 0.3]);
    let root = Snarl::compound("root", 5, 1, vec![child]).with_allele_frequencies(vec![0.2; 5]);
    let schema = FeatureBuilder::new().optimize(&[root]);

    assert_eq!(schema.sites.len(), 1);
    assert_eq!(schema.sites[0].snarl_id, "child");
    assert_eq!(
        schema.sites[0].conditional_on,
        vec![TraversalCondition {
            snarl_id: "root".to_string(),
            allowed_parent_skeleton_alleles: vec![0]
        }]
    );
}

#[test]
fn reference_aware_biallelic_encoding() {
    assert_eq!(
        encode_haploid_acyclic_with_reference(Some(1), 2, 1),
        vec![Some(0.0)]
    );
    assert_eq!(
        encode_haploid_acyclic_with_reference(Some(0), 2, 1),
        vec![Some(1.0)]
    );
}

#[test]
fn reference_aware_multiallelic_encoding() {
    // k=4, reference=2 -> columns correspond to alleles [0,1,3].
    assert_eq!(
        encode_haploid_acyclic_with_reference(Some(2), 4, 2),
        vec![Some(0.0), Some(0.0), Some(0.0)]
    );
    assert_eq!(
        encode_haploid_acyclic_with_reference(Some(3), 4, 2),
        vec![Some(0.0), Some(0.0), Some(1.0)]
    );
}

#[test]
fn reference_allele_picks_most_common_frequency() {
    assert_eq!(reference_allele_index(&[0.1, 0.5, 0.4]), Some(1));
    assert_eq!(reference_allele_index(&[]), None);
}

#[test]
fn diploid_sum_keeps_traversing_haplotype_when_other_is_missing() {
    let left = vec![Some(1.0)];
    let right = vec![None];
    assert_eq!(sum_diploid_site(&left, &right), vec![Some(1.0)]);
}

#[test]
fn diploid_sum_adds_observed_haplotypes() {
    let left = vec![Some(1.0), Some(0.0), None];
    let right = vec![Some(1.0), None, Some(1.0)];
    assert_eq!(
        sum_diploid_site(&left, &right),
        vec![Some(2.0), Some(0.0), Some(1.0)]
    );
}

#[test]
#[should_panic(expected = "equal length")]
fn diploid_sum_panics_on_length_mismatch() {
    let _ = sum_diploid_site(&[Some(1.0)], &[Some(1.0), Some(0.0)]);
}

#[test]
fn infers_counts_and_frequencies_from_panel_walks() {
    let topology = SnarlTopology::acyclic(
        "outer",
        1,
        4,
        vec![SnarlTopology::acyclic("inner", 2, 3, vec![])],
    );
    let panel = vec![
        walk("h1", &[(1, true), (2, true), (3, true), (4, true)]),
        walk("h2", &[(1, true), (5, true), (4, true)]),
        walk(
            "h3",
            &[(1, true), (2, true), (6, true), (3, true), (4, true)],
        ),
        walk("h4", &[(1, true), (2, true), (3, true), (4, true)]),
        walk("h5", &[(1, true), (5, true), (4, true)]),
        walk(
            "h6",
            &[(1, true), (2, true), (6, true), (3, true), (4, true)],
        ),
    ];

    let inferred = infer_snarl_panel(&[topology], &panel);
    let root = &inferred.roots[0];
    assert_eq!(root.id, "outer");
    assert_eq!(root.allele_count, 3);
    assert_eq!(root.skeleton_allele_count, Some(2));
    assert_eq!(
        root.allele_frequencies,
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    );
    assert_eq!(
        root.skeleton_allele_frequencies,
        Some(vec![2.0 / 3.0, 1.0 / 3.0])
    );
    assert_eq!(root.children.len(), 1);
    assert_eq!(root.children[0].parent_skeleton_alleles, Some(vec![0]));
}

#[test]
fn runtime_encodes_nested_missingness_from_path_traversal() {
    let topology = SnarlTopology::acyclic(
        "outer",
        1,
        4,
        vec![SnarlTopology::acyclic("inner", 2, 3, vec![])],
    );
    let panel = vec![
        walk("h1", &[(1, true), (2, true), (3, true), (4, true)]),
        walk("h2", &[(1, true), (5, true), (4, true)]),
        walk(
            "h3",
            &[(1, true), (2, true), (6, true), (3, true), (4, true)],
        ),
        walk("h4", &[(1, true), (2, true), (3, true), (4, true)]),
        walk("h5", &[(1, true), (5, true), (4, true)]),
        walk(
            "h6",
            &[(1, true), (2, true), (6, true), (3, true), (4, true)],
        ),
    ];

    let runtime = build_runtime_from_walks(&[topology], &panel, HashMap::new());
    assert_eq!(runtime.schema.sites.len(), 2);
    assert_eq!(runtime.schema.sites[0].snarl_id, "outer");
    assert_eq!(runtime.schema.sites[1].snarl_id, "inner");

    let skip_walk = walk("skip", &[(1, true), (5, true), (4, true)]);
    let inner_walk = walk("inner", &[(1, true), (2, true), (3, true), (4, true)]);

    let encoded_skip = runtime.encode_haplotype(&skip_walk);
    let encoded_inner = runtime.encode_haplotype(&inner_walk);

    assert_eq!(encoded_skip.len(), runtime.schema.total_features);
    assert_eq!(encoded_inner.len(), runtime.schema.total_features);
    assert_eq!(encoded_skip[0], Some(1.0));
    assert_eq!(encoded_skip[1], None);
    assert_eq!(encoded_inner[0], Some(0.0));
    assert_eq!(encoded_inner[1], Some(0.0));
}

#[test]
fn singleton_alleles_pool_into_other() {
    let topology = SnarlTopology::acyclic("site", 1, 4, vec![]);
    let panel = vec![
        walk("h1", &[(1, true), (2, true), (4, true)]),
        walk("h2", &[(1, true), (2, true), (4, true)]),
        walk("h3", &[(1, true), (3, true), (4, true)]),
        walk("h4", &[(1, true), (5, true), (4, true)]),
    ];
    let runtime = build_runtime_from_walks(&[topology], &panel, HashMap::new());
    let site = &runtime.schema.sites[0];
    assert_eq!(site.allele_count, 2);
    assert_eq!(site.allele_frequencies, vec![0.5, 0.5]);
}

#[test]
fn noncanonical_snarl_traversals_are_not_forced_to_missing() {
    let topology = SnarlTopology::acyclic("site", 2, 3, vec![]);
    let panel = vec![
        walk("h1", &[(1, true), (2, true), (4, true)]),
        walk("h2", &[(1, true), (2, true), (5, true)]),
        walk("h3", &[(1, true), (3, true), (4, true)]),
        walk("h4", &[(1, true), (3, true), (5, true)]),
    ];
    let runtime = build_runtime_from_walks(&[topology], &panel, HashMap::new());
    assert_eq!(runtime.schema.sites.len(), 1);
    let site = &runtime.schema.sites[0];
    assert_eq!(site.allele_count, 2);

    let encoded_entry = runtime.encode_haplotype(&walk("entry_only", &[(2, true)]));
    let encoded_exit = runtime.encode_haplotype(&walk("exit_only", &[(3, true)]));
    assert_eq!(encoded_entry.len(), runtime.schema.total_features);
    assert_eq!(encoded_exit.len(), runtime.schema.total_features);
    assert!(encoded_entry[0].is_some());
    assert!(encoded_exit[0].is_some());
}

#[test]
fn cyclic_repeat_count_not_stuck_at_one() {
    let topology = SnarlTopology::cyclic("vntr", 1, 2);
    let panel = vec![walk("p1", &[(1, true), (3, true), (2, true)])];
    let runtime = build_runtime_from_walks(&[topology], &panel, HashMap::new());
    assert_eq!(runtime.schema.sites.len(), 1);
    assert_eq!(runtime.schema.sites[0].class, SiteClass::Cyclic);

    let repeated = walk(
        "repeat2",
        &[(1, true), (3, true), (2, true), (3, true), (2, true)],
    );
    let encoded = runtime.encode_haplotype(&repeated);
    assert_eq!(encoded.len(), 1);
    assert_eq!(encoded[0], Some(2.0));
}

#[test]
fn catalog_binary_round_trip() {
    let topology = SnarlTopology::acyclic(
        "outer",
        1,
        4,
        vec![SnarlTopology::acyclic("inner", 2, 3, vec![])],
    );
    let panel = vec![
        walk("h1", &[(1, true), (2, true), (3, true), (4, true)]),
        walk("h2", &[(1, true), (5, true), (4, true)]),
        walk(
            "h3",
            &[(1, true), (2, true), (6, true), (3, true), (4, true)],
        ),
        walk("h4", &[(1, true), (2, true), (3, true), (4, true)]),
        walk("h5", &[(1, true), (5, true), (4, true)]),
        walk(
            "h6",
            &[(1, true), (2, true), (6, true), (3, true), (4, true)],
        ),
    ];
    let runtime = build_runtime_from_walks(&[topology], &panel, HashMap::new());
    let catalog = catalog_from_runtime(&runtime);

    let tmp = tempdir().expect("tempdir create failed");
    let features_path = tmp.path().join("features.bin");
    let traversals_path = tmp.path().join("traversals.bin");
    catalog
        .write_bins(&features_path, &traversals_path)
        .expect("write bins failed");
    let loaded =
        FeatureCatalog::read_bins(&features_path, &traversals_path).expect("read bins failed");

    assert_eq!(loaded.features.len(), catalog.features.len());
    assert_eq!(loaded.features[0].feature_id, 0);
    assert_eq!(loaded.features[0].boundary_entry, 1);
    assert_eq!(loaded.features[0].boundary_exit, 4);
    assert_eq!(loaded.features[0].k, 2);
    assert!(!loaded.features[0].is_cyclic);
    assert_eq!(loaded.features[1].parent_feature_id, Some(0));
}

#[test]
fn catalog_feature_id_uses_feature_vector_position() {
    let topology = SnarlTopology::acyclic("multi", 1, 4, vec![]);
    let panel = vec![
        walk("h1", &[(1, true), (2, true), (4, true)]),
        walk("h2", &[(1, true), (2, true), (4, true)]),
        walk("h3", &[(1, true), (3, true), (4, true)]),
        walk("h4", &[(1, true), (5, true), (4, true)]),
        walk("h5", &[(1, true), (5, true), (4, true)]),
        walk("h6", &[(1, true), (6, true), (4, true)]),
    ];
    let runtime = build_runtime_from_walks(&[topology], &panel, HashMap::new());
    let catalog = catalog_from_runtime(&runtime);
    assert_eq!(runtime.schema.sites[0].feature_start, 0);
    assert_eq!(catalog.features[0].feature_id, 0);
    assert_eq!(catalog.features[0].k, 3);
}

#[test]
fn manifest_round_trip() {
    let manifest = FeatureCatalogManifest {
        graph_build_id: "abc123".to_string(),
        graph_construction_pipeline: "Minigraph-Cactus 2.7".to_string(),
        reference_coordinates: "CHM13".to_string(),
        hprc_release: "HPRC v2.0".to_string(),
        haplotype_count: 464,
        snarl_decomposition_tool: "vg snarls 1.59.0".to_string(),
    };

    let tmp = tempdir().expect("tempdir create failed");
    let path = tmp.path().join("manifest.tsv");
    write_manifest(&path, &manifest).expect("write manifest failed");
    let loaded = read_manifest(&path).expect("read manifest failed");
    assert_eq!(loaded, manifest);
}

#[test]
fn write_dir_with_manifest_and_read_dir() {
    let catalog = FeatureCatalog {
        features: vec![graphome::pangenome_catalog::CatalogFeature {
            feature_id: 0,
            boundary_entry: 1,
            boundary_exit: 2,
            k: 2,
            is_cyclic: false,
            parent_feature_id: None,
            alleles: vec![
                CatalogAlleleTraversal {
                    node_ids: vec![1, 2],
                },
                CatalogAlleleTraversal {
                    node_ids: vec![1, 3, 2],
                },
            ],
        }],
    };
    let manifest = FeatureCatalogManifest {
        graph_build_id: "hash".to_string(),
        graph_construction_pipeline: "PGGB".to_string(),
        reference_coordinates: "GRCh38".to_string(),
        hprc_release: "HPRC v2.0".to_string(),
        haplotype_count: 464,
        snarl_decomposition_tool: "vg snarls 1.x".to_string(),
    };

    let tmp = tempdir().expect("tempdir create failed");
    catalog
        .write_dir_with_manifest(tmp.path(), &manifest)
        .expect("write dir with manifest failed");
    let loaded_catalog = FeatureCatalog::read_dir(tmp.path()).expect("read dir failed");
    let loaded_manifest =
        read_manifest(&tmp.path().join("manifest.tsv")).expect("read manifest failed");

    assert_eq!(loaded_catalog, catalog);
    assert_eq!(loaded_manifest, manifest);
}

#[test]
fn catalog_other_allele_is_empty_path() {
    let catalog = FeatureCatalog {
        features: vec![graphome::pangenome_catalog::CatalogFeature {
            feature_id: 0,
            boundary_entry: 10,
            boundary_exit: 11,
            k: 2,
            is_cyclic: false,
            parent_feature_id: None,
            alleles: vec![
                CatalogAlleleTraversal {
                    node_ids: vec![10, 11],
                },
                CatalogAlleleTraversal { node_ids: vec![] },
            ],
        }],
    };
    let tmp = tempdir().expect("tempdir create failed");
    let features_path = tmp.path().join("features.bin");
    let traversals_path = tmp.path().join("traversals.bin");
    catalog
        .write_bins(&features_path, &traversals_path)
        .expect("write bins failed");
    let loaded =
        FeatureCatalog::read_bins(&features_path, &traversals_path).expect("read bins failed");
    assert_eq!(loaded.features[0].alleles[1].node_ids, Vec::<u64>::new());
}

#[test]
fn runtime_encodes_from_node_value_traversals() {
    let topology = SnarlTopology::acyclic(
        "outer",
        1,
        4,
        vec![SnarlTopology::acyclic("inner", 2, 3, vec![])],
    );
    let panel = vec![
        walk("h1", &[(1, true), (2, true), (3, true), (4, true)]),
        walk("h2", &[(1, true), (5, true), (4, true)]),
        walk(
            "h3",
            &[(1, true), (2, true), (6, true), (3, true), (4, true)],
        ),
        walk("h4", &[(1, true), (2, true), (3, true), (4, true)]),
        walk("h5", &[(1, true), (5, true), (4, true)]),
        walk(
            "h6",
            &[(1, true), (2, true), (6, true), (3, true), (4, true)],
        ),
    ];
    let runtime = build_runtime_from_walks(&[topology], &panel, HashMap::new());

    let skip = vec![
        NodeTraversal {
            node_id: 1,
            value: 1.0,
        },
        NodeTraversal {
            node_id: 5,
            value: 1.0,
        },
        NodeTraversal {
            node_id: 4,
            value: 1.0,
        },
    ];
    let ins = vec![
        NodeTraversal {
            node_id: 1,
            value: 1.0,
        },
        NodeTraversal {
            node_id: 2,
            value: 1.0,
        },
        NodeTraversal {
            node_id: 3,
            value: 1.0,
        },
        NodeTraversal {
            node_id: 4,
            value: 1.0,
        },
    ];

    let encoded_skip = runtime.encode_haplotype_node_values(&skip);
    let encoded_ins = runtime.encode_haplotype_node_values(&ins);
    assert_eq!(encoded_skip[0], Some(1.0));
    assert_eq!(encoded_skip[1], None);
    assert_eq!(encoded_ins[0], Some(0.0));
    assert_eq!(encoded_ins[1], Some(0.0));
}

fn walk(id: &str, nodes: &[(usize, bool)]) -> HaplotypeWalk {
    HaplotypeWalk {
        id: id.to_string(),
        steps: nodes
            .iter()
            .map(|(node_id, forward)| HaplotypeStep {
                node_id: *node_id,
                orientation: if *forward {
                    Orientation::Forward
                } else {
                    Orientation::Reverse
                },
            })
            .collect(),
    }
}

#[test]
fn loads_topology_from_tsv() {
    let mut file = NamedTempFile::new().expect("tempfile create failed");
    writeln!(
        file,
        "outer\tacyclic\t1\t4\t.\tchr1:1-10\ninner\tacyclic\t2\t3\touter\tchr1:3-8"
    )
    .expect("write failed");

    let roots = load_topology_tsv(file.path().to_str().expect("utf8 path")).expect("load failed");
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].id, "outer");
    assert_eq!(roots[0].children.len(), 1);
    assert_eq!(roots[0].children[0].id, "inner");
    assert_eq!(
        roots[0].children[0].genomic_region.as_deref(),
        Some("chr1:3-8")
    );
}

#[test]
fn allows_empty_parent_skeleton_alleles_for_never_traversed_child() {
    let child = leaf_with_freq("child", vec![1.0]).with_parent_skeleton_alleles(vec![]);
    let root = Snarl::compound("root", 2, 2, vec![child])
        .with_allele_frequencies(vec![0.5, 0.5])
        .with_skeleton_allele_frequencies(vec![0.5, 0.5]);
    let schema = FeatureBuilder::new().optimize(&[root]);
    // Child has no variation and should not become a selected feature.
    assert_eq!(schema.sites.len(), 1);
    assert_eq!(schema.sites[0].snarl_id, "root");
}
