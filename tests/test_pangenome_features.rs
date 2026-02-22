use graphome::pangenome_features::{
    encode_haploid_acyclic, encode_haploid_cyclic, sum_diploid_site, FeatureBuilder, FeatureKind,
    SiteClass, Snarl,
};

#[test]
fn ld_block_prefers_flattening() {
    let children = (0..50)
        .map(|i| Snarl::leaf(format!("leaf_{i}"), 2))
        .collect();
    let root = Snarl::compound("ld_block", 2, 2, children);

    let schema = FeatureBuilder::new().optimize(&[root]);

    assert_eq!(schema.total_features, 1);
    assert_eq!(schema.sites.len(), 1);
    assert_eq!(schema.sites[0].snarl_id, "ld_block");
    assert_eq!(schema.sites[0].kind, FeatureKind::Flat);
    assert_eq!(schema.sites[0].class, SiteClass::Biallelic);
}

#[test]
fn nested_tie_prefers_decomposition() {
    let inner = Snarl::leaf("inner_snp", 2);
    let root = Snarl::compound("outer_sv", 3, 2, vec![inner]);

    let schema = FeatureBuilder::new().optimize(&[root]);

    assert_eq!(schema.total_features, 2);
    assert_eq!(schema.sites.len(), 2);
    assert_eq!(schema.sites[0].snarl_id, "outer_sv");
    assert_eq!(schema.sites[0].kind, FeatureKind::Skeleton);
    assert_eq!(schema.sites[1].snarl_id, "inner_snp");
    assert_eq!(schema.sites[1].kind, FeatureKind::Leaf);
    assert_eq!(schema.sites[1].parent_snarl_id.as_deref(), Some("outer_sv"));
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
