# Pangenome Feature Representation for Polygenic Prediction

## 1. Overview

This document specifies a method for constructing per-individual numeric feature vectors from a human pangenome graph for use in polygenic score (PGS) estimation. The method produces a feature representation that:

- Captures SNPs, indels, structural variants, inversions, and tandem repeats in a single unified framework.
- Handles nested variation (e.g., a SNP inside a structural variant) with an encoding that allows shared structural effects to be estimated once, not redundantly.
- Achieves the minimum number of features that resolves all non-singleton haplotype variation observed in the reference panel (alleles observed in ≥2 HPRC haplotypes are individually resolved; singletons are pooled).
- Uses standard allele dosage at biallelic sites and standard dummy coding at multi-allelic sites.

The method defines the **representation layer**: the transformation from a pangenome graph and individual node traversals into a numeric feature vector. It specifies both the encoding rules and the concrete output format. It does not specify the downstream PGS estimation model or the upstream genotype-to-graph projection pipeline.

---

## 2. Motivation

Current polygenic scores are built on SNP genotypes. This representation has blind spots:

- **Structural variants** (insertions, deletions, inversions, complex rearrangements) are absent or poorly tagged by nearby SNPs. SVs comprise a substantial fraction of inter-individual nucleotide divergence and have known phenotypic effects.
- **Multi-allelic sites** are awkwardly handled or collapsed into biallelic approximations.
- **Tandem repeats** (VNTRs, STRs) are invisible to SNP arrays. Copy-number variation at these loci can affect gene expression and disease risk.
- **Nested variation** — structural variants containing internal SNPs, overlapping indels, inversions with internal polymorphism — cannot be decomposed into independent biallelic sites without information loss.

The pangenome graph represents all of these variant classes as topological features: bubbles, nested subgraphs, and cycles. A haplotype is a path through the graph. The points where paths diverge and rejoin define the variant sites ("snarls").

The representation specified here extracts features from these snarls, producing a numeric vector that captures the full spectrum of variation accessible from the pangenome. In simple regions, it reduces exactly to standard allele dosage. In complex regions, it captures variation that SNP-based PGS cannot.

---

## 3. Inputs

### 3.1 Pangenome graph and haplotype index (GBZ)

The primary computational input is a **GBZ file**: vg's compressed format bundling the pangenome graph and the GBWT (Graph BWT) haplotype index into a single artifact. The GBZ contains both the graph topology and the embedded reference haplotype paths, providing everything needed for snarl decomposition, feature site selection, and allele definition.

The recommended source is the **HPRC v2.0 pangenome**, constructed from phased haplotype assemblies of 232 individuals (~464 haplotypes). Both Minigraph-Cactus and PGGB graphs are suitable; the method is graph-construction-agnostic. The graph is available in CHM13 and GRCh38 reference coordinates.

"Base-level" graph resolution is assumed: the graph resolves variation down to single nucleotides, so individual SNPs are distinct bubbles. The graph is treated as a fixed reference resource, analogous to hg38 for SNP-based analyses.

A GFA representation of the same graph may be used for inspection or interchange but is not the primary computational input (the HPRC v2.0 MC-CHM13 GFA is ~60GB uncompressed vs. ~5GB for the GBZ).

### 3.2 Snarl decomposition

The snarl decomposition of the graph, computable via `vg snarls` or equivalent tooling. This produces a hierarchy of snarls and chains (the **snarl tree**), where snarls can be nested inside other snarls. The snarl decomposition is derivable from the GBZ and does not require a separate input file.

### 3.3 Per-individual node traversals

For each individual to be scored, a per-haplotype list of graph nodes traversed. Each entry is a (node_id, value) pair, where value is 1 for hard genotype calls or a float in [0, 1] for probabilistic calls. How these node traversals are obtained (graph alignment, imputation, linear-to-graph liftover) is out of scope for this spec.

### 3.4 Feature catalog manifest

To ensure reproducibility, the feature catalog is accompanied by a manifest recording:

| Field | Description |
|---|---|
| Graph build ID | Hash of the GBZ file used |
| Graph construction pipeline | Minigraph-Cactus or PGGB, with version and parameters |
| Reference coordinates | CHM13 or GRCh38 |
| HPRC release | Release identifier (e.g., HPRC v2.0) |
| Haplotype count | Number of HPRC haplotypes defining alleles (e.g., ~464 for v2.0) |
| Snarl decomposition tool | Tool and version used (e.g., `vg snarls` from vg 1.x.x) |

Different manifest values produce different feature catalogs. Models trained on one catalog are not portable to another without re-mapping.

---

## 4. Definitions

**Snarl.** A subgraph bounded by two boundary nodes (entry and exit), where all paths from the rest of the graph enter through one boundary and exit through the other. Snarls correspond to variant sites. Formally, the boundary node sides are separable (splitting them disconnects the snarl from the rest of the graph) and minimal (no internal node is separable with the boundaries).

**Chain.** A sequence of consecutive snarls sharing boundary nodes, alternating with the shared nodes between them. In vg's snarl tree, the hierarchy alternates between snarls and chains: each snarl contains child chains, and each chain contains child snarls and nodes. Chains correspond to runs of adjacent variant sites — roughly analogous to LD blocks in linear genomics.

**Leaf snarl.** A snarl containing no child snarls. The minimal, irreducible unit of variation. A biallelic SNP is a leaf snarl with two path alleles.

**Compound snarl.** A snarl containing one or more child snarls (nested variation). Example: a deletion that contains two internal SNPs on one branch.

**Snarl tree.** The hierarchy of snarls and chains defined by their nesting relationships, computable from the graph topology. In vg, this is represented by the Distance Index or Snarl Manager.

**Path allele.** At a given snarl, each distinct traversal from entry to exit observed among the HPRC haplotypes constitutes one allele. The number of distinct path alleles at a snarl is *k*.

**Netgraph / Skeleton.** At a compound snarl, the **netgraph** (vg terminology) is the view of the snarl where each child chain is collapsed to a single placeholder node. The distinct HPRC haplotype traversals through the netgraph define the **skeleton routes**: these capture "which outer route was taken" independent of choices made inside children. Each child is abstracted to its snarl ID only — the specific allele taken within the child is not distinguished at the skeleton level. The number of distinct skeleton routes is *k*_skel.

**Cyclic snarl.** A snarl whose netgraph contains a directed cycle. This arises from tandem repeats or VNTRs where the graph represents variable copy number via back-edges. Detection rule: a snarl is cyclic if and only if its netgraph (with child chains abstracted) contains a directed cycle. Unrolled repeats represented as acyclic multi-path bubbles are *not* cyclic snarls; they are standard multi-allelic snarls.

**Non-start-end-connected snarl.** A valid snarl where no path connects the entry boundary to the exit boundary through the snarl interior. These can arise as artifacts of the snarl-finding algorithm, particularly when the snarl tree is rooted on a long internal sequence. Such snarls are treated as standard snarls; their alleles are defined by observed HPRC traversals, which may enter or exit through non-canonical routes.

**Feature site.** A snarl selected by the optimization algorithm (Section 6) to contribute features to the final vector. Not every snarl in the tree becomes a feature site — some are absorbed into their parent's representation.

---

## 5. Encoding: Snarl Dosage

### 5.1 Principle

At each feature site, individuals who traverse the snarl receive a standard genotype encoding (allele dosage for biallelic sites, dummy coding for multi-allelic sites). Individuals who do not traverse the snarl — because a routing decision at a parent snarl directed them elsewhere — receive **NA (missing)** for all features at that snarl.

This handles nested variation correctly. When a snarl exists only on one branch of a parent snarl (e.g., a SNP inside an insertion), individuals who took the other branch (e.g., skipped the insertion) are marked missing at the inner snarl. They are excluded entirely from estimating the inner snarl's effect. The outer snarl has its own feature capturing the structural choice (insertion vs. skip), estimated from all individuals. The inner snarl is a standard biallelic site estimated from the subpopulation that carries the insertion — with full statistical power, correct heterozygote contributions, and no contamination from non-carriers.

At non-nested sites (the vast majority of the genome), every individual traverses the snarl, no values are missing, and the encoding is standard allele dosage — identical to existing GWAS/PGS practice.

### 5.2 Biallelic snarls (k = 2)

One feature: **allele dosage**. Designate the more common HPRC allele as reference (ref) and the less common as alternative (alt). For a haploid individual:

| Allele carried | Feature value |
|---|---|
| Reference allele | 0 |
| Alternative allele | 1 |

For a diploid individual (summing two haplotypes), the feature value is in {0, 1, 2} — the standard allele dosage used in GWAS.

This is identical to existing practice. No transformation is needed for compatibility with any PGS tool.

### 5.3 Multi-allelic snarls (k > 2)

*k* − 1 features using **dummy coding**. Designate the most common HPRC allele as reference (omitted category). Each remaining allele gets one indicator feature (0 or 1 per haplotype; 0, 1, or 2 per diploid individual). Each coefficient is the effect of that allele relative to the reference.

#### Singleton pooling

Alleles observed in only one HPRC haplotype (singletons) are pooled into a single **OTHER** allele. The OTHER allele receives one dummy indicator, like any other non-reference allele. Singletons are pooled because: a single observation in ~464 haplotypes provides no estimation power, singletons are enriched for assembly errors, and they inflate feature count at complex snarls without benefit.

After pooling, *k* is the number of remaining alleles (common alleles + OTHER). A snarl with 20 observed alleles, 5 seen in ≥2 haplotypes and 15 singletons, produces 5 features: 4 common allele dummies + 1 OTHER dummy.

Every allele observed in ≥2 HPRC haplotypes is individually resolved.

### 5.4 Cyclic snarls (tandem repeats / VNTRs)

One feature: the **repeat count** (number of cycle traversals), encoded as a non-negative integer.

For diploid individuals, sum the two haplotype counts.

Rationale: tandem repeat effects typically scale with copy number. A single continuous feature captures this in one dimension, regardless of how many distinct copy-number alleles exist. Dummy coding *k* repeat lengths as *k* − 1 indicators would waste degrees of freedom on what is fundamentally a one-dimensional dose-response signal.

### 5.5 Non-traversing individuals

For any snarl that an individual's haplotype does not traverse (because a routing decision at a parent snarl directed them elsewhere), all features at that snarl are set to **NA (missing)**.

This is the biologically correct representation: the variant site does not exist in this individual's genome. They do not carry the reference allele, the alternative allele, or any allele. The site is absent.

Downstream estimation methods handle this by excluding the individual from that feature's effect estimation. This is standard missing-data handling in regression: the individual contributes to coefficient estimates at all other features but is excluded from this one. For summary-statistics-based PGS methods, the per-snarl GWAS is computed using only traversing individuals.

At non-nested sites (the vast majority of features), every individual traverses the snarl, and no values are missing. Missingness arises only at the small fraction of snarls that are conditional on a parent structural variant. The missingness pattern is structurally determined (by the snarl tree), not random.

### 5.6 Diploid encoding

The diploid feature value is computed per-feature from two independent haplotype-level calls. Each haplotype at a given snarl is either traversing (producing a 0 or 1 for that feature column) or non-traversing (contributing nothing).

**Rules:**
- If both haplotypes traverse the snarl: diploid value = sum of two haplotype values. Standard {0, 1, 2} dosage.
- If exactly one haplotype traverses: diploid value = that haplotype's value alone (0 or 1). The non-traversing haplotype contributes nothing — it is not counted as 0, it is absent.
- If neither haplotype traverses: diploid value = NA.

**Complete truth table for an inner biallelic SNP (T=ref, G=alt) inside an insertion:**

| Haplotype 1 | Haplotype 2 | Outer insertion dosage | Inner SNP dosage |
|---|---|---|---|
| skip | skip | 0 | NA |
| skip | ins+T | 1 | 0 |
| skip | ins+G | 1 | 1 |
| ins+T | ins+T | 2 | 0 |
| ins+T | ins+G | 2 | 1 |
| ins+G | ins+G | 2 | 2 |

This produces the correct additive scores. For example, under a model with outer effect β_out and inner effect β_in:

- skip/ins+G scores: β_out(1) + β_in(1)
- ins+T/ins+G scores: β_out(2) + β_in(1)

The difference is exactly β_out — one additional copy of the insertion.

**Hemizygosity.** When one haplotype traverses and the other skips, the inner feature value is based on a single haplotype observation. This is analogous to X-chromosome dosage in males: the value range is {0, 1} rather than {0, 1, 2}, and the effective sample size for estimating the inner effect is reduced. Standard GWAS tools do not adjust for per-individual-per-site ploidy variation; this is a known limitation that affects only the small fraction of sites that are conditional on a structural variant. Effective ploidy (0, 1, or 2) per individual per feature site is derivable at runtime from the parent snarl's genotype.

### 5.7 Relationship to the node traversal matrix

#### The underlying model

Both the snarl encoding and any alternative pangenome feature representation are instances of a single underlying model: the **path-additive model**.

An individual's polygenic score is the sum of learned weights along the elements their haplotypes traverse:

PGS_i = Σ_{e ∈ path(i)} w_e

The "elements" can be graph nodes, graph edges, snarl alleles, or any other decomposition of the graph into pieces with learnable weights. All such decompositions produce predictions in the same subspace — they differ only in parameterization, not in expressive power.

#### Node traversal as the primitive representation

The most direct representation of a pangenome genotype is the **node traversal matrix** X_node ∈ ℝ^{n×p}, where X_{h,j} indicates whether (and with what confidence) haplotype h traverses variable node j. For hard genotype calls, entries are in {0, 1}. For probabilistic calls, entries are floats in [0, 1]. p = |variable nodes|.

This matrix is rank-deficient. The graph imposes **flow conservation constraints**: at every snarl boundary, traversal counts entering equal those exiting. These structural constraints make columns linearly dependent. Let rank(X_node) = r < p.

#### The snarl encoding is a linear projection of the node matrix

The snarl feature matrix Z_snarl is related to the node matrix by a linear projection:

Z_snarl = X_node × A

where A ∈ ℝ^{p×r} is a projection matrix determined entirely by graph topology and the snarl decomposition. Specifically, A is constructed by three deterministic operations on the node matrix:

1. **Merge equivalence classes.** Nodes with identical traversal patterns across all HPRC haplotypes (e.g., backbone nodes of the same structural variant) are merged into single features.

2. **Apply flow conservation.** At each snarl boundary, the constraint that entering traversals equal exiting traversals eliminates one degree of freedom per constraint. The reference allele's node group is dropped (absorbed into the intercept).

3. **Eliminate remaining linear dependencies.** Any residual collinearity from the snarl hierarchy is removed, choosing to retain features that align with the snarl tree's biological hierarchy (keeping skeleton/backbone features, eliminating child-level redundancies).

The result is a full-rank matrix with r columns, where r = Σ(k_s − 1) across all selected feature sites. This r is the number of haplotype degrees of freedom in the pangenome — the true dimensionality of genetic variation as captured by the reference panel.

#### Both representations span the same prediction space

Because Z_snarl = X_node × A and A has rank r, the column spaces satisfy col(Z_snarl) ⊆ col(X_node). Because Z_snarl has rank r = rank(X_node), the column spaces are equal: col(Z_snarl) = col(X_node). The set of achievable predictions is identical regardless of which representation is used.

Coefficients transform between representations: β_node = A × β_snarl. A snarl-level effect is distributed back onto its constituent nodes, weighted by the projection. A node-level effect is aggregated up to snarl level.

#### Why the snarl encoding is preferred in practice

Both representations produce the same predictions under OLS. Under regularized estimation, they differ (see Section 9). The snarl encoding is preferred for practical reasons:

- **Compact:** r features instead of p >> r.
- **Interpretable:** each feature corresponds to a variant site with a genetic interpretation.
- **Compatible:** standard {0, 1, 2} dosage at most sites, directly usable by existing tools.
- **Correct nesting:** NA for non-traversing individuals at nested sites, rather than a numeric zero that conflates "absent" with "reference allele."

The node traversal matrix remains the conceptual foundation. The snarl encoding is a specific full-rank basis for its column space, chosen for interpretability and compatibility.

---

## 6. Feature Site Selection: Snarl Tree Optimization

Not every snarl in the tree becomes a feature site. The optimization algorithm selects the set of snarls that minimizes total feature count while remaining lossless with respect to HPRC haplotype paths.

### 6.1 The problem

A compound snarl can be represented in two ways:

**Flat:** treat the compound snarl as a single feature site. Its alleles are the complete HPRC paths from entry to exit. Cost: *k* − 1 features, where *k* is the number of distinct full-length paths.

**Decomposed:** represent the skeleton as one feature site and each child snarl as a separate feature site (recursively optimized). Cost: (k_skel − 1) + Σ cost(child_i), where k_skel is the number of distinct skeleton routes.

In high-LD regions, many consecutive snarls have few distinct joint haplotype paths. Flattening these into a compound feature achieves lossless compression: 50 biallelic SNPs in perfect LD have only 2 haplotype paths, so 1 feature replaces 50.

In regions with nested conditional variation (e.g., a SNP inside an insertion), decomposition is preferred because it separates the outer structural effect from the inner conditional effect, enabling the outer effect to be estimated with full statistical power.

### 6.2 The algorithm

Traverse the snarl tree bottom-up. At each snarl *s*:

**If *s* is a leaf snarl:**

cost(*s*) = k_s − 1

**If *s* is a cyclic snarl:**

cost(*s*) = 1 (repeat count feature)

**If *s* is a compound acyclic snarl:**

1. Compute flat_cost(*s*) = k_s − 1, where k_s = number of distinct HPRC paths through the entire snarl.
2. Compute skeleton_cost(*s*) = k_skel − 1, where k_skel = number of distinct skeleton routes (child traversals replaced by placeholders).
3. Compute decomp_cost(*s*) = skeleton_cost(*s*) + Σ cost(child_i), where each child's cost has already been optimized.
4. If flat_cost < decomp_cost: represent *s* as a flat feature site. Mark all descendants as absorbed (they do not become feature sites).
5. If flat_cost ≥ decomp_cost: decompose. The skeleton of *s* becomes a feature site (if k_skel > 1). Each child becomes a feature site (using its own optimized representation).

**When flat_cost equals decomp_cost, prefer decomposition.** The decomposed representation produces smaller-arity categorical sites (more stable allele calling for new individuals), keeps feature groups aligned with the snarl tree, and avoids large allele catalogs that arise from flattening when the column count is equal but interpretability differs.

#### Computing skeleton routes (k_skel)

To compute the skeleton routes for a compound snarl *s*:

1. For each HPRC haplotype that traverses *s*, extract the subpath from entry boundary to exit boundary.
2. For each child snarl *c* of *s*: identify the segment of the subpath that traverses *c* and replace it with a single placeholder token `<child_snarl_id>`. The placeholder is the child's snarl ID only — the specific allele taken within the child is **not** recorded. This is the key abstraction: the skeleton sees which children were visited and in what order, but not what happened inside them.
3. Hash the resulting token sequence (mix of literal graph nodes and placeholder tokens) to produce a skeleton signature.
4. k_skel = number of distinct skeleton signatures across all HPRC haplotypes traversing *s*.

This is equivalent to computing distinct paths through the snarl's **netgraph** (vg terminology), where child chains are collapsed to single nodes. The computation is linear in the total length of observed HPRC traversals through *s*, not exponential in the number of children.

### 6.3 Key properties of the algorithm

**Optimality.** Because the cost of a subtree depends only on the subtrees below it (no cross-snarl interactions), the bottom-up greedy algorithm is globally optimal for minimizing total feature count.

**LD compression.** In a region of perfect LD where 50 biallelic leaf snarls have only 2 joint haplotype paths: flat_cost = 1, decomp_cost ≥ 50. The algorithm flattens, replacing 50 features with 1. This is lossless — every HPRC haplotype is perfectly distinguished.

**Nesting preservation.** For an insertion containing a SNP: flat_cost = 2 (three paths: skip, ins+T, ins+G), decomp_cost = 1 (skeleton: skip vs. insertion) + 1 (inner SNP: T vs. G) = 2. Tied, so decompose. The insertion effect and the SNP effect become separate features.

**Automatic granularity.** In low-LD regions where every snarl is roughly independent, the algorithm keeps leaf-level resolution. In high-LD regions, it collapses blocks. In nested regions, it decomposes. The snarl tree topology and HPRC haplotype paths jointly determine the optimal granularity everywhere.

### 6.4 Computational substrate

All path counts (k_s, k_skel) are computed by querying the GBWT index, which stores all HPRC haplotype paths and supports efficient local queries of the form "how many distinct paths traverse this snarl?" The algorithm is a single bottom-up traversal of the snarl tree with one GBWT query per snarl. It runs once on the reference pangenome and produces the feature catalog (Section 8).

---

## 7. Feature Vector Construction

The feature catalog (Section 8) defines the projection from node traversals to snarl features. Given an individual's per-haplotype node traversal list (Section 3.3), the feature vector is computed as follows:

1. At each feature site, the catalog lists the node paths defining each allele. Match the individual's traversed nodes against these allele definitions to determine which allele each haplotype carries.

2. Apply the encoding rules from Section 5: dosage for biallelic sites, dummy indicators for multi-allelic sites, repeat count for cyclic sites.

3. Apply the diploid rules from Section 5.6: sum haplotype values when both traverse, use the single haplotype's value when one skips, assign NA when both skip.

4. Concatenate features from all feature sites into a single numeric vector.

The output per individual is a numeric vector. Its length equals the total number of feature columns: Σ (k_i − 1) across all selected acyclic feature sites, plus 1 per selected cyclic feature site. Values are integers for hard genotype calls, floats for probabilistic calls.

**No normalization is applied.** Features are stored as raw dosage/indicator values. Frequency-dependent scaling or variance standardization are deferred to the downstream model.

**No interaction or cross-snarl features are computed.** The feature vector consists exclusively of single-site features.

---

## 8. Output Format

The spec produces one artifact: the **feature catalog**, stored as two binary files. These files fully define the feature space and the projection from node traversals to snarl features. No per-individual genotype matrix is materialized.

### 8.1 features.bin

Fixed-width records, one per feature site. Memory-mappable. Feature N is at byte offset (header_size + N × record_size).

```
Header (24 bytes):
  magic:            uint8[4]   "SFC\0"
  version:          uint16
  n_features:       uint32
  n_traversals:     uint32     (total allele entries across all features)
  reserved:         uint8[10]

Per feature (32 bytes):
  feature_id:        uint32    (position in feature vector)
  boundary_entry:    uint64    (entry boundary node ID)
  boundary_exit:     uint64    (exit boundary node ID)
  k:                 uint16    (allele count, including OTHER if present)
  is_cyclic:         uint8     (0 or 1)
  parent_feature_id: uint32    (0xFFFFFFFF = top-level, no parent)
  traversal_offset:  uint32    (byte offset into traversals.bin for this feature's first allele)
  reserved:          uint8[1]
```

32 bytes × 20M features = 640MB.

The six fields that define the feature space:
- **feature_id**: ordering in the feature vector
- **boundary_entry, boundary_exit**: snarl identity. Also serves as the group ID for multi-allelic dummy features from the same snarl.
- **k**: determines encoding (k=2 → one dosage feature; k>2 → k−1 dummy features; cyclic → one count feature)
- **is_cyclic**: distinguishes repeat-count encoding from dummy encoding
- **parent_feature_id**: determines NA propagation (if parent allele = skip, all children are NA)

### 8.2 traversals.bin

Variable-length allele definitions. Each allele is a list of node IDs defining the path through the snarl for that allele. This is the projection: given an individual's node traversals, match against these paths to determine allele assignment.

```
Per allele:
  n_nodes:   uint16              (number of nodes in this path)
  node_ids:  uint64[n_nodes]     (the node sequence)
```

The alleles for feature N start at the byte offset stored in features.bin's traversal_offset field. The first allele listed is the reference allele (coded as 0 / omitted category). Subsequent alleles correspond to dummy indicators in order. If the feature has an OTHER allele, it is the last allele listed and has n_nodes = 0 (empty path, since OTHER is a catch-all with no specific node path).

Average ~3 nodes per allele, ~2.5 alleles per feature: 20M × 2.5 × (2 + 3×8) ≈ 1.3GB.

### 8.3 Total size

~2GB for the complete feature catalog. Loaded once at pipeline start, kept in memory for the duration of any streaming computation.

### 8.4 No materialized genotype matrix

Per-individual feature vectors are computed on-the-fly from node traversals (Section 7) and consumed by streaming accumulators (e.g., GWAS sufficient statistics). They are not stored. The feature catalog is the only artifact the spec produces. LD matrices, summary statistics, and PGS weights are downstream pipeline outputs, not part of the representation layer.

---

## 9. Encoding, Regularization, and Implicit Priors

### 9.1 The entanglement

Under ordinary least squares (no regularization), the choice of basis is irrelevant: any full-rank representation of the same column space produces identical predictions. The snarl encoding, node traversal matrix, or any other full-rank parameterization give the same OLS fit.

Under regularized estimation — ridge, LASSO, elastic net, or Bayesian priors — this invariance breaks. The basis choice and the regularization penalty interact: applying the same isotropic penalty (e.g., λ||β||²) in two different coordinate systems produces different predictions. Concretely, isotropic ridge on the snarl feature matrix is equivalent to a specific anisotropic penalty on the node traversal matrix, and vice versa.

This means: **the choice of feature encoding is not purely a representation decision. Under regularization, the encoding implicitly defines a prior over genetic effect architectures.** Representation and prior are inherently entangled.

### 9.2 What each encoding implies

**Snarl dosage with isotropic ridge** penalizes each snarl feature equally. A biallelic SNP and a 50kb insertion each contribute one feature, each penalized by the same λ. This implicitly assumes that SNP effects and large SV effects are drawn from the same distribution — regardless of structural complexity, node count, or allele frequency.

**Node traversal matrix with isotropic ridge** produces a frequency-dependent penalty in the prediction space. Directions of common haplotype variation (large singular values in the node matrix) are penalized less; directions of rare variation are penalized more. Additionally, structural variants spanning many nodes are effectively penalized less than single-node SNPs: the ridge-optimal solution distributes a shared effect across co-traversed backbone nodes, reducing the per-node penalty by a factor of 1/(number of nodes). The graph topology directly shapes the implicit prior.

Neither is objectively correct. The right prior depends on the true genetic architecture of the trait being predicted, which varies across phenotypes.

### 9.3 Breaking the entanglement

The entanglement between encoding and prior can be broken if the estimation layer has sufficient flexibility to impose per-feature penalty weights. Specifically: for any target prior geometry and any full-rank basis, there exists an anisotropic penalty that recovers the target predictions.

This is why structural information derivable from the feature catalog and the biobank is important. It provides the information the estimation layer needs to construct feature-specific penalties that compensate for the encoding's implicit prior:

- **Allele frequency** (computed from the biobank during the GWAS streaming pass) enables frequency-dependent shrinkage (rare variants penalized more), approximating the spectral weighting that the overcomplete node matrix achieves automatically.
- **Group membership** (snarl ID, from the feature catalog) enables group sparsity (multi-allelic snarl features enter/exit the model together).

A naive estimation method (isotropic ridge on snarl features) ignores this information and accepts the encoding's implicit prior. A sophisticated method (Bayesian PGS with annotation-dependent variance components) uses it to construct a prior appropriate for the trait, independent of the encoding choice.

### 9.4 Practical implication

The snarl encoding specified in this document is a compressed, interpretable basis. Under isotropic regularization, it imposes a specific implicit prior (uniform per-variant penalization). This prior is adequate for many applications but is not optimal.

For best prediction accuracy, the estimation layer should construct per-feature or per-group penalties using information derivable from the feature catalog (snarl grouping, parent-child structure) and from the biobank (allele frequencies, LD structure). The feature catalog provides the structural skeleton; the biobank provides the population-specific quantities.

---

## 10. Properties of the Representation

**Panel-lossless (modulo singletons).** Every HPRC allele observed in ≥2 haplotypes is individually resolved. Singleton alleles (observed in exactly one haplotype) are pooled into OTHER. No information about non-singleton HPRC haplotype variation is discarded.

**Minimal.** The snarl tree optimization minimizes total feature count subject to panel-losslessness. No representation with fewer features can distinguish all HPRC haplotypes.

**Generalization of SNP dosage.** At non-nested biallelic sites, the encoding is standard {0, 1, 2} allele dosage — identical to existing GWAS practice. The representation is a strict superset of SNP-based PGS features: it includes all SNP dosages plus structural variant features, multi-allelic features, VNTR counts, and correctly handled nested variation.

**Correct nested variation handling.** At sites with conditional inner variation (e.g., SNP inside SV), outer structural effects and inner conditional effects occupy separate features. Outer effects are estimated from all individuals. Inner effects are estimated exclusively from individuals who traverse the inner snarl — non-traversing individuals are marked NA and excluded entirely. This provides full statistical power (heterozygotes contribute normally), zero contamination (non-carriers cannot influence the inner effect estimate), and biological correctness (missing means absent, not "zero effect").

**Explicit structural sharing.** Effects shared across multiple alleles (e.g., the insertion backbone shared by ins+T and ins+G) are captured by the outer snarl's feature, not duplicated across inner allele indicators. The snarl tree decomposition ensures that each level of the hierarchy is its own feature, and shared effects at each level are estimated from the appropriate subpopulation.

**Unified variant representation.** SNPs, indels, SVs, inversions, VNTRs, nested variation, and multi-allelic sites are all encoded in the same framework. No separate pipelines for different variant types.

**Fixed feature space.** Feature definitions are determined entirely by the pangenome graph and HPRC haplotype paths. They do not depend on any downstream cohort. Once defined, the same feature space is used for training, validation, testing, and deployment.

**Basis-dependent regularization.** Under regularized estimation, the snarl encoding imposes a specific implicit prior: isotropic penalization across variant sites regardless of structural complexity or allele frequency (Section 9). This prior is adequate but not optimal. Allele frequencies computed from the biobank and structural information from the feature catalog enable the estimation layer to construct per-feature penalties that approximate any desired prior geometry.

---

## 11. Variant Type Reference

| Variant type | Graph representation | Feature encoding |
|---|---|---|
| Biallelic SNP | 2-path bubble | 1 dosage feature, {0, 1, 2} diploid |
| Multi-allelic SNP | 3+ path bubble | k−1 dummy indicators |
| Simple insertion | 2-path bubble (extra nodes on one path) | 1 dosage feature, {0, 1, 2} diploid |
| Simple deletion | 2-path bubble (missing nodes on one path) | 1 dosage feature, {0, 1, 2} diploid |
| Complex indel | Multi-path bubble | k−1 dummy indicators |
| Inversion | 2-path bubble (reverse node orientation) | 1 dosage feature, {0, 1, 2} diploid |
| VNTR / tandem repeat | Cyclic subgraph | Integer repeat count |
| SNP inside SV | Outer SV snarl + nested inner SNP snarl | Separate features: outer SV dosage + inner SNP dosage (NA for skip/skip; hemizygous for skip/ins) |
| Complex nested SV | Multi-level snarl tree | Features at each level selected by optimization; inner features NA for non-traversers |
| LD block (perfect LD) | Consecutive snarls, ≤2 joint haplotype paths | Collapsed to 1 dosage feature by optimization |
| Non-tandem segdup | Separate graph regions per copy | Independent features per copy |

---

## 12. Practical Notes

### 12.1 Expected dimensionality

Total feature count depends on the pangenome's haplotype complexity. With HPRC v2.0 (~464 haplotypes from 232 individuals), the number of distinct path alleles per snarl is bounded by 464. Without the snarl tree optimization (leaf snarls only), feature count would be on the order of 100 million (comparable to the full variant count in a diverse pangenome). With the optimization, LD compression reduces this substantially in high-LD regions. The expected range is **20–50 million features**, comparable to a whole-genome imputation panel and within reach of existing PGS estimation methods.

The exact feature count is a property of the pangenome that has not yet been computed: the total effective degrees of freedom of human haplotype variation as captured by the snarl tree. Computing this is itself a contribution.

### 12.2 Encoding properties

At non-nested biallelic sites (the vast majority of features), the encoding is standard {0, 1, 2} allele dosage. Multi-allelic sites use dummy indicators. Nested sites introduce structured missingness (NA for non-traversing individuals). The missingness pattern is deterministic, determined by the snarl tree, not random.

### 12.3 Population structure

The feature vector naturally encodes population structure. Snarls where path alleles differ in frequency across populations produce features correlated with ancestry. This is intentional and desirable for within-cohort prediction: ancestry-associated phenotypic variation improves prediction accuracy. If ancestry adjustment is needed for other purposes (causal inference, cross-population portability), it should be handled in the modeling layer (e.g., ancestry PCs as covariates), not by modifying features.

### 12.4 Reference allele assignment

The reference allele at each snarl (coded as 0 in the biallelic case, or the omitted category in dummy coding) is the **most common HPRC allele**, not necessarily the allele matching a linear reference genome. This avoids biasing the encoding toward the single individual whose genome was chosen as a linear reference.

### 12.5 GBWT as computational substrate

The GBWT index (bundled in the GBZ file) is the computational substrate for both the snarl tree optimization (Section 6) and the feature catalog construction. It supports efficient queries for enumerating distinct paths through a snarl and their frequencies.

---

## 13. Summary

| Component | Specification |
|---|---|
| **Underlying model** | Path-additive: PGS = sum of learned weights along haplotype path through graph |
| **Input** | GBZ file (HPRC v2.0, ~464 haplotypes from 232 individuals) + per-individual node traversals (hard or probabilistic) |
| **Unit of variation** | Feature site: a snarl selected by snarl tree optimization |
| **Allele definition** | Distinct HPRC haplotype paths (or skeleton routes) through the snarl; singletons pooled into OTHER |
| **Encoding** | Allele dosage (biallelic), dummy indicators (multi-allelic), repeat count (cyclic); NA for non-traversing individuals |
| **Biallelic sites** | {0, 1} haploid; {0, 1, 2} diploid |
| **Multi-allelic sites** | k−1 dummy indicators |
| **Singleton alleles** | Pooled into OTHER (allele count < 2 in HPRC) |
| **VNTRs** | Integer repeat count (cyclic snarls: directed cycle in netgraph) |
| **Nested variation** | Outer and inner snarls encoded separately; inner features NA for non-traversers; hemizygous values at ploidy 1 |
| **LD compression** | Lossless collapsing by snarl tree optimization when joint paths < sum of individual features |
| **Relation to node matrix** | Snarl features = linear projection of node traversal matrix via flow conservation and snarl tree hierarchy |
| **Normalization** | None (deferred to downstream model) |
| **Interaction features** | None |
| **Feature space** | Fixed by pangenome graph + HPRC haplotype paths |
| **Output artifact** | Feature catalog: features.bin (fixed-width, 640MB) + traversals.bin (variable-length, ~1.3GB) |
| **No materialized genotype matrix** | Per-individual features computed on-the-fly from node traversals, consumed by streaming accumulators |
| **Reproducibility** | Feature catalog header records graph build hash, pipeline, HPRC release |
| **Expected dimensionality** | 20–50 million features genome-wide |
