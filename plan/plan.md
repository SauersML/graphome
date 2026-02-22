# Pangenome Feature Representation for Polygenic Prediction

## 1. Overview

This document specifies a method for constructing per-individual numeric feature vectors from a human pangenome graph for use in polygenic score (PGS) estimation. The method produces a feature representation that:

- Captures SNPs, indels, structural variants, inversions, and tandem repeats in a single unified framework.
- Handles nested variation (e.g., a SNP inside a structural variant) with an encoding that allows shared structural effects to be estimated once, not redundantly.
- Achieves the minimum number of features that losslessly represents all haplotype variation observed in the reference panel.
- Uses standard allele dosage at biallelic sites and standard dummy coding at multi-allelic sites — directly compatible with existing PGS tools.

The method defines only the **representation layer**: the transformation from a pangenome graph and individual genotype data into a numeric vector suitable for downstream modeling. It does not specify the downstream PGS estimation model or the upstream genotype-to-graph projection pipeline.

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

### 3.1 Pangenome graph

A base-level pangenome graph (GFA or equivalent), constructed from high-quality phased haplotype assemblies. "Base-level" means the graph resolves variation down to single nucleotides: individual SNPs are distinct bubbles. The recommended source is the HPRC pangenome, built via Minigraph-Cactus or PGGB. The graph is treated as a fixed reference resource, analogous to hg38 for SNP-based analyses.

### 3.2 Reference haplotype paths

The set of haplotype paths through the graph from the HPRC reference panel, embedded as path/walk annotations (W-lines or P-lines in GFA) and indexed via the GBWT (Graph BWT). These paths define the **alleles** at each variant site: an allele is a distinct haplotype traversal through a snarl.

### 3.3 Snarl decomposition

The snarl decomposition of the graph, computable via `vg snarls` or equivalent tooling. This produces a hierarchy of snarls (the **snarl tree**), where snarls can be nested inside other snarls.

---

## 4. Definitions

**Snarl.** A subgraph bounded by two boundary nodes (entry and exit), where all paths from the rest of the graph enter through one boundary and exit through the other. Snarls correspond to variant sites.

**Leaf snarl.** A snarl containing no child snarls. The minimal, irreducible unit of variation. A biallelic SNP is a leaf snarl with two path alleles.

**Compound snarl.** A snarl containing one or more child snarls (nested variation). Example: a deletion that contains two internal SNPs on one branch.

**Snarl tree.** The hierarchy of snarls defined by their nesting relationships. Children are snarls nested inside a parent snarl.

**Path allele.** At a given snarl, each distinct traversal from entry to exit observed among the HPRC haplotypes constitutes one allele. The number of distinct path alleles at a snarl is *k*.

**Skeleton.** At a compound snarl, the skeleton is the outer routing structure with all child snarls abstracted to generic placeholders. Concretely: take all HPRC paths through the snarl, replace each child-snarl traversal with a placeholder symbol, count distinct resulting strings. The skeleton captures "which outer route was taken" independent of choices made inside children. The number of distinct skeleton paths is *k*_skel.

**Cyclic snarl.** A snarl whose internal subgraph contains cycles, arising from tandem repeats or VNTRs. Haplotypes traverse the cycle a variable number of times, encoding copy number.

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

*k* − 1 features using dummy coding. Designate the most common HPRC allele as reference. Each remaining allele gets one indicator feature (0 or 1 per haplotype; 0, 1, or 2 per diploid individual).

The specific choice of coding scheme (dummy, effect, or sum-to-zero contrasts) is left to the implementation. Any full-rank coding of a *k*-category variable spans the same information; differences affect only how isotropic regularization acts on coefficients. The representation layer emits the allele identity per haplotype; the downstream tool applies its preferred coding.

**Recommended default:** dummy coding (one indicator per non-reference allele), as this is the most widely supported in existing tools and the most interpretable (each coefficient is the effect of that allele relative to the reference).

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

Sum the two haploid feature vectors. For a biallelic snarl, diploid values are {0, 1, 2} — standard dosage.

When one haplotype traverses a snarl and the other does not, the traversing haplotype contributes its allele code and the non-traversing haplotype contributes NA. The diploid value for that feature is the traversing haplotype's value alone (effectively treating the missing haplotype as excluded, not as zero).

Example: a diploid individual heterozygous for skip and ins+T at a nested insertion. Outer feature: 0 + 1 = 1 (one copy of insertion). Inner SNP feature: one haplotype carries T (dosage contribution = 0, if T is reference among insertion alleles), the other haplotype is NA (skip, doesn't traverse). The inner feature value is 0, representing this individual's single observation of T at the inner SNP. The model estimates the inner SNP effect from this individual's single-haplotype observation alongside other insertion carriers' observations.

### 5.7 Relationship to the node traversal matrix

#### The underlying model

Both the snarl encoding and any alternative pangenome feature representation are instances of a single underlying model: the **path-additive model**.

An individual's polygenic score is the sum of learned weights along the elements their haplotypes traverse:

PGS_i = Σ_{e ∈ path(i)} w_e

The "elements" can be graph nodes, graph edges, snarl alleles, or any other decomposition of the graph into pieces with learnable weights. All such decompositions produce predictions in the same subspace — they differ only in parameterization, not in expressive power.

#### Node traversal as the primitive representation

The most direct representation of a pangenome genotype is the **node traversal matrix** X_node ∈ {0,1}^{n×p}, where X_{h,j} = 1 if haplotype h traverses variable node j, and p = |variable nodes|.

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

**When flat_cost equals decomp_cost, prefer decomposition.** The decomposed representation aligns features with the biological effect hierarchy (outer structural effects in their own features, inner conditional effects in theirs), which produces better-conditioned estimation even at equal dimensionality.

### 6.3 Key properties of the algorithm

**Optimality.** Because the cost of a subtree depends only on the subtrees below it (no cross-snarl interactions), the bottom-up greedy algorithm is globally optimal for minimizing total feature count.

**LD compression.** In a region of perfect LD where 50 biallelic leaf snarls have only 2 joint haplotype paths: flat_cost = 1, decomp_cost ≥ 50. The algorithm flattens, replacing 50 features with 1. This is lossless — every HPRC haplotype is perfectly distinguished.

**Nesting preservation.** For an insertion containing a SNP: flat_cost = 2 (three paths: skip, ins+T, ins+G), decomp_cost = 1 (skeleton: skip vs. insertion) + 1 (inner SNP: T vs. G) = 2. Tied, so decompose. The insertion effect and the SNP effect become separate features.

**Automatic granularity.** In low-LD regions where every snarl is roughly independent, the algorithm keeps leaf-level resolution. In high-LD regions, it collapses blocks. In nested regions, it decomposes. The snarl tree topology and HPRC haplotype paths jointly determine the optimal granularity everywhere.

### 6.4 Computational substrate

All path counts (k_s, k_skel) are computed by querying the GBWT index, which stores all HPRC haplotype paths and supports efficient local queries of the form "how many distinct paths traverse this snarl?" The algorithm is a single bottom-up traversal of the snarl tree with one GBWT query per snarl. It runs once on the reference pangenome and produces a fixed feature schema for all downstream use.

---

## 7. Feature Vector Construction

For each individual, the feature vector is constructed as follows:

1. Determine the individual's haplotype path(s) through the pangenome graph (out of scope for this spec; handled by genotype-to-graph projection tools such as vg giraffe).

2. At each feature site selected by the optimization algorithm:
   - Determine which path allele (or skeleton allele) the individual's haplotype matches.
   - If the haplotype traverses the snarl: assign the dosage or dummy code for the matched allele.
   - If the haplotype does not traverse the snarl (bypassed due to a parent routing decision): assign NA for all features at this snarl.

3. For diploid individuals: sum the two haploid feature vectors.

4. Concatenate features from all feature sites across the genome into a single numeric vector.

The output per individual is a single vector of real numbers. Its length equals the total number of feature columns: Σ (k_i − 1) across all selected acyclic feature sites, plus 1 per selected cyclic feature site.

**No normalization is applied.** Features are stored as raw dosage/indicator values. Frequency-dependent scaling, variance standardization, or MAF-based weighting are deferred to the downstream model, which can apply its own architecture-appropriate prior (e.g., MAF-dependent effect-size variance in Bayesian PGS methods). Baking normalization into the features would impose an implicit prior that may be suboptimal for specific traits.

**No interaction or cross-snarl features are computed.** The feature vector consists exclusively of single-site features. For common complex diseases, essentially all heritable variance is additive at the variant level, and interaction effects contribute negligibly to prediction. Encoding interaction features would increase dimensionality without improving prediction.

---

## 8. Structural Metadata

Alongside the feature vector, the method emits structural metadata for each feature site. This metadata is not used by the representation layer but is available for downstream models that can exploit it.

**Per feature site:**

| Field | Description |
|---|---|
| Snarl ID | Identifier (boundary node pair) of the snarl |
| Snarl type | Acyclic biallelic, acyclic multi-allelic, or cyclic |
| Snarl tree depth | Nesting depth (0 = top-level, higher = more deeply nested) |
| Parent snarl ID | Which snarl this site is nested inside (if any) |
| Conditional on | Which parent skeleton allele(s) must be taken to traverse this snarl |
| Allele count (k) | Number of distinct path alleles at this site |
| Allele frequencies | HPRC frequency of each allele |
| Node count | Number of variable graph nodes underlying this feature site (structural complexity) |
| Genomic region | Approximate linear genome coordinates (for compatibility with existing tools) |
| Feature columns | Column indices in the feature vector |

**Recommended downstream use of metadata:**

This metadata serves a critical role beyond documentation: it enables the estimation layer to construct per-feature priors that compensate for the implicit prior imposed by the encoding choice (see Section 9).

- **Group sparsity.** Multi-allelic feature sites produce k − 1 columns that belong to a single biological locus. Downstream models using group sparsity (e.g., group LASSO, group spike-and-slab) should treat each feature site as one group. The snarl ID defines the grouping.

- **LD computation.** Pairwise LD between feature sites should be computed directly from the snarl dosage feature matrix using HPRC haplotypes, not from linear genomic coordinates. Graph topology (nearby snarls in the graph) can serve as a heuristic for which pairs to evaluate, replacing linear distance-based windowing.

- **Frequency-dependent priors.** Allele frequency metadata enables per-feature shrinkage that penalizes rare variants more heavily. This approximates the frequency-dependent weighting that ridge regression on the overcomplete node traversal matrix achieves automatically through the singular value structure (Section 9.2).

- **Complexity-dependent priors.** Node count metadata enables per-feature shrinkage that penalizes large structural variants less per unit effect. Under ridge on the raw node matrix, a structural variant spanning m nodes is effectively penalized 1/m as much as a single-node SNP (Section 9.2). The node count metadata allows snarl-level estimation to approximate this structural complexity weighting.

- **Depth-dependent priors.** Nesting depth metadata enables stronger shrinkage for deeply nested conditional features. Inner variants (e.g., a SNP inside an SV inside an inversion) are less likely to be independently causal and should be penalized more aggressively.

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

### 9.3 Breaking the entanglement with metadata

The entanglement between encoding and prior can be broken if the estimation layer has sufficient flexibility to impose per-feature penalty weights. Specifically: for any target prior geometry and any full-rank basis, there exists an anisotropic penalty that recovers the target predictions.

This is why the structural metadata emitted alongside the feature vector (Section 8) is not optional. It provides the information the estimation layer needs to construct feature-specific penalties that compensate for the encoding's implicit prior:

- **Allele frequency** enables frequency-dependent shrinkage (rare variants penalized more), approximating the spectral weighting that the overcomplete node matrix achieves automatically.
- **Node count / structural complexity** enables complexity-dependent shrinkage (large SVs penalized less per unit effect), recovering the structural complexity weighting inherent in the node representation.
- **Nesting depth** enables depth-dependent shrinkage (deeply nested conditional features penalized more).
- **Group membership** (snarl ID) enables group sparsity (multi-allelic snarl features enter/exit the model together).

A naive estimation method (isotropic ridge on snarl features) ignores the metadata and accepts the encoding's implicit prior. A sophisticated method (Bayesian PGS with annotation-dependent variance components) uses the metadata to construct a prior that is appropriate for the trait, independent of the encoding choice.

### 9.4 Practical implication

The snarl encoding specified in this document is a compressed, interpretable, tool-compatible basis. Under isotropic regularization, it imposes a specific implicit prior (uniform per-variant penalization). This prior is adequate for many applications but is not optimal.

For best prediction accuracy, the estimation layer should use the emitted metadata to construct per-feature or per-group penalties. The recommended approach is a Bayesian PGS method (e.g., PRS-CS, SBayesR, or extensions) that accepts per-SNP annotations and learns variance components from data. The annotations derived from the pangenome — allele frequency, structural complexity, nesting depth, variant type — are novel inputs that existing annotation-stratified methods can consume directly.

The representation layer's role is to provide both the features (the compressed basis) and the metadata (the information needed to construct an appropriate prior). Together, they enable the estimation layer to approximate any reasonable prior geometry regardless of the basis choice.

---

## 10. Properties of the Representation

**Panel-lossless.** Every distinct HPRC haplotype path produces a unique feature vector. No information about HPRC haplotype variation is discarded.

**Minimal.** The snarl tree optimization minimizes total feature count subject to panel-losslessness. No representation with fewer features can distinguish all HPRC haplotypes.

**Generalization of SNP dosage.** At non-nested biallelic sites, the encoding is standard {0, 1, 2} allele dosage — identical to existing GWAS practice. The representation is a strict superset of SNP-based PGS features: it includes all SNP dosages plus structural variant features, multi-allelic features, VNTR counts, and correctly handled nested variation.

**Correct nested variation handling.** At sites with conditional inner variation (e.g., SNP inside SV), outer structural effects and inner conditional effects occupy separate features. Outer effects are estimated from all individuals. Inner effects are estimated exclusively from individuals who traverse the inner snarl — non-traversing individuals are marked NA and excluded entirely. This provides full statistical power (heterozygotes contribute normally), zero contamination (non-carriers cannot influence the inner effect estimate), and biological correctness (missing means absent, not "zero effect").

**Explicit structural sharing.** Effects shared across multiple alleles (e.g., the insertion backbone shared by ins+T and ins+G) are captured by the outer snarl's feature, not duplicated across inner allele indicators. The snarl tree decomposition ensures that each level of the hierarchy is its own feature, and shared effects at each level are estimated from the appropriate subpopulation.

**Unified variant representation.** SNPs, indels, SVs, inversions, VNTRs, nested variation, and multi-allelic sites are all encoded in the same framework. No separate pipelines for different variant types.

**Fixed feature space.** Feature definitions are determined entirely by the pangenome graph and HPRC haplotype paths. They do not depend on any downstream cohort. Once defined, the same feature space is used for training, validation, testing, and deployment.

**Basis-dependent regularization.** Under regularized estimation, the snarl encoding imposes a specific implicit prior: isotropic penalization across variant sites regardless of structural complexity or allele frequency (Section 9). This prior is adequate but not optimal. The emitted structural metadata (Section 8) enables the estimation layer to construct per-feature penalties that approximate any desired prior geometry, decoupling the representation choice from the modeling assumptions.

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
| SNP inside SV | Outer SV snarl + nested inner SNP snarl | Separate features: outer SV dosage + inner SNP dosage (NA for non-traversers) |
| Complex nested SV | Multi-level snarl tree | Features at each level selected by optimization; inner features NA for non-traversers |
| LD block (perfect LD) | Consecutive snarls, ≤2 joint haplotype paths | Collapsed to 1 dosage feature by optimization |
| Non-tandem segdup | Separate graph regions per copy | Independent features per copy |

---

## 12. Practical Notes

### 12.1 Expected dimensionality

Total feature count depends on the pangenome's haplotype complexity. Without the snarl tree optimization (leaf snarls only), feature count would be on the order of 100 million (comparable to the full variant count in a diverse pangenome). With the optimization, LD compression reduces this substantially in high-LD regions. The expected range is **20–50 million features**, comparable to a whole-genome imputation panel and within reach of existing PGS estimation methods.

The exact feature count is a property of the pangenome that has not yet been computed: the total effective degrees of freedom of human haplotype variation as captured by the snarl tree. Computing this is itself a contribution.

### 12.2 Compatibility with existing tools

At non-nested biallelic sites (the vast majority of features), the encoding is standard {0, 1, 2} allele dosage — directly compatible with all existing GWAS and PGS tools. Multi-allelic sites use standard dummy indicators. Nested sites introduce structured missingness (NA for non-traversing individuals); tools that handle missing data natively (e.g., glmnet, most Bayesian PGS methods) require no modification. For tools that do not support per-feature missingness, mean imputation at nested sites is an acceptable approximation (see Section 5.5).

### 12.3 Population structure

The feature vector naturally encodes population structure. Snarls where path alleles differ in frequency across populations produce features correlated with ancestry. This is intentional and desirable for within-cohort prediction: ancestry-associated phenotypic variation improves prediction accuracy. If ancestry adjustment is needed for other purposes (causal inference, cross-population portability), it should be handled in the modeling layer (e.g., ancestry PCs as covariates), not by modifying features.

### 12.4 Reference allele assignment

The reference allele at each snarl (coded as 0 in the biallelic case, or the omitted category in dummy coding) is the **most common HPRC allele**, not necessarily the allele matching a linear reference genome. This avoids biasing the encoding toward the single individual whose genome was chosen as a linear reference.

### 12.5 GBWT as computational substrate

The GBWT index is recommended for implementing both the optimization algorithm (Section 6) and the individual feature computation (Section 7). It supports efficient queries for enumerating distinct paths through a snarl and their frequencies, and for matching a new individual's local haplotype to the closest HPRC path allele at each feature site.

---

## 13. Summary

| Component | Specification |
|---|---|
| **Underlying model** | Path-additive: PGS = sum of learned weights along haplotype path through graph |
| **Unit of variation** | Feature site: a snarl selected by snarl tree optimization |
| **Allele definition** | Distinct HPRC haplotype paths (or skeleton routes) through the snarl |
| **Encoding** | Standard allele dosage (biallelic) or dummy indicators (multi-allelic); NA for non-traversing individuals |
| **Biallelic sites** | {0, 1} haploid; {0, 1, 2} diploid |
| **Multi-allelic sites** | k−1 dummy indicators (coding scheme flexible) |
| **VNTRs** | Integer repeat count |
| **Nested variation** | Outer and inner snarls encoded separately; inner features NA for non-traversers |
| **LD compression** | Lossless collapsing by snarl tree optimization when joint paths < sum of individual features |
| **Relation to node matrix** | Snarl features = linear projection of node traversal matrix via flow conservation and snarl tree hierarchy |
| **Normalization** | None (deferred to downstream model) |
| **Interaction features** | None |
| **Feature space** | Fixed by pangenome graph + HPRC haplotype paths |
| **Metadata emitted** | Snarl tree structure, depth, allele frequencies, node count, group membership, conditional dependencies |
| **Metadata role** | Enables estimation layer to construct per-feature priors that compensate for encoding's implicit regularization geometry |
| **Output per individual** | Numeric vector with structured missingness at nested sites |
| **Expected dimensionality** | 20–50 million features genome-wide |
