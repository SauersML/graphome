# Analysis Scripts

This directory contains scripts for analyzing genome graphs with graphome.

## AMY Gene Cluster Analysis

Analyzes the AMY (amylase) gene cluster on chromosome 1 using sliding window eigendecomposition.

### Files

- **`run_amy_analysis.sh`** - Main analysis script that runs eigen-region across 5kb sliding windows
- **`plot_amy_ngec.py`** - Python script to plot NGEC values and annotate gene locations

### Usage

```bash
# Run analysis
bash scripts/run_amy_analysis.sh \
  "s3://human-pangenomics/.../hprc-v2.0-mc-grch38.gbz" \
  "output_directory"

# Or use local GBZ file
bash scripts/run_amy_analysis.sh \
  "data/hprc/hprc-v2.0-mc-grch38.gbz" \
  "output_directory"
```

### Output

- `output_directory/ngec_results.txt` - Raw NGEC values for each window
- `output_directory/amy_ngec_plot.png` - Visualization with gene annotations

### Region Details

**Analysis region:** chr1:103,514,644-103,798,692 (284kb)
- Original AMY cluster: 103,554,644-103,758,692 (204kb)
- Expanded 40kb in each direction for context

**Genes annotated:**
- **AMY2B** (103554644-103579534) - Pancreatic amylase
- **AMY2A** (103616651-103625780) - Pancreatic amylase  
- **AMY1A** (103655519-103664554) - Salivary amylase
- **AMY1B** (103687415-103696453) - Salivary amylase (minus strand)
- **AMY2Ap** (103713720-103719905) - Pseudogene (AMYP1)
- **AMY1C** (103749654-103758692) - Salivary amylase

### GitHub Actions

The analysis runs automatically via GitHub Actions:
- **Workflow:** `.github/workflows/amy-analysis.yml`
- **Trigger:** Manual dispatch or weekly schedule
- **Artifacts:** Results and plots uploaded for 90 days

To run manually:
1. Go to Actions tab in GitHub
2. Select "AMY Gene Cluster Analysis"
3. Click "Run workflow"

### Requirements

- Rust (nightly)
- Python 3 with matplotlib and numpy
- HPRC GBZ file (5.1GB, downloaded automatically in CI)

### Analysis Details

- **Window size:** 2kb (smaller windows = higher resolution)
- **Number of windows:** 142
- **Total region:** 284kb (AMY cluster + 40kb flanking each side)
- **Metric:** NGEC (Normalized Graph Eigen-Complexity)
  - Measures structural complexity of the graph
  - Range: [0, 1]
  - Higher values = more complex/diverse structure
- **Method:** Laplacian eigendecomposition of subgraph for each window
