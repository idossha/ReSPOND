# Voxelwise Statistical Analysis for Neuroimaging Data

## Overview

A professional Python toolkit for non-parametric voxelwise statistical analysis to identify brain regions with significantly different current intensity between responders and non-responders.

### Project Structure

```
stats/
├── src/                             # Source code
│   ├── main.py                      # Main entry point
│   ├── utils.py                     # Core functionality (stats, atlas, I/O)
│   ├── posthoc_atlas_analysis.py   # Post-hoc atlas overlap tool
│   ├── requirements.txt             # Python dependencies
│   └── venv/                        # Virtual environment
├── data/                            # Input data
│   ├── subject_class.csv            # Subject classifications
│   └── *_TI_MNI_MNI_TI_max.nii.gz  # Subject NIfTI files
├── assets/                          # Brain atlases
│   ├── HarvardOxford-cort-maxprob-thr0-1mm.nii.gz
│   ├── Talairach-labels-1mm.nii.gz
│   └── MNI_Glasser_HCP_v1.0.nii.gz
├── output/                          # Analysis outputs (auto-created)
│   ├── significant_voxels_mask.nii.gz
│   ├── pvalues_map.nii.gz
│   ├── analysis_summary.txt
│   └── ...
├── README.md                        # This file
└── CLI.md                           # Programmatic usage guide
```

## Statistical Approach

- **Test**: T-test (configurable: paired or unpaired)
  - Paired: For matched/repeated measures designs
  - Unpaired: For independent groups (default)
- **Multiple Comparison Correction**: Cluster-based permutation testing
  - Cluster-forming threshold: p < 0.01 (uncorrected, configurable)
  - 1000 permutations to build null distribution (configurable)
  - Tests all valid voxels in permutations for comprehensive null distribution
  - Cluster-level significance: α = 0.05 (configurable)
  - Paired test uses sign-flipping; unpaired uses label permutation
- This approach has more power than voxelwise corrections for spatially contiguous effects

## Features

1. Voxelwise comparison between two groups (responders vs non-responders)
2. Cluster-based permutation correction for multiple comparisons
3. Cluster identification in significant regions
4. Atlas-based region overlap analysis with automatic dimension checking and resampling
5. Generation of average intensity maps (responders, non-responders, difference)
6. Post-hoc atlas analysis for existing masks
7. Comprehensive summary statistics

## Installation

Install required packages:

```bash
cd src
pip install -r requirements.txt
```

Or use the existing virtual environment:

```bash
source src/venv/bin/activate  # On macOS/Linux
# or
src\venv\Scripts\activate     # On Windows
```

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
cd src
python main.py
```

### Programmatic Usage

For custom analyses and integration with your workflow, see [`CLI.md`](CLI.md) for detailed examples:

```python
import sys
sys.path.append('src')  # Add src to path
from utils import load_subject_data, ttest_voxelwise, cluster_based_correction

# Load data
responders, non_responders, img = load_subject_data('data/subject_class.csv', 'data')

# Run analysis (unpaired t-test)
p_values, _, valid_mask = ttest_voxelwise(responders, non_responders, test_type='unpaired')
sig_mask, _, clusters, _, _ = cluster_based_correction(
    responders, non_responders, p_values, valid_mask, test_type='unpaired'
)
```

### Post-Hoc Atlas Analysis

Test existing results against new atlases (without re-running analysis):

```bash
cd src
python posthoc_atlas_analysis.py
```

## Input Files Required

- `subject_class.csv` - Subject classifications (responder/non-responder)
- `*_grey_*_TI_MNI_MNI_TI_max.nii.gz` - Individual subject NIfTI files
- `HarvardOxford-cort-maxprob-thr0-1mm.nii.gz` - Harvard-Oxford cortical atlas
- `Talairach-labels-1mm.nii.gz` - Talairach atlas
- `MNI_Glasser_HCP_v1.0.nii.gz` - Glasser HCP atlas

**Note:** Atlases with different dimensions are automatically resampled to match your data.

## Output Files

1. **`significant_voxels_mask.nii.gz`** - Binary mask of significant voxels (0/1)
2. **`pvalues_map.nii.gz`** - Map of -log10(p-values) for visualization
3. **`analysis_summary.txt`** - Detailed text summary including:
   - Sample information
   - Number of significant voxels
   - Cluster information
   - Atlas overlap rankings
4. **`average_responders.nii.gz`** - Average current intensity map for responders
5. **`average_non_responders.nii.gz`** - Average current intensity map for non-responders
6. **`difference_map.nii.gz`** - Difference map (responders - non-responders)
7. **`permutation_null_distribution.pdf`** - Visualization of permutation null distribution with:
   - Histogram of maximum cluster sizes from permutations
   - Cluster-level significance threshold
   - Observed cluster sizes (color-coded by significance)

## Visualization

### Quick Visualization (Python)

Generate PNG visualizations automatically:

```bash
python visualize_results.py
```

This creates:
- `statistical_results_visualization.png` - P-value maps
- `significant_mask_visualization.png` - Binary mask overlay

### Professional Neuroimaging Software

You can also visualize the results using:

- **FSLeyes**: `fsleyes pvalues_map.nii.gz -cm red-yellow -dr 2 10`
- **FreeSurfer**: Overlay on standard MNI152 template
- **SPM**: Load as overlay in SPM viewer

## Atlas Overlap Analysis

The script automatically analyzes overlap between significant voxels and atlas regions, ranking them by:
- Number of significant voxels in each region
- Percentage of region covered by significant voxels

This helps identify which anatomical regions show the strongest effects.

## Post-Hoc Atlas Analysis

If you want to test an existing significant voxel mask against new atlases without re-running the full analysis:

```bash
python posthoc_atlas_analysis.py
```

Or specify custom files:

```bash
python posthoc_atlas_analysis.py --mask your_mask.nii.gz --atlases atlas1.nii.gz atlas2.nii.gz
```

This will:
- Load your existing significant voxel mask
- Check dimensions and resample atlases if needed
- Calculate overlap statistics for each atlas region
- Generate CSV files and summary report

**Use cases:**
- You obtained a new atlas and want to test it
- You want to test different parcellations
- You have masks from different analyses to compare

## Code Organization

### Core Modules

**`utils.py`** - Core functionality organized into sections:
- **Data Loading & I/O**: `load_subject_data()`, `save_nifti()`
- **Statistical Analysis**: `mannwhitneyu_voxelwise()`, `cluster_based_correction()`
- **Cluster Analysis**: `cluster_analysis()`
- **Atlas Processing**: `check_and_resample_atlas()`, `atlas_overlap_analysis()`
- **Reporting**: `generate_summary()`

**`main.py`** - Clean entry point with:
- Configuration dictionary for easy parameter adjustment
- 7-step workflow with clear progress indicators
- Comprehensive output generation

**`posthoc_atlas_analysis.py`** - Standalone tool for testing existing masks against new atlases

**`CLI.md`** - Comprehensive guide for programmatic usage with examples

### Benefits of This Structure

✅ **Modular**: Import only what you need  
✅ **Reusable**: Use functions in your own scripts  
✅ **Maintainable**: Clear separation of concerns  
✅ **Documented**: Inline docstrings and CLI.md examples  
✅ **Extensible**: Easy to add new functionality  
✅ **Professional**: Industry-standard Python project layout  

## Notes

- All analyses are performed in MNI standard space
- T-tests can be configured as paired (for matched designs) or unpaired (for independent groups)
- Cluster-based permutation correction controls for family-wise error while maintaining power
- Permutation testing builds an empirical null distribution:
  - Unpaired: Label permutation (shuffles group assignments)
  - Paired: Sign-flipping (preserves pairing structure)
- Clusters are identified using 26-connectivity (3D)
- This approach is particularly effective for detecting spatially contiguous effects
- Atlases are automatically resampled to match data dimensions using nearest-neighbor interpolation

## Documentation

- **[README.md](README.md)** - This file: Overview, installation, quick start
- **[CLI.md](CLI.md)** - Programmatic usage guide with code examples
- **Code docstrings** - Detailed function documentation in `utils.py`

