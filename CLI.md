# Command Line Interface & Programmatic Usage

This document provides examples of how to interact with the neuroimaging analysis functions programmatically without running the full pipeline.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Loading Data](#loading-data)
3. [Statistical Testing](#statistical-testing)
4. [Cluster Analysis](#cluster-analysis)
5. [Atlas Analysis](#atlas-analysis)
6. [Customization Examples](#customization-examples)
7. [Integration with Other Tools](#integration-with-other-tools)

---

## Quick Start

### Running the Full Pipeline

```bash
python main.py
```

### Running Post-Hoc Atlas Analysis

```bash
# Use default configuration
python posthoc_atlas_analysis.py

# Or specify custom files
python posthoc_atlas_analysis.py --mask my_mask.nii.gz --atlases atlas1.nii.gz atlas2.nii.gz
```

### Visualizing Results

```bash
python visualize_results.py
```

---

## Loading Data

### Basic Data Loading

```python
from utils import load_subject_data

# Load responders and non-responders
responders, non_responders, template_img = load_subject_data(
    csv_file="/path/to/subject_class.csv",
    data_dir="/path/to/nifti/files/"
)

print(f"Responders shape: {responders.shape}")
print(f"Non-responders shape: {non_responders.shape}")
```

### Loading Custom Subject Lists

```python
import pandas as pd
import nibabel as nib
import numpy as np

# Create custom subject list
subjects = pd.DataFrame({
    'subject_id': ['sub-101', 'sub-102', 'sub-103'],
    'response': [1, -1, 1],
    'simulation_name': ['Nav_3', 'Nav_2', 'Nav_3']
})

# Save and load
subjects.to_csv('my_subjects.csv', index=False)
data = load_subject_data('my_subjects.csv', '/path/to/data/')
```

---

## Statistical Testing

### Voxelwise Mann-Whitney U Test

```python
from utils import mannwhitneyu_voxelwise

# Run statistical test
p_values, u_statistics, valid_mask = mannwhitneyu_voxelwise(
    responders, 
    non_responders,
    verbose=True  # Set to False to suppress progress bars
)

# Examine results
print(f"Minimum p-value: {np.min(p_values[valid_mask])}")
print(f"Number of voxels with p < 0.01: {np.sum((p_values < 0.01) & valid_mask)}")
```

### Cluster-Based Permutation Correction

```python
from utils import cluster_based_correction

# Standard parameters (recommended)
sig_mask, threshold, sig_clusters = cluster_based_correction(
    responders, 
    non_responders, 
    p_values, 
    valid_mask,
    cluster_threshold=0.01,
    n_permutations=500,
    alpha=0.05
)

# More conservative (fewer false positives)
sig_mask_conservative, _, _ = cluster_based_correction(
    responders, non_responders, p_values, valid_mask,
    cluster_threshold=0.001,  # Stricter cluster formation
    n_permutations=1000,       # More permutations
    alpha=0.01                 # Stricter cluster significance
)

# More liberal (more sensitivity)
sig_mask_liberal, _, _ = cluster_based_correction(
    responders, non_responders, p_values, valid_mask,
    cluster_threshold=0.05,   # More lenient cluster formation
    n_permutations=500,
    alpha=0.05
)
```

### Custom Statistical Tests

```python
import numpy as np
from scipy import stats

# Run your own test at each voxel
def custom_test(responders, non_responders, valid_mask):
    """Example: t-test instead of Mann-Whitney U"""
    shape = responders.shape[:3]
    p_values = np.ones(shape)
    
    for i, j, k in np.argwhere(valid_mask):
        resp_vals = responders[i, j, k, :]
        non_resp_vals = non_responders[i, j, k, :]
        
        t_stat, p_val = stats.ttest_ind(resp_vals, non_resp_vals)
        p_values[i, j, k] = p_val
    
    return p_values

# Use with cluster correction
custom_p_values = custom_test(responders, non_responders, valid_mask)
sig_mask, _, _ = cluster_based_correction(
    responders, non_responders, custom_p_values, valid_mask
)
```

---

## Cluster Analysis

### Analyze Significant Clusters

```python
from utils import cluster_analysis

# Get detailed cluster information
clusters = cluster_analysis(
    sig_mask, 
    template_img.affine,
    verbose=True
)

# Access cluster information
for cluster in clusters:
    print(f"Cluster {cluster['cluster_id']}:")
    print(f"  Size: {cluster['size']} voxels")
    print(f"  MNI center: {cluster['center_mni']}")
    print(f"  Voxel center: {cluster['center_voxel']}")
```

### Extract Voxel Values from Clusters

```python
import nibabel as nib

# Get values from largest cluster
largest_cluster = clusters[0]
cluster_id = largest_cluster['cluster_id']

# Create mask for this cluster
labeled_array, _ = label(sig_mask)
cluster_mask = (labeled_array == cluster_id)

# Extract values
resp_values_in_cluster = responders[cluster_mask, :]
non_resp_values_in_cluster = non_responders[cluster_mask, :]

print(f"Mean in responders: {np.mean(resp_values_in_cluster)}")
print(f"Mean in non-responders: {np.mean(non_resp_values_in_cluster)}")
```

### Save Individual Clusters

```python
from utils import save_nifti

# Save each cluster separately
labeled_array, num_clusters = label(sig_mask)

for cluster_id in range(1, num_clusters + 1):
    cluster_mask = (labeled_array == cluster_id).astype(np.uint8)
    save_nifti(
        cluster_mask,
        template_img.affine,
        template_img.header,
        f"cluster_{cluster_id}.nii.gz",
        dtype=np.uint8
    )
```

---

## Atlas Analysis

### Analyze Atlas Overlap

```python
from utils import atlas_overlap_analysis

# Analyze overlap with multiple atlases
atlas_files = [
    "HarvardOxford-cort-maxprob-thr0-1mm.nii.gz",
    "Talairach-labels-1mm.nii.gz",
    "MNI_Glasser_HCP_v1.0.nii.gz"
]

atlas_results = atlas_overlap_analysis(
    sig_mask,
    atlas_files,
    data_dir="/path/to/atlases/",
    reference_img=template_img,
    verbose=True
)

# Access results
for atlas_name, regions in atlas_results.items():
    print(f"\n{atlas_name}:")
    for i, region in enumerate(regions[:5], 1):
        print(f"  {i}. Region {region['region_id']}: "
              f"{region['overlap_voxels']} voxels")
```

### Check Atlas Dimensions

```python
from utils import check_and_resample_atlas
import nibabel as nib

# Load atlas and check dimensions
atlas_img = nib.load("your_atlas.nii.gz")
reference_img = nib.load("your_reference.nii.gz")

# Automatically resample if needed
atlas_data = check_and_resample_atlas(
    atlas_img,
    reference_img,
    atlas_name="Custom Atlas",
    verbose=True
)
```

### Extract Region-Specific Values

```python
# Load atlas
atlas_img = nib.load("HarvardOxford-cort-maxprob-thr0-1mm.nii.gz")
atlas_data = atlas_img.get_fdata().astype(int)

# Extract values for specific region (e.g., region 10)
region_id = 10
region_mask = (atlas_data == region_id)

# Get mean values in this region
resp_in_region = np.mean(responders[region_mask, :], axis=0)
non_resp_in_region = np.mean(non_responders[region_mask, :], axis=0)

# Statistical test for this region
from scipy.stats import mannwhitneyu
u_stat, p_val = mannwhitneyu(resp_in_region, non_resp_in_region)
print(f"Region {region_id} p-value: {p_val}")
```

---

## Customization Examples

### Subset Analysis

```python
# Analyze only a specific brain region
# Create ROI mask (e.g., only analyze frontal cortex)
roi_mask = np.zeros_like(valid_mask)
roi_mask[50:120, :, :] = True  # Adjust coordinates for your ROI

# Combine with valid mask
analysis_mask = valid_mask & roi_mask

# Run analysis on ROI
p_values_roi, _, _ = mannwhitneyu_voxelwise(
    responders, non_responders, verbose=False
)

sig_mask_roi, _, _ = cluster_based_correction(
    responders, non_responders, p_values_roi, analysis_mask
)
```

### Batch Processing

```python
# Process multiple contrasts
contrasts = [
    ('responders_vs_nonresponders', responders, non_responders),
    ('high_vs_low', high_group, low_group),
    ('pre_vs_post', pre_treatment, post_treatment)
]

results = {}
for name, group1, group2 in contrasts:
    print(f"\nProcessing: {name}")
    
    p_vals, _, mask = mannwhitneyu_voxelwise(group1, group2, verbose=False)
    sig_mask, thresh, clusters = cluster_based_correction(
        group1, group2, p_vals, mask, verbose=False
    )
    
    results[name] = {
        'sig_mask': sig_mask,
        'n_voxels': np.sum(sig_mask),
        'clusters': clusters
    }
    
    # Save results
    save_nifti(sig_mask, template_img.affine, template_img.header,
               f"sig_mask_{name}.nii.gz", dtype=np.uint8)
```

### Parameter Sensitivity Analysis

```python
# Test different cluster thresholds
thresholds = [0.001, 0.005, 0.01, 0.05]

for thresh in thresholds:
    sig_mask, _, clusters = cluster_based_correction(
        responders, non_responders, p_values, valid_mask,
        cluster_threshold=thresh,
        n_permutations=500,
        verbose=False
    )
    
    print(f"Threshold p < {thresh}:")
    print(f"  Significant voxels: {np.sum(sig_mask)}")
    print(f"  Number of clusters: {len(clusters)}")
```

### Export Results to Different Formats

```python
import pandas as pd

# Export significant voxel coordinates to CSV
sig_coords = np.argwhere(sig_mask)
mni_coords = nib.affines.apply_affine(template_img.affine, sig_coords)

df = pd.DataFrame({
    'voxel_x': sig_coords[:, 0],
    'voxel_y': sig_coords[:, 1],
    'voxel_z': sig_coords[:, 2],
    'mni_x': mni_coords[:, 0],
    'mni_y': mni_coords[:, 1],
    'mni_z': mni_coords[:, 2],
    'p_value': p_values[sig_mask.astype(bool)]
})

df.to_csv('significant_voxels.csv', index=False)
```

---

## Integration with Other Tools

### Use with FSL

```python
# Save results in FSL-compatible format
from utils import save_nifti

# Save as integer mask (for FSL cluster)
save_nifti(sig_mask, template_img.affine, template_img.header,
          "fsl_mask.nii.gz", dtype=np.uint8)

# Save as float (for FSL overlay)
save_nifti(1 - p_values, template_img.affine, template_img.header,
          "fsl_1minusp.nii.gz", dtype=np.float32)
```

### Use with SPM

```python
# Create SPM-style stat map
z_scores = -stats.norm.ppf(p_values)
z_scores[~valid_mask] = 0
z_scores[np.isinf(z_scores)] = 10  # Cap infinite values

save_nifti(z_scores, template_img.affine, template_img.header,
          "spm_zscores.nii.gz")
```

### Use with nilearn

```python
from nilearn import plotting

# Create nilearn-compatible image
from nibabel import Nifti1Image
sig_img = Nifti1Image(sig_mask.astype(float), template_img.affine)

# Plot on glass brain
plotting.plot_glass_brain(sig_img, threshold=0.5, colorbar=True)
plotting.show()

# Plot on statistical map
plotting.plot_stat_map(sig_img, threshold=0.5, display_mode='z',
                       cut_coords=5, title="Significant Voxels")
plotting.show()
```

### Interactive Python Session Example

```python
# Start interactive session
python

# Import and load
from utils import *
import numpy as np

# Quick analysis
resp, non_resp, img = load_subject_data('subject_class.csv', '.')
p, u, mask = mannwhitneyu_voxelwise(resp, non_resp)
sig, thresh, clust = cluster_based_correction(resp, non_resp, p, mask)

# Quick stats
print(f"Significant voxels: {np.sum(sig)}")
print(f"Largest cluster: {clust[0]['size']} voxels" if clust else "No clusters")
```

---

## Advanced Examples

### Customize Summary Output

The `generate_summary` function is now fully configurable:

```python
from utils import generate_summary

# Custom parameters
params = {
    'cluster_threshold': 0.01,
    'n_permutations': 1000,
    'alpha': 0.05,
    'n_jobs': 8
}

# Generate summary with custom labels
generate_summary(
    group1_data,
    group2_data,
    sig_mask,
    cluster_threshold,
    clusters,
    atlas_results,
    'my_summary.txt',
    correction_method='cluster',
    params=params,
    group1_name='High Performers',      # Custom group name
    group2_name='Low Performers',       # Custom group name
    value_metric='Activation Level'     # Custom metric name
)
```

### Create Custom Summary (JSON)

```python
def create_custom_summary(sig_mask, clusters, atlas_results):
    """Create a custom summary tailored to your needs"""
    
    summary = {
        'total_sig_voxels': int(np.sum(sig_mask)),
        'n_clusters': len(clusters),
        'largest_cluster_size': clusters[0]['size'] if clusters else 0,
        'atlas_regions': {}
    }
    
    for atlas_name, regions in atlas_results.items():
        summary['atlas_regions'][atlas_name] = [
            {'region': r['region_id'], 'voxels': r['overlap_voxels']}
            for r in regions[:5]  # Top 5 regions
        ]
    
    return summary

# Use it
summary = create_custom_summary(sig_mask, clusters, atlas_results)

import json
with open('my_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

### Parallel Processing

```python
from joblib import Parallel, delayed

def process_subject(subject_file):
    """Process individual subject"""
    img = nib.load(subject_file)
    return img.get_fdata()

# Load subjects in parallel
subject_files = [f"sub-{i:03d}_data.nii.gz" for i in range(101, 137)]
data = Parallel(n_jobs=4)(delayed(process_subject)(f) for f in subject_files)
```

---

## Tips & Best Practices

1. **Always check your data dimensions** before running analysis
2. **Use verbose=False** in loops to avoid cluttering output
3. **Save intermediate results** for long-running analyses
4. **Test on a small subset** before processing the full dataset
5. **Document your parameters** for reproducibility
6. **Check log files** after analysis for timing and diagnostics
7. **Keep log files** with your results for full provenance

## Logging

The `main.py` script automatically creates timestamped log files. For custom scripts:

```python
import logging
from datetime import datetime

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'my_analysis_{timestamp}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use throughout your analysis
logger.info("Starting analysis...")
logger.info(f"Processing {n_subjects} subjects")
```

For more examples and documentation, see:
- `README.md` - Overview and installation
- `utils.py` - Full function documentation
- `main.py` - Complete pipeline example

