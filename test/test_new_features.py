#!/usr/bin/env python3
"""
Quick test of new features:
1. Top 10 observed clusters in log and summary
2. Permutation details logging
"""

import os
import sys
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_subject_data, ttest_voxelwise, cluster_based_correction
from scipy.ndimage import label

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(PROJECT_ROOT, 'data')
csv_file = os.path.join(data_dir, 'subject_class.csv')
output_dir = os.path.join(PROJECT_ROOT, 'test_output')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("TESTING NEW FEATURES")
print("="*70)

# Load data WITH subject IDs
print("\n[1] Loading data with subject IDs...")
responders, non_responders, template_img, resp_ids, non_resp_ids = load_subject_data(
    csv_file, 
    data_dir,
    return_ids=True
)

print(f"Responder IDs: {resp_ids}")
print(f"Non-Responder IDs: {non_resp_ids}")

# Run t-test
print("\n[2] Running voxelwise t-test...")
start = time.time()
p_values, t_statistics, valid_mask = ttest_voxelwise(
    responders, 
    non_responders,
    test_type='unpaired',
    alternative='greater'
)
print(f"Completed in {time.time() - start:.2f} seconds")

# Get observed clusters
print("\n[3] Identifying observed clusters...")
cluster_threshold = 0.01
observed_mask = (p_values < cluster_threshold) & valid_mask
observed_labeled, n_observed_clusters = label(observed_mask)

print(f"Found {n_observed_clusters} clusters at p < {cluster_threshold}")

if n_observed_clusters > 0:
    observed_cluster_sizes = []
    for cluster_id in range(1, n_observed_clusters + 1):
        size = np.sum(observed_labeled == cluster_id)
        observed_cluster_sizes.append(size)
    
    observed_cluster_sizes.sort(reverse=True)
    
    print("\nTop 10 Observed Clusters:")
    for i, size in enumerate(observed_cluster_sizes[:10], 1):
        print(f"  {i:2d}. {size:6d} voxels")

# Run cluster correction with permutation logging (small number for speed)
print("\n[4] Running cluster correction with permutation logging...")
print("Using only 20 permutations for speed...")
permutation_log_file = os.path.join(output_dir, 'permutation_details.txt')

start = time.time()
sig_mask, cluster_threshold_val, sig_clusters, null_distribution, all_clusters = cluster_based_correction(
    responders, 
    non_responders, 
    p_values, 
    valid_mask,
    cluster_threshold=0.01,
    n_permutations=20,
    alpha=0.01,
    test_type='unpaired',
    alternative='greater',
    n_jobs=-1,
    save_permutation_log=True,
    permutation_log_file=permutation_log_file,
    subject_ids_resp=resp_ids,
    subject_ids_non_resp=non_resp_ids
)
print(f"Completed in {time.time() - start:.2f} seconds")

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
print(f"\nPermutation details saved to: {permutation_log_file}")
print("\nCheck the file to verify it contains:")
print("  - Original responder and non-responder IDs")
print("  - Each permutation's group assignments")
print("  - Maximum cluster size for each permutation")

# Show first few lines of the file
print("\nFirst 30 lines of permutation details file:")
print("-" * 70)
with open(permutation_log_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 30:
            break
        print(line, end='')


