#!/usr/bin/env python3
"""
Example script demonstrating how to save permutation details
"""

import os
import sys
import numpy as np

# Import utilities
from utils import (
    load_subject_data,
    ttest_voxelwise,
    cluster_based_correction
)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(PROJECT_ROOT, 'data')
csv_file = os.path.join(data_dir, 'subject_class.csv')
output_dir = os.path.join(PROJECT_ROOT, 'output')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("EXAMPLE: Saving Permutation Details")
print("="*70)

# Load data WITH subject IDs
print("\n[1] Loading data...")
responders, non_responders, template_img, resp_ids, non_resp_ids = load_subject_data(
    csv_file, 
    data_dir,
    return_ids=True  # Request subject IDs
)

print(f"\nResponder IDs: {resp_ids}")
print(f"Non-Responder IDs: {non_resp_ids}")

# Run t-test
print("\n[2] Running voxelwise t-test...")
p_values, t_statistics, valid_mask = ttest_voxelwise(
    responders, 
    non_responders,
    test_type='unpaired',
    alternative='greater'
)

# Run cluster correction with permutation logging
print("\n[3] Running cluster correction (with permutation logging)...")
permutation_log_file = os.path.join(output_dir, 'permutation_details.txt')

sig_mask, cluster_threshold, sig_clusters, null_distribution, all_clusters = cluster_based_correction(
    responders, 
    non_responders, 
    p_values, 
    valid_mask,
    cluster_threshold=0.01,
    n_permutations=50,  # Small number for demo
    alpha=0.01,
    test_type='unpaired',
    alternative='greater',
    n_jobs=-1,
    save_permutation_log=True,  # Enable logging
    permutation_log_file=permutation_log_file,
    subject_ids_resp=resp_ids,  # Pass subject IDs
    subject_ids_non_resp=non_resp_ids
)

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nPermutation details saved to: {permutation_log_file}")
print("\nYou can now inspect which subjects were assigned to which group")
print("in each permutation, and see the resulting maximum cluster size.")

