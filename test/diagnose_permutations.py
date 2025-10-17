#!/usr/bin/env python3
"""
Run multiple permutations to find which ones produce huge clusters
"""
import numpy as np
import sys
import os
from scipy.ndimage import label
from joblib import Parallel, delayed
import multiprocessing

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_subject_data, ttest_voxelwise

# Load data
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(PROJECT_ROOT, 'data')
csv_file = os.path.join(data_dir, 'subject_class.csv')

print("Loading data...")
responders, non_responders, template_img = load_subject_data(csv_file, data_dir)

# Combine data
all_data = np.concatenate([responders, non_responders], axis=-1)
n_resp = responders.shape[-1]
n_total = all_data.shape[-1]

print(f"Sample sizes: {n_resp} vs {n_total - n_resp}")

# Get valid mask from original test
p_values, t_statistics, valid_mask = ttest_voxelwise(
    responders, 
    non_responders,
    test_type='unpaired',
    alternative='greater'
)

cluster_threshold = 0.01

def run_single_perm(perm_i, all_data, n_resp, n_total, valid_mask, cluster_threshold):
    """Run a single permutation and return results"""
    # Random permutation
    np.random.seed(perm_i)
    perm_idx = np.random.permutation(n_total)
    
    # Create permuted groups
    perm_data = all_data[:, :, :, perm_idx]
    perm_resp = perm_data[:, :, :, :n_resp]
    perm_non_resp = perm_data[:, :, :, n_resp:]
    
    # Run t-test
    perm_p_values, _, _ = ttest_voxelwise(
        perm_resp, 
        perm_non_resp,
        test_type='unpaired',
        alternative='greater'
    )
    
    # Find clusters
    perm_mask = (perm_p_values < cluster_threshold) & valid_mask
    perm_labeled, perm_n_clusters = label(perm_mask)
    
    # Get max cluster size
    if perm_n_clusters > 0:
        perm_cluster_sizes = [np.sum(perm_labeled == cid) 
                             for cid in range(1, perm_n_clusters + 1)]
        max_size = max(perm_cluster_sizes)
        total_sig = np.sum(perm_mask)
    else:
        max_size = 0
        total_sig = 0
    
    return {
        'perm_i': perm_i,
        'max_size': max_size,
        'total_sig': total_sig,
        'n_clusters': perm_n_clusters,
        'perm_idx': perm_idx,
        'perm_resp': perm_resp if max_size > 10000 else None,
        'perm_non_resp': perm_non_resp if max_size > 10000 else None,
        'perm_p_values': perm_p_values if max_size > 10000 else None
    }

n_perms = 50  # Increased from 20
n_jobs = multiprocessing.cpu_count()

print(f"\nRunning {n_perms} permutations in parallel using {n_jobs} cores...")
print("="*70)

# Run permutations in parallel
results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_single_perm)(perm_i, all_data, n_resp, n_total, valid_mask, cluster_threshold)
    for perm_i in range(n_perms)
)

max_sizes = []

# Sort results by permutation index
results = sorted(results, key=lambda x: x['perm_i'])

# Process results
for result in results:
    perm_i = result['perm_i']
    max_size = result['max_size']
    total_sig = result['total_sig']
    perm_n_clusters = result['n_clusters']
    
    max_sizes.append(max_size)
    
    print(f"Perm {perm_i:3d}: max_cluster={max_size:6d} voxels, "
          f"total_sig={total_sig:6d}, n_clusters={perm_n_clusters:4d}")
    
    # If we find a huge one, analyze it
    if max_size > 10000:
        print(f"\n  ⚠️  FOUND HUGE CLUSTER! Analyzing permutation {perm_i}...")
        perm_idx = result['perm_idx']
        perm_resp = result['perm_resp']
        perm_non_resp = result['perm_non_resp']
        perm_p_values = result['perm_p_values']
        
        print(f"  Group assignment: {perm_idx[:n_resp]} vs {perm_idx[n_resp:]}")
        
        # Check variance in each group
        resp_mean = np.mean(perm_resp[valid_mask], axis=0)
        non_resp_mean = np.mean(perm_non_resp[valid_mask], axis=0)
        
        print(f"  Permuted group 1 mean values: min={np.min(resp_mean):.4f}, "
              f"max={np.max(resp_mean):.4f}, mean={np.mean(resp_mean):.4f}")
        print(f"  Permuted group 2 mean values: min={np.min(non_resp_mean):.4f}, "
              f"max={np.max(non_resp_mean):.4f}, mean={np.mean(non_resp_mean):.4f}")
        
        # Check how many voxels have p < various thresholds
        print(f"  P-value distribution:")
        for thresh in [0.001, 0.01, 0.05, 0.1]:
            n_sig = np.sum((perm_p_values < thresh) & valid_mask)
            pct = 100 * n_sig / np.sum(valid_mask)
            print(f"    p < {thresh}: {n_sig:7d} voxels ({pct:5.2f}%)")

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print(f"Max cluster sizes across {len(max_sizes)} permutations:")
print(f"  Min:    {np.min(max_sizes):7.1f}")
print(f"  Mean:   {np.mean(max_sizes):7.1f}")
print(f"  Median: {np.median(max_sizes):7.1f}")
print(f"  95th:   {np.percentile(max_sizes, 95):7.1f}")
print(f"  99th:   {np.percentile(max_sizes, 99):7.1f}")
print(f"  Max:    {np.max(max_sizes):7.1f}")

print(f"\nPermutations with >1000 voxel clusters: {np.sum(np.array(max_sizes) > 1000)}")
print(f"Permutations with >5000 voxel clusters: {np.sum(np.array(max_sizes) > 5000)}")
print(f"Permutations with >10000 voxel clusters: {np.sum(np.array(max_sizes) > 10000)}")

