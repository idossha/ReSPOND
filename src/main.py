#!/usr/bin/env python3
"""
Main entry point for voxelwise neuroimaging statistical analysis

This script performs non-parametric voxelwise statistical analysis to identify
brain regions with significantly different current intensity between responders
and non-responders using cluster-based permutation correction.

Usage:
    python main.py

For programmatic usage, see CLI.md
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime
import time

from utils import (
    load_subject_data,
    ttest_voxelwise,
    cluster_based_correction,
    cluster_analysis,
    atlas_overlap_analysis,
    generate_summary,
    save_nifti,
    plot_permutation_null_distribution
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Customize these parameters for your specific study.
# All labels, paths, and statistical parameters can be easily modified here.

# Determine project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # Directory paths
    'data_dir': os.path.join(PROJECT_ROOT, 'data'),
    'assets_dir': os.path.join(PROJECT_ROOT, 'assets'),
    'output_dir': os.path.join(PROJECT_ROOT, 'output'),
    
    # Input files
    'csv_file': "subject_class.csv",
    
    # Output files (log file is auto-generated with timestamp)
    'output_mask': "significant_voxels_mask.nii.gz",
    'output_pvalues': "pvalues_map.nii.gz",
    'output_summary': "analysis_summary.txt",
    'output_avg_responders': "average_responders.nii.gz",
    'output_avg_non_responders': "average_non_responders.nii.gz",
    'output_difference': "difference_map.nii.gz",
    'output_permutation_plot': "permutation_null_distribution.pdf",
    
    # Atlas files (in assets/)
    'atlas_files': [
        "HarvardOxford-cort-maxprob-thr0-1mm.nii.gz",
        "Talairach-labels-1mm.nii.gz",
        "MNI_Glasser_HCP_v1.0.nii.gz"
    ],
    
    # Statistical parameters
    'test_type': 'unpaired',        # 'paired' or 'unpaired' t-test
    'alternative': 'greater',       # 'two-sided', 'greater' (resp > non-resp), or 'less'
    'cluster_threshold': 0.01,      # p < 0.01 for cluster formation
    'n_permutations': 1000,         # Number of permutations
    'alpha': 0.05,                  # Cluster-level significance
    'n_jobs': -1,                   # Number of parallel jobs (-1 = all cores, 1 = sequential)
    
    # Group and metric labels (customize for your study)
    'group1_name': 'Responders',
    'group2_name': 'Non-Responders',
    'value_metric': 'Current Intensity'
}


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(output_dir):
    """
    Set up logging to both file and console
    
    Parameters:
    -----------
    output_dir : str
        Directory where log file will be saved
    
    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    log_file : str
        Path to log file
    """
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"analysis_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('VoxelwiseAnalysis')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simple)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    """Main analysis workflow"""
    
    # Start timing
    analysis_start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Set up logging
    logger, log_file = setup_logging(CONFIG['output_dir'])
    
    logger.info("="*70)
    logger.info("VOXELWISE NON-PARAMETRIC STATISTICAL ANALYSIS")
    logger.info("="*70)
    logger.info(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    # Log configuration
    logger.info("CONFIGURATION:")
    logger.info(f"  Data directory: {CONFIG['data_dir']}")
    logger.info(f"  Output directory: {CONFIG['output_dir']}")
    logger.info(f"  Assets directory: {CONFIG['assets_dir']}")
    logger.info(f"  CSV file: {CONFIG['csv_file']}")
    logger.info(f"  Statistical test: {CONFIG['test_type'].capitalize()} t-test")
    alt_text = {
        'two-sided': 'two-sided (≠)',
        'greater': f"one-sided ({CONFIG['group1_name']} > {CONFIG['group2_name']})",
        'less': f"one-sided ({CONFIG['group1_name']} < {CONFIG['group2_name']})"
    }
    logger.info(f"  Alternative hypothesis: {alt_text.get(CONFIG['alternative'], CONFIG['alternative'])}")
    logger.info(f"  Cluster threshold: {CONFIG['cluster_threshold']}")
    logger.info(f"  Number of permutations: {CONFIG['n_permutations']}")
    logger.info(f"  Alpha level: {CONFIG['alpha']}")
    logger.info(f"  Parallel jobs: {CONFIG['n_jobs']}")
    logger.info(f"  Group 1: {CONFIG['group1_name']}")
    logger.info(f"  Group 2: {CONFIG['group2_name']}")
    logger.info(f"  Metric: {CONFIG['value_metric']}")
    logger.info("")
    
    # Get full paths
    csv_file = os.path.join(CONFIG['data_dir'], CONFIG['csv_file'])
    
    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    logger.info("\n[1/8] LOADING SUBJECT DATA")
    logger.info("-" * 70)
    step_start = time.time()
    
    responders, non_responders, template_img, resp_ids, non_resp_ids = load_subject_data(
        csv_file, 
        CONFIG['data_dir'],
        return_ids=True
    )
    
    logger.info(f"Loaded {responders.shape[-1]} {CONFIG['group1_name']}: {resp_ids}")
    logger.info(f"Loaded {non_responders.shape[-1]} {CONFIG['group2_name']}: {non_resp_ids}")
    logger.info(f"Image shape: {responders.shape[:3]}")
    logger.info(f"Step completed in {time.time() - step_start:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 2. VOXELWISE STATISTICAL TEST
    # -------------------------------------------------------------------------
    logger.info("\n[2/8] RUNNING VOXELWISE STATISTICAL TESTS")
    logger.info("-" * 70)
    step_start = time.time()
    
    p_values, t_statistics, valid_mask = ttest_voxelwise(
        responders, 
        non_responders,
        test_type=CONFIG['test_type'],
        alternative=CONFIG['alternative']
    )
    
    n_valid = np.sum(valid_mask)
    logger.info(f"Tested {n_valid} valid voxels")
    logger.info(f"Minimum p-value: {np.min(p_values[valid_mask]):.6e}")
    logger.info(f"Voxels with p<0.01: {np.sum((p_values < 0.01) & valid_mask)}")
    logger.info(f"Step completed in {time.time() - step_start:.2f} seconds")
    
    # Log observed clusters before permutation correction
    from scipy.ndimage import label as scipy_label
    observed_mask = (p_values < CONFIG['cluster_threshold']) & valid_mask
    observed_labeled, n_observed_clusters = scipy_label(observed_mask)
    
    logger.info(f"\nObserved clusters at p < {CONFIG['cluster_threshold']} (before permutation correction):")
    logger.info(f"Total clusters found: {n_observed_clusters}")
    
    if n_observed_clusters > 0:
        observed_cluster_sizes = []
        for cluster_id in range(1, n_observed_clusters + 1):
            size = np.sum(observed_labeled == cluster_id)
            observed_cluster_sizes.append(size)
        
        observed_cluster_sizes.sort(reverse=True)
        
        logger.info(f"\nTop 10 largest observed clusters (before permutation correction):")
        for i, size in enumerate(observed_cluster_sizes[:10], 1):
            logger.info(f"  Cluster {i:2d}: {size:6d} voxels")
        
        logger.info(f"\nLargest observed cluster: {observed_cluster_sizes[0]} voxels")
        logger.info(f"Total voxels in all clusters: {sum(observed_cluster_sizes)}")
    
    # -------------------------------------------------------------------------
    # 3. CLUSTER-BASED PERMUTATION CORRECTION
    # -------------------------------------------------------------------------
    logger.info("\n[3/8] APPLYING CLUSTER-BASED PERMUTATION CORRECTION")
    logger.info("-" * 70)
    step_start = time.time()
    
    # Set up permutation log file path
    permutation_log_file = os.path.join(CONFIG['output_dir'], 'permutation_details.txt')
    
    sig_mask, cluster_threshold, sig_clusters, null_distribution, all_clusters = cluster_based_correction(
        responders, 
        non_responders, 
        p_values, 
        valid_mask,
        cluster_threshold=CONFIG['cluster_threshold'],
        n_permutations=CONFIG['n_permutations'],
        alpha=CONFIG['alpha'],
        test_type=CONFIG['test_type'],
        alternative=CONFIG['alternative'],
        n_jobs=CONFIG['n_jobs'],
        save_permutation_log=True,
        permutation_log_file=permutation_log_file,
        subject_ids_resp=resp_ids,
        subject_ids_non_resp=non_resp_ids
    )
    
    logger.info(f"Cluster size threshold: {cluster_threshold:.1f} voxels")
    logger.info(f"Significant clusters found: {len(sig_clusters)}")
    logger.info(f"Total significant voxels: {np.sum(sig_mask)}")
    logger.info(f"Step completed in {time.time() - step_start:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 4. CLUSTER ANALYSIS
    # -------------------------------------------------------------------------
    logger.info("\n[4/8] ANALYZING SIGNIFICANT CLUSTERS")
    logger.info("-" * 70)
    step_start = time.time()
    
    clusters = cluster_analysis(sig_mask, template_img.affine)
    
    if clusters:
        logger.info(f"Largest cluster: {clusters[0]['size']} voxels")
        logger.info(f"MNI center: ({clusters[0]['center_mni'][0]:.1f}, "
                   f"{clusters[0]['center_mni'][1]:.1f}, {clusters[0]['center_mni'][2]:.1f})")
    logger.info(f"Step completed in {time.time() - step_start:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 5. PLOT PERMUTATION NULL DISTRIBUTION
    # -------------------------------------------------------------------------
    logger.info("\n[5/8] PLOTTING PERMUTATION NULL DISTRIBUTION")
    logger.info("-" * 70)
    step_start = time.time()
    
    perm_plot_file = os.path.join(CONFIG['output_dir'], CONFIG['output_permutation_plot'])
    plot_permutation_null_distribution(
        null_distribution,
        cluster_threshold,
        all_clusters,
        perm_plot_file,
        alpha=CONFIG['alpha']
    )
    logger.info(f"Step completed in {time.time() - step_start:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 6. GENERATE AVERAGE MAPS
    # -------------------------------------------------------------------------
    logger.info("\n[6/8] GENERATING AVERAGE INTENSITY MAPS")
    logger.info("-" * 70)
    step_start = time.time()
    
    # Average responders
    avg_responders = np.mean(responders, axis=-1)
    avg_resp_file = os.path.join(CONFIG['output_dir'], CONFIG['output_avg_responders'])
    save_nifti(avg_responders, template_img.affine, template_img.header, avg_resp_file)
    logger.info(f"Saved: {CONFIG['output_avg_responders']}")
    
    # Average non-responders
    avg_non_responders = np.mean(non_responders, axis=-1)
    avg_non_resp_file = os.path.join(CONFIG['output_dir'], CONFIG['output_avg_non_responders'])
    save_nifti(avg_non_responders, template_img.affine, template_img.header, avg_non_resp_file)
    logger.info(f"Saved: {CONFIG['output_avg_non_responders']}")
    
    # Difference map
    diff_map = avg_responders - avg_non_responders
    diff_file = os.path.join(CONFIG['output_dir'], CONFIG['output_difference'])
    save_nifti(diff_map, template_img.affine, template_img.header, diff_file)
    logger.info(f"Saved: {CONFIG['output_difference']}")
    
    logger.info(f"Mean difference in brain: {np.mean(diff_map[valid_mask]):.4f}")
    logger.info(f"Max absolute difference: {np.max(np.abs(diff_map[valid_mask])):.4f}")
    logger.info(f"Step completed in {time.time() - step_start:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 7. ATLAS OVERLAP ANALYSIS
    # -------------------------------------------------------------------------
    logger.info("\n[7/8] PERFORMING ATLAS OVERLAP ANALYSIS")
    logger.info("-" * 70)
    step_start = time.time()
    
    atlas_results = atlas_overlap_analysis(
        sig_mask, 
        CONFIG['atlas_files'], 
        CONFIG['assets_dir'], 
        reference_img=template_img
    )
    
    for atlas_name, regions in atlas_results.items():
        if regions:
            logger.info(f"{atlas_name}: {len(regions)} regions with overlap")
    logger.info(f"Step completed in {time.time() - step_start:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 8. SAVE OUTPUTS
    # -------------------------------------------------------------------------
    logger.info("\n[8/8] SAVING RESULTS")
    logger.info("-" * 70)
    step_start = time.time()
    
    # Binary mask
    output_mask = os.path.join(CONFIG['output_dir'], CONFIG['output_mask'])
    save_nifti(sig_mask, template_img.affine, template_img.header, output_mask, dtype=np.uint8)
    logger.info(f"Saved: {CONFIG['output_mask']}")
    
    # P-values map (as -log10 for visualization)
    log_p = -np.log10(p_values + 1e-10)
    log_p[~valid_mask] = 0
    output_pvalues = os.path.join(CONFIG['output_dir'], CONFIG['output_pvalues'])
    save_nifti(log_p, template_img.affine, template_img.header, output_pvalues)
    logger.info(f"Saved: {CONFIG['output_pvalues']}")
    
    # Summary report
    output_summary = os.path.join(CONFIG['output_dir'], CONFIG['output_summary'])
    
    # Prepare parameters dictionary for summary
    summary_params = {
        'cluster_threshold': CONFIG['cluster_threshold'],
        'n_permutations': CONFIG['n_permutations'],
        'alpha': CONFIG['alpha'],
        'n_jobs': CONFIG['n_jobs']
    }
    
    generate_summary(
        responders, 
        non_responders, 
        sig_mask, 
        cluster_threshold,
        clusters, 
        atlas_results, 
        output_summary, 
        correction_method="cluster",
        params=summary_params,
        group1_name=CONFIG['group1_name'],
        group2_name=CONFIG['group2_name'],
        value_metric=CONFIG['value_metric'],
        test_type=CONFIG['test_type'],
        observed_cluster_sizes=observed_cluster_sizes if n_observed_clusters > 0 else None
    )
    logger.info(f"Saved: {CONFIG['output_summary']}")
    logger.info(f"Step completed in {time.time() - step_start:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------
    total_time = time.time() - analysis_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*70)
    logger.info(f"Total analysis time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("OUTPUT FILES:")
    logger.info(f"  1. Binary mask: {output_mask}")
    logger.info(f"  2. P-values map (-log10): {output_pvalues}")
    logger.info(f"  3. Summary report: {output_summary}")
    logger.info(f"  4. Average {CONFIG['group1_name']}: {avg_resp_file}")
    logger.info(f"  5. Average {CONFIG['group2_name']}: {avg_non_resp_file}")
    logger.info(f"  6. Difference map: {diff_file}")
    logger.info(f"  7. Permutation null distribution plot: {perm_plot_file}")
    logger.info(f"  8. Permutation details log: {permutation_log_file}")
    logger.info(f"  9. Log file: {log_file}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  - Visualize: Use FSLeyes, SPM, or nilearn")
    logger.info("  - Post-hoc atlas analysis: python posthoc_atlas_analysis.py")
    logger.info("  - Programmatic usage: See CLI.md for examples")
    logger.info("")
    
    # Close log handlers
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    main()

