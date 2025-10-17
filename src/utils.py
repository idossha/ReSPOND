"""
Utility functions for voxelwise neuroimaging statistical analysis

This module contains all core functionality for:
- Loading and processing neuroimaging data
- Non-parametric statistical testing
- Cluster-based permutation correction
- Atlas overlap analysis
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from scipy.ndimage import label
from nibabel.processing import resample_from_to
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation

# ==============================================================================
# DATA LOADING AND I/O
# ==============================================================================

def load_subject_data(csv_file, data_dir):
    """
    Load subject classifications and corresponding NIfTI files
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with columns: subject_id, response, simulation_name
    data_dir : str
        Directory containing NIfTI files
    
    Returns:
    --------
    responders : ndarray (x, y, z, n_subjects)
        4D array of responder data
    non_responders : ndarray (x, y, z, n_subjects)
        4D array of non-responder data
    template_img : nibabel image
        Template image for affine/header information
    """
    df = pd.read_csv(csv_file)
    
    responders = []
    non_responders = []
    responder_ids = []
    non_responder_ids = []
    
    for _, row in df.iterrows():
        subject_num = row['subject_id'].replace('sub-', '')
        sim_name = row['simulation_name']
        response = row['response']
        
        # Construct filename
        filename = f"{subject_num}_grey_{sim_name}_TI_MNI_MNI_TI_max.nii.gz"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found - {filename}")
            continue
        
        # Load NIfTI
        img = nib.load(filepath)
        data = img.get_fdata()
        
        # Ensure 3D data (squeeze out extra dimensions if present)
        while data.ndim > 3:
            data = np.squeeze(data, axis=-1)
        
        if response == 1:
            responders.append(data)
            responder_ids.append(subject_num)
        else:
            non_responders.append(data)
            non_responder_ids.append(subject_num)
    
    print(f"\nLoaded {len(responders)} responders: {responder_ids}")
    print(f"Loaded {len(non_responders)} non-responders: {non_responder_ids}")
    
    # Stack into 4D arrays (subjects x volume)
    responders = np.stack(responders, axis=-1)
    non_responders = np.stack(non_responders, axis=-1)
    
    print(f"Responders shape: {responders.shape}")
    print(f"Non-responders shape: {non_responders.shape}")
    
    return responders, non_responders, img


def save_nifti(data, affine, header, filepath, dtype=np.float32):
    """
    Save data as NIfTI file
    
    Parameters:
    -----------
    data : ndarray
        Data to save
    affine : ndarray
        Affine transformation matrix
    header : nibabel header
        NIfTI header
    filepath : str
        Output file path
    dtype : numpy dtype
        Data type for output
    """
    img = nib.Nifti1Image(data.astype(dtype), affine, header)
    nib.save(img, filepath)
    print(f"Saved: {filepath}")


# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

def _fast_ttest_ind(a, b, alternative='two-sided'):
    """
    Fast manual computation of independent samples t-test
    ~13x faster than scipy.stats.ttest_ind
    
    Parameters:
    -----------
    a, b : array-like
        Sample arrays (group 1 and group 2)
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis (default: 'two-sided'):
        * 'two-sided': means are different (a ≠ b)
        * 'greater': mean of a is greater than mean of b (a > b)
        * 'less': mean of a is less than mean of b (a < b)
    
    Returns:
    --------
    t_stat : float
        T-statistic
    p_val : float
        P-value for the specified alternative hypothesis
    """
    n1, n2 = len(a), len(b)
    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    # Avoid division by zero
    if pooled_std == 0:
        return 0.0, 1.0
    
    # T-statistic
    t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2))
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    # P-value based on alternative hypothesis
    if alternative == 'two-sided':
        p_val = 2 * stats.t.sf(np.abs(t_stat), df)
    elif alternative == 'greater':
        # One-sided: test if mean(a) > mean(b)
        # P(T > t_stat) = sf(t_stat)
        p_val = stats.t.sf(t_stat, df)
    elif alternative == 'less':
        # One-sided: test if mean(a) < mean(b)
        # P(T < t_stat) = cdf(t_stat) = sf(-t_stat)
        p_val = stats.t.sf(-t_stat, df)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    return t_stat, p_val


def _fast_ttest_rel(a, b, alternative='two-sided'):
    """
    Fast manual computation of paired samples t-test
    ~10x faster than scipy.stats.ttest_rel
    
    Parameters:
    -----------
    a, b : array-like
        Paired sample arrays
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis (default: 'two-sided'):
        * 'two-sided': means are different (a ≠ b)
        * 'greater': mean of a is greater than mean of b (a > b)
        * 'less': mean of a is less than mean of b (a < b)
    
    Returns:
    --------
    t_stat : float
        T-statistic
    p_val : float
        P-value for the specified alternative hypothesis
    """
    # Compute differences
    diff = a - b
    n = len(diff)
    
    # Mean and std of differences
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # Avoid division by zero
    if std_diff == 0:
        return 0.0, 1.0
    
    # T-statistic
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    
    # Degrees of freedom
    df = n - 1
    
    # P-value based on alternative hypothesis
    if alternative == 'two-sided':
        p_val = 2 * stats.t.sf(np.abs(t_stat), df)
    elif alternative == 'greater':
        # One-sided: test if mean(a) > mean(b)
        p_val = stats.t.sf(t_stat, df)
    elif alternative == 'less':
        # One-sided: test if mean(a) < mean(b)
        p_val = stats.t.sf(-t_stat, df)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    return t_stat, p_val


def ttest_voxelwise(responders, non_responders, test_type='unpaired', alternative='two-sided', verbose=True):
    """
    Perform t-test (paired or unpaired) at each voxel using optimized manual computation
    
    Uses fast manual t-test implementation (~13x faster than scipy.stats.ttest_ind)
    
    Parameters:
    -----------
    responders : ndarray (x, y, z, n_subjects)
        Responder data (group 1)
    non_responders : ndarray (x, y, z, n_subjects)
        Non-responder data (group 2)
    test_type : str
        Either 'paired' or 'unpaired' t-test
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis (default: 'two-sided'):
        * 'two-sided': means are different (responders ≠ non-responders)
        * 'greater': responders have higher values (responders > non-responders)
        * 'less': responders have lower values (responders < non-responders)
    verbose : bool
        Print progress information
    
    Returns:
    --------
    p_values : ndarray (x, y, z)
        P-value at each voxel
    t_statistics : ndarray (x, y, z)
        T-statistic at each voxel
    valid_mask : ndarray (x, y, z)
        Boolean mask of valid voxels
    """
    if verbose:
        test_name = "Paired" if test_type == 'paired' else "Unpaired (Independent Samples)"
        alt_text = ""
        if alternative == 'greater':
            alt_text = " (one-sided: responders > non-responders)"
        elif alternative == 'less':
            alt_text = " (one-sided: responders < non-responders)"
        print(f"\nPerforming voxelwise {test_name} t-tests{alt_text} (optimized)...")
    
    # Validate test type
    if test_type not in ['paired', 'unpaired']:
        raise ValueError("test_type must be 'paired' or 'unpaired'")
    
    # For paired test, check that sample sizes match
    if test_type == 'paired':
        if responders.shape[-1] != non_responders.shape[-1]:
            raise ValueError(f"Paired t-test requires equal sample sizes. "
                           f"Got {responders.shape[-1]} vs {non_responders.shape[-1]} subjects")
    
    shape = responders.shape[:3]
    p_values = np.ones(shape)
    t_statistics = np.zeros(shape)
    
    # Create mask of valid voxels (non-zero in at least some subjects)
    responder_mask = np.any(responders > 0, axis=-1)
    non_responder_mask = np.any(non_responders > 0, axis=-1)
    valid_mask = responder_mask | non_responder_mask
    
    total_voxels = np.sum(valid_mask)
    if verbose:
        print(f"Testing {total_voxels} valid voxels...")
    
    # Get coordinates of valid voxels
    valid_coords = np.argwhere(valid_mask)
    
    # Perform test at each voxel
    iterator = tqdm(valid_coords, desc="Testing voxels") if verbose else valid_coords
    for coord in iterator:
        i, j, k = coord[0], coord[1], coord[2]
        
        resp_vals = responders[i, j, k, :]
        non_resp_vals = non_responders[i, j, k, :]
        
        # Only test if we have variance
        if test_type == 'paired':
            # For paired test, check variance of differences
            diff = resp_vals - non_resp_vals
            if np.std(diff) > 0:
                try:
                    t_stat, p_val = _fast_ttest_rel(resp_vals, non_resp_vals, alternative=alternative)
                    t_statistics[i, j, k] = t_stat
                    p_values[i, j, k] = p_val
                except:
                    pass
        else:
            # For unpaired test, check variance in at least one group
            if np.std(resp_vals) > 0 or np.std(non_resp_vals) > 0:
                try:
                    t_stat, p_val = _fast_ttest_ind(resp_vals, non_resp_vals, alternative=alternative)
                    t_statistics[i, j, k] = t_stat
                    p_values[i, j, k] = p_val
                except:
                    pass
    
    return p_values, t_statistics, valid_mask


def _run_single_permutation(test_data, test_coords, n_resp, n_total, cluster_threshold, 
                           valid_mask, p_values_shape, test_type='unpaired', 
                           alternative='two-sided', seed=None):
    """
    Helper function to run a single permutation (for parallel processing)
    
    Parameters:
    -----------
    test_data : ndarray
        Pre-extracted test voxel data
    test_coords : ndarray
        Coordinates of test voxels
    n_resp : int
        Number of responders
    n_total : int
        Total number of subjects
    cluster_threshold : float
        P-value threshold for cluster formation
    valid_mask : ndarray
        Boolean mask of valid voxels
    p_values_shape : tuple
        Shape of p_values array
    test_type : str
        Either 'paired' or 'unpaired' t-test
    alternative : {'two-sided', 'greater', 'less'}, optional
        Alternative hypothesis (default: 'two-sided')
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    max_cluster_size : int
        Maximum cluster size in this permutation
    """
    if seed is not None:
        np.random.seed(seed)
    
    if test_type == 'paired':
        # For paired test, randomly flip signs of differences
        # This preserves the pairing structure
        perm_test_data = test_data.copy()
        n_voxels = test_data.shape[0]
        
        # Split back into groups
        resp_data = test_data[:, :n_resp]
        non_resp_data = test_data[:, n_resp:]
        
        # Randomly flip signs for each pair
        sign_flips = np.random.choice([-1, 1], size=n_resp)
        
        # Create permuted data by flipping differences
        for i in range(n_voxels):
            mean_pair = (resp_data[i, :] + non_resp_data[i, :]) / 2
            diff_pair = (resp_data[i, :] - non_resp_data[i, :]) / 2
            perm_test_data[i, :n_resp] = mean_pair + sign_flips * diff_pair
            perm_test_data[i, n_resp:] = mean_pair - sign_flips * diff_pair
        
        perm_resp_data = perm_test_data[:, :n_resp]
        perm_non_resp_data = perm_test_data[:, n_resp:]
    else:
        # For unpaired test, randomly shuffle group labels
        perm_idx = np.random.permutation(n_total)
        perm_test_data = test_data[:, perm_idx]
        
        # Split into groups
        perm_resp_data = perm_test_data[:, :n_resp]
        perm_non_resp_data = perm_test_data[:, n_resp:]
    
    # Compute permuted p-values
    perm_p_values = np.ones(p_values_shape)
    
    for idx, coord in enumerate(test_coords):
        i, j, k = coord[0], coord[1], coord[2]
        resp_vals = perm_resp_data[idx, :]
        non_resp_vals = perm_non_resp_data[idx, :]
        
        try:
            if test_type == 'paired':
                diff = resp_vals - non_resp_vals
                if np.std(diff) > 0:
                    _, p_val = _fast_ttest_rel(resp_vals, non_resp_vals, alternative=alternative)
                    perm_p_values[i, j, k] = p_val
            else:
                if np.std(resp_vals) > 0 or np.std(non_resp_vals) > 0:
                    _, p_val = _fast_ttest_ind(resp_vals, non_resp_vals, alternative=alternative)
                    perm_p_values[i, j, k] = p_val
        except:
            pass
    
    # Find clusters in permuted data
    perm_mask = (perm_p_values < cluster_threshold) & valid_mask
    perm_labeled, perm_n_clusters = label(perm_mask)
    
    # Return maximum cluster size
    if perm_n_clusters > 0:
        perm_cluster_sizes = [np.sum(perm_labeled == cid) 
                             for cid in range(1, perm_n_clusters + 1)]
        return max(perm_cluster_sizes)
    else:
        return 0


def cluster_based_correction(responders, non_responders, p_values, valid_mask, 
                            cluster_threshold=0.01, n_permutations=500, alpha=0.05,
                            test_type='unpaired', alternative='two-sided', n_jobs=-1, verbose=True):
    """
    Apply cluster-based permutation correction for multiple comparisons
    
    This implements the cluster-mass approach commonly used in neuroimaging.
    Tests all valid voxels in permutations and uses parallel processing for speed.
    
    Parameters:
    -----------
    responders : ndarray (x, y, z, n_subjects)
        Responder data
    non_responders : ndarray (x, y, z, n_subjects)
        Non-responder data
    p_values : ndarray (x, y, z)
        Uncorrected p-values from initial test
    valid_mask : ndarray (x, y, z)
        Boolean mask of valid voxels
    cluster_threshold : float
        Initial p-value threshold for cluster formation (uncorrected)
    n_permutations : int
        Number of permutations for null distribution (500-1000 recommended)
    alpha : float
        Significance level for cluster-level correction
    test_type : str
        Either 'paired' or 'unpaired' t-test for permutations
    alternative : {'two-sided', 'greater', 'less'}, optional
        Alternative hypothesis (default: 'two-sided')
    n_jobs : int
        Number of parallel jobs. -1 uses all available CPU cores. 1 disables parallelization.
    verbose : bool
        Print progress information
    
    Returns:
    --------
    sig_mask : ndarray (x, y, z)
        Binary mask of significant voxels
    cluster_size_threshold : float
        Cluster size threshold from permutation distribution
    sig_clusters : list of dict
        Information about significant clusters
    null_max_cluster_sizes : ndarray
        Maximum cluster sizes from permutation null distribution
    cluster_sizes : list of dict
        All clusters from observed data (for plotting)
    """
    if verbose:
        print(f"\n{'='*70}")
        print("CLUSTER-BASED PERMUTATION CORRECTION")
        print(f"{'='*70}")
        print(f"Cluster-forming threshold: p < {cluster_threshold}")
        print(f"Number of permutations: {n_permutations}")
        print(f"Cluster-level alpha: {alpha}")
    
    # Step 1: Form clusters based on initial threshold
    initial_mask = (p_values < cluster_threshold) & valid_mask
    labeled_array, n_clusters = label(initial_mask)
    
    if verbose:
        print(f"\nFound {n_clusters} clusters at p < {cluster_threshold} (uncorrected)")
    
    if n_clusters == 0:
        if verbose:
            print("No clusters found. Try increasing cluster_threshold (e.g., 0.05)")
        return np.zeros_like(p_values, dtype=int), cluster_threshold, []
    
    # Calculate cluster sizes
    cluster_sizes = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = (labeled_array == cluster_id)
        size = np.sum(cluster_mask)
        cluster_sizes.append({'id': cluster_id, 'size': size, 'mask': cluster_mask})
        if verbose:
            print(f"  Cluster {cluster_id}: {size} voxels")
    
    cluster_sizes.sort(key=lambda x: x['size'], reverse=True)
    max_cluster_size = cluster_sizes[0]['size']
    
    if verbose:
        print(f"\nLargest cluster: {max_cluster_size} voxels")
    
    # Step 2: Test all valid voxels in permutations
    test_mask = valid_mask
    test_coords = np.argwhere(test_mask)
    n_test_voxels = len(test_coords)
    
    if verbose:
        print(f"\nTesting all {n_test_voxels} valid voxels in permutations")
    
    # Step 3: Permutation testing
    if verbose:
        print(f"\nRunning {n_permutations} permutations...")
    
    # Combine all subjects
    all_data = np.concatenate([responders, non_responders], axis=-1)
    n_resp = responders.shape[-1]
    n_non_resp = non_responders.shape[-1]
    n_total = n_resp + n_non_resp
    
    # Pre-extract data for test voxels
    if verbose:
        print("Pre-extracting voxel data for faster permutations...")
    test_data = np.zeros((n_test_voxels, n_total))
    for idx, coord in enumerate(test_coords):
        i, j, k = coord[0], coord[1], coord[2]
        test_data[idx, :] = all_data[i, j, k, :]
    
    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    if verbose:
        if n_jobs == 1:
            print("Running permutations sequentially (1 core)...")
        else:
            print(f"Running permutations in parallel using {n_jobs} cores...")
    
    # Run permutations in parallel
    # Use seeds for reproducibility
    seeds = np.random.randint(0, 2**31, size=n_permutations)
    
    if n_jobs == 1:
        # Sequential execution with progress bar
        null_max_cluster_sizes = []
        iterator = tqdm(range(n_permutations), desc="Permutations") if verbose else range(n_permutations)
        for perm in iterator:
            max_size = _run_single_permutation(
                test_data, test_coords, n_resp, n_total, 
                cluster_threshold, valid_mask, p_values.shape, 
                test_type=test_type,
                alternative=alternative,
                seed=seeds[perm]
            )
            null_max_cluster_sizes.append(max_size)
    else:
        # Parallel execution
        null_max_cluster_sizes = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_run_single_permutation)(
                test_data, test_coords, n_resp, n_total,
                cluster_threshold, valid_mask, p_values.shape,
                test_type=test_type,
                alternative=alternative,
                seed=seeds[perm]
            ) for perm in range(n_permutations)
        )
    
    # Step 4: Determine cluster size threshold
    null_max_cluster_sizes = np.array(null_max_cluster_sizes)
    cluster_size_threshold = np.percentile(null_max_cluster_sizes, 100 * (1 - alpha))
    
    if verbose:
        print(f"\n{100*(1-alpha)}th percentile of null distribution: {cluster_size_threshold:.1f} voxels")
        print(f"Null distribution stats: min={np.min(null_max_cluster_sizes):.0f}, "
              f"mean={np.mean(null_max_cluster_sizes):.1f}, "
              f"max={np.max(null_max_cluster_sizes):.0f}")
    
    # Step 5: Identify significant clusters
    sig_mask = np.zeros_like(p_values, dtype=int)
    sig_clusters = []
    
    for cluster_info in cluster_sizes:
        if cluster_info['size'] > cluster_size_threshold:
            sig_mask[cluster_info['mask']] = 1
            sig_clusters.append(cluster_info)
            if verbose:
                print(f"  ✓ Cluster {cluster_info['id']} is SIGNIFICANT "
                      f"({cluster_info['size']} voxels > {cluster_size_threshold:.1f})")
        else:
            if verbose:
                print(f"  ✗ Cluster {cluster_info['id']} is NOT significant "
                      f"({cluster_info['size']} voxels < {cluster_size_threshold:.1f})")
    
    if verbose:
        print(f"\nNumber of significant clusters: {len(sig_clusters)}")
        print(f"Total significant voxels: {np.sum(sig_mask)}")
    
    return sig_mask, cluster_size_threshold, sig_clusters, null_max_cluster_sizes, cluster_sizes


# ==============================================================================
# CLUSTER ANALYSIS
# ==============================================================================

def cluster_analysis(sig_mask, affine, verbose=True):
    """
    Perform cluster analysis on significant voxels
    
    Parameters:
    -----------
    sig_mask : ndarray (x, y, z)
        Binary mask of significant voxels
    affine : ndarray
        Affine transformation matrix
    verbose : bool
        Print progress information
    
    Returns:
    --------
    clusters : list of dict
        Cluster information including size, center of mass in voxel and MNI coordinates
    """
    if verbose:
        print("\nPerforming cluster analysis...")
    
    # Find connected clusters
    labeled_array, num_clusters = label(sig_mask)
    
    if num_clusters == 0:
        if verbose:
            print("No clusters found")
        return []
    
    clusters = []
    for cluster_id in range(1, num_clusters + 1):
        cluster_mask = (labeled_array == cluster_id)
        cluster_size = np.sum(cluster_mask)
        
        # Get center of mass in voxel coordinates
        coords = np.argwhere(cluster_mask)
        com_voxel = np.mean(coords, axis=0)
        
        # Convert to MNI coordinates
        com_mni = nib.affines.apply_affine(affine, com_voxel)
        
        clusters.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'center_voxel': com_voxel,
            'center_mni': com_mni
        })
    
    # Sort by size
    clusters = sorted(clusters, key=lambda x: x['size'], reverse=True)
    
    if verbose:
        print(f"Found {num_clusters} clusters")
        for c in clusters[:10]:  # Show top 10
            print(f"  Cluster {c['cluster_id']}: {c['size']} voxels, "
                  f"MNI center: ({c['center_mni'][0]:.1f}, {c['center_mni'][1]:.1f}, {c['center_mni'][2]:.1f})")
    
    return clusters


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_permutation_null_distribution(null_distribution, threshold, observed_clusters, 
                                       output_file, alpha=0.05):
    """
    Plot permutation null distribution with threshold and observed clusters
    
    Parameters:
    -----------
    null_distribution : ndarray
        Maximum cluster sizes from permutation null distribution
    threshold : float
        Cluster size threshold at given alpha level
    observed_clusters : list of dict
        List of observed cluster information (with 'size' key)
    output_file : str
        Path to save PDF file
    alpha : float
        Significance level used
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of null distribution
    plt.hist(null_distribution, bins=50, alpha=0.7, color='gray', 
             edgecolor='black', label='Null Distribution')
    
    # Plot threshold line
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold (α={alpha}): {threshold:.1f} voxels')
    
    # Plot observed cluster sizes
    for i, cluster in enumerate(observed_clusters):
        size = cluster['size']
        is_significant = size > threshold
        color = 'green' if is_significant else 'orange'
        label = 'Significant Cluster' if i == 0 and is_significant else None
        if i == 0 and not is_significant:
            label = 'Non-significant Cluster'
        
        plt.axvline(size, color=color, linestyle='-', linewidth=1.5, 
                   alpha=0.8, label=label)
    
    plt.xlabel('Maximum Cluster Size (voxels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Permutation Null Distribution of Maximum Cluster Sizes', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved permutation null distribution plot: {output_file}")


# ==============================================================================
# ATLAS PROCESSING
# ==============================================================================

def check_and_resample_atlas(atlas_img, reference_img, atlas_name, verbose=True):
    """
    Check if atlas dimensions match reference, resample if needed
    
    Parameters:
    -----------
    atlas_img : nibabel image
        Atlas to check/resample
    reference_img : nibabel image
        Reference image (from subject data)
    atlas_name : str
        Name of atlas for logging
    verbose : bool
        Print information
    
    Returns:
    --------
    atlas_data : ndarray
        Atlas data in correct dimensions
    """
    atlas_shape = atlas_img.shape
    ref_shape = reference_img.shape
    
    if verbose:
        print(f"  Atlas shape: {atlas_shape}")
        print(f"  Reference shape: {ref_shape[:3]}")
    
    # Check if dimensions match (only compare spatial dimensions)
    if atlas_shape[:3] != ref_shape[:3]:
        if verbose:
            print(f"  ⚠ Dimensions don't match! Resampling atlas...")
        
        try:
            # Create a clean 3D reference image for resampling
            # Extract just the first 3D volume if reference is 4D
            if len(ref_shape) > 3:
                ref_data_3d = reference_img.get_fdata()[:, :, :, 0]
            else:
                ref_data_3d = reference_img.get_fdata()
            
            # Create a new 3D reference image with standard 4x4 affine
            ref_img_3d = nib.Nifti1Image(
                ref_data_3d.astype(np.float32),
                reference_img.affine[:4, :4],  # Ensure 4x4 affine matrix
                None
            )
            
            # Ensure atlas is also 3D with 4x4 affine
            atlas_data_raw = atlas_img.get_fdata()
            if len(atlas_data_raw.shape) > 3:
                atlas_data_raw = atlas_data_raw[:, :, :, 0]
            
            atlas_img_3d = nib.Nifti1Image(
                atlas_data_raw.astype(np.float32),
                atlas_img.affine[:4, :4],  # Ensure 4x4 affine matrix
                None
            )
            
            # Resample atlas to match reference image
            resampled_atlas = resample_from_to(
                atlas_img_3d, 
                ref_img_3d, 
                order=0  # Use nearest neighbor for label data
            )
            
            atlas_data = resampled_atlas.get_fdata().astype(int)
            if verbose:
                print(f"  ✓ Resampled to: {atlas_data.shape}")
                
        except Exception as e:
            if verbose:
                print(f"  ✗ Resampling failed: {e}")
                print(f"  Attempting alternative resampling method...")
            
            # Fallback: use scipy for resampling
            from scipy.ndimage import zoom
            
            atlas_data_raw = atlas_img.get_fdata()
            if len(atlas_data_raw.shape) > 3:
                atlas_data_raw = atlas_data_raw[:, :, :, 0]
            
            # Calculate zoom factors
            zoom_factors = [
                ref_shape[i] / atlas_shape[i] for i in range(3)
            ]
            
            # Resample using nearest neighbor
            atlas_data = zoom(atlas_data_raw, zoom_factors, order=0).astype(int)
            
            if verbose:
                print(f"  ✓ Resampled to: {atlas_data.shape}")
    else:
        if verbose:
            print(f"  ✓ Dimensions match!")
        atlas_data = atlas_img.get_fdata().astype(int)
        
        # Ensure 3D
        if len(atlas_data.shape) > 3:
            atlas_data = atlas_data[:, :, :, 0]
    
    return atlas_data


def atlas_overlap_analysis(sig_mask, atlas_files, data_dir, reference_img=None, verbose=True):
    """
    Analyze overlap between significant voxels and atlas regions
    
    Parameters:
    -----------
    sig_mask : ndarray (x, y, z)
        Binary mask of significant voxels
    atlas_files : list of str
        List of atlas file names
    data_dir : str
        Directory containing atlas files
    reference_img : nibabel image, optional
        Reference image for resampling
    verbose : bool
        Print progress information
    
    Returns:
    --------
    results : dict
        Dictionary mapping atlas names to DataFrames of region overlap statistics
    """
    if verbose:
        print("\n" + "="*60)
        print("ATLAS OVERLAP ANALYSIS")
        print("="*60)
    
    results = {}
    
    for atlas_file in atlas_files:
        atlas_path = os.path.join(data_dir, atlas_file)
        if not os.path.exists(atlas_path):
            if verbose:
                print(f"Warning: Atlas file not found - {atlas_file}")
            continue
        
        if verbose:
            print(f"\n--- {atlas_file} ---")
        atlas_img = nib.load(atlas_path)
        
        # Check dimensions and resample if needed
        if reference_img is not None:
            atlas_data = check_and_resample_atlas(atlas_img, reference_img, atlas_file, verbose)
        else:
            atlas_data = atlas_img.get_fdata().astype(int)
        
        # Get unique regions (excluding 0 = background)
        regions = np.unique(atlas_data[atlas_data > 0])
        
        region_counts = []
        for region_id in regions:
            region_mask = (atlas_data == region_id)
            overlap = np.sum(sig_mask & region_mask)
            
            if overlap > 0:
                region_counts.append({
                    'region_id': int(region_id),
                    'overlap_voxels': int(overlap),
                    'region_size': int(np.sum(region_mask))
                })
        
        # Sort by overlap count
        region_counts = sorted(region_counts, key=lambda x: x['overlap_voxels'], reverse=True)
        
        if verbose:
            print(f"\nTop regions by significant voxel count:")
            for i, r in enumerate(region_counts[:15], 1):
                pct = 100 * r['overlap_voxels'] / r['region_size']
                print(f"{i:2d}. Region {r['region_id']:3d}: {r['overlap_voxels']:4d} sig. voxels "
                      f"({pct:.1f}% of region)")
        
        results[atlas_file] = region_counts
    
    return results


# ==============================================================================
# SUMMARY GENERATION
# ==============================================================================

def generate_summary(responders, non_responders, sig_mask, correction_threshold, 
                    clusters, atlas_results, output_file, 
                    correction_method="cluster",
                    params=None,
                    group1_name="Responders",
                    group2_name="Non-Responders",
                    value_metric="Current Intensity",
                    test_type="unpaired"):
    """
    Generate comprehensive summary report
    
    Parameters:
    -----------
    responders : ndarray
        Responder data (group 1)
    non_responders : ndarray
        Non-responder data (group 2)
    sig_mask : ndarray
        Binary mask of significant voxels
    correction_threshold : float
        Threshold used for multiple comparison correction
    clusters : list
        List of cluster dictionaries
    atlas_results : dict
        Atlas overlap results
    output_file : str
        Path to output summary file
    correction_method : str
        Method used: 'cluster' or 'fdr'
    params : dict, optional
        Dictionary of analysis parameters (cluster_threshold, n_permutations, alpha, etc.)
        If None, uses defaults
    group1_name : str
        Name for first group (default: "Responders")
    group2_name : str
        Name for second group (default: "Non-Responders")
    value_metric : str
        Name of the metric being compared (default: "Current Intensity")
    test_type : str
        Type of t-test used: 'paired' or 'unpaired' (default: "unpaired")
    """
    # Set default parameters if not provided
    if params is None:
        params = {
            'cluster_threshold': 0.01,
            'n_permutations': 500,
            'alpha': 0.05
        }
    
    # Extract parameters with defaults
    cluster_threshold_param = params.get('cluster_threshold', 0.01)
    n_permutations = params.get('n_permutations', 500)
    alpha = params.get('alpha', 0.05)
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VOXELWISE STATISTICAL ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("ANALYSIS DETAILS:\n")
        f.write("-" * 70 + "\n")
        test_name = "Paired t-test" if test_type == "paired" else "Unpaired (Independent Samples) t-test"
        f.write(f"Statistical Test: {test_name}\n")
        
        if correction_method == "cluster":
            f.write(f"Multiple Comparison Correction: Cluster-based Permutation\n")
            f.write(f"Cluster-forming threshold: p < {cluster_threshold_param} (uncorrected)\n")
            f.write(f"Number of permutations: {n_permutations}\n")
            f.write(f"Cluster-level alpha: {alpha}\n")
            f.write(f"Cluster size threshold: {correction_threshold:.1f} voxels\n")
            if 'n_jobs' in params:
                n_jobs = params['n_jobs']
                if n_jobs == -1:
                    import multiprocessing
                    n_jobs_actual = multiprocessing.cpu_count()
                    f.write(f"Parallel processing: {n_jobs_actual} cores\n")
                elif n_jobs == 1:
                    f.write(f"Parallel processing: Sequential (1 core)\n")
                else:
                    f.write(f"Parallel processing: {n_jobs} cores\n")
            f.write("\n")
        else:
            f.write(f"Multiple Comparison Correction: FDR (False Discovery Rate)\n")
            f.write(f"Significance Level: alpha = {alpha}\n")
            f.write(f"FDR-corrected p-value threshold: {correction_threshold:.6f}\n\n")
        
        f.write("SAMPLE INFORMATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of {group1_name}: {responders.shape[-1]}\n")
        f.write(f"Number of {group2_name}: {non_responders.shape[-1]}\n")
        f.write(f"Total Subjects: {responders.shape[-1] + non_responders.shape[-1]}\n\n")
        
        f.write("RESULTS:\n")
        f.write("-" * 70 + "\n")
        n_sig = np.sum(sig_mask)
        f.write(f"Number of Significant Voxels: {n_sig}\n")
        
        if n_sig > 0:
            # Calculate mean values in significant voxels
            sig_bool = sig_mask.astype(bool)
            group1_mean = np.mean(responders[sig_bool, :])
            group2_mean = np.mean(non_responders[sig_bool, :])
            
            f.write(f"\nMean {value_metric} in Significant Voxels:\n")
            f.write(f"  {group1_name}: {group1_mean:.4f}\n")
            f.write(f"  {group2_name}: {group2_mean:.4f}\n")
            f.write(f"  Difference ({group1_name} - {group2_name}): {group1_mean - group2_mean:.4f}\n")
        
        f.write(f"\nNumber of Clusters: {len(clusters)}\n\n")
        
        if clusters:
            f.write("TOP 10 CLUSTERS (by size):\n")
            f.write("-" * 70 + "\n")
            for i, c in enumerate(clusters[:10], 1):
                f.write(f"{i}. Cluster {c['cluster_id']}: {c['size']} voxels\n")
                f.write(f"   MNI Center: ({c['center_mni'][0]:.1f}, "
                       f"{c['center_mni'][1]:.1f}, {c['center_mni'][2]:.1f})\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("ATLAS OVERLAP ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        for atlas_name, region_counts in atlas_results.items():
            f.write(f"\n{atlas_name}\n")
            f.write("-" * 70 + "\n")
            
            if region_counts:
                f.write(f"Number of regions with significant voxels: {len(region_counts)}\n\n")
                f.write("Top 20 regions:\n")
                for i, r in enumerate(region_counts[:20], 1):
                    pct = 100 * r['overlap_voxels'] / r['region_size']
                    f.write(f"{i:2d}. Region {r['region_id']:3d}: "
                           f"{r['overlap_voxels']:4d} voxels ({pct:5.1f}% of region)\n")
            else:
                f.write("No overlapping regions found.\n")
    
    print(f"\nSummary written to: {output_file}")

