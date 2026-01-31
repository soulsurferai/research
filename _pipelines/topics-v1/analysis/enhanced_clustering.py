"""
enhanced_clustering.py - Advanced clustering algorithms for better topic discovery
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import hdbscan
from typing import Tuple, List, Dict, Optional
import umap
import sys
sys.path.append('..')
from config import PCA_COMPONENTS, MIN_CLUSTER_SIZE, MIN_SAMPLES


def reduce_dimensions_umap(embeddings: np.ndarray, 
                          n_components: int = 50,
                          n_neighbors: int = 15,
                          min_dist: float = 0.1) -> np.ndarray:
    """
    Reduce dimensions using UMAP (better for preserving local structure)
    
    Args:
        embeddings: Array of embeddings
        n_components: Target dimensions
        n_neighbors: Balance local vs global structure
        min_dist: How tightly points are packed
        
    Returns:
        Reduced embeddings
    """
    print(f"Reducing dimensions with UMAP from {embeddings.shape[1]} to {n_components}...")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    
    embeddings_reduced = reducer.fit_transform(embeddings)
    return embeddings_reduced


def hierarchical_clustering(embeddings: np.ndarray,
                          min_topics: int = 3,
                          max_topics: int = 10) -> Dict[str, np.ndarray]:
    """
    Perform hierarchical clustering to get topics at different granularities
    
    Args:
        embeddings: Reduced embeddings
        min_topics: Minimum number of topics
        max_topics: Maximum number of topics
        
    Returns:
        Dictionary of granularity_level -> cluster labels
    """
    from sklearn.cluster import AgglomerativeClustering
    
    results = {}
    
    for n_clusters in range(min_topics, max_topics + 1):
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        labels = clusterer.fit_predict(embeddings)
        results[f'{n_clusters}_topics'] = labels
        
    return results


def smart_hdbscan(embeddings: np.ndarray,
                 min_cluster_size_ratio: float = 0.02,
                 max_cluster_size_ratio: float = 0.3) -> Tuple[np.ndarray, Dict]:
    """
    HDBSCAN with smart parameter selection based on data size
    
    Args:
        embeddings: Reduced embeddings
        min_cluster_size_ratio: Min cluster size as ratio of total docs
        max_cluster_size_ratio: Max cluster size as ratio of total docs
        
    Returns:
        Cluster labels and clustering metrics
    """
    n_samples = len(embeddings)
    
    # Dynamic parameter calculation
    min_cluster_size = max(5, int(n_samples * min_cluster_size_ratio))
    min_samples = max(3, min_cluster_size // 3)
    
    print(f"Smart HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.1,  # Allow some flexibility
        prediction_data=True
    )
    
    labels = clusterer.fit_predict(embeddings)
    
    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Check for mega-clusters
    cluster_sizes = {}
    for label in set(labels):
        if label != -1:
            size = list(labels).count(label)
            cluster_sizes[label] = size
            if size > n_samples * max_cluster_size_ratio:
                print(f"Warning: Cluster {label} is too large ({size}/{n_samples} docs)")
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / n_samples,
        'cluster_sizes': cluster_sizes,
        'probabilities': clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
    }
    
    return labels, metrics


def ensemble_clustering(embeddings: np.ndarray,
                       methods: List[str] = ['hdbscan', 'kmeans', 'hierarchical']) -> np.ndarray:
    """
    Combine multiple clustering methods for robustness
    
    Args:
        embeddings: Reduced embeddings
        methods: List of clustering methods to use
        
    Returns:
        Consensus cluster labels
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    
    n_samples = len(embeddings)
    all_labels = []
    
    # Get labels from each method
    if 'hdbscan' in methods:
        labels, _ = smart_hdbscan(embeddings)
        all_labels.append(labels)
    
    if 'kmeans' in methods:
        # Try different k values
        for k in [5, 7, 10]:
            if k < n_samples:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                all_labels.append(labels)
    
    if 'hierarchical' in methods:
        for n in [5, 7]:
            if n < n_samples:
                agg = AgglomerativeClustering(n_clusters=n)
                labels = agg.fit_predict(embeddings)
                all_labels.append(labels)
    
    # Find consensus using majority voting
    consensus_labels = np.zeros(n_samples, dtype=int)
    
    # Simple implementation: use the clustering with median number of clusters
    n_clusters_list = [len(set(labels)) - (1 if -1 in labels else 0) for labels in all_labels]
    median_idx = np.argsort(n_clusters_list)[len(n_clusters_list)//2]
    
    return all_labels[median_idx]


def subclustering(embeddings: np.ndarray, 
                 labels: np.ndarray,
                 large_cluster_threshold: float = 0.3) -> np.ndarray:
    """
    Break up large clusters into subclusters
    
    Args:
        embeddings: Reduced embeddings
        labels: Initial cluster labels
        large_cluster_threshold: Ratio threshold for "too large" clusters
        
    Returns:
        Refined cluster labels
    """
    n_samples = len(embeddings)
    refined_labels = labels.copy()
    max_label = labels.max()
    
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
            
        cluster_mask = labels == cluster_id
        cluster_size = cluster_mask.sum()
        
        # If cluster is too large, subcluster it
        if cluster_size > n_samples * large_cluster_threshold:
            print(f"Subclustering cluster {cluster_id} ({cluster_size} docs)")
            
            cluster_embeddings = embeddings[cluster_mask]
            
            # Use KMeans for subclustering
            n_subclusters = min(5, cluster_size // 20)
            if n_subclusters > 1:
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                sublabels = kmeans.fit_predict(cluster_embeddings)
                
                # Assign new labels
                cluster_indices = np.where(cluster_mask)[0]
                for i, sublabel in enumerate(sublabels):
                    if sublabel > 0:  # Keep first subcluster with original label
                        refined_labels[cluster_indices[i]] = max_label + sublabel
                
                max_label += n_subclusters - 1
    
    return refined_labels
