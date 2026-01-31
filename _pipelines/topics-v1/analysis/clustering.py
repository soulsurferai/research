"""
clustering.py - Clustering algorithms for topic discovery
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from typing import Tuple, List
import sys
sys.path.append('..')
from config import PCA_COMPONENTS, MIN_CLUSTER_SIZE, MIN_SAMPLES


def reduce_dimensions(embeddings: np.ndarray, n_components: int = PCA_COMPONENTS) -> np.ndarray:
    """
    Reduce embedding dimensions using PCA
    
    Args:
        embeddings: Array of embeddings
        n_components: Number of components to reduce to
        
    Returns:
        Reduced embeddings
    """
    print(f"Reducing dimensions from {embeddings.shape[1]} to {n_components}...")
    
    # Standardize the features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")
    
    return embeddings_reduced


def cluster_embeddings(embeddings: np.ndarray, 
                      min_cluster_size: int = MIN_CLUSTER_SIZE,
                      min_samples: int = MIN_SAMPLES) -> Tuple[np.ndarray, float]:
    """
    Cluster embeddings using HDBSCAN
    
    Args:
        embeddings: Array of (reduced) embeddings
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples for core points
        
    Returns:
        cluster_labels: Array of cluster labels (-1 for noise)
        noise_ratio: Proportion of points classified as noise
    """
    print(f"Clustering {len(embeddings)} documents...")
    
    # Configure HDBSCAN for better topic discovery
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',  # Excess of Mass
        prediction_data=True
    )
    
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Calculate statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_ratio = n_noise / len(cluster_labels)
    
    print(f"Found {n_clusters} clusters ({n_noise} noise points, {noise_ratio:.1%})")
    
    # Get cluster persistence (stability)
    if hasattr(clusterer, 'cluster_persistence_'):
        persistence = clusterer.cluster_persistence_
        print(f"Cluster persistence: {persistence}")
    
    return cluster_labels, noise_ratio


def adaptive_clustering(embeddings: np.ndarray, target_clusters: int = 5) -> np.ndarray:
    """
    Try different clustering parameters to get reasonable number of clusters
    
    Args:
        embeddings: Array of embeddings
        target_clusters: Desired number of clusters (approximate)
        
    Returns:
        Best cluster labels found
    """
    best_labels = None
    best_score = float('inf')
    
    # Try different parameter combinations
    for min_cluster in [5, 10, 15, 20]:
        for min_samp in [1, 3, 5]:
            labels, noise_ratio = cluster_embeddings(
                embeddings, 
                min_cluster_size=min_cluster,
                min_samples=min_samp
            )
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Score based on distance from target and noise ratio
            score = abs(n_clusters - target_clusters) + noise_ratio * 10
            
            if score < best_score and n_clusters > 0:
                best_score = score
                best_labels = labels
                print(f"  Better parameters: min_cluster={min_cluster}, min_samples={min_samp}")
    
    return best_labels if best_labels is not None else np.zeros(len(embeddings))


def get_cluster_centers(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calculate cluster centers for visualization or further analysis
    
    Args:
        embeddings: Array of embeddings
        labels: Cluster labels
        
    Returns:
        Dictionary of cluster_id -> center coordinates
    """
    centers = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    
    for label in unique_labels:
        mask = labels == label
        centers[label] = embeddings[mask].mean(axis=0)
    
    return centers
