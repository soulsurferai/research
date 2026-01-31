#!/usr/bin/env python3
"""
topics_simple.py - Simplified topic analysis without BERTopic
Uses HDBSCAN directly on pre-computed embeddings
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import hdbscan


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env', '.env'))

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
OUTPUT_DIR = 'quick_results'


class SimpleTopicAnalyzer:
    """Simple topic analysis using HDBSCAN directly"""
    
    def __init__(self):
        """Initialize Qdrant client"""
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
    def fetch_subreddit_data(self, subreddit: str, limit: int = 500) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
        """Fetch posts and comments from Qdrant"""
        print(f"\nFetching data for r/{subreddit}...")
        
        embeddings = []
        texts = []
        metadata = []
        
        # Fetch from both collections
        for collection in ['reddit_posts', 'reddit_comments']:
            try:
                results = self.client.scroll(
                    collection_name=collection,
                    scroll_filter={
                        "must": [
                            {"key": "subreddit", "match": {"value": subreddit}}
                        ]
                    },
                    limit=limit // 2,
                    with_payload=True,
                    with_vectors=True
                )
                
                for point in results[0]:
                    # Get embedding
                    if hasattr(point.vector, 'tolist'):
                        embedding = point.vector
                    else:
                        embedding = point.vector['semantic'] if isinstance(point.vector, dict) else point.vector
                    
                    # Get text
                    text = point.payload.get('content', '')
                    if collection == 'reddit_posts' and 'title' in point.payload:
                        text = f"{point.payload['title']} {text}"
                    
                    if len(text) < 50:
                        continue
                    
                    embeddings.append(embedding)
                    texts.append(text)
                    metadata.append({
                        'id': point.id,
                        'score': point.payload.get('score', 0),
                        'type': 'post' if collection == 'reddit_posts' else 'comment'
                    })
                    
            except Exception as e:
                print(f"Error fetching from {collection}: {e}")
                continue
        
        print(f"Fetched {len(texts)} documents from r/{subreddit}")
        return embeddings, texts, metadata
    
    def analyze_topics(self, subreddit: str, n_samples: int = 500) -> pd.DataFrame:
        """Run simple topic analysis"""
        
        # Fetch data
        embeddings, texts, metadata = self.fetch_subreddit_data(subreddit, n_samples)
        
        if len(texts) < 50:
            print(f"Not enough data for r/{subreddit}")
            return pd.DataFrame()
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Reduce dimensions with PCA
        print(f"Reducing dimensions for r/{subreddit}...")
        pca = PCA(n_components=50)
        embeddings_reduced = pca.fit_transform(embeddings_array)
        
        # Cluster with HDBSCAN (relaxed parameters for more clusters)
        print(f"Clustering documents for r/{subreddit}...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3)
        cluster_labels = clusterer.fit_predict(embeddings_reduced)
        
        # Extract topics using TF-IDF
        print(f"Extracting topic words for r/{subreddit}...")
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        
        # Create topic summaries
        topics = []
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise cluster
        
        for label in unique_labels:
            # Get texts in this cluster
            cluster_texts = [texts[i] for i, l in enumerate(cluster_labels) if l == label]
            
            if cluster_texts:
                # Get top words using TF-IDF
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top words
                tfidf_scores = tfidf_matrix.sum(axis=0).A1
                top_indices = tfidf_scores.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                
                topics.append({
                    'Topic': label,
                    'Count': len(cluster_texts),
                    'Name': f"Topic_{label}",
                    'Representation': top_words[:5],
                    'Representative_Docs': cluster_texts[:3]
                })
        
        # Create DataFrame
        topic_df = pd.DataFrame(topics)
        topic_df['subreddit'] = subreddit
        
        # Add noise cluster info
        noise_count = sum(1 for l in cluster_labels if l == -1)
        print(f"Found {len(topics)} topics in r/{subreddit} ({noise_count} noise points)")
        
        # Save results
        output_path = os.path.join(OUTPUT_DIR, f'{subreddit}_topics.csv')
        topic_df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
        
        return topic_df


def main():
    """Run topic analysis on specified subreddits"""
    subreddits = ['cannabis', 'weed', 'trees', 'marijuana', 'weedstocks', 'weedbiz']
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SimpleTopicAnalyzer()
    
    # Track results
    results = {}
    
    # Analyze each subreddit
    for subreddit in subreddits:
        try:
            topic_df = analyzer.analyze_topics(subreddit, n_samples=400)
            
            # Ensure all values are JSON serializable
            if not topic_df.empty:
                n_topics = len(topic_df)
                n_docs = topic_df['Count'].sum()
                
                # Convert numpy types to Python native types
                results[subreddit] = {
                    'n_topics': int(n_topics) if hasattr(n_topics, 'item') else int(n_topics),
                    'n_docs': int(n_docs) if hasattr(n_docs, 'item') else int(n_docs)
                }
            else:
                results[subreddit] = {
                    'n_topics': 0,
                    'n_docs': 0
                }
                
        except Exception as e:
            print(f"Error analyzing r/{subreddit}: {e}")
            import traceback
            traceback.print_exc()
            results[subreddit] = {'error': str(e)}
    
    # Save summary with custom encoder for numpy types
    summary_path = os.path.join(OUTPUT_DIR, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nAnalysis complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
