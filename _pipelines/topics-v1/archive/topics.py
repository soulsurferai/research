#!/usr/bin/env python3
"""
topics.py - Fetch Reddit data from Qdrant and perform topic analysis using BERTopic
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env', '.env'))

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
OUTPUT_DIR = 'quick_results'

# Cannabis-specific stopwords
CANNABIS_STOPWORDS = [
    'deleted', 'removed', 'bot', 'moderator', 'automod',
    'lol', 'lmao', 'edit', 'update', 'reddit', 'subreddit',
    'http', 'https', 'www', 'com'
]


class TopicAnalyzer:
    """Analyze topics in Reddit communities using Qdrant embeddings"""
    
    def __init__(self):
        """Initialize Qdrant client and BERTopic model"""
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.setup_bertopic()
        
    def setup_bertopic(self):
        """Configure BERTopic for memory efficiency"""
        # Reduce dimensionality aggressively for Mac
        self.umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric='euclidean',
            random_state=42,
            low_memory=True
        )
        
        # Configure HDBSCAN
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=15,
            prediction_data=False,
            core_dist_n_jobs=1
        )
        
        # Vectorizer with custom stopwords
        self.vectorizer_model = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Remove cannabis stopwords
        stop_words = list(self.vectorizer_model.get_stop_words())
        stop_words.extend(CANNABIS_STOPWORDS)
        self.vectorizer_model.stop_words = stop_words
        
    def fetch_subreddit_data(self, subreddit: str, limit: int = 500) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
        """
        Fetch posts and comments from Qdrant for a subreddit
        
        Returns:
            embeddings: List of embedding vectors
            texts: List of text content
            metadata: List of metadata dicts
        """
        print(f"\nFetching data for r/{subreddit}...")
        
        embeddings = []
        texts = []
        metadata = []
        
        # Fetch from both collections
        for collection in ['reddit_posts', 'reddit_comments']:
            try:
                # Query with filter
                results = self.client.scroll(
                    collection_name=collection,
                    scroll_filter={
                        "must": [
                            {"key": "subreddit", "match": {"value": subreddit}}
                        ]
                    },
                    limit=limit // 2,  # Split limit between posts and comments
                    with_payload=True,
                    with_vectors=True
                )
                
                # Process results
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
                    
                    # Skip if text is too short
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
        """Run topic analysis on a subreddit"""
        
        # Fetch data
        embeddings, texts, metadata = self.fetch_subreddit_data(subreddit, n_samples)
        
        if len(texts) < 50:
            print(f"Not enough data for r/{subreddit}")
            return pd.DataFrame()
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=None,  # Use pre-computed embeddings
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            calculate_probabilities=False,
            verbose=False
        )
        
        print(f"Running topic modeling for r/{subreddit}...")
        topics, _ = topic_model.fit_transform(texts, embeddings_array)
        
        # Get topic info
        topic_info = topic_model.get_topic_info()
        
        # Add subreddit info
        topic_info['subreddit'] = subreddit
        
        # Save model
        output_path = os.path.join(OUTPUT_DIR, f'{subreddit}_topics.csv')
        topic_info.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
        
        # Print summary
        n_topics = len(topic_info) - 1  # Exclude -1 topic
        print(f"Found {n_topics} topics in r/{subreddit}")
        
        return topic_info


def main():
    """Run topic analysis on specified subreddits"""
    subreddits = ['cannabis', 'weed', 'trees', 'marijuana', 'weedstocks', 'weedbiz']
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize analyzer
    analyzer = TopicAnalyzer()
    
    # Track results
    results = {}
    
    # Analyze each subreddit
    for subreddit in subreddits:
        try:
            topic_info = analyzer.analyze_topics(subreddit, n_samples=400)
            results[subreddit] = {
                'n_topics': len(topic_info) - 1,
                'n_docs': topic_info['Count'].sum()
            }
        except Exception as e:
            print(f"Error analyzing r/{subreddit}: {e}")
            results[subreddit] = {'error': str(e)}
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
