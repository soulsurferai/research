#!/usr/bin/env python3
"""
topics_modular.py - Modular topic analysis using refactored components
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any

from config import (
    CANNABIS_SUBREDDITS, OUTPUT_DIR, DEFAULT_SAMPLE_SIZE,
    MIN_CLUSTER_SIZE, MIN_SAMPLES
)
from utils import QdrantFetcher, save_json, filter_boilerplate
from analysis import (
    reduce_dimensions, adaptive_clustering, 
    extract_topics_c_tfidf, create_topic_summary,
    get_topic_diversity
)


class ModularTopicAnalyzer:
    """Modular topic analyzer using refactored components"""
    
    def __init__(self):
        """Initialize with Qdrant fetcher"""
        self.fetcher = QdrantFetcher()
        
    def analyze_subreddit(self, subreddit: str, n_samples: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
        """
        Analyze topics in a single subreddit
        
        Args:
            subreddit: Name of subreddit
            n_samples: Number of samples to analyze
            
        Returns:
            DataFrame of topics
        """
        print(f"\n{'='*50}")
        print(f"Analyzing r/{subreddit}")
        print(f"{'='*50}")
        
        # 1. Fetch data
        embeddings, texts, metadata = self.fetcher.fetch_subreddit_data(subreddit, n_samples)
        
        if len(texts) < MIN_CLUSTER_SIZE * 2:
            print(f"⚠️  Not enough data for r/{subreddit} ({len(texts)} docs)")
            return pd.DataFrame()
        
        # 2. Filter boilerplate
        texts, metadata = filter_boilerplate(texts, metadata)
        embeddings = [embeddings[i] for i in range(len(embeddings)) if i < len(texts)]
        
        print(f"After filtering: {len(texts)} documents")
        
        if len(texts) < MIN_CLUSTER_SIZE * 2:
            print(f"⚠️  Not enough data after filtering")
            return pd.DataFrame()
        
        # 3. Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # 4. Reduce dimensions
        embeddings_reduced = reduce_dimensions(embeddings_array)
        
        # 5. Cluster with adaptive parameters
        cluster_labels = adaptive_clustering(embeddings_reduced, target_clusters=5)
        
        # 6. Extract topics
        topics = extract_topics_c_tfidf(texts, cluster_labels)
        
        if not topics:
            print(f"⚠️  No topics found for r/{subreddit}")
            return pd.DataFrame()
        
        # 7. Calculate diversity
        diversity = get_topic_diversity(topics)
        print(f"Topic diversity score: {diversity:.2f}")
        
        # 8. Create summary
        topic_df = create_topic_summary(topics, cluster_labels, subreddit)
        
        # 9. Save results
        output_path = os.path.join(OUTPUT_DIR, f'{subreddit}_topics_v2.csv')
        topic_df.to_csv(output_path, index=False)
        print(f"✅ Saved results to {output_path}")
        
        return topic_df


def main():
    """Run modular topic analysis on all subreddits"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ModularTopicAnalyzer()
    
    # Track results
    results = {}
    all_topics = []
    
    # Analyze each subreddit
    for subreddit in CANNABIS_SUBREDDITS:
        try:
            topic_df = analyzer.analyze_subreddit(subreddit)
            
            if not topic_df.empty:
                results[subreddit] = {
                    'n_topics': len(topic_df),
                    'n_docs': int(topic_df['Count'].sum()),
                    'topics': topic_df['Representation'].tolist()[:3]  # Top 3 topics
                }
                all_topics.append(topic_df)
            else:
                results[subreddit] = {
                    'n_topics': 0,
                    'n_docs': 0,
                    'error': 'Insufficient data or no topics found'
                }
                
        except Exception as e:
            print(f"❌ Error analyzing r/{subreddit}: {e}")
            import traceback
            traceback.print_exc()
            results[subreddit] = {'error': str(e)}
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'analysis_summary_v2.json')
    save_json(results, summary_path)
    
    # Create combined topics file
    if all_topics:
        combined_df = pd.concat(all_topics, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, 'all_topics_v2.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f"\n✅ Saved combined topics to {combined_path}")
    
    print(f"\n✅ Analysis complete! Summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for sub, info in results.items():
        if 'error' not in info:
            print(f"r/{sub}: {info['n_topics']} topics, {info['n_docs']} docs")
        else:
            print(f"r/{sub}: ERROR - {info.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
