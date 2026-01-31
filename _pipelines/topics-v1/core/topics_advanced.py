#!/usr/bin/env python3
"""
topics_advanced.py - Advanced topic analysis with multiple improvements
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CANNABIS_SUBREDDITS, OUTPUT_DIR, DEFAULT_SAMPLE_SIZE,
    MIN_CLUSTER_SIZE, MIN_SAMPLES
)
from utils import QdrantFetcher, save_json, filter_boilerplate
from analysis.enhanced_clustering import (
    reduce_dimensions_umap, smart_hdbscan, subclustering
)
from analysis.enhanced_topic_extraction import (
    extract_topics_bm25, extract_topics_mmr, generate_topic_labels,
    calculate_topic_coherence
)


class AdvancedTopicAnalyzer:
    """Advanced topic analyzer with multiple improvements"""
    
    def __init__(self):
        """Initialize with Qdrant fetcher"""
        self.fetcher = QdrantFetcher()
        
    def analyze_subreddit(self, subreddit: str, 
                         n_samples: int = DEFAULT_SAMPLE_SIZE,
                         min_topics: int = 3,
                         max_topics: int = 15) -> Dict[str, Any]:
        """
        Perform advanced topic analysis on a subreddit
        
        Args:
            subreddit: Name of subreddit
            n_samples: Number of samples to analyze
            min_topics: Minimum expected topics
            max_topics: Maximum expected topics
            
        Returns:
            Dictionary with topics and analysis results
        """
        print(f"\n{'='*60}")
        print(f"🔍 Advanced Analysis of r/{subreddit}")
        print(f"{'='*60}")
        
        # 1. Fetch data with higher sample size
        embeddings, texts, metadata = self.fetcher.fetch_subreddit_data(
            subreddit, 
            n_samples * 2  # Fetch more, filter down
        )
        
        if len(texts) < MIN_CLUSTER_SIZE * 3:
            print(f"⚠️  Insufficient data for r/{subreddit} ({len(texts)} docs)")
            return {'error': 'Insufficient data'}
        
        # 2. Enhanced filtering
        texts, metadata = filter_boilerplate(texts, metadata)
        embeddings = [embeddings[i] for i in range(len(embeddings)) if i < len(texts)]
        
        # Additional filtering: remove very short posts
        filtered_data = []
        for emb, txt, meta in zip(embeddings, texts, metadata):
            if len(txt.split()) >= 20:  # At least 20 words
                filtered_data.append((emb, txt, meta))
        
        if len(filtered_data) < MIN_CLUSTER_SIZE * 3:
            print(f"⚠️  Insufficient data after filtering")
            return {'error': 'Insufficient data after filtering'}
        
        embeddings, texts, metadata = zip(*filtered_data)
        embeddings_array = np.array(embeddings)
        
        print(f"📊 Working with {len(texts)} quality documents")
        
        # 3. Better dimensionality reduction
        embeddings_reduced = reduce_dimensions_umap(
            embeddings_array,
            n_components=50,
            n_neighbors=min(15, len(texts) // 10)
        )
        
        # 4. Smart clustering with subclustering
        labels, metrics = smart_hdbscan(
            embeddings_reduced,
            min_cluster_size_ratio=0.02,
            max_cluster_size_ratio=0.25
        )
        
        # Subcluster large clusters
        labels = subclustering(embeddings_reduced, labels, large_cluster_threshold=0.25)
        
        # 5. Multiple topic extraction methods
        topics_bm25 = extract_topics_bm25(texts, labels)
        topics_mmr = extract_topics_mmr(texts, labels, diversity=0.6)
        
        # 6. Use the method with better coherence
        coherence_bm25 = calculate_topic_coherence(topics_bm25, texts)
        coherence_mmr = calculate_topic_coherence(topics_mmr, texts)
        
        avg_coherence_bm25 = np.mean(list(coherence_bm25.values())) if coherence_bm25 else 0
        avg_coherence_mmr = np.mean(list(coherence_mmr.values())) if coherence_mmr else 0
        
        if avg_coherence_mmr > avg_coherence_bm25:
            topics = topics_mmr
            method = 'MMR'
            coherence = coherence_mmr
        else:
            topics = topics_bm25
            method = 'BM25'
            coherence = coherence_bm25
        
        print(f"📈 Using {method} extraction (coherence: {np.mean(list(coherence.values())):.3f})")
        
        # 7. Generate human-readable labels
        topic_labels = generate_topic_labels(topics, method='thematic')
        
        # 8. Create enhanced summary
        rows = []
        for topic_id, topic_info in topics.items():
            # Get top document snippets
            rep_docs = topic_info['representative_docs']
            doc_snippets = [doc[:200] + '...' if len(doc) > 200 else doc for doc in rep_docs]
            
            row = {
                'topic_id': topic_id,
                'label': topic_labels[topic_id],
                'size': topic_info['size'],
                'top_words': topic_info['words'][:10],
                'coherence': coherence.get(topic_id, 0),
                'representative_snippets': doc_snippets,
                'subreddit': subreddit
            }
            rows.append(row)
        
        # Sort by size
        topic_df = pd.DataFrame(rows).sort_values('size', ascending=False)
        
        # 9. Save detailed results
        output_path = os.path.join(OUTPUT_DIR, f'{subreddit}_topics_advanced.csv')
        topic_df.to_csv(output_path, index=False)
        
        # Also save detailed JSON
        detailed_results = {
            'subreddit': subreddit,
            'n_documents': len(texts),
            'n_topics': len(topics),
            'extraction_method': method,
            'avg_coherence': float(np.mean(list(coherence.values()))),
            'noise_ratio': metrics['noise_ratio'],
            'topics': []
        }
        
        for _, row in topic_df.iterrows():
            detailed_results['topics'].append({
                'id': int(row['topic_id']),
                'label': row['label'],
                'size': int(row['size']),
                'coherence': float(row['coherence']),
                'words': row['top_words']
            })
        
        json_path = os.path.join(OUTPUT_DIR, f'{subreddit}_analysis_advanced.json')
        save_json(detailed_results, json_path)
        
        print(f"✅ Saved results to {output_path}")
        
        return detailed_results


def main():
    """Run advanced topic analysis on all subreddits"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize analyzer
    analyzer = AdvancedTopicAnalyzer()
    
    # Track results
    all_results = {}
    comparison_data = []
    
    # Analyze each subreddit with increased samples
    sample_sizes = {
        'cannabis': 800,
        'weed': 800,
        'trees': 800,
        'marijuana': 1000,  # Try harder for this one
        'weedstocks': 800,
        'weedbiz': 800
    }
    
    for subreddit in CANNABIS_SUBREDDITS:
        try:
            n_samples = sample_sizes.get(subreddit, DEFAULT_SAMPLE_SIZE)
            results = analyzer.analyze_subreddit(subreddit, n_samples=n_samples)
            
            if 'error' not in results:
                all_results[subreddit] = results
                
                # Create comparison entry
                comparison_data.append({
                    'subreddit': subreddit,
                    'n_topics': results['n_topics'],
                    'n_documents': results['n_documents'],
                    'avg_coherence': results['avg_coherence'],
                    'method': results['extraction_method'],
                    'top_topics': [t['label'] for t in results['topics'][:3]]
                })
            else:
                all_results[subreddit] = results
                
        except Exception as e:
            print(f"❌ Error analyzing r/{subreddit}: {e}")
            import traceback
            traceback.print_exc()
            all_results[subreddit] = {'error': str(e)}
    
    # Save master summary
    summary_path = os.path.join(OUTPUT_DIR, 'analysis_summary_advanced.json')
    save_json(all_results, summary_path)
    
    # Create comparison DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(OUTPUT_DIR, 'subreddit_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"\n{'='*60}")
        print("📊 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False))
    
    print(f"\n✅ Complete analysis saved to {summary_path}")


if __name__ == "__main__":
    main()
