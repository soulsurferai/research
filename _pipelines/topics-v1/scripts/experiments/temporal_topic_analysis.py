#!/usr/bin/env python3
"""
temporal_topic_analysis.py - Analyze how topics evolve over time
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import OUTPUT_DIR
from utils import QdrantFetcher
from core.topics_advanced import AdvancedTopicAnalyzer


class TemporalTopicAnalyzer:
    """Analyze topic evolution over time"""
    
    def __init__(self):
        self.fetcher = QdrantFetcher()
        self.analyzer = AdvancedTopicAnalyzer()
    
    def analyze_time_windows(self, subreddit: str, 
                           window_size: int = 30,  # days
                           num_windows: int = 6,
                           samples_per_window: int = 200):
        """
        Analyze topics in time windows going backwards from present
        
        Args:
            subreddit: Name of subreddit to analyze
            window_size: Size of each time window in days
            num_windows: Number of windows to analyze
            samples_per_window: Samples per time window
        """
        results = []
        
        # Get current date (you might want to adjust this based on your data)
        end_date = datetime(2025, 7, 1)  # Adjust based on your data cutoff
        
        print(f"\n📅 Analyzing r/{subreddit} over {num_windows} time windows of {window_size} days each")
        
        for i in range(num_windows):
            # Calculate window boundaries
            window_end = end_date - timedelta(days=i * window_size)
            window_start = window_end - timedelta(days=window_size)
            
            print(f"\n🔍 Window {i+1}: {window_start.date()} to {window_end.date()}")
            
            # Fetch data for this time window
            try:
                docs, embeddings = self.fetcher.fetch_subreddit_data(
                    subreddit=subreddit,
                    n_samples=samples_per_window,
                    date_range=(window_start, window_end)  # This would need implementation in fetcher
                )
                
                if len(docs) < 10:
                    print(f"  ⚠️  Only {len(docs)} documents found, skipping window")
                    continue
                
                # Run topic analysis for this window
                window_topics = self._analyze_window(docs, embeddings, window_start, window_end)
                
                results.append({
                    'window': i,
                    'start': window_start.isoformat(),
                    'end': window_end.isoformat(),
                    'num_docs': len(docs),
                    'topics': window_topics
                })
                
            except Exception as e:
                print(f"  ❌ Error analyzing window: {e}")
                continue
        
        return results
    
    def _analyze_window(self, docs, embeddings, start_date, end_date):
        """Analyze topics for a single time window"""
        # This is a simplified version - you'd want to use the actual
        # topic analysis pipeline from topics_advanced.py
        
        # For now, just extract key terms as a placeholder
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = [doc['content'] for doc in docs]
        
        # Extract top terms for this window
        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average tf-idf scores
        avg_scores = tfidf_matrix.mean(axis=0).A1
        
        # Get top terms
        top_indices = avg_scores.argsort()[-20:][::-1]
        top_terms = [(feature_names[i], avg_scores[i]) for i in top_indices]
        
        return {
            'top_terms': top_terms,
            'doc_count': len(docs)
        }
    
    def identify_trends(self, temporal_results):
        """Identify rising and falling topics across time windows"""
        # Track term frequencies across windows
        term_trajectories = defaultdict(list)
        
        for window_result in temporal_results:
            window_terms = {term: score for term, score in window_result['topics']['top_terms']}
            
            # Add scores for all terms
            all_terms = set()
            for w in temporal_results:
                all_terms.update([t[0] for t in w['topics']['top_terms']])
            
            for term in all_terms:
                term_trajectories[term].append(window_terms.get(term, 0))
        
        # Calculate trends
        trends = {}
        for term, trajectory in term_trajectories.items():
            if len(trajectory) > 1:
                # Simple trend: compare first half to second half average
                first_half = np.mean(trajectory[:len(trajectory)//2])
                second_half = np.mean(trajectory[len(trajectory)//2:])
                
                if second_half > first_half * 1.5 and second_half > 0.01:
                    trends[term] = {
                        'trend': 'rising',
                        'change': (second_half - first_half) / (first_half + 0.001),
                        'trajectory': trajectory
                    }
                elif second_half < first_half * 0.5 and first_half > 0.01:
                    trends[term] = {
                        'trend': 'falling', 
                        'change': (second_half - first_half) / (first_half + 0.001),
                        'trajectory': trajectory
                    }
        
        return trends
    
    def generate_temporal_report(self, subreddit, results, trends):
        """Generate a report on temporal topic evolution"""
        report = [f"# Temporal Topic Analysis - r/{subreddit}\n"]
        
        # Timeline overview
        report.append("## Timeline Overview\n")
        for window in results:
            report.append(f"- **Window {window['window'] + 1}** ({window['start'][:10]} to {window['end'][:10]}): {window['num_docs']} documents")
        
        # Rising topics
        report.append("\n## 📈 Rising Topics\n")
        rising = {k: v for k, v in trends.items() if v['trend'] == 'rising'}
        for term, data in sorted(rising.items(), key=lambda x: x[1]['change'], reverse=True)[:10]:
            report.append(f"- **{term}**: {data['change']:.1%} increase")
        
        # Falling topics
        report.append("\n## 📉 Declining Topics\n")
        falling = {k: v for k, v in trends.items() if v['trend'] == 'falling'}
        for term, data in sorted(falling.items(), key=lambda x: x[1]['change'])[:10]:
            report.append(f"- **{term}**: {data['change']:.1%} decrease")
        
        # Insights
        report.append("\n## 💡 Key Insights\n")
        report.append(f"- Analyzed {len(results)} time windows")
        report.append(f"- Identified {len(rising)} rising topics")
        report.append(f"- Identified {len(falling)} declining topics")
        
        return '\n'.join(report)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze topic evolution over time')
    parser.add_argument('--subreddit', default='cannabis', help='Subreddit to analyze')
    parser.add_argument('--windows', type=int, default=6, help='Number of time windows')
    parser.add_argument('--window-size', type=int, default=30, help='Window size in days')
    parser.add_argument('--samples', type=int, default=200, help='Samples per window')
    
    args = parser.parse_args()
    
    print(f"⏰ Starting Temporal Analysis for r/{args.subreddit}")
    
    analyzer = TemporalTopicAnalyzer()
    
    # Note: This is a simplified implementation. 
    # You'll need to modify the QdrantFetcher to support date filtering
    # For now, this shows the structure of temporal analysis
    
    print("\n⚠️  Note: This is a demonstration of temporal analysis structure.")
    print("To fully implement, you'll need to:")
    print("1. Add date filtering to QdrantFetcher")
    print("2. Ensure your Qdrant data includes timestamps")
    print("3. Integrate with the full topic analysis pipeline")
    
    # Placeholder demonstration
    print("\n📊 Demonstration of temporal analysis approach:")
    print("- Divide time into windows (e.g., monthly)")
    print("- Run topic analysis on each window")
    print("- Track topic prevalence across windows")
    print("- Identify rising/falling trends")
    print("- Find event-driven topic spikes")
    
    # Save template
    template = {
        'description': 'Temporal topic analysis template',
        'approach': {
            'window_size': args.window_size,
            'num_windows': args.windows,
            'samples_per_window': args.samples
        },
        'next_steps': [
            'Implement date filtering in Qdrant queries',
            'Add temporal metadata to embeddings',
            'Create visualization of topic evolution',
            'Correlate topic changes with external events'
        ]
    }
    
    output_path = os.path.join(OUTPUT_DIR, 'temporal_analysis_template.json')
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✅ Saved temporal analysis template to {output_path}")


if __name__ == "__main__":
    main()
