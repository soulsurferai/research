#!/usr/bin/env python3
"""
run_full_metaphor_analysis.py - Run metaphor analysis on full Reddit dataset
"""

import json
from utils import QdrantFetcher
from analysis.nlp import MetaphorDetector
from collections import defaultdict


def analyze_full_subreddit(subreddit: str, limit: int = 1000):
    """Analyze metaphors in a full subreddit dataset"""
    
    print(f"\nAnalyzing r/{subreddit} (up to {limit} documents)...")
    print("-"*60)
    
    # Fetch data
    fetcher = QdrantFetcher()
    embeddings, texts, metadata = fetcher.fetch_subreddit_data(subreddit, limit)
    
    if not texts:
        print(f"No data found for r/{subreddit}")
        return None
    
    print(f"Analyzing {len(texts)} documents...")
    
    # Run metaphor detection
    detector = MetaphorDetector()
    results = detector.detect_metaphors(texts)
    
    # Add community info
    results['community'] = f"r/{subreddit}"
    results['n_documents'] = len(texts)
    
    # Print summary
    print(f"Total metaphors found: {results['total_metaphors']}")
    print(f"Metaphors per document: {results['total_metaphors']/len(texts):.2f}")
    
    print("\nTop 5 metaphors:")
    for m_type, count in sorted(results['metaphor_counts'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]:
        pct = results['metaphor_distribution'][m_type]
        print(f"  {m_type:15} : {count:4} ({pct:5.1%})")
    
    return results


def create_metaphor_heatmap(all_results):
    """Create data for a heatmap visualization"""
    
    communities = sorted(all_results.keys())
    metaphors = ['journey', 'medicine', 'nature', 'freedom', 'war', 
                 'enlightenment', 'relationship', 'food']
    
    # Create matrix
    heatmap_data = []
    for metaphor in metaphors:
        row = []
        for comm in communities:
            pct = all_results[comm]['metaphor_distribution'].get(metaphor, 0)
            row.append(round(pct * 100, 1))
        heatmap_data.append(row)
    
    # Save for visualization
    heatmap = {
        'communities': communities,
        'metaphors': metaphors,
        'data': heatmap_data
    }
    
    with open('quick_results/metaphor_heatmap_data.json', 'w') as f:
        json.dump(heatmap, f, indent=2)
    
    print("\nHeatmap data saved to metaphor_heatmap_data.json")
    print("\nHeatmap Preview:")
    print(f"{'':15}", end='')
    for comm in communities:
        print(f"{comm:>10}", end='')
    print()
    
    for i, metaphor in enumerate(metaphors):
        print(f"{metaphor:15}", end='')
        for val in heatmap_data[i]:
            if val > 20:
                print(f"{'█'*int(val/10):>10}", end='')
            elif val > 10:
                print(f"{'▓'*int(val/10):>10}", end='')
            elif val > 5:
                print(f"{'▒'*int(val/10):>10}", end='')
            else:
                print(f"{'░'*max(1, int(val/10)):>10}", end='')
        print()


def main():
    """Run full analysis on all subreddits"""
    
    print("FULL METAPHOR ANALYSIS OF CANNABIS REDDIT COMMUNITIES")
    print("="*80)
    
    subreddits = ['cannabis', 'weed', 'trees', 'Marijuana', 'weedstocks', 'weedbiz']
    all_results = {}
    
    # Analyze each subreddit
    for subreddit in subreddits:
        results = analyze_full_subreddit(subreddit, limit=500)
        if results:
            all_results[subreddit] = results
    
    # Create comparison visualizations
    if len(all_results) > 1:
        print("\n\n" + "="*80)
        print("CREATING VISUALIZATION DATA")
        print("="*80)
        
        create_metaphor_heatmap(all_results)
    
    # Save complete results
    output_file = 'quick_results/full_metaphor_analysis.json'
    
    # Prepare for JSON serialization
    output = {}
    for sub, results in all_results.items():
        output[sub] = {
            'n_documents': results['n_documents'],
            'total_metaphors': results['total_metaphors'],
            'metaphors_per_doc': results['total_metaphors'] / results['n_documents'],
            'metaphor_counts': results['metaphor_counts'],
            'metaphor_distribution': results['metaphor_distribution'],
            'top_examples': {
                m_type: [ex['context'] for ex in examples[:2]]
                for m_type, examples in results.get('examples', {}).items()
            }
        }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nFull results saved to {output_file}")


if __name__ == "__main__":
    main()
