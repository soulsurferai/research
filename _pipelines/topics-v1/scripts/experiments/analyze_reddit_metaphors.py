#!/usr/bin/env python3
"""
analyze_reddit_metaphors.py - Run metaphor analysis on actual Reddit cannabis data
"""

import json
import pandas as pd
from collections import defaultdict
from analysis.nlp import MetaphorDetector, analyze_community_metaphors

def load_reddit_texts():
    """Load text data from the advanced topic analysis"""
    
    # Load the detailed CSV files which contain representative docs
    subreddit_texts = {}
    
    subreddits = ['cannabis', 'weed', 'trees', 'Marijuana', 'weedstocks', 'weedbiz']
    
    for subreddit in subreddits:
        try:
            # Load the topic CSV
            df = pd.read_csv(f'quick_results/{subreddit}_topics_advanced.csv')
            
            # Extract texts from representative_snippets column
            texts = []
            for _, row in df.iterrows():
                if pd.notna(row.get('representative_snippets')):
                    # Parse the snippets (they're stored as string representation of list)
                    snippets = eval(row['representative_snippets'])
                    texts.extend(snippets)
            
            subreddit_texts[subreddit] = texts
            print(f"Loaded {len(texts)} text snippets from r/{subreddit}")
            
        except Exception as e:
            print(f"Could not load r/{subreddit}: {e}")
    
    return subreddit_texts


def analyze_all_communities():
    """Run metaphor analysis on all communities"""
    
    print("METAPHOR ANALYSIS OF CANNABIS REDDIT COMMUNITIES")
    print("="*80)
    
    # Load texts
    subreddit_texts = load_reddit_texts()
    
    # Initialize detector
    detector = MetaphorDetector()
    
    # Store results for comparison
    all_results = {}
    
    # Analyze each community
    for subreddit, texts in subreddit_texts.items():
        if texts:
            print(f"\n\nAnalyzing r/{subreddit}")
            print("-"*60)
            
            results = analyze_community_metaphors(texts, f"r/{subreddit}")
            all_results[subreddit] = results
            
            # Print summary
            print(f"Total metaphors: {results['total_metaphors']}")
            print(f"Interpretation: {results['interpretation']}")
            
            print("\nTop metaphors:")
            for metaphor, count in results['dominant_metaphors']:
                pct = results['metaphor_distribution'].get(metaphor, 0)
                print(f"  {metaphor:15} : {count:3} ({pct:.1%})")
            
            # Show examples
            if results['examples']:
                print("\nExample usage:")
                for m_type, examples in list(results['examples'].items())[:2]:
                    if examples:
                        ex = examples[0]
                        print(f"  {m_type}: '{ex['keyword']}' in \"{ex['context'][:80]}...\"")
    
    return all_results


def compare_communities(all_results):
    """Compare metaphor usage across communities"""
    
    print("\n\n" + "="*80)
    print("CROSS-COMMUNITY METAPHOR COMPARISON")
    print("="*80)
    
    # Create comparison matrix
    communities = list(all_results.keys())
    metaphor_types = ['journey', 'medicine', 'nature', 'freedom', 'war', 
                      'enlightenment', 'relationship', 'food']
    
    print(f"\n{'Metaphor':15}", end='')
    for comm in communities:
        print(f"{comm:>12}", end='')
    print()
    print("-"*80)
    
    for metaphor in metaphor_types:
        print(f"{metaphor:15}", end='')
        for comm in communities:
            pct = all_results[comm]['metaphor_distribution'].get(metaphor, 0)
            print(f"{pct:11.1%}", end=' ')
        print()
    
    # Find distinctive metaphors for each community
    print("\n\nDISTINCTIVE METAPHORS BY COMMUNITY")
    print("-"*80)
    
    for comm in communities:
        # Find metaphors that are more common here than average
        comm_dist = all_results[comm]['metaphor_distribution']
        
        # Calculate average usage across other communities
        distinctive = []
        for metaphor in metaphor_types:
            comm_usage = comm_dist.get(metaphor, 0)
            
            # Average in other communities
            other_usage = []
            for other_comm in communities:
                if other_comm != comm:
                    other_usage.append(
                        all_results[other_comm]['metaphor_distribution'].get(metaphor, 0)
                    )
            
            avg_other = sum(other_usage) / len(other_usage) if other_usage else 0
            
            # If this community uses it significantly more
            if comm_usage > avg_other * 1.5 and comm_usage > 0.1:
                distinctive.append((metaphor, comm_usage, avg_other))
        
        if distinctive:
            print(f"\nr/{comm}:")
            for metaphor, comm_pct, avg_pct in sorted(distinctive, 
                                                     key=lambda x: x[1], 
                                                     reverse=True):
                print(f"  {metaphor}: {comm_pct:.1%} (vs avg {avg_pct:.1%})")


def analyze_medical_vs_recreational():
    """Compare medical-focused vs recreational communities"""
    
    print("\n\n" + "="*80)
    print("MEDICAL VS RECREATIONAL METAPHOR ANALYSIS")
    print("="*80)
    
    detector = MetaphorDetector()
    subreddit_texts = load_reddit_texts()
    
    # Group communities
    medical_texts = []
    recreational_texts = []
    
    # Medical/serious: cannabis, Marijuana, weedbiz
    for sub in ['cannabis', 'Marijuana', 'weedbiz']:
        if sub in subreddit_texts:
            medical_texts.extend(subreddit_texts[sub])
    
    # Recreational: weed, trees
    for sub in ['weed', 'trees']:
        if sub in subreddit_texts:
            recreational_texts.extend(subreddit_texts[sub])
    
    if medical_texts and recreational_texts:
        comparison = detector.compare_metaphor_usage(
            medical_texts,
            recreational_texts,
            'Medical/Policy Focus',
            'Recreational Focus'
        )
        
        print(f"\n{'Metaphor':15} {'Medical/Policy':>15} {'Recreational':>15} {'Difference':>12}")
        print("-"*60)
        
        for metaphor in ['medicine', 'journey', 'nature', 'freedom', 'war']:
            med_pct = comparison['Medical/Policy Focus'].get(metaphor, 0)
            rec_pct = comparison['Recreational Focus'].get(metaphor, 0)
            diff = comparison['differences'].get(metaphor, 0)
            
            print(f"{metaphor:15} {med_pct:14.1%} {rec_pct:14.1%} {diff:+11.1%}")


def save_results(all_results):
    """Save analysis results"""
    
    # Convert to serializable format
    output = {}
    for comm, results in all_results.items():
        output[comm] = {
            'total_metaphors': results['total_metaphors'],
            'metaphor_distribution': results['metaphor_distribution'],
            'dominant_metaphors': results['dominant_metaphors'],
            'interpretation': results['interpretation']
        }
    
    with open('quick_results/metaphor_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n\nResults saved to quick_results/metaphor_analysis.json")


def main():
    """Run complete metaphor analysis"""
    
    # Analyze all communities
    all_results = analyze_all_communities()
    
    # Compare across communities
    if len(all_results) > 1:
        compare_communities(all_results)
        
    # Medical vs recreational comparison
    analyze_medical_vs_recreational()
    
    # Save results
    save_results(all_results)


if __name__ == "__main__":
    main()
