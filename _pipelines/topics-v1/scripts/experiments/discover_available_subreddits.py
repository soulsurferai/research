#!/usr/bin/env python3
"""
discover_available_subreddits.py - Query Qdrant to find all available subreddits
"""

import json
import os
import sys
from collections import defaultdict
from qdrant_client import QdrantClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import QDRANT_URL, QDRANT_API_KEY, OUTPUT_DIR


def discover_subreddits():
    """Query Qdrant to find all available subreddits and their document counts"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    subreddit_counts = defaultdict(lambda: {'posts': 0, 'comments': 0, 'total': 0})
    
    print("🔍 Discovering available subreddits in Qdrant...\n")
    
    # Check both collections
    for collection in ['reddit_posts', 'reddit_comments']:
        print(f"Scanning {collection}...")
        
        try:
            # Use scroll to get all unique subreddits
            # We'll get a sample and extract unique values
            offset = None
            seen_subreddits = set()
            
            while True:
                results = client.scroll(
                    collection_name=collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # Don't need vectors for this
                )
                
                points, next_offset = results
                
                if not points:
                    break
                    
                for point in points:
                    subreddit = point.payload.get('subreddit', '')
                    if subreddit and subreddit not in seen_subreddits:
                        seen_subreddits.add(subreddit)
                        
                        # Count documents for this subreddit
                        count_result = client.count(
                            collection_name=collection,
                            count_filter={
                                "must": [
                                    {"key": "subreddit", "match": {"value": subreddit}}
                                ]
                            }
                        )
                        
                        if collection == 'reddit_posts':
                            subreddit_counts[subreddit]['posts'] = count_result.count
                        else:
                            subreddit_counts[subreddit]['comments'] = count_result.count
                        
                        subreddit_counts[subreddit]['total'] = (
                            subreddit_counts[subreddit]['posts'] + 
                            subreddit_counts[subreddit]['comments']
                        )
                
                offset = next_offset
                
                # Stop after finding a reasonable number of subreddits
                if len(seen_subreddits) > 50:
                    print(f"  Found {len(seen_subreddits)} subreddits (stopping scan)...")
                    break
                    
        except Exception as e:
            print(f"Error scanning {collection}: {e}")
    
    return dict(subreddit_counts)


def categorize_subreddits(subreddit_data):
    """Categorize subreddits into suggested comparison groups"""
    categories = {
        'cannabis': [],
        'politics': [],
        'finance': [],
        'technology': [],
        'gaming': [],
        'lifestyle': [],
        'regional': [],
        'other': []
    }
    
    # Categorize based on name patterns
    for sub, data in subreddit_data.items():
        sub_lower = sub.lower()
        
        if any(term in sub_lower for term in ['weed', 'cannabis', 'marijuana', 'trees', 'mmj', 'ent']):
            categories['cannabis'].append((sub, data))
        elif any(term in sub_lower for term in ['politics', 'liberal', 'conservative', 'libertarian', 'socialist']):
            categories['politics'].append((sub, data))
        elif any(term in sub_lower for term in ['stocks', 'investing', 'finance', 'crypto', 'bitcoin']):
            categories['finance'].append((sub, data))
        elif any(term in sub_lower for term in ['tech', 'programming', 'gaming', 'pcmr', 'games']):
            categories['technology'].append((sub, data))
        elif any(term in sub_lower for term in ['gaming', 'games', 'gamer']):
            categories['gaming'].append((sub, data))
        elif any(term in sub_lower for term in ['fitness', 'food', 'cooking', 'diy', 'art']):
            categories['lifestyle'].append((sub, data))
        elif any(term in sub_lower for term in ['city', 'state', 'country']) or len(sub) == 2:
            categories['regional'].append((sub, data))
        else:
            categories['other'].append((sub, data))
    
    # Sort each category by total document count
    for cat in categories:
        categories[cat].sort(key=lambda x: x[1]['total'], reverse=True)
    
    return categories


def suggest_comparisons(categories):
    """Suggest interesting comparison groups"""
    suggestions = []
    
    # Cannabis communities (if we have them)
    if len(categories['cannabis']) >= 3:
        cannabis_subs = [sub for sub, _ in categories['cannabis'][:6]]
        suggestions.append({
            'name': 'Cannabis Communities',
            'subreddits': cannabis_subs,
            'description': 'Compare emotional landscapes across different cannabis communities'
        })
    
    # Political spectrum (if we have different political subs)
    if len(categories['politics']) >= 3:
        political_subs = [sub for sub, _ in categories['politics'][:5]]
        suggestions.append({
            'name': 'Political Spectrum',
            'subreddits': political_subs,
            'description': 'Compare emotional patterns across different political ideologies'
        })
    
    # Finance communities
    if len(categories['finance']) >= 3:
        finance_subs = [sub for sub, _ in categories['finance'][:5]]
        suggestions.append({
            'name': 'Financial Communities',
            'subreddits': finance_subs,
            'description': 'Compare hope/anxiety in investment and finance communities'
        })
    
    # Cross-domain comparison
    if categories['cannabis'] and categories['politics'] and categories['finance']:
        cross_domain = [
            categories['cannabis'][0][0] if categories['cannabis'] else None,
            categories['politics'][0][0] if categories['politics'] else None,
            categories['finance'][0][0] if categories['finance'] else None,
            categories['technology'][0][0] if categories['technology'] else None,
            categories['lifestyle'][0][0] if categories['lifestyle'] else None
        ]
        cross_domain = [s for s in cross_domain if s][:5]
        
        if len(cross_domain) >= 3:
            suggestions.append({
                'name': 'Cross-Domain Emotions',
                'subreddits': cross_domain,
                'description': 'Compare emotional patterns across completely different communities'
            })
    
    return suggestions


def save_results(subreddit_data, categories, suggestions):
    """Save discovered subreddits and suggestions"""
    output = {
        'total_subreddits': len(subreddit_data),
        'subreddit_data': subreddit_data,
        'categories': {
            cat: [(sub, data) for sub, data in subs] 
            for cat, subs in categories.items()
        },
        'suggestions': suggestions
    }
    
    output_path = os.path.join(OUTPUT_DIR, 'available_subreddits.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    return output_path


def print_summary(subreddit_data, categories, suggestions):
    """Print a summary of findings"""
    print(f"\n📊 AVAILABLE SUBREDDITS SUMMARY")
    print(f"{'='*50}")
    print(f"Total subreddits found: {len(subreddit_data)}")
    print(f"\nBy category:")
    
    for cat, subs in categories.items():
        if subs:
            print(f"\n{cat.upper()} ({len(subs)} subreddits):")
            for sub, data in subs[:5]:  # Show top 5
                print(f"  - r/{sub}: {data['total']:,} docs ({data['posts']} posts, {data['comments']} comments)")
            if len(subs) > 5:
                print(f"  ... and {len(subs) - 5} more")
    
    print(f"\n💡 SUGGESTED COMPARISONS")
    print(f"{'='*50}")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['name']}")
        print(f"   {suggestion['description']}")
        print(f"   Subreddits: {', '.join(['r/' + s for s in suggestion['subreddits']])}")


def main():
    print("🔍 Discovering Available Subreddits in Qdrant Database\n")
    
    # Discover subreddits
    subreddit_data = discover_subreddits()
    
    if not subreddit_data:
        print("❌ No subreddits found in Qdrant!")
        return
    
    # Categorize them
    categories = categorize_subreddits(subreddit_data)
    
    # Generate suggestions
    suggestions = suggest_comparisons(categories)
    
    # Save results
    output_path = save_results(subreddit_data, categories, suggestions)
    
    # Print summary
    print_summary(subreddit_data, categories, suggestions)
    
    print(f"\n✅ Full results saved to: {output_path}")
    print("\n🚀 To run emotion analysis on any set of subreddits:")
    print("   python scripts/experiments/flexible_emotion_analysis.py --subreddits sub1 sub2 sub3")


if __name__ == "__main__":
    main()
