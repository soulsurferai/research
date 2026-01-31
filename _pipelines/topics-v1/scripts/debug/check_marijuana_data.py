#!/usr/bin/env python3
"""
check_marijuana_data.py - Spot check r/marijuana data availability
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env', '.env'))

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

def check_subreddit_data(subreddit: str):
    """Check data availability for a specific subreddit"""
    print(f"\n{'='*60}")
    print(f"Checking data for r/{subreddit}")
    print(f"{'='*60}")
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    total_count = 0
    
    for collection in ['reddit_posts', 'reddit_comments']:
        try:
            # Count points
            count_result = client.count(
                collection_name=collection,
                count_filter={
                    "must": [
                        {"key": "subreddit", "match": {"value": subreddit}}
                    ]
                }
            )
            
            count = count_result.count
            print(f"\n{collection}: {count} documents")
            total_count += count
            
            # Get a few samples
            if count > 0:
                results = client.scroll(
                    collection_name=collection,
                    scroll_filter={
                        "must": [
                            {"key": "subreddit", "match": {"value": subreddit}}
                        ]
                    },
                    limit=3,
                    with_payload=True,
                    with_vectors=False
                )
                
                print(f"\nSample documents from {collection}:")
                for i, point in enumerate(results[0]):
                    print(f"\n{i+1}. ID: {point.id}")
                    if 'title' in point.payload:
                        print(f"   Title: {point.payload['title'][:100]}...")
                    if 'content' in point.payload:
                        content = point.payload.get('content', '')
                        if content:
                            print(f"   Content: {content[:100]}...")
                    print(f"   Score: {point.payload.get('score', 'N/A')}")
                    print(f"   Created: {point.payload.get('created_at', 'N/A')}")
                    
        except Exception as e:
            print(f"Error checking {collection}: {e}")
    
    print(f"\n{'='*30}")
    print(f"TOTAL for r/{subreddit}: {total_count} documents")
    print(f"{'='*30}")
    
    return total_count

def main():
    """Check marijuana and compare with other subreddits"""
    
    # Check all cannabis subreddits
    subreddits = ['marijuana', 'cannabis', 'weed', 'trees', 'weedstocks', 'weedbiz']
    
    counts = {}
    for sub in subreddits:
        count = check_subreddit_data(sub)
        counts[sub] = count
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Document counts per subreddit:")
    print(f"{'='*60}")
    for sub, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"r/{sub:15} : {count:6} documents")

if __name__ == "__main__":
    main()
