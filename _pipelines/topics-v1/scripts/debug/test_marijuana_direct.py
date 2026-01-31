#!/usr/bin/env python3
"""
test_marijuana_direct.py - Direct Qdrant query for r/marijuana
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env', '.env')
load_dotenv(env_path)

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

print(f"Environment loaded from: {env_path}")
print(f"Qdrant URL: {QDRANT_URL[:30]}...")

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Check both collections
for collection in ['reddit_posts', 'reddit_comments']:
    print(f"\n{'='*50}")
    print(f"Checking {collection} for r/marijuana")
    print(f"{'='*50}")
    
    try:
        # First, just count
        count_result = client.count(
            collection_name=collection,
            count_filter={
                "must": [
                    {"key": "subreddit", "match": {"value": "marijuana"}}
                ]
            }
        )
        print(f"Count: {count_result.count}")
        
        # If there's data, get samples
        if count_result.count > 0:
            results = client.scroll(
                collection_name=collection,
                scroll_filter={
                    "must": [
                        {"key": "subreddit", "match": {"value": "marijuana"}}
                    ]
                },
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            print(f"\nFound {len(results[0])} documents in scroll")
            for i, point in enumerate(results[0][:3]):
                print(f"\nDocument {i+1}:")
                print(f"  Subreddit: {point.payload.get('subreddit', 'N/A')}")
                if 'title' in point.payload:
                    print(f"  Title: {point.payload['title'][:80]}...")
                if 'content' in point.payload and point.payload['content']:
                    print(f"  Content: {point.payload['content'][:80]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Also check what subreddits ARE in the database
print(f"\n{'='*50}")
print("Checking what cannabis subreddits exist in posts...")
print(f"{'='*50}")

try:
    # Get a sample of posts
    sample = client.scroll(
        collection_name='reddit_posts',
        limit=1000,
        with_payload=['subreddit'],
        with_vectors=False
    )
    
    # Count subreddits
    subreddit_counts = {}
    for point in sample[0]:
        sub = point.payload.get('subreddit', 'unknown')
        if any(term in sub.lower() for term in ['cannabis', 'weed', 'marijuana', 'trees']):
            subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
    
    print("\nCannabis-related subreddits found:")
    for sub, count in sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sub}: {count}")
        
except Exception as e:
    print(f"Error sampling: {e}")
