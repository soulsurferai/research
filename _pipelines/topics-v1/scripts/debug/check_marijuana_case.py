#!/usr/bin/env python3
"""
check_marijuana_case.py - Check if r/marijuana needs capital M
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env', '.env')
load_dotenv(env_path)

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

print("Checking different case variations of marijuana...\n")

# Test different case variations
variants = [
    'marijuana',
    'Marijuana', 
    'MARIJUANA',
    'r/marijuana',
    'r/Marijuana',
    'marijuanaenthusiasts',  # This is actually the trees subreddit!
    'MarijuanaEnthusiasts'
]

found_variants = []

for variant in variants:
    print(f"Checking '{variant}'...")
    
    for collection in ['reddit_posts', 'reddit_comments']:
        try:
            count = client.count(
                collection_name=collection,
                count_filter={
                    "must": [
                        {"key": "subreddit", "match": {"value": variant}}
                    ]
                }
            )
            
            if count.count > 0:
                print(f"  ✓ Found {count.count} in {collection}!")
                found_variants.append((variant, collection, count.count))
                
                # Get a sample to see exact formatting
                sample = client.scroll(
                    collection_name=collection,
                    scroll_filter={
                        "must": [
                            {"key": "subreddit", "match": {"value": variant}}
                        ]
                    },
                    limit=1,
                    with_payload=['subreddit']
                )
                
                if sample[0]:
                    actual_value = sample[0][0].payload.get('subreddit', 'N/A')
                    print(f"    Actual value in database: '{actual_value}'")
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")

print(f"\n{'='*60}")
print("SUMMARY:")
print(f"{'='*60}")

if found_variants:
    print("Found data with these variants:")
    for variant, collection, count in found_variants:
        print(f"  '{variant}' in {collection}: {count} documents")
else:
    print("No data found with any case variation!")
    
# Also check what cannabis-related subreddits exist
print(f"\n{'='*60}")
print("Checking all cannabis-related subreddits in database...")
print(f"{'='*60}")

# Get a larger sample to find all subreddit names
sample = client.scroll(
    collection_name='reddit_posts',
    limit=5000,
    with_payload=['subreddit']
)

subreddit_counts = {}
for point in sample[0]:
    sub = point.payload.get('subreddit', '')
    if any(term in sub.lower() for term in ['marijuana', 'cannabis', 'weed', 'trees', 'pot']):
        subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1

print("\nCannabis-related subreddits found:")
for sub, count in sorted(subreddit_counts.items()):
    print(f"  '{sub}': {count} (in sample)")
