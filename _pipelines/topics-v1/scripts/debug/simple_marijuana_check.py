#!/usr/bin/env python3
"""Simple check for marijuana data"""

from utils.qdrant_utils import QdrantFetcher

fetcher = QdrantFetcher()
print("Fetching r/marijuana data...")

# Try different sample sizes
for limit in [100, 500, 1000]:
    embeddings, texts, metadata = fetcher.fetch_subreddit_data('marijuana', limit)
    print(f"\nLimit {limit}: Found {len(texts)} documents")
    
    if texts:
        # Show first few
        print("Sample texts:")
        for i, text in enumerate(texts[:3]):
            print(f"{i+1}. {text[:100]}...")
        break
else:
    print("\nNo data found for r/marijuana in any sample size!")

# Also check exact subreddit names in a sample
print("\n" + "="*50)
print("Checking exact subreddit names in database...")

# Get a broader sample
embeddings, texts, metadata = fetcher.fetch_subreddit_data('cannabis', 10)
if metadata:
    print(f"Sample subreddit value from cannabis data: '{metadata[0].get('subreddit', 'N/A')}'")
