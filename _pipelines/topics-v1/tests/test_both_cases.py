#!/usr/bin/env python3
"""
test_both_cases.py - Quick test with both marijuana and Marijuana
"""

from utils import QdrantFetcher

fetcher = QdrantFetcher()

print("Testing case sensitivity...\n")

for subreddit_name in ['marijuana', 'Marijuana']:
    print(f"Trying '{subreddit_name}':")
    embeddings, texts, metadata = fetcher.fetch_subreddit_data(subreddit_name, 100)
    print(f"  Found {len(texts)} documents")
    
    if texts:
        print(f"  First document type: {metadata[0]['type']}")
        print(f"  First 100 chars: {texts[0][:100]}...")
        print(f"  Subreddit value in metadata: '{metadata[0].get('subreddit', 'N/A')}'")
    print()

# Also try the trees/marijuana confusion
print("Also checking marijuanaenthusiasts (which is actually about trees!):")
embeddings, texts, metadata = fetcher.fetch_subreddit_data('marijuanaenthusiasts', 100)
print(f"  Found {len(texts)} documents")
if texts:
    print(f"  First doc: {texts[0][:100]}...")
