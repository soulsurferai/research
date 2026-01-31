#!/usr/bin/env python3
"""
quick_marijuana_check.py - Quick check for r/marijuana data
"""

import os
import sys
sys.path.append('.')
from utils import QdrantFetcher

def main():
    fetcher = QdrantFetcher()
    
    print("Checking r/marijuana data...")
    
    # Try to fetch some data
    embeddings, texts, metadata = fetcher.fetch_subreddit_data('marijuana', limit=100)
    
    print(f"\nFound {len(texts)} documents")
    
    if texts:
        print("\nFirst 5 documents:")
        for i, (text, meta) in enumerate(zip(texts[:5], metadata[:5])):
            print(f"\n{i+1}. Type: {meta['type']}")
            print(f"   Text: {text[:200]}...")
            print(f"   Score: {meta['score']}")

if __name__ == "__main__":
    main()
