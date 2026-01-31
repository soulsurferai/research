#!/usr/bin/env python3
"""
debug_marijuana.py - Debug why r/marijuana isn't being analyzed
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Topics_V1.utils import QdrantFetcher, filter_boilerplate
from Topics_V1.config import MIN_CLUSTER_SIZE

def debug_marijuana_analysis():
    """Step through the analysis process for r/marijuana"""
    
    print("="*60)
    print("DEBUGGING r/marijuana ANALYSIS")
    print("="*60)
    
    # Step 1: Initialize fetcher
    print("\n1. Initializing QdrantFetcher...")
    fetcher = QdrantFetcher()
    
    # Step 2: Fetch data with different sample sizes
    for sample_size in [100, 500, 1000, 2000]:
        print(f"\n2. Fetching r/marijuana with sample_size={sample_size}...")
        embeddings, texts, metadata = fetcher.fetch_subreddit_data('marijuana', sample_size)
        
        print(f"   Raw results: {len(texts)} documents")
        
        if len(texts) == 0:
            print("   ❌ No documents found!")
            continue
            
        # Step 3: Check document lengths
        doc_lengths = [len(text.split()) for text in texts]
        print(f"   Document word counts: min={min(doc_lengths)}, max={max(doc_lengths)}, avg={np.mean(doc_lengths):.1f}")
        
        # Step 4: Apply filtering
        print(f"\n3. Filtering boilerplate...")
        filtered_texts, filtered_metadata = filter_boilerplate(texts, metadata)
        print(f"   After filtering: {len(filtered_texts)} documents")
        
        # Step 5: Apply additional filtering (20+ words)
        print(f"\n4. Filtering short documents (<20 words)...")
        long_docs = [(t, m) for t, m in zip(filtered_texts, filtered_metadata) if len(t.split()) >= 20]
        print(f"   After length filter: {len(long_docs)} documents")
        
        # Step 6: Check against minimum cluster size
        min_required = MIN_CLUSTER_SIZE * 3
        print(f"\n5. Checking against MIN_CLUSTER_SIZE * 3 = {min_required}")
        if len(long_docs) >= min_required:
            print(f"   ✅ Sufficient data! {len(long_docs)} >= {min_required}")
            
            # Show some sample documents
            print(f"\n6. Sample documents:")
            for i, (text, meta) in enumerate(long_docs[:3]):
                print(f"\n   Document {i+1} ({meta['type']}):")
                print(f"   {text[:200]}...")
            
            return True
        else:
            print(f"   ❌ Insufficient data! {len(long_docs)} < {min_required}")
    
    return False

def check_subreddit_variations():
    """Check if marijuana might be stored with different case/spelling"""
    print("\n" + "="*60)
    print("CHECKING SUBREDDIT NAME VARIATIONS")
    print("="*60)
    
    from qdrant_client import QdrantClient
    from dotenv import load_dotenv
    
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env', '.env')
    load_dotenv(env_path)
    
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )
    
    # Try different variations
    variations = ['marijuana', 'Marijuana', 'MARIJUANA', 'r/marijuana', 'marijuanaenthusiasts']
    
    for variant in variations:
        print(f"\nChecking '{variant}'...")
        try:
            count = client.count(
                collection_name='reddit_posts',
                count_filter={
                    "must": [
                        {"key": "subreddit", "match": {"value": variant}}
                    ]
                }
            )
            if count.count > 0:
                print(f"  ✓ Found {count.count} posts!")
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    # First check name variations
    check_subreddit_variations()
    
    # Then debug the analysis
    print("\n")
    success = debug_marijuana_analysis()
    
    if not success:
        print("\n" + "="*60)
        print("CONCLUSION: r/marijuana likely has insufficient data in Qdrant")
        print("or is filtered out during preprocessing.")
        print("="*60)
