#!/usr/bin/env python3
"""
marijuana_investigation.py - Detailed investigation of r/marijuana data
"""

import os
import sys
sys.path.append('.')

from utils import QdrantFetcher
from config import MIN_CLUSTER_SIZE

print("=" * 80)
print("INVESTIGATING r/marijuana DATA AVAILABILITY")
print("=" * 80)

# Initialize fetcher
fetcher = QdrantFetcher()

# Try progressively larger sample sizes
sample_sizes = [100, 500, 1000, 2000, 5000]

for sample_size in sample_sizes:
    print(f"\n{'='*60}")
    print(f"Testing with sample_size = {sample_size}")
    print(f"{'='*60}")
    
    # Fetch data
    embeddings, texts, metadata = fetcher.fetch_subreddit_data('marijuana', sample_size)
    
    print(f"\n1. Raw fetch results:")
    print(f"   - Documents found: {len(texts)}")
    
    if not texts:
        print("   ❌ No documents found at all!")
        continue
    
    # Analyze document types
    post_count = sum(1 for m in metadata if m['type'] == 'post')
    comment_count = sum(1 for m in metadata if m['type'] == 'comment')
    print(f"   - Posts: {post_count}")
    print(f"   - Comments: {comment_count}")
    
    # Check text lengths
    text_lengths = [len(text.split()) for text in texts]
    print(f"\n2. Text length analysis:")
    print(f"   - Min words: {min(text_lengths)}")
    print(f"   - Max words: {max(text_lengths)}")
    print(f"   - Avg words: {sum(text_lengths)/len(text_lengths):.1f}")
    
    # Count how many would survive 20-word filter
    long_texts = [t for t in texts if len(t.split()) >= 20]
    print(f"\n3. After 20-word minimum filter:")
    print(f"   - Surviving documents: {len(long_texts)} ({len(long_texts)/len(texts)*100:.1f}%)")
    
    # Check minimum requirement
    min_required = MIN_CLUSTER_SIZE * 3
    print(f"\n4. Minimum requirement check:")
    print(f"   - Required: {min_required} (MIN_CLUSTER_SIZE * 3)")
    print(f"   - Have: {len(long_texts)}")
    print(f"   - {'✅ SUFFICIENT' if len(long_texts) >= min_required else '❌ INSUFFICIENT'}")
    
    # Show some sample texts
    if long_texts:
        print(f"\n5. Sample texts that would be analyzed:")
        for i, text in enumerate(long_texts[:3]):
            print(f"\n   Text {i+1} ({len(text.split())} words):")
            print(f"   {text[:200]}...")
    
    # If we have enough, break
    if len(long_texts) >= min_required:
        print(f"\n✅ Found sufficient data with sample_size={sample_size}")
        break
else:
    print(f"\n❌ Could not find sufficient data even with sample_size={sample_sizes[-1]}")

# Final summary
print(f"\n{'='*80}")
print("SUMMARY:")
print(f"{'='*80}")
print(f"MIN_CLUSTER_SIZE: {MIN_CLUSTER_SIZE}")
print(f"Required documents: {MIN_CLUSTER_SIZE * 3}")
print(f"Maximum found (after filtering): {len(long_texts) if 'long_texts' in locals() else 0}")
print("\nCONCLUSION: r/marijuana likely has very little content in Qdrant,")
print("or most of its content is too short (< 20 words) to be useful for topic analysis.")
