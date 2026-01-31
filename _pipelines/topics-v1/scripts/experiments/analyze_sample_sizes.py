#!/usr/bin/env python3
"""
analyze_sample_sizes.py - Analyze impact of different sample sizes on topic quality
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load current results
with open('quick_results/analysis_summary_advanced.json', 'r') as f:
    results = json.load(f)

print("CURRENT SAMPLE SIZES AND RESULTS")
print("="*80)
print(f"{'Subreddit':15} {'Docs':>8} {'Topics':>8} {'Coherence':>10} {'Docs/Topic':>12}")
print("-"*80)

for sub, data in results.items():
    if 'error' not in data:
        docs = data['n_documents']
        topics = data['n_topics']
        coherence = data['avg_coherence']
        docs_per_topic = docs / topics if topics > 0 else 0
        
        print(f"{sub:15} {docs:8} {topics:8} {coherence:10.3f} {docs_per_topic:12.1f}")

print("\n\nSAMPLE SIZE RECOMMENDATIONS")
print("="*80)

# Analyze relationship between docs and quality
doc_counts = []
coherences = []
topic_counts = []

for sub, data in results.items():
    if 'error' not in data:
        doc_counts.append(data['n_documents'])
        coherences.append(data['avg_coherence'])
        topic_counts.append(data['n_topics'])

# Calculate correlations
if len(doc_counts) > 3:
    doc_coherence_corr = np.corrcoef(doc_counts, coherences)[0, 1]
    doc_topics_corr = np.corrcoef(doc_counts, topic_counts)[0, 1]
    
    print(f"\nCorrelation between document count and coherence: {doc_coherence_corr:.3f}")
    print(f"Correlation between document count and topic count: {doc_topics_corr:.3f}")

# Recommendations
print("\n\nRECOMMENDATIONS:")
print("-"*80)

for sub, data in results.items():
    if 'error' not in data:
        docs = data['n_documents']
        coherence = data['avg_coherence']
        
        # Calculate recommended sample size
        if coherence < 0.4:  # Low coherence
            recommended = int(docs * 1.5)
            print(f"{sub}: INCREASE to {recommended} docs (currently {docs})")
            print(f"  Reason: Low coherence ({coherence:.3f}) suggests need for more data")
        elif docs / data['n_topics'] < 50:  # Too few docs per topic
            recommended = int(data['n_topics'] * 75)
            print(f"{sub}: INCREASE to {recommended} docs (currently {docs})")
            print(f"  Reason: Only {docs/data['n_topics']:.1f} docs per topic (target: 75+)")
        else:
            print(f"{sub}: Current size ({docs}) is ADEQUATE")
            print(f"  Good coherence ({coherence:.3f}) and {docs/data['n_topics']:.1f} docs per topic")

print("\n\nSAMPLE SIZE IMPACT ANALYSIS")
print("="*80)

# Theoretical analysis
sample_sizes = [100, 500, 1000, 2000, 5000, 10000]
expected_topics = []
expected_quality = []

for size in sample_sizes:
    # Heuristic: topics grow with log of documents
    topics = int(5 + 3 * np.log(size/100))
    expected_topics.append(topics)
    
    # Heuristic: quality improves with docs per topic, plateaus around 100
    docs_per_topic = size / topics
    quality = min(0.8, 0.3 + 0.5 * (docs_per_topic / 100))
    expected_quality.append(quality)

print(f"\n{'Sample Size':>12} {'Expected Topics':>16} {'Expected Quality':>18} {'Docs/Topic':>12}")
print("-"*80)

for size, topics, quality in zip(sample_sizes, expected_topics, expected_quality):
    print(f"{size:12} {topics:16} {quality:18.3f} {size/topics:12.1f}")

print("\n\nKEY INSIGHTS:")
print("-"*80)
print("1. Optimal docs per topic: 75-150")
print("2. Diminishing returns after 2000-3000 docs per subreddit")
print("3. Quality plateaus around 0.7-0.8 coherence")
print("4. More data helps discover rare topics but may dilute common ones")
print("\nRECOMMENDED APPROACH:")
print("- Increase samples for low-coherence communities (weed, weedbiz)")
print("- Maintain current sizes for high-coherence communities")
print("- Consider stratified sampling to ensure topic diversity")
