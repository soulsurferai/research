#!/usr/bin/env python3
"""
analyze_other_topics.py - Investigate topics that don't fit main categories
"""

import json
import pandas as pd

# Load results
with open('quick_results/analysis_summary_advanced.json', 'r') as f:
    results = json.load(f)

# Theme mapping from our visualization
theme_map = {
    'Legalization Movement': 'Policy/Legal',
    'Federal Rescheduling': 'Policy/Legal',
    'Political Landscape': 'Policy/Legal',
    'State Politics': 'Policy/Legal',
    'Legal Issues': 'Policy/Legal',
    
    'Medical Cannabis': 'Medical/Health',
    'Cannabinoid Science': 'Medical/Health',
    'Therapeutic Use': 'Medical/Health',
    'Scientific Research': 'Medical/Health',
    
    'Investment & Markets': 'Business/Finance',
    'Stock Market': 'Business/Finance',
    'Financial Performance': 'Business/Finance',
    'Financial Health': 'Business/Finance',
    'Business & Retail': 'Business/Finance',
    'Industry News': 'Business/Finance',
    'Retail Operations': 'Business/Finance',
    
    'Consumption & Effects': 'Lifestyle/Culture',
    'Smoking Culture': 'Lifestyle/Culture',
    'Effects & Experiences': 'Lifestyle/Culture',
    'Positive Vibes': 'Lifestyle/Culture',
    'Community & Culture': 'Lifestyle/Culture',
    
    'Cultivation': 'Growing/Production',
    'Growing Methods': 'Growing/Production',
    'Genetics & Strains': 'Growing/Production',
}

print("TOPICS IN 'OTHER' CATEGORY")
print("="*80)

other_topics = []

for sub, data in results.items():
    if 'topics' in data:
        for topic in data['topics']:
            label = topic['label']
            # Check if it maps to a known theme
            if label not in theme_map and not label.startswith('Topic '):
                other_topics.append({
                    'subreddit': sub,
                    'label': label,
                    'size': topic['size'],
                    'words': topic['words'][:5],
                    'coherence': topic.get('coherence', 0)
                })

# Also check generic "Topic X" labels
generic_topics = []
for sub, data in results.items():
    if 'topics' in data:
        for topic in data['topics']:
            if topic['label'].startswith('Topic '):
                generic_topics.append({
                    'subreddit': sub,
                    'label': topic['label'],
                    'size': topic['size'],
                    'words': topic['words'],
                    'coherence': topic.get('coherence', 0)
                })

print("\nLabeled topics not in main categories:")
for t in other_topics:
    print(f"\n{t['subreddit']}: {t['label']} ({t['size']} docs, coherence: {t['coherence']:.3f})")
    print(f"  Words: {', '.join(t['words'])}")

print("\n\nGeneric 'Topic X' labels that need better categorization:")
for t in sorted(generic_topics, key=lambda x: x['size'], reverse=True)[:10]:
    print(f"\n{t['subreddit']}: {t['label']} ({t['size']} docs, coherence: {t['coherence']:.3f})")
    print(f"  Words: {', '.join(t['words'][:10])}")

# Suggest categories for generic topics
print("\n\nSUGGESTED CATEGORIZATIONS:")
print("="*80)

for t in generic_topics:
    words = ' '.join(t['words'][:5]).lower()
    suggested = None
    
    if any(w in words for w in ['test', 'drug test', 'detox', 'clean']):
        suggested = "Drug Testing"
    elif any(w in words for w in ['hemp', 'cbd', 'delta']):
        suggested = "Hemp/CBD Products"  
    elif any(w in words for w in ['state', 'texas', 'florida', 'california']):
        suggested = "State-Specific Issues"
    elif any(w in words for w in ['company', 'brand', 'product']):
        suggested = "Brand/Product Discussion"
    elif any(w in words for w in ['price', 'cost', 'expensive', 'cheap']):
        suggested = "Pricing/Economics"
    elif any(w in words for w in ['anxiety', 'depression', 'pain', 'sleep']):
        suggested = "Medical Conditions"
    elif any(w in words for w in ['job', 'work', 'career', 'hire']):
        suggested = "Employment"
    
    if suggested:
        print(f"{t['subreddit']} {t['label']}: → {suggested}")
        print(f"  Based on: {', '.join(t['words'][:5])}")
