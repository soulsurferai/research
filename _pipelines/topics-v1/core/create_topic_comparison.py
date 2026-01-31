#!/usr/bin/env python3
"""
create_topic_comparison.py - Compare topics across cannabis subreddits
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import Counter

# Load all topic data
with open('quick_results/analysis_summary_advanced.json', 'r') as f:
    results = json.load(f)

# Extract topic themes across all subreddits
topic_themes = []
for sub, data in results.items():
    if 'topics' in data:
        for topic in data['topics']:
            topic_themes.append({
                'subreddit': sub,
                'topic': topic['label'],
                'size': topic['size'],
                'coherence': topic['coherence']
            })

df = pd.DataFrame(topic_themes)

# Categorize topics into broader themes
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

# Apply theme mapping
df['theme'] = df['topic'].map(lambda x: theme_map.get(x, 'Other'))

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# 1. Stacked bar chart of themes by subreddit
theme_pivot = df.groupby(['subreddit', 'theme'])['size'].sum().unstack(fill_value=0)
theme_pivot = theme_pivot.div(theme_pivot.sum(axis=1), axis=0) * 100

# Order subreddits by knowledge axis
subreddit_order = ['trees', 'weed', 'Marijuana', 'cannabis', 'weedbiz', 'weedstocks']
theme_pivot = theme_pivot.reindex(subreddit_order)

theme_pivot.plot(kind='bar', stacked=True, ax=ax1, 
                colormap='tab10', width=0.8)
ax1.set_title('Topic Theme Distribution by Subreddit', fontsize=16, fontweight='bold')
ax1.set_xlabel('Subreddit', fontsize=12)
ax1.set_ylabel('Percentage of Discussion', fontsize=12)
ax1.legend(title='Theme', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# 2. Topic quality scatter plot
ax2.scatter(df['size'], df['coherence'], 
           c=df['subreddit'].astype('category').cat.codes,
           s=100, alpha=0.6, cmap='tab10')
ax2.set_xlabel('Topic Size (number of documents)', fontsize=12)
ax2.set_ylabel('Topic Coherence Score', fontsize=12)
ax2.set_title('Topic Quality vs Size Across Subreddits', fontsize=16, fontweight='bold')
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Good coherence threshold')
ax2.legend()

# Add coherence quality zones
ax2.axhspan(0, 0.3, alpha=0.1, color='red', label='Poor coherence')
ax2.axhspan(0.3, 0.5, alpha=0.1, color='yellow', label='Moderate coherence')
ax2.axhspan(0.5, 1.5, alpha=0.1, color='green', label='Good coherence')

plt.tight_layout()
plt.savefig('quick_results/topic_theme_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary statistics
print("\n" + "="*60)
print("TOPIC THEME SUMMARY")
print("="*60)
print(theme_pivot.round(1))

# Find dominant themes per subreddit
print("\n" + "="*60)
print("DOMINANT THEMES")
print("="*60)
for sub in subreddit_order:
    if sub in theme_pivot.index:
        dominant = theme_pivot.loc[sub].idxmax()
        percentage = theme_pivot.loc[sub].max()
        print(f"{sub:15} : {dominant:20} ({percentage:.1f}%)")
