#!/usr/bin/env python3
"""
create_gartner_matrix.py - Create a Gartner-style positioning matrix for cannabis subreddits
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

# Load the analysis results
with open('quick_results/analysis_summary_advanced.json', 'r') as f:
    results = json.load(f)

# Define positioning based on topic analysis
# X-axis: Knowledge Type (Personal/Experiential <-> Professional/Informational)
# Y-axis: Engagement Style (Transactional/Business <-> Community/Cultural)

positions = {
    'cannabis': {
        'x': 0,  # Center - mix of personal and professional
        'y': 0.5,  # Advocacy/civic engagement
        'label': 'r/cannabis',
        'desc': 'Policy & Advocacy'
    },
    'weed': {
        'x': -0.8,  # Personal/experiential
        'y': 0.3,  # Community focused
        'label': 'r/weed',
        'desc': 'Casual Culture'
    },
    'trees': {
        'x': -0.7,  # Personal/experiential
        'y': 0.7,  # Strong community
        'label': 'r/trees',
        'desc': 'Lifestyle Community'
    },
    'Marijuana': {
        'x': -0.2,  # Slightly personal
        'y': 0.2,  # Mixed engagement
        'label': 'r/Marijuana',
        'desc': 'General Discussion'
    },
    'weedstocks': {
        'x': 0.9,  # Professional/financial
        'y': -0.8,  # Transactional
        'label': 'r/weedstocks',
        'desc': 'Investment Focus'
    },
    'weedbiz': {
        'x': 0.8,  # Professional/business
        'y': -0.5,  # Business operations
        'label': 'r/weedbiz',
        'desc': 'Industry B2B'
    }
}

# Create the figure
fig, ax = plt.subplots(figsize=(12, 10))

# Set up the quadrants
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# Add quadrant labels
quadrant_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor='gray')
ax.text(-0.5, 0.85, "Personal\nCommunity", transform=ax.transData, fontsize=11,
        verticalalignment='top', bbox=quadrant_props, alpha=0.7)
ax.text(0.5, 0.85, "Professional\nCommunity", transform=ax.transData, fontsize=11,
        verticalalignment='top', bbox=quadrant_props, alpha=0.7)
ax.text(-0.5, -0.85, "Personal\nTransactional", transform=ax.transData, fontsize=11,
        verticalalignment='bottom', bbox=quadrant_props, alpha=0.7)
ax.text(0.5, -0.85, "Professional\nTransactional", transform=ax.transData, fontsize=11,
        verticalalignment='bottom', bbox=quadrant_props, alpha=0.7)

# Plot each subreddit
for sub, pos in positions.items():
    if sub in results and 'error' not in results[sub]:
        # Size based on number of documents
        size = results[sub].get('n_documents', 100)
        # Color based on topic coherence
        coherence = results[sub].get('avg_coherence', 0.5)
        
        # Plot the bubble
        scatter = ax.scatter(pos['x'], pos['y'], 
                           s=size, 
                           c=coherence, 
                           cmap='RdYlGn',
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=2,
                           vmin=0.2, vmax=0.8)
        
        # Add label
        ax.annotate(pos['label'], 
                   (pos['x'], pos['y']), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=12,
                   fontweight='bold')
        
        # Add description
        ax.annotate(pos['desc'], 
                   (pos['x'], pos['y']), 
                   xytext=(5, -15), 
                   textcoords='offset points',
                   fontsize=10,
                   style='italic',
                   alpha=0.8)

# Set labels and title
ax.set_xlabel('Knowledge Type\n← Personal/Experiential    |    Professional/Informational →', 
              fontsize=14, fontweight='bold')
ax.set_ylabel('Engagement Style\n← Transactional/Business    |    Community/Cultural →', 
              fontsize=14, fontweight='bold')
ax.set_title('Cannabis Subreddit Positioning Matrix\n(Size = Document Count, Color = Topic Quality)', 
             fontsize=16, fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

# Add colorbar for coherence
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Topic Coherence Score', fontsize=12)

# Add grid
ax.grid(True, alpha=0.2)

# Add note about data
note_text = f"Based on analysis of {sum(r.get('n_documents', 0) for r in results.values() if 'error' not in r):,} documents across 6 subreddits"
ax.text(0.5, -1.25, note_text, transform=ax.transData, 
        fontsize=10, ha='center', alpha=0.7)

plt.tight_layout()
plt.savefig('quick_results/cannabis_gartner_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('quick_results/cannabis_gartner_matrix.pdf', bbox_inches='tight')
plt.show()

print("✅ Saved Gartner matrix to quick_results/")
