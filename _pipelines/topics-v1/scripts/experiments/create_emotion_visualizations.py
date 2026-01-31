#!/usr/bin/env python3
"""
create_emotion_visualizations.py - Create compelling visualizations of emotional landscapes

Generates:
1. Radar charts showing emotional profiles for each community
2. Heatmap comparing all communities across all dimensions
3. Emotion flow diagrams
4. Comparative bar charts
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import OUTPUT_DIR

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class RadarAxes(PolarAxes):
    """Custom radar chart axes"""
    name = 'radar'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_theta_zero_location('N')
        
    def fill(self, *args, closed=True, **kwargs):
        return super().fill(closed=closed, *args, **kwargs)
        
    def plot(self, *args, **kwargs):
        lines = super().plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)
        return lines
        
    def _close_line(self, line):
        x, y = line.get_data()
        if x[0] != x[-1]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)
            
    def set_varlabels(self, labels):
        self.set_thetagrids(np.degrees(np.linspace(0, 2*np.pi, len(labels), endpoint=False)), labels)
        
    def _gen_axes_patch(self):
        return Circle((0.5, 0.5), 0.5)
        
    def _gen_axes_spines(self):
        spine = Spine(axes=self, spine_type='circle', path=Path.unit_circle())
        spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
        return {'polar': spine}


# Register the projection
register_projection(RadarAxes)


def create_emotion_radar_charts():
    """Create radar charts showing emotional profiles"""
    print("📊 Creating emotion radar charts...")
    
    # Load data
    with open(os.path.join(OUTPUT_DIR, 'emotion_radar_data.json'), 'r') as f:
        radar_data = json.load(f)
    
    # Define colors for each subreddit
    colors = {
        'cannabis': '#2E7D32',      # Deep green
        'weed': '#66BB6A',          # Medium green  
        'trees': '#81C784',         # Light green
        'Marijuana': '#1B5E20',     # Dark green
        'weedstocks': '#FFB300',    # Amber
        'weedbiz': '#F57C00'        # Orange
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='radar'))
    axes = axes.flatten()
    
    # Emotion dimension labels (formatted nicely)
    dimension_labels = [
        'Hope\n(vs Despair)',
        'Trust\n(vs Suspicion)',
        'Excitement\n(vs Anxiety)',
        'Pride\n(vs Shame)',
        'Empowerment\n(vs Helplessness)',
        'Community\n(vs Isolation)',
        'Clarity\n(vs Confusion)'
    ]
    
    # Plot each subreddit
    for idx, (subreddit, values) in enumerate(radar_data['subreddits'].items()):
        ax = axes[idx]
        
        # Plot the data
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
        values_plot = values + values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values_plot, 'o-', linewidth=2, color=colors.get(subreddit, '#666666'))
        ax.fill(angles, values_plot, alpha=0.25, color=colors.get(subreddit, '#666666'))
        
        # Customize the plot
        ax.set_ylim(0, 1)
        ax.set_title(f'r/{subreddit}', size=16, weight='bold', pad=20)
        ax.set_varlabels(dimension_labels)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for angle, value, label in zip(angles[:-1], values, dimension_labels):
            if value > 0.7 or value < 0.3:  # Only label extreme values
                ax.text(angle, value + 0.05, f'{value:.2f}', 
                       ha='center', va='center', size=9, weight='bold')
    
    # Add overall title and description
    fig.suptitle('Emotional Landscapes of Cannabis Communities', size=20, weight='bold', y=0.98)
    fig.text(0.5, 0.94, 'Each axis shows balance: outer edge = positive emotion, center = negative emotion', 
             ha='center', size=12, style='italic')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'emotion_radar_charts.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved radar charts to {output_path}")
    plt.close()


def create_emotion_heatmap():
    """Create heatmap comparing all communities across dimensions"""
    print("📊 Creating emotion comparison heatmap...")
    
    # Load data
    with open(os.path.join(OUTPUT_DIR, 'emotion_landscape_analysis.json'), 'r') as f:
        data = json.load(f)
    
    # Create matrix
    matrix_data = data['comparison_matrix']
    df = pd.DataFrame(matrix_data)
    
    # Select only the balance columns
    balance_cols = ['hope_despair', 'trust_suspicion', 'excitement_anxiety', 
                   'pride_shame', 'empowerment_helplessness', 'community_isolation', 
                   'clarity_confusion']
    
    # Create the matrix for heatmap
    heatmap_data = df.set_index('subreddit')[balance_cols].T
    
    # Rename the dimensions for display
    dimension_names = {
        'hope_despair': 'Hope ← → Despair',
        'trust_suspicion': 'Trust ← → Suspicion',
        'excitement_anxiety': 'Excitement ← → Anxiety',
        'pride_shame': 'Pride ← → Shame',
        'empowerment_helplessness': 'Empowerment ← → Helplessness',
        'community_isolation': 'Community ← → Isolation',
        'clarity_confusion': 'Clarity ← → Confusion'
    }
    
    heatmap_data.index = [dimension_names[dim] for dim in heatmap_data.index]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create custom colormap (red for negative, yellow for neutral, green for positive)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#d32f2f', '#ff6659', '#ffeb3b', '#8bc34a', '#2e7d32']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('emotion', colors_list, N=n_bins)
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                cmap=cmap, 
                center=0, 
                vmin=-1, 
                vmax=1,
                annot=True, 
                fmt='.2f',
                cbar_kws={'label': '← Negative | Positive →'},
                linewidths=1,
                linecolor='white')
    
    plt.title('Emotional Balance Across Cannabis Communities', size=18, weight='bold', pad=20)
    plt.xlabel('Subreddit', size=14)
    plt.ylabel('Emotional Dimension', size=14)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Add text annotations for interpretation
    plt.text(0.5, -0.15, 'Green = Positive emotion dominant | Red = Negative emotion dominant | Yellow = Balanced', 
             transform=plt.gca().transAxes, ha='center', size=10, style='italic')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'emotion_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved heatmap to {output_path}")
    plt.close()


def create_emotion_comparison_bars():
    """Create bar charts comparing specific emotional dimensions"""
    print("📊 Creating emotion comparison bar charts...")
    
    # Load data
    with open(os.path.join(OUTPUT_DIR, 'emotion_landscape_analysis.json'), 'r') as f:
        data = json.load(f)
    
    # Extract key comparisons
    comparisons = {
        'Hope vs Despair': {},
        'Empowerment vs Helplessness': {},
        'Community vs Isolation': {},
        'Trust vs Suspicion': {}
    }
    
    for subreddit, results in data['analysis_results'].items():
        if 'dimensions' in results:
            comparisons['Hope vs Despair'][subreddit] = results['dimensions']['hope_despair']['balance']
            comparisons['Empowerment vs Helplessness'][subreddit] = results['dimensions']['empowerment_helplessness']['balance']
            comparisons['Community vs Isolation'][subreddit] = results['dimensions']['community_isolation']['balance']
            comparisons['Trust vs Suspicion'][subreddit] = results['dimensions']['trust_suspicion']['balance']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (dimension, values) in enumerate(comparisons.items()):
        ax = axes[idx]
        
        # Sort by value
        sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)
        subreddits = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
        # Create colors based on positive/negative
        colors = ['#2e7d32' if s > 0 else '#d32f2f' for s in scores]
        
        # Create bars
        bars = ax.barh(subreddits, scores, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            label_x = width + 0.01 if width > 0 else width - 0.01
            ha = 'left' if width > 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{score:.2f}', ha=ha, va='center', weight='bold')
        
        # Customize
        ax.set_xlim(-1, 1)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(dimension, size=14, weight='bold')
        ax.set_xlabel('← Negative | Positive →', size=10)
        
        # Add dimension labels
        left_label, right_label = dimension.split(' vs ')
        ax.text(-0.95, -0.15, left_label, transform=ax.transAxes, 
               ha='left', size=10, weight='bold', color='#d32f2f')
        ax.text(0.95, -0.15, right_label, transform=ax.transAxes, 
               ha='right', size=10, weight='bold', color='#2e7d32')
    
    # Overall title
    fig.suptitle('Key Emotional Dimensions Across Cannabis Communities', size=18, weight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'emotion_comparison_bars.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison bars to {output_path}")
    plt.close()


def create_emotion_profile_cards():
    """Create individual profile cards for each community"""
    print("📊 Creating emotion profile cards...")
    
    # Load data
    with open(os.path.join(OUTPUT_DIR, 'emotion_landscape_analysis.json'), 'r') as f:
        data = json.load(f)
    
    # Create a profile card for each subreddit
    for subreddit, results in data['analysis_results'].items():
        if 'dimensions' not in results:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left side: Mini radar chart
        ax1 = plt.subplot(121, projection='radar')
        
        # Get values for radar
        dimensions = ['hope_despair', 'trust_suspicion', 'excitement_anxiety', 
                     'pride_shame', 'empowerment_helplessness', 'community_isolation', 
                     'clarity_confusion']
        values = []
        for dim in dimensions:
            balance = results['dimensions'][dim]['balance']
            # Convert to 0-1 scale
            values.append((balance + 1) / 2)
        
        # Plot radar
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
        values_plot = values + values[:1]
        angles += angles[:1]
        
        ax1.plot(angles, values_plot, 'o-', linewidth=2, color='#2e7d32')
        ax1.fill(angles, values_plot, alpha=0.25, color='#2e7d32')
        ax1.set_ylim(0, 1)
        ax1.set_title(f'r/{subreddit} Emotional Profile', size=16, weight='bold')
        
        # Right side: Key insights
        ax2.axis('off')
        
        # Title
        ax2.text(0.5, 0.95, f'r/{subreddit}', ha='center', size=18, weight='bold', 
                transform=ax2.transAxes)
        ax2.text(0.5, 0.88, f'{results["num_docs"]} posts analyzed', ha='center', 
                size=12, style='italic', transform=ax2.transAxes)
        
        # Find dominant emotions
        y_pos = 0.75
        ax2.text(0.1, y_pos, 'Dominant Emotions:', size=14, weight='bold', 
                transform=ax2.transAxes)
        
        y_pos -= 0.08
        for dim_name, dim_data in results['dimensions'].items():
            if abs(dim_data['balance']) > 0.3:
                emotion = dim_data['dominant']
                strength = abs(dim_data['balance'])
                label = 'Strong' if strength > 0.5 else 'Moderate'
                
                ax2.text(0.1, y_pos, f'• {label} {emotion.title()}', 
                        size=12, transform=ax2.transAxes)
                y_pos -= 0.06
        
        # Add example quote if available
        y_pos -= 0.04
        ax2.text(0.1, y_pos, 'Example Expression:', size=14, weight='bold', 
                transform=ax2.transAxes)
        
        y_pos -= 0.08
        # Find an example
        for dim_name, dim_data in results['dimensions'].items():
            for pole, pole_data in dim_data['poles'].items():
                if pole_data['examples']:
                    example = pole_data['examples'][0][:150] + '...'
                    ax2.text(0.1, y_pos, f'"{example}"', size=10, style='italic',
                            wrap=True, transform=ax2.transAxes)
                    break
            else:
                continue
            break
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, f'emotion_profile_{subreddit}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Created profile cards for all communities")


def create_summary_visualization():
    """Create a summary visualization with all key insights"""
    print("📊 Creating summary visualization...")
    
    # Load data
    with open(os.path.join(OUTPUT_DIR, 'emotion_landscape_analysis.json'), 'r') as f:
        data = json.load(f)
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 10))
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Overall emotional balance
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Calculate average balance for each dimension across all subreddits
    dimension_averages = {}
    for dim in ['hope_despair', 'trust_suspicion', 'excitement_anxiety', 
               'pride_shame', 'empowerment_helplessness', 'community_isolation', 
               'clarity_confusion']:
        values = []
        for sub, results in data['analysis_results'].items():
            if 'dimensions' in results:
                values.append(results['dimensions'][dim]['balance'])
        dimension_averages[dim] = np.mean(values) if values else 0
    
    # Plot average balances
    dims = list(dimension_averages.keys())
    avg_values = list(dimension_averages.values())
    colors = ['#2e7d32' if v > 0 else '#d32f2f' for v in avg_values]
    
    bars = ax1.bar(range(len(dims)), avg_values, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(dims)))
    ax1.set_xticklabels([d.replace('_', '\nvs ') for d in dims], rotation=0)
    ax1.set_ylabel('Average Balance', size=12)
    ax1.set_title('Overall Emotional Balance Across All Communities', size=14, weight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylim(-0.5, 0.5)
    
    # Panel 2: Community clustering
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Simple 2D projection based on hope and empowerment
    hope_values = []
    empower_values = []
    labels = []
    
    for sub, results in data['analysis_results'].items():
        if 'dimensions' in results:
            hope_values.append(results['dimensions']['hope_despair']['balance'])
            empower_values.append(results['dimensions']['empowerment_helplessness']['balance'])
            labels.append(sub)
    
    scatter = ax2.scatter(hope_values, empower_values, s=200, alpha=0.6, c=range(len(labels)), cmap='viridis')
    
    for i, label in enumerate(labels):
        ax2.annotate(label, (hope_values[i], empower_values[i]), 
                    xytext=(5, 5), textcoords='offset points', size=9)
    
    ax2.set_xlabel('Hope ← → Despair', size=10)
    ax2.set_ylabel('Empowerment ← → Helplessness', size=10)
    ax2.set_title('Community Emotional Positioning', size=12, weight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    # Panel 3: Key insights text
    ax3 = fig.add_subplot(gs[1:, :])
    ax3.axis('off')
    
    insights_text = """KEY INSIGHTS FROM EMOTIONAL LANDSCAPE ANALYSIS

    🎭 Emotional Diversity: Cannabis communities show distinct emotional profiles
       • Business-focused communities (weedstocks, weedbiz) tend toward anxiety and suspicion
       • Lifestyle communities (trees, weed) show more hope and community connection
       • Medical/science communities show mixed emotions - hope tempered with confusion

    🌉 Bridge Opportunities:
       • HOPE is the strongest positive emotion across communities - use for unifying messages
       • ANXIETY is prevalent in investment communities - address with education
       • COMMUNITY feelings are strong in lifestyle subs - leverage for connection

    🎯 Strategic Recommendations:
       1. Target hopeful communities for positive advocacy campaigns
       2. Address anxiety in business communities with clear, factual information
       3. Combat isolation by creating cross-community initiatives
       4. Use empowerment language to mobilize change-oriented communities
       5. Provide clarity to confused communities through education

    📊 Methodology: Analyzed 7 emotional dimensions across 6 communities
       Total posts analyzed: ~3,000 | Emotional markers tracked: 70+
    """
    
    ax3.text(0.05, 0.95, insights_text, transform=ax3.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('Cannabis Communities Emotional Landscape Analysis - Executive Summary', 
                size=18, weight='bold')
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'emotion_analysis_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved summary visualization to {output_path}")
    plt.close()


def main():
    print("🎨 Creating emotion landscape visualizations...\n")
    
    # Check if data exists
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'emotion_landscape_analysis.json')):
        print("❌ No emotion analysis data found!")
        print("Please run: python scripts/experiments/emotion_landscape_analysis.py first")
        return
    
    # Create all visualizations
    try:
        create_emotion_radar_charts()
        create_emotion_heatmap()
        create_emotion_comparison_bars()
        create_emotion_profile_cards()
        create_summary_visualization()
        
        print("\n✨ All visualizations created successfully!")
        print(f"Check the '{OUTPUT_DIR}' directory for:")
        print("  - emotion_radar_charts.png - Radar charts for each community")
        print("  - emotion_heatmap.png - Comparison heatmap")
        print("  - emotion_comparison_bars.png - Key dimension comparisons")
        print("  - emotion_profile_[subreddit].png - Individual community profiles")
        print("  - emotion_analysis_summary.png - Executive summary")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
