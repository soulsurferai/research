#!/usr/bin/env python3
"""
create_flexible_emotion_visualizations.py - Create visualizations for any emotion analysis

Automatically detects and visualizes any emotion analysis output files.
"""

import json
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import OUTPUT_DIR
from scripts.experiments.create_emotion_visualizations import (
    RadarAxes, register_projection, create_emotion_heatmap,
    create_emotion_comparison_bars
)

# Register radar projection
register_projection(RadarAxes)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


def find_emotion_analyses():
    """Find all emotion analysis files in the output directory"""
    pattern = os.path.join(OUTPUT_DIR, '*_emotion_analysis.json')
    files = glob.glob(pattern)
    
    # Extract analysis info
    analyses = []
    for file in files:
        basename = os.path.basename(file)
        name = basename.replace('_emotion_analysis.json', '')
        
        # Load to get subreddit list
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                analyses.append({
                    'name': name,
                    'file': file,
                    'subreddits': data.get('subreddits', []),
                    'comparison_name': data.get('comparison_name', name)
                })
        except:
            continue
    
    return analyses


def create_flexible_radar_chart(data, output_prefix):
    """Create radar chart for any number of subreddits"""
    radar_data = data['radar_data']
    colors = data.get('visualization_config', {}).get('radar', {}).get('colors', {})
    
    # Determine layout based on number of subreddits
    n_subs = len(radar_data['subreddits'])
    
    if n_subs <= 6:
        # Use individual subplots
        cols = 3
        rows = (n_subs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), 
                                subplot_kw=dict(projection='radar'))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
    else:
        # Use single combined chart
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), subplot_kw=dict(projection='radar'))
        axes = [ax]
    
    # Emotion dimension labels
    dimension_labels = [
        'Hope\n(vs Despair)',
        'Trust\n(vs Suspicion)',
        'Excitement\n(vs Anxiety)',
        'Pride\n(vs Shame)',
        'Empowerment\n(vs Helplessness)',
        'Community\n(vs Isolation)',
        'Clarity\n(vs Confusion)'
    ]
    
    if n_subs <= 6:
        # Individual charts
        for idx, (subreddit, values) in enumerate(radar_data['subreddits'].items()):
            if idx < len(axes):
                ax = axes[idx]
                
                # Plot the data
                angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
                values_plot = values + values[:1]
                angles += angles[:1]
                
                color = colors.get(subreddit, f'C{idx}')
                ax.plot(angles, values_plot, 'o-', linewidth=2, color=color)
                ax.fill(angles, values_plot, alpha=0.25, color=color)
                
                ax.set_ylim(0, 1)
                ax.set_title(f'r/{subreddit}', size=16, weight='bold', pad=20)
                ax.set_varlabels(dimension_labels)
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_subs, len(axes)):
            axes[idx].set_visible(False)
    else:
        # Combined chart for many subreddits
        ax = axes[0]
        
        for idx, (subreddit, values) in enumerate(radar_data['subreddits'].items()):
            angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
            values_plot = values + values[:1]
            angles += angles[:1]
            
            color = colors.get(subreddit, f'C{idx % 10}')
            ax.plot(angles, values_plot, 'o-', linewidth=2, color=color, 
                   label=f'r/{subreddit}', alpha=0.7)
            ax.fill(angles, values_plot, alpha=0.1, color=color)
        
        ax.set_ylim(0, 1)
        ax.set_varlabels(dimension_labels)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.3, 1), loc='upper left', borderaxespad=0)
    
    # Title
    comparison_name = data.get('comparison_name', 'Subreddit')
    fig.suptitle(f'Emotional Landscapes: {comparison_name.replace("_", " ").title()}', 
                size=20, weight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_radar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved radar chart to {output_path}")
    plt.close()


def create_flexible_heatmap(data, output_prefix):
    """Create heatmap for any emotion analysis"""
    # Create matrix
    matrix_data = data['comparison_matrix']
    df = pd.DataFrame(matrix_data)
    
    # Select only the balance columns
    balance_cols = ['hope_despair', 'trust_suspicion', 'excitement_anxiety', 
                   'pride_shame', 'empowerment_helplessness', 'community_isolation', 
                   'clarity_confusion']
    
    # Create the matrix for heatmap
    heatmap_data = df.set_index('subreddit')[balance_cols].T
    
    # Rename dimensions
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
    
    # Adjust figure size based on number of subreddits
    n_subs = len(heatmap_data.columns)
    fig_width = max(12, n_subs * 1.5)
    
    plt.figure(figsize=(fig_width, 8))
    
    # Create colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#d32f2f', '#ff6659', '#ffeb3b', '#8bc34a', '#2e7d32']
    cmap = LinearSegmentedColormap.from_list('emotion', colors_list, N=100)
    
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
    
    comparison_name = data.get('comparison_name', 'Communities')
    plt.title(f'Emotional Balance: {comparison_name.replace("_", " ").title()}', 
             size=18, weight='bold', pad=20)
    plt.xlabel('Subreddit', size=14)
    plt.ylabel('Emotional Dimension', size=14)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved heatmap to {output_path}")
    plt.close()


def create_emotion_profile_summary(data, output_prefix):
    """Create a summary visualization with key insights"""
    fig = plt.figure(figsize=(16, 10))
    
    # Calculate statistics
    all_balances = []
    emotion_stats = {}
    
    for sub, results in data['analysis_results'].items():
        if 'dimensions' in results:
            for dim, dim_data in results['dimensions'].items():
                balance = dim_data['balance']
                all_balances.append(balance)
                
                if dim not in emotion_stats:
                    emotion_stats[dim] = []
                emotion_stats[dim].append((sub, balance))
    
    # Sort subreddits by overall emotional positivity
    sub_scores = {}
    for sub, results in data['analysis_results'].items():
        if 'dimensions' in results:
            scores = [d['balance'] for d in results['dimensions'].values()]
            sub_scores[sub] = np.mean(scores)
    
    sorted_subs = sorted(sub_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create text summary
    summary_text = f"""EMOTIONAL LANDSCAPE ANALYSIS SUMMARY
{'-' * 50}
Comparison: {data.get('comparison_name', 'Subreddits').replace('_', ' ').title()}
Subreddits Analyzed: {len(data['subreddits'])}
Total Documents: {sum(r.get('num_docs', 0) for r in data['analysis_results'].values())}

EMOTIONAL RANKINGS (Most Positive → Most Negative):
"""
    
    for i, (sub, score) in enumerate(sorted_subs, 1):
        summary_text += f"\n{i}. r/{sub}: Overall Balance = {score:.2f}"
        
        # Find strongest emotions for this sub
        if sub in data['analysis_results'] and 'dimensions' in data['analysis_results'][sub]:
            strong_emotions = []
            for dim, dim_data in data['analysis_results'][sub]['dimensions'].items():
                if abs(dim_data['balance']) > 0.5:
                    emotion = dim_data['dominant']
                    strong_emotions.append(emotion.title())
            
            if strong_emotions:
                summary_text += f"\n   Strong emotions: {', '.join(strong_emotions)}"
    
    # Add dimension insights
    summary_text += "\n\nKEY EMOTIONAL PATTERNS:"
    
    for dim in emotion_stats:
        sorted_dim = sorted(emotion_stats[dim], key=lambda x: x[1], reverse=True)
        most_positive = sorted_dim[0]
        most_negative = sorted_dim[-1]
        
        dim_name = dim.replace('_', ' vs ').title()
        summary_text += f"\n\n{dim_name}:"
        summary_text += f"\n  Most {dim.split('_')[0]}: r/{most_positive[0]} ({most_positive[1]:.2f})"
        summary_text += f"\n  Most {dim.split('_')[1]}: r/{most_negative[0]} ({most_negative[1]:.2f})"
    
    # Add recommendations
    summary_text += "\n\nSTRATEGIC INSIGHTS:"
    summary_text += "\n• Communities with high hope scores are ideal for positive messaging"
    summary_text += "\n• Address anxiety in communities showing negative excitement balance"
    summary_text += "\n• Leverage community feelings to build connections"
    summary_text += "\n• Use clear information to combat confusion"
    
    # Create the plot
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved summary to {output_path}")
    plt.close()


def create_all_visualizations(analysis_file):
    """Create all visualizations for a given analysis file"""
    # Load data
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    output_prefix = data.get('comparison_name', 'emotion_analysis')
    
    print(f"\n🎨 Creating visualizations for: {output_prefix}")
    print(f"   Subreddits: {', '.join(['r/' + s for s in data['subreddits']])}")
    
    try:
        # Create visualizations
        create_flexible_radar_chart(data, output_prefix)
        create_flexible_heatmap(data, output_prefix)
        create_emotion_profile_summary(data, output_prefix)
        
        # Also create comparison bars if not too many subreddits
        if len(data['subreddits']) <= 8:
            # Reuse the existing function but save with custom prefix
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # We'll modify the global OUTPUT_DIR temporarily
            original_data_file = os.path.join(OUTPUT_DIR, 'emotion_landscape_analysis.json')
            
            # Save our data temporarily with expected name
            temp_saved = False
            if not os.path.exists(original_data_file):
                with open(original_data_file, 'w') as f:
                    json.dump(data, f)
                temp_saved = True
            
            try:
                create_emotion_comparison_bars()
                # Rename the output
                default_path = os.path.join(OUTPUT_DIR, 'emotion_comparison_bars.png')
                if os.path.exists(default_path):
                    custom_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_comparison_bars.png')
                    os.rename(default_path, custom_path)
                    print(f"✅ Saved comparison bars to {custom_path}")
            finally:
                if temp_saved:
                    os.remove(original_data_file)
        
        print(f"\n✨ All visualizations created for {output_prefix}!")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Create emotion visualizations for any analysis'
    )
    
    parser.add_argument(
        '--analysis',
        help='Specific analysis file to visualize (without path)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Create visualizations for all emotion analyses'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available emotion analyses'
    )
    
    args = parser.parse_args()
    
    # Find available analyses
    analyses = find_emotion_analyses()
    
    if args.list or (not args.analysis and not args.all):
        print("📊 Available Emotion Analyses:")
        print("-" * 50)
        
        for i, analysis in enumerate(analyses):
            print(f"\n{i+1}. {analysis['comparison_name']}")
            print(f"   File: {os.path.basename(analysis['file'])}")
            print(f"   Subreddits: {', '.join(['r/' + s for s in analysis['subreddits']])}")
        
        if not args.list:
            print("\nUse --analysis <filename> to visualize a specific analysis")
            print("Use --all to visualize all analyses")
        return
    
    # Process requested analyses
    if args.all:
        print(f"\n🎨 Creating visualizations for all {len(analyses)} analyses...")
        for analysis in analyses:
            create_all_visualizations(analysis['file'])
    
    elif args.analysis:
        # Find the matching analysis
        found = False
        for analysis in analyses:
            if args.analysis in os.path.basename(analysis['file']):
                create_all_visualizations(analysis['file'])
                found = True
                break
        
        if not found:
            print(f"❌ Analysis file not found: {args.analysis}")
            print("Use --list to see available analyses")


if __name__ == "__main__":
    main()
