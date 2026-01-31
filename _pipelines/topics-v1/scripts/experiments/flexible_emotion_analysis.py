#!/usr/bin/env python3
"""
flexible_emotion_analysis.py - Run emotion analysis on any set of subreddits

Usage:
    python flexible_emotion_analysis.py --subreddits cannabis trees marijuana
    python flexible_emotion_analysis.py --subreddits politics conservative liberal
    python flexible_emotion_analysis.py --from-file subreddit_list.txt
    python flexible_emotion_analysis.py --suggestion 1  # Use suggestion from discovery
"""

import argparse
import json
import os
import sys
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import OUTPUT_DIR
from scripts.experiments.emotion_landscape_analysis import (
    EmotionLandscapeAnalyzer, 
    create_visualization_config
)


def load_suggestions():
    """Load suggestions from discovery file if it exists"""
    suggestions_file = os.path.join(OUTPUT_DIR, 'available_subreddits.json')
    if os.path.exists(suggestions_file):
        with open(suggestions_file, 'r') as f:
            data = json.load(f)
            return data.get('suggestions', [])
    return []


def get_subreddits_from_args(args) -> List[str]:
    """Get list of subreddits from various input methods"""
    subreddits = []
    
    # From direct argument
    if args.subreddits:
        subreddits = args.subreddits
    
    # From file
    elif args.from_file:
        with open(args.from_file, 'r') as f:
            subreddits = [line.strip() for line in f if line.strip()]
    
    # From suggestion
    elif args.suggestion is not None:
        suggestions = load_suggestions()
        if 0 <= args.suggestion < len(suggestions):
            subreddits = suggestions[args.suggestion]['subreddits']
            print(f"Using suggestion: {suggestions[args.suggestion]['name']}")
            print(f"Description: {suggestions[args.suggestion]['description']}")
        else:
            print(f"❌ Invalid suggestion number. Available: 0-{len(suggestions)-1}")
            sys.exit(1)
    
    # Validate we have subreddits
    if not subreddits:
        print("❌ No subreddits specified!")
        print("Use --subreddits, --from-file, or --suggestion")
        sys.exit(1)
    
    return subreddits


def create_comparison_name(subreddits: List[str]) -> str:
    """Create a descriptive name for this comparison"""
    if len(subreddits) <= 3:
        return "_vs_".join(subreddits)
    else:
        return f"{subreddits[0]}_and_{len(subreddits)-1}_others"


def run_flexible_analysis(subreddits: List[str], output_prefix: str = None):
    """Run emotion analysis on specified subreddits"""
    print(f"🎭 Running Emotion Analysis on: {', '.join(['r/' + s for s in subreddits])}\n")
    
    # Create output prefix if not specified
    if not output_prefix:
        output_prefix = create_comparison_name(subreddits)
    
    # Initialize analyzer
    analyzer = EmotionLandscapeAnalyzer()
    all_results = {}
    
    # Analyze each subreddit
    for subreddit in subreddits:
        try:
            results = analyzer.analyze_subreddit_emotions(subreddit, n_samples=500)
            if results:
                all_results[subreddit] = results
                print(f"✅ Completed analysis for r/{subreddit}")
            else:
                print(f"⚠️  No data for r/{subreddit}")
        except Exception as e:
            print(f"❌ Error analyzing r/{subreddit}: {e}")
            continue
    
    if not all_results:
        print("\n❌ No successful analyses!")
        return
    
    # Create visualization data
    print("\n📊 Creating visualization data...")
    radar_data = analyzer.create_emotion_radar_data(all_results)
    comparison_matrix = analyzer.create_emotion_comparison_matrix(all_results)
    
    # Create visualization config with custom colors
    # Generate colors dynamically based on number of subreddits
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    n_subs = len(subreddits)
    colormap = cm.get_cmap('tab20' if n_subs <= 20 else 'hsv')
    colors = {}
    for i, sub in enumerate(subreddits):
        color = colormap(i / n_subs)
        colors[sub] = mcolors.to_hex(color)
    
    viz_config = create_visualization_config(radar_data, comparison_matrix)
    viz_config['radar']['colors'] = colors
    
    # Generate insights report
    print("📝 Generating insights report...")
    report = analyzer.generate_insights_report(all_results, radar_data, comparison_matrix)
    
    # Save all results with custom prefix
    output_data = {
        'comparison_name': output_prefix,
        'subreddits': subreddits,
        'analysis_results': all_results,
        'radar_data': radar_data,
        'comparison_matrix': comparison_matrix.to_dict('records'),
        'visualization_config': viz_config
    }
    
    # Save JSON data
    json_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_emotion_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Saved analysis data to {json_path}")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_emotion_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Saved report to {report_path}")
    
    # Save visualization data
    radar_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_radar_data.json')
    with open(radar_path, 'w') as f:
        json.dump(radar_data, f, indent=2)
    
    # Create a summary of key findings
    print_analysis_summary(all_results, subreddits)
    
    return output_data


def print_analysis_summary(all_results, subreddits):
    """Print a summary of key emotional findings"""
    print("\n🎯 EMOTIONAL LANDSCAPE SUMMARY")
    print("=" * 60)
    
    # Find most extreme emotions across all dimensions
    dimension_extremes = {}
    
    for sub in subreddits:
        if sub not in all_results or 'dimensions' not in all_results[sub]:
            continue
            
        for dim, data in all_results[sub]['dimensions'].items():
            balance = data['balance']
            if dim not in dimension_extremes:
                dimension_extremes[dim] = {'most_positive': (sub, balance), 'most_negative': (sub, balance)}
            else:
                if balance > dimension_extremes[dim]['most_positive'][1]:
                    dimension_extremes[dim]['most_positive'] = (sub, balance)
                if balance < dimension_extremes[dim]['most_negative'][1]:
                    dimension_extremes[dim]['most_negative'] = (sub, balance)
    
    # Print extremes
    print("\n🏆 Emotional Champions:")
    for dim, extremes in dimension_extremes.items():
        dim_name = dim.replace('_', ' vs ')
        pos_sub, pos_score = extremes['most_positive']
        neg_sub, neg_score = extremes['most_negative']
        
        if pos_score > 0.3:
            print(f"  Most {dim_name.split(' vs ')[0]}: r/{pos_sub} ({pos_score:.2f})")
        if neg_score < -0.3:
            print(f"  Most {dim_name.split(' vs ')[1]}: r/{neg_sub} ({neg_score:.2f})")
    
    print("\n💡 Quick Insights:")
    print("  - Run visualization script to see emotional radar charts")
    print("  - Check report file for detailed analysis and examples")
    print(f"  - Comparing {len(subreddits)} communities across 7 emotional dimensions")


def main():
    parser = argparse.ArgumentParser(
        description='Run emotion analysis on any set of subreddits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific subreddits
  python flexible_emotion_analysis.py --subreddits politics conservative liberal

  # Use a suggestion from discovery
  python flexible_emotion_analysis.py --suggestion 0

  # Load from file (one subreddit per line)
  python flexible_emotion_analysis.py --from-file my_subreddits.txt

  # Custom output name
  python flexible_emotion_analysis.py --subreddits gaming pcmasterrace --output-prefix gaming_comparison
        """
    )
    
    # Input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--subreddits', 
        nargs='+', 
        help='List of subreddits to analyze'
    )
    input_group.add_argument(
        '--from-file', 
        help='File containing subreddits (one per line)'
    )
    input_group.add_argument(
        '--suggestion', 
        type=int, 
        help='Use a suggestion from discovery (0-based index)'
    )
    
    # Other options
    parser.add_argument(
        '--output-prefix',
        help='Custom prefix for output files (default: auto-generated)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=500,
        help='Number of samples per subreddit (default: 500)'
    )
    parser.add_argument(
        '--show-suggestions',
        action='store_true',
        help='Show available suggestions and exit'
    )
    
    args = parser.parse_args()
    
    # Show suggestions if requested
    if args.show_suggestions:
        suggestions = load_suggestions()
        if suggestions:
            print("📋 Available Suggestions:")
            for i, sugg in enumerate(suggestions):
                print(f"\n{i}. {sugg['name']}")
                print(f"   {sugg['description']}")
                print(f"   Subreddits: {', '.join(['r/' + s for s in sugg['subreddits']])}")
        else:
            print("No suggestions available. Run discover_available_subreddits.py first.")
        return
    
    # Get subreddits
    subreddits = get_subreddits_from_args(args)
    
    # Run analysis
    run_flexible_analysis(subreddits, args.output_prefix)
    
    print("\n✨ Analysis complete!")
    print("Next steps:")
    print("1. Run visualization script to create charts")
    print("2. Check the output files in the results directory")


if __name__ == "__main__":
    main()
