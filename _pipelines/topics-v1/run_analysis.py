#!/usr/bin/env python3
"""
run_analysis.py - Main entry point for cannabis subreddit topic analysis
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.topics_advanced import AdvancedTopicAnalyzer
from core.create_gartner_matrix import create_gartner_matrix
from core.create_topic_comparison import create_topic_comparison
from core.generate_insights_report import generate_insights_report
from config import CANNABIS_SUBREDDITS, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description='Analyze topics in cannabis-related subreddits'
    )
    
    parser.add_argument(
        '--subreddit',
        choices=CANNABIS_SUBREDDITS + ['all'],
        default='all',
        help='Which subreddit to analyze (default: all)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=500,
        help='Number of samples per subreddit (default: 500)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations after analysis'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate insights report after analysis'
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip analysis and only generate visualizations/reports from existing data'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run analysis unless skipped
    if not args.skip_analysis:
        analyzer = AdvancedTopicAnalyzer()
        
        if args.subreddit == 'all':
            print(f"🚀 Analyzing all {len(CANNABIS_SUBREDDITS)} cannabis subreddits...")
            for subreddit in CANNABIS_SUBREDDITS:
                try:
                    analyzer.analyze_subreddit(subreddit, n_samples=args.samples)
                except Exception as e:
                    print(f"❌ Error analyzing r/{subreddit}: {e}")
                    continue
        else:
            print(f"🔍 Analyzing r/{args.subreddit}...")
            analyzer.analyze_subreddit(args.subreddit, n_samples=args.samples)
    
    # Generate visualizations if requested
    if args.visualize:
        print("\n📊 Generating visualizations...")
        try:
            create_gartner_matrix()
            print("✅ Created Gartner matrix")
        except Exception as e:
            print(f"❌ Error creating Gartner matrix: {e}")
            
        try:
            create_topic_comparison()
            print("✅ Created topic comparison")
        except Exception as e:
            print(f"❌ Error creating topic comparison: {e}")
    
    # Generate report if requested
    if args.report:
        print("\n📝 Generating insights report...")
        try:
            generate_insights_report()
            print("✅ Generated insights report")
        except Exception as e:
            print(f"❌ Error generating report: {e}")
    
    print("\n✨ Analysis complete! Check the 'results' directory for outputs.")


if __name__ == "__main__":
    main()
