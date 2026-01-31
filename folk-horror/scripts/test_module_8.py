"""
Test script for Module 8: JSON Export Functions

Run this after opening movie_insights.ipynb to test the export functionality
"""

# Import the export functions
import sys
sys.path.append('/Users/jamesroot/Desktop/JAMES/Noetheca/Reviews/scripts')
from module_8_export import *

print("\n" + "="*80)
print("MODULE 8 TEST: JSON Export Functions")
print("="*80 + "\n")

# Test 1: Export single movie
print("TEST 1: Export single movie (The Witch)")
print("-" * 80)
result = quick_export("The Witch", output_dir='/Users/jamesroot/Desktop/JAMES/Noetheca/Reviews/insights')

# Test 2: Check if file was created
import os
test_file = '/Users/jamesroot/Desktop/JAMES/Noetheca/Reviews/insights/the_witch.json'
if os.path.exists(test_file):
    print(f"\n✅ SUCCESS: File created at {test_file}")
    
    # Show file size
    file_size = os.path.getsize(test_file) / 1024
    print(f"📊 File size: {file_size:.1f} KB")
    
    # Show structure
    import json
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n📋 Data structure:")
    print(f"   Movie: {data.get('movie')}")
    print(f"   Analysis date: {data.get('analysis_date')}")
    print(f"   Modules: {list(data.get('modules', {}).keys())}")
    
    # Show sample from Module 1
    if 'audience_breakdown' in data.get('modules', {}):
        mod1 = data['modules']['audience_breakdown']
        print(f"\n   Sample (Module 1 - Audience Breakdown):")
        print(f"      Total reviews: {mod1.get('total_reviews')}")
        print(f"      Avg rating: {mod1.get('avg_rating'):.2f}")
        print(f"      Lovers: {mod1.get('rating_segments', {}).get('lovers_8_10')}")
else:
    print(f"\n❌ ERROR: File not created at {test_file}")

print("\n" + "="*80)
print("Test complete!")
print("="*80 + "\n")
