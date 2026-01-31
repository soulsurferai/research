"""
Add this cell to movie_insights.ipynb after Module 7 to integrate Module 8
"""

# ============================================================================
# MODULE 8: JSON EXPORT FUNCTIONS
# ============================================================================

import sys
sys.path.append('/Users/USER/Desktop/JAMES/Noetheca/Reviews/scripts')
from module_8_export import export_movie_analysis, export_all_movies, create_combined_export, quick_export

print("="*80)
print("MODULE 8: JSON Export Functions")
print("="*80)
print("\n✅ Functions loaded:")
print("   • export_movie_analysis(movie_name, output_dir)")
print("   • export_all_movies(output_dir)")
print("   • create_combined_export(output_dir, output_file)")
print("   • quick_export(movie_name)")
print("\n📁 Default output directory: ../insights/")
print("\n")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

print("📝 Usage Examples:\n")
print("# Export single movie:")
print("result = quick_export('The Witch')")
print("\n# Export all 10 movies:")
print("all_results = export_all_movies()")
print("\n# Create combined file:")
print("combined = create_combined_export()")
print("\n" + "="*80 + "\n")
