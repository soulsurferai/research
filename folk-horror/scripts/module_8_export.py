# MODULE 8: JSON EXPORT FUNCTIONS
# This module exports all analysis results to JSON files

import json
from pathlib import Path
from datetime import datetime

def export_movie_analysis(movie_name, output_dir='../insights'):
    """
    Run all analysis modules for a single movie and export to JSON
    
    Args:
        movie_name: Name of the movie to analyze
        output_dir: Directory to save JSON file (default: ../insights)
    
    Returns: Dictionary with all analysis results
    """
    
    print(f"\n{'='*80}")
    print(f"🎬 Analyzing: {movie_name}")
    print(f"{'='*80}\n")
    
    # Run all analysis modules
    results = {
        'movie': movie_name,
        'analysis_date': datetime.now().isoformat(),
        'modules': {}
    }
    
    # Module 1: Audience Breakdown
    print("📊 Module 1: Audience Breakdown...")
    try:
        results['modules']['audience_breakdown'] = audience_breakdown(movie_name)
        print("   ✅ Complete")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['modules']['audience_breakdown'] = {'error': str(e)}
    
    # Module 2: What Resonated
    print("❤️  Module 2: What Resonated...")
    try:
        results['modules']['what_resonated'] = what_resonated(movie_name)
        print("   ✅ Complete")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['modules']['what_resonated'] = {'error': str(e)}
    
    # Module 3: What Didn't Work
    print("💔 Module 3: What Didn't Work...")
    try:
        results['modules']['what_didnt_work'] = what_didnt_work(movie_name)
        print("   ✅ Complete")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['modules']['what_didnt_work'] = {'error': str(e)}
    
    # Module 4: Polarization Analysis
    print("⚡ Module 4: Polarization Analysis...")
    try:
        results['modules']['polarization'] = polarization_analysis(movie_name)
        print("   ✅ Complete")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['modules']['polarization'] = {'error': str(e)}
    
    # Module 5: Marketing Disconnect
    print("📊 Module 5: Marketing Disconnect...")
    try:
        results['modules']['marketing_disconnect'] = marketing_disconnect_analysis(movie_name)
        print("   ✅ Complete")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['modules']['marketing_disconnect'] = {'error': str(e)}
    
    # Module 6: Risk Factors
    print("⚠️  Module 6: Risk Factors...")
    try:
        results['modules']['risk_factors'] = risk_factors_analysis(movie_name)
        print("   ✅ Complete")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['modules']['risk_factors'] = {'error': str(e)}
    
    # Module 7: Target Audience (if function exists)
    if 'target_audience_recommendation' in globals():
        print("🎯 Module 7: Target Audience...")
        try:
            results['modules']['target_audience'] = target_audience_recommendation(movie_name)
            print("   ✅ Complete")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results['modules']['target_audience'] = {'error': str(e)}
    else:
        print("🎯 Module 7: Target Audience... ⏭️  Skipped (function not defined)")
        results['modules']['target_audience'] = {'status': 'skipped', 'reason': 'function not defined'}
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename (sanitize movie name)
    safe_movie_name = movie_name.lower().replace(' ', '_').replace("'", "")
    filename = f"{safe_movie_name}.json"
    filepath = output_path / filename
    
    # Export to JSON
    print(f"\n💾 Exporting to JSON...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Saved: {filepath}")
    print(f"\n{'='*80}\n")
    
    return results


def export_all_movies(output_dir='../insights'):
    """
    Run analysis on all 10 movies and export individual JSON files
    
    Args:
        output_dir: Directory to save JSON files (default: ../insights)
    
    Returns: Dictionary with all movie results
    """
    
    all_results = {
        'export_date': datetime.now().isoformat(),
        'total_movies': len(MOVIES),
        'movies': {}
    }
    
    print(f"\n🎬 BATCH EXPORT: Analyzing {len(MOVIES)} movies")
    print(f"📁 Output directory: {output_dir}\n")
    
    for i, movie in enumerate(MOVIES, 1):
        print(f"\n[{i}/{len(MOVIES)}] Processing: {movie}")
        
        try:
            result = export_movie_analysis(movie, output_dir)
            all_results['movies'][movie] = {
                'status': 'success',
                'filepath': f"{movie.lower().replace(' ', '_').replace(\"'\", '')}.json"
            }
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            all_results['movies'][movie] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Export master index file
    output_path = Path(output_dir)
    master_file = output_path / '_index.json'
    
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ BATCH EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"   Total movies: {len(MOVIES)}")
    print(f"   Successful: {sum(1 for m in all_results['movies'].values() if m['status'] == 'success')}")
    print(f"   Failed: {sum(1 for m in all_results['movies'].values() if m['status'] == 'failed')}")
    print(f"   Master index: {master_file}")
    print(f"\n")
    
    return all_results


def create_combined_export(output_dir='../insights', output_file='all_movies_combined.json'):
    """
    Create a single JSON file with all movie analyses combined
    
    Args:
        output_dir: Directory containing individual JSON files
        output_file: Name of combined output file
    
    Returns: Combined results dictionary
    """
    
    print(f"\n📦 Creating combined export...")
    
    output_path = Path(output_dir)
    combined_data = {
        'export_date': datetime.now().isoformat(),
        'total_movies': len(MOVIES),
        'movies': {}
    }
    
    # Read all individual movie JSON files
    for movie in MOVIES:
        safe_movie_name = movie.lower().replace(' ', '_').replace("'", "")
        filepath = output_path / f"{safe_movie_name}.json"
        
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                movie_data = json.load(f)
                combined_data['movies'][movie] = movie_data
            print(f"   ✅ Loaded: {movie}")
        else:
            print(f"   ⚠️  Missing: {movie}")
            combined_data['movies'][movie] = {'status': 'file_not_found'}
    
    # Save combined file
    combined_filepath = output_path / output_file
    with open(combined_filepath, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n   💾 Combined file saved: {combined_filepath}")
    print(f"   📊 Size: {combined_filepath.stat().st_size / 1024:.1f} KB")
    print(f"\n")
    
    return combined_data


# Quick export helper for testing
def quick_export(movie_name, output_dir='../insights'):
    """
    Quick export for a single movie (with progress output)
    """
    return export_movie_analysis(movie_name, output_dir)

print("✅ Module 8: JSON Export Functions defined")
