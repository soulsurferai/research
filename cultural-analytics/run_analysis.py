#!/usr/bin/env python3
# run_analysis.py - Wrapper script to run the subreddit worldview analysis workflow

import os
import sys
import json
import argparse
import subprocess
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    # Find the env directory relative to the current file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # This would be the parent of Worldview folder
    env_path = os.path.join(parent_dir, 'env', '.env')
    
    if os.path.exists(env_path):
        print(f"✅ Found .env file at: {env_path}")
        load_dotenv(env_path)
        return True
    else:
        print(f"⚠️ Warning: .env file not found at {env_path}")
        # Try alternate location - sibling to Worldview folder
        alternate_path = os.path.join(os.path.dirname(parent_dir), 'env', '.env')
        if os.path.exists(alternate_path):
            print(f"✅ Found .env file at alternate location: {alternate_path}")
            load_dotenv(alternate_path)
            return True
        else:
            # Try default location as fallback
            load_dotenv()
            return os.environ.get("QDRANT_URL") is not None

def check_dependencies():
    """Check if required Python dependencies are installed"""
    try:
        import qdrant_client
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import sklearn
        from dotenv import load_dotenv
        print("✅ All required dependencies are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install all required dependencies:")
        print("pip install qdrant-client numpy pandas matplotlib seaborn scikit-learn python-dotenv")
        return False

def run_chooser():
    """Run the reddit_chooser.py script to select subreddits"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chooser_path = os.path.join(script_dir, "reddit_chooser.py")
    
    if not os.path.exists(chooser_path):
        print(f"❌ Chooser script not found at: {chooser_path}")
        return False
    
    try:
        print("\n🚀 Running subreddit selection tool...")
        subprocess.run([sys.executable, chooser_path], check=True)
        
        # Check if selection file was created
        selection_file = os.path.join(script_dir, "subreddit_selection.json")
        if os.path.exists(selection_file):
            return True
        else:
            print("❌ No selection was made.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running chooser script: {e}")
        return False

def run_analyzer():
    """Run the worldview_analyzer.py script with the selected subreddits"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyzer_path = os.path.join(script_dir, "worldview_analyzer.py")
    
    if not os.path.exists(analyzer_path):
        print(f"❌ Analyzer script not found at: {analyzer_path}")
        return False
    
    try:
        print("\n🚀 Running worldview analyzer...")
        subprocess.run([sys.executable, analyzer_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analyzer script: {e}")
        return False

def find_latest_result():
    """Find the most recent analysis result file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "analysis_results")
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    # Find all analysis JSON files and sort by modification time
    result_files = [
        os.path.join(results_dir, f) for f in os.listdir(results_dir)
        if f.startswith("analysis_") and f.endswith(".json")
    ]
    
    if not result_files:
        print("❌ No analysis result files found.")
        return None
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return result_files[0]

def create_ai_prompt(result_file):
    """Create a prompt for AI interpretation based on the analysis results"""
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Extract metadata
        metadata = results.get("run_metadata", {})
        subreddit_a = metadata.get("subreddit_a", "subreddit_a")
        subreddit_b = metadata.get("subreddit_b", "subreddit_b")
        
        # Create prompt
        prompt = f"""# Subreddit Worldview Analysis Results

## Task: Interpret the semantic differences between r/{subreddit_a} and r/{subreddit_b}

I've analyzed the semantic embeddings and language patterns of two subreddits using vector similarity and NLP techniques. Below are the key findings. Please provide a thoughtful interpretation of what these results reveal about the worldviews, values, and communication patterns of these communities.

### Key Metrics:
- Jensen-Shannon Distance: {results.get("semantic_distance", {}).get("jensen_shannon_distance", "N/A")}
- Cosine Similarity: {results.get("semantic_distance", {}).get("cosine_similarity", "N/A")}

### Distinctive Concepts for r/{subreddit_a}:
{", ".join([term for term, score in results.get("key_concepts", {}).get(f"distinctive_concepts_{subreddit_a}", [])[:15]])}

### Distinctive Concepts for r/{subreddit_b}:
{", ".join([term for term, score in results.get("key_concepts", {}).get(f"distinctive_concepts_{subreddit_b}", [])[:15]])}

### Common Concepts:
{", ".join([term for term, score_a, score_b in results.get("key_concepts", {}).get("common_concepts", [])[:10]])}

### Lexical Emotional Profile Comparison:
- Top emotions in r/{subreddit_a}: {", ".join([f"{emotion} ({score:.3f})" for emotion, score in sorted(results.get("emotional_profile", {}).get("emotion_profile_a", {}).items(), key=lambda x: x[1], reverse=True)[:5]])}
- Top emotions in r/{subreddit_b}: {", ".join([f"{emotion} ({score:.3f})" for emotion, score in sorted(results.get("emotional_profile", {}).get("emotion_profile_b", {}).items(), key=lambda x: x[1], reverse=True)[:5]])}
"""

        # Add embedding-based emotional analysis if available
        if "embedding_emotional_analysis" in results:
            embedding_emotions = results["embedding_emotional_analysis"]
            prompt += f"""
### Embedding-Based Emotional Profile:
- Dominant emotions in r/{subreddit_a}: {", ".join([f"{emotion} ({score:.3f})" for emotion, score in embedding_emotions.get("profile_a", {}).get("dominant_emotions", [])[:5]])}
- Dominant emotions in r/{subreddit_b}: {", ".join([f"{emotion} ({score:.3f})" for emotion, score in embedding_emotions.get("profile_b", {}).get("dominant_emotions", [])[:5]])}

### Largest Emotional Differences:
{", ".join([f"{emotion} ({diff:.3f} {'higher in r/'+subreddit_a if diff > 0 else 'higher in r/'+subreddit_b})" for emotion, diff in embedding_emotions.get("differences", [])[:5]])}

### Emotional Bridges (Shared Emotions):
{", ".join([f"{bridge.get('emotion')} ({bridge.get('score_a', 0):.3f} vs {bridge.get('score_b', 0):.3f})" for bridge in embedding_emotions.get("emotional_bridges", [])[:3]])}
"""

        # Add sample comments and bridges
        prompt += f"""
### Sample Comment from r/{subreddit_a}:
"{results.get("comment_samples", {}).get(f"r/{subreddit_a}_samples", [""])[0][:200]}..."

### Sample Comment from r/{subreddit_b}:
"{results.get("comment_samples", {}).get(f"r/{subreddit_b}_samples", [""])[0][:200]}..."

### Potential Bridge Comments (High Cross-Community Similarity):
1. r/{subreddit_a}: "{results.get("worldview_bridges", {}).get("bridge_candidates", [{}])[0].get("comment_a", "")[:100]}..."
   r/{subreddit_b}: "{results.get("worldview_bridges", {}).get("bridge_candidates", [{}])[0].get("comment_b", "")[:100]}..."

### Thematic Bridges:
{", ".join([bridge.get("theme", "") for bridge in results.get("worldview_bridges", {}).get("thematic_bridges", [])[:5]])}

## Questions for interpretation:
1. What are the defining characteristics of each community's worldview?
2. How do these communities differ in their emotional and cognitive approaches?
3. What potential bridges exist for communication between these communities?
4. What underlying values or assumptions might explain these semantic differences?
5. How might these differences influence how members of each community interpret the same information?

Please provide a thoughtful analysis that could help understand the perspective gaps between these communities and potential approaches to bridge them."""
        
        # Write prompt to file
        prompt_file = os.path.join(os.path.dirname(result_file), f"prompt_{os.path.basename(result_file)}")
        with open(prompt_file, 'w') as f:
            f.write(prompt)
        
        print(f"\n✅ AI interpretation prompt saved to: {prompt_file}")
        print("\nYou can copy this prompt and paste it to Claude or another AI for interpretation.")
        print("The prompt includes the key findings from your analysis in a format ready for interpretation.")
        
        return prompt_file
    except Exception as e:
        print(f"❌ Error creating AI prompt: {e}")
        return None

def main():
    """Main function to run the analysis workflow"""
    parser = argparse.ArgumentParser(description="Run subreddit worldview analysis")
    parser.add_argument("--skip-chooser", action="store_true", help="Skip the subreddit selection step")
    parser.add_argument("--skip-analyzer", action="store_true", help="Skip the analysis step")
    parser.add_argument("--prompt-only", action="store_true", help="Only generate AI prompt from latest results")
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("🔍 SUBREDDIT WORLDVIEW ANALYSIS WORKFLOW")
    print("="*50)
    
    # Check environment and dependencies
    if not load_environment():
        print("❌ Failed to load environment variables. Please check your .env file.")
        return 1
    
    if not check_dependencies():
        return 1
    
    # Handle prompt-only mode
    if args.prompt_only:
        latest_result = find_latest_result()
        if latest_result:
            create_ai_prompt(latest_result)
        else:
            print("❌ No analysis results found. Run the analyzer first.")
        return 0
    
    # Run the chooser if not skipped
    if not args.skip_chooser:
        if not run_chooser():
            print("❌ Subreddit selection failed. Stopping workflow.")
            return 1
    else:
        print("\n🔍 Skipping subreddit selection as requested.")
    
    # Run the analyzer if not skipped
    if not args.skip_analyzer:
        if not run_analyzer():
            print("❌ Analysis failed. Stopping workflow.")
            return 1
    else:
        print("\n🔍 Skipping analysis as requested.")
    
    # Create AI prompt from latest results
    latest_result = find_latest_result()
    if latest_result:
        create_ai_prompt(latest_result)
    
    print("\n" + "="*50)
    print("✅ ANALYSIS WORKFLOW COMPLETE")
    print("="*50)
    return 0

if __name__ == "__main__":
    sys.exit(main())