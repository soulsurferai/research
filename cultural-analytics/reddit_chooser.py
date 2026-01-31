#!/usr/bin/env python3
# reddit_chooser.py - Lists and selects subreddits available in Qdrant

import os
import json
from datetime import datetime
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    # Find the env directory relative to the current file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    env_path = os.path.join(parent_dir, 'env', '.env')
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
        return True
    else:
        print(f"⚠️ Warning: .env file not found at {env_path}")
        # Try default location as fallback
        load_dotenv()
        return os.environ.get("QDRANT_URL") is not None

class SubredditChooser:
    """Utility for listing and selecting subreddits from Qdrant"""
    
    def __init__(self):
        self._connect_to_qdrant()
        self.collection_name = "reddit_comments"
        
    def _connect_to_qdrant(self):
        """Connect to Qdrant using environment variables"""
        try:
            self.qdrant_client = QdrantClient(
                url=os.environ.get("QDRANT_URL"),
                api_key=os.environ.get("QDRANT_API_KEY")
            )
            return True
        except Exception as e:
            print(f"❌ Error connecting to Qdrant: {e}")
            return False
    
    def list_available_subreddits(self):
        """List all subreddits available in the Qdrant database with their comment counts"""
        print("📊 Analyzing subreddit distribution from database sample...")
        
        try:
            # Sample the database to get subreddit distribution
            print("🔍 Sampling database to discover available subreddits...")
            
            # Enhanced sampling with metadata collection
            subreddit_counts = {}
            date_range = {'min': None, 'max': None}
            quality_metrics = {'with_subreddit': 0, 'total_points': 0}
            offset = None
            total_processed = 0
            batch_size = 10000
            max_batches = 5  # Process up to 5 batches for comprehensive sampling
            
            for batch_num in range(max_batches):
                print(f"  - Fetching batch {batch_num+1}/{max_batches}...")
                
                # Get a batch of points with enhanced metadata
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=["subreddit", "created_at"],
                    with_vectors=False
                )
                
                points = scroll_result[0]
                if not points:
                    break
                
                # Process points and collect metadata
                for point in points:
                    quality_metrics['total_points'] += 1
                    
                    if hasattr(point, 'payload'):
                        # Track subreddit distribution
                        if 'subreddit' in point.payload:
                            subreddit = point.payload['subreddit']
                            subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
                            quality_metrics['with_subreddit'] += 1
                        
                        # Track date range
                        if 'created_at' in point.payload:
                            try:
                                # Handle various date formats
                                created_at = point.payload['created_at']
                                if isinstance(created_at, str):
                                    # Try parsing common formats
                                    try:
                                        date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                    except:
                                        date_obj = datetime.fromtimestamp(float(created_at))
                                elif isinstance(created_at, (int, float)):
                                    date_obj = datetime.fromtimestamp(created_at)
                                else:
                                    continue
                                    
                                if date_range['min'] is None or date_obj < date_range['min']:
                                    date_range['min'] = date_obj
                                if date_range['max'] is None or date_obj > date_range['max']:
                                    date_range['max'] = date_obj
                            except Exception:
                                # Skip date processing if format is unrecognizable
                                pass
                
                # Update offset and counts
                offset = scroll_result[1]
                total_processed += len(points)
                print(f"  - Processed {total_processed:,} comments, found {len(subreddit_counts)} subreddits")
                
                if offset is None:
                    break
            
            # Sort subreddits by count
            sorted_subreddits = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate data quality percentage
            data_quality_pct = (quality_metrics['with_subreddit'] / quality_metrics['total_points'] * 100) if quality_metrics['total_points'] > 0 else 0
            
            # Print enhanced results
            print("\n=== Available Subreddits (Sample-Based Analysis) ===")
            print(f"📈 Sample size: {total_processed:,} comments analyzed")
            print(f"🎯 Data quality: {data_quality_pct:.1f}% of comments have subreddit labels")
            print(f"🔍 Found {len(sorted_subreddits)} unique subreddits")
            
            # Add date range information if available
            if date_range['min'] and date_range['max']:
                print(f"📅 Date range: {date_range['min'].strftime('%b %d, %Y')} to {date_range['max'].strftime('%b %d, %Y')}")
            
            # Enhanced table display
            print(f"\n{'#':<4}| {'Subreddit':<25} | {'Sample Count':<12} | {'Activity Level':<15}")
            print("-" * 65)
            
            for i, (subreddit, count) in enumerate(sorted_subreddits, 1):
                # Determine activity level based on sample count
                if count >= 500:
                    activity_level = "Very High"
                elif count >= 200:
                    activity_level = "High"
                elif count >= 50:
                    activity_level = "Medium"
                elif count >= 10:
                    activity_level = "Low"
                else:
                    activity_level = "Very Low"
                    
                print(f"{i:<4}| r/{subreddit:<23} | {count:<12,} | {activity_level:<15}")
            
            return sorted_subreddits
            
        except Exception as e:
            print(f"❌ Error analyzing subreddits: {e}")
            print("🛠️ Falling back to basic collection check...")
            
            # Fallback: try basic collection existence check
            try:
                collections = self.qdrant_client.get_collections()
                if self.collection_name in [c.name for c in collections.collections]:
                    print(f"✅ Collection '{self.collection_name}' exists but analysis failed")
                    print("You may need to select subreddits manually or check your data")
                else:
                    print(f"❌ Collection '{self.collection_name}' not found")
            except Exception as fallback_error:
                print(f"❌ Complete connection failure: {fallback_error}")
            
            return []
    
    def interactive_selection(self):
        """Interactively select two subreddits for comparison"""
        subreddits = self.list_available_subreddits()
        
        if not subreddits:
            print("No subreddits available for selection.")
            return None
        
        selected = {}
        
        # Create a lookup dict for easier selection by number
        subreddit_lookup = {i: name for i, (name, _) in enumerate(subreddits, 1)}
        
        # First subreddit selection
        print("\n⬆️ Select first subreddit (enter number or name):")
        choice_1 = input("> ").strip()
        
        # Handle selection by number or name
        if choice_1.isdigit() and int(choice_1) in subreddit_lookup:
            selected["subreddit_a"] = subreddit_lookup[int(choice_1)]
        elif choice_1 in [s[0] for s in subreddits]:
            selected["subreddit_a"] = choice_1
        else:
            print(f"Invalid selection: {choice_1}")
            return None
        
        # Second subreddit selection
        print(f"\n⬇️ Select second subreddit to compare with r/{selected['subreddit_a']} (enter number or name):")
        choice_2 = input("> ").strip()
        
        # Handle selection by number or name
        if choice_2.isdigit() and int(choice_2) in subreddit_lookup:
            selected["subreddit_b"] = subreddit_lookup[int(choice_2)]
        elif choice_2 in [s[0] for s in subreddits]:
            selected["subreddit_b"] = choice_2
        else:
            print(f"Invalid selection: {choice_2}")
            return None
        
        # Topic filter (optional)
        print("\n🔍 Optional: Enter a topic filter (e.g., 'economy', 'immigration')")
        print("   This will limit analysis to comments containing this term")
        print("   Or press Enter to analyze all comments")
        topic = input("> ").strip()
        if topic:
            selected["topic_filter"] = topic
        else:
            selected["topic_filter"] = None
        
        # Sample size
        print("\n📊 Enter sample size (number of comments to analyze from each subreddit)")
        print("   Recommended: 200-500, higher numbers give better results but take longer")
        print("   Press Enter for default (300)")
        sample_size = input("> ").strip()
        if sample_size and sample_size.isdigit() and int(sample_size) > 0:
            selected["sample_size"] = int(sample_size)
        else:
            selected["sample_size"] = 300
        
        print(f"\n✅ Selected comparison: r/{selected['subreddit_a']} vs r/{selected['subreddit_b']}")
        if selected["topic_filter"]:
            print(f"   Topic filter: '{selected['topic_filter']}'")
        print(f"   Sample size: {selected['sample_size']} comments per subreddit")
        
        return selected
    
    def save_selection(self, selection, output_file="subreddit_selection.json"):
        """Save the selection to a JSON file"""
        if not selection:
            return False
            
        try:
            with open(output_file, 'w') as f:
                json.dump(selection, f, indent=2)
            print(f"\n💾 Selection saved to {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving selection: {e}")
            return False


if __name__ == "__main__":
    # Stand-alone usage
    if not load_environment():
        print("❌ Failed to load environment variables. Please check your .env file.")
        exit(1)
        
    chooser = SubredditChooser()
    selection = chooser.interactive_selection()
    
    if selection:
        chooser.save_selection(selection)
        print("\n🚀 Now you can run the analyzer with this selection!")
        print("   Run: python run_analysis.py")
    else:
        print("\n❌ Selection failed or was cancelled.")