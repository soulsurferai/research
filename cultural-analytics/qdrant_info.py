#!/usr/bin/env python3
# qdrant_minimal.py - Get minimal stats about the Qdrant database

import os
import sys
import json
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    # Find the env directory relative to the current file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # This would be the parent of Worldview folder
    env_path = os.path.join(parent_dir, 'env', '.env')
    
    if os.path.exists(env_path):
        print(f"Found .env file at: {env_path}")
        load_dotenv(env_path)
        return True
    else:
        print(f"Warning: .env file not found at {env_path}")
        # Try alternate location - sibling to Worldview folder
        alternate_path = os.path.join(os.path.dirname(parent_dir), 'env', '.env')
        if os.path.exists(alternate_path):
            print(f"Found .env file at alternate location: {alternate_path}")
            load_dotenv(alternate_path)
            return True
        else:
            # Try default location as fallback
            load_dotenv()
            return os.environ.get("QDRANT_URL") is not None

def get_minimal_stats():
    """Get absolutely minimal statistics about the Qdrant database"""
    try:
        # Connect to Qdrant
        print("Connecting to Qdrant...")
        qdrant_client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )
        
        # Get collections list
        print("Getting collections list...")
        collections_response = qdrant_client.get_collections()
        
        # Print raw collections response for debugging
        print("\nRaw collections response:")
        print(collections_response)
        
        # Try to extract collection names
        collection_names = []
        try:
            if hasattr(collections_response, 'collections'):
                for collection in collections_response.collections:
                    if hasattr(collection, 'name'):
                        collection_names.append(collection.name)
        except Exception as e:
            print(f"Error extracting collection names: {e}")
        
        print(f"\nFound collections: {collection_names}")
        
        # For each collection, try to get basic info
        for collection_name in collection_names:
            print(f"\nCollection: {collection_name}")
            
            try:
                # Get collection info - but don't try to format it
                collection_info = qdrant_client.get_collection(collection_name)
                print("Raw collection info:")
                print(collection_info)
                
                # Try to get vector count
                if hasattr(collection_info, 'vectors_count'):
                    print(f"Vector count: {collection_info.vectors_count}")
                else:
                    print("Vector count not found in collection info")
                
                # Try to get a single point
                print("\nTrying to get one sample point...")
                try:
                    sample_result = qdrant_client.scroll(
                        collection_name=collection_name,
                        limit=1,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    if sample_result and len(sample_result) > 0:
                        points = sample_result[0]
                        print(f"Retrieved {len(points)} points")
                        
                        if len(points) > 0:
                            sample_point = points[0]
                            print("Sample point structure:")
                            print(sample_point)
                            
                            if hasattr(sample_point, 'payload'):
                                print("Payload keys:")
                                print(list(sample_point.payload.keys()))
                except Exception as e:
                    print(f"Error getting sample point: {e}")
            
            except Exception as e:
                print(f"Error getting collection info: {e}")
        
        # Try to count subreddits in reddit_comments collection
        if 'reddit_comments' in collection_names:
            try:
                print("\nTrying to count subreddits in reddit_comments collection...")
                subreddit_counts = {}
                
                # Get a small sample
                sample_result = qdrant_client.scroll(
                    collection_name='reddit_comments',
                    limit=100,  # Just get a small sample to avoid errors
                    with_payload=["subreddit"],
                    with_vectors=False
                )
                
                if sample_result and len(sample_result) > 0:
                    points = sample_result[0]
                    print(f"Retrieved {len(points)} points")
                    
                    for point in points:
                        if hasattr(point, 'payload') and 'subreddit' in point.payload:
                            subreddit = point.payload['subreddit']
                            subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
                    
                    print(f"Found {len(subreddit_counts)} unique subreddits in sample")
                    
                    if subreddit_counts:
                        print("Top subreddits:")
                        for subreddit, count in sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                            print(f"- r/{subreddit}: {count}")
            except Exception as e:
                print(f"Error counting subreddits: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return False

if __name__ == "__main__":
    if not load_environment():
        print("Failed to load environment variables. Please check your .env file.")
        sys.exit(1)
    
    get_minimal_stats()