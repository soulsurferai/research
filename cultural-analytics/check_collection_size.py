#!/usr/bin/env python3
# check_collection_size.py - Simple script to check Qdrant collection size

import os
import sys
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    # Find the env directory relative to the current file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    env_path = os.path.join(parent_dir, 'env', '.env')
    
    if os.path.exists(env_path):
        print(f"✅ Found .env file at: {env_path}")
        load_dotenv(env_path)
        return True
    else:
        print(f"⚠️ Warning: .env file not found at {env_path}")
        # Try alternate location
        alternate_path = os.path.join(os.path.dirname(parent_dir), 'env', '.env')
        if os.path.exists(alternate_path):
            print(f"✅ Found .env file at alternate location: {alternate_path}")
            load_dotenv(alternate_path)
            return True
        else:
            print("❌ No .env file found!")
            return False

def check_collection_size():
    """Check the size of Qdrant collections"""
    try:
        # Connect to Qdrant
        print("Connecting to Qdrant...")
        qdrant_client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )
        
        # Get collections
        collections = qdrant_client.get_collections().collections
        print(f"Found {len(collections)} collections")
        
        for collection in collections:
            print(f"\nChecking collection: {collection.name}")
            info = qdrant_client.get_collection(collection.name)
            
            # Print all collection info fields
            print("Collection info fields:")
            for attr in dir(info):
                if not attr.startswith('_') and not callable(getattr(info, attr)):
                    print(f"  - {attr}: {getattr(info, attr)}")
            
            # Print all relevant count fields
            print("\nRelevant count fields:")
            for field in ['vectors_count', 'indexed_vectors_count', 'points_count']:
                if hasattr(info, field):
                    print(f"  - {field}: {getattr(info, field)}")
            
            # Get a sample of subreddits
            print("\nGetting sample of subreddits...")
            sample = qdrant_client.scroll(
                collection_name=collection.name,
                limit=1000,
                with_payload=["subreddit", "type"],
                with_vectors=False
            )[0]
            
            # Count types
            type_counts = {}
            for point in sample:
                if hasattr(point, 'payload') and 'type' in point.payload:
                    point_type = point.payload['type']
                    type_counts[point_type] = type_counts.get(point_type, 0) + 1
            
            print(f"Found {len(type_counts)} different types in sample:")
            for point_type, count in type_counts.items():
                print(f"  - {point_type}: {count}")
            
            # Count subreddits for comments only
            if 'comment' in type_counts:
                comment_subreddits = {}
                for point in sample:
                    if (hasattr(point, 'payload') and 
                        'type' in point.payload and 
                        point.payload['type'] == 'comment' and
                        'subreddit' in point.payload):
                        subreddit = point.payload['subreddit']
                        comment_subreddits[subreddit] = comment_subreddits.get(subreddit, 0) + 1
                
                print(f"\nFound {len(comment_subreddits)} subreddits among {type_counts.get('comment', 0)} comments:")
                for i, (subreddit, count) in enumerate(sorted(comment_subreddits.items(), key=lambda x: x[1], reverse=True)[:10], 1):
                    print(f"  {i}. r/{subreddit}: {count}")
            
            # Get query filter for comments
            if 'comment' in type_counts:
                print("\nTesting query filter for comments...")
                query_filter = {
                    "must": [
                        {"key": "type", "match": {"value": "comment"}}
                    ]
                }
                
                comment_sample = qdrant_client.scroll(
                    collection_name=collection.name,
                    scroll_filter=query_filter,
                    limit=10,
                    with_payload=["subreddit", "content"],
                    with_vectors=False
                )[0]
                
                print(f"Retrieved {len(comment_sample)} comments with filter")
                if comment_sample:
                    print("Sample comment:")
                    sample_comment = comment_sample[0]
                    print(f"  - Subreddit: {sample_comment.payload.get('subreddit')}")
                    content = sample_comment.payload.get('content', '')
                    print(f"  - Content: {content[:100]}...")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if not load_environment():
        sys.exit(1)
    
    check_collection_size()