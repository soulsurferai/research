"""
qdrant_utils.py - Qdrant vector database utilities
"""

import numpy as np
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
import sys
sys.path.append('..')
from config import QDRANT_URL, QDRANT_API_KEY, MIN_TEXT_LENGTH


class QdrantFetcher:
    """Handle fetching embeddings and text from Qdrant"""
    
    def __init__(self):
        """Initialize Qdrant client"""
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
    def fetch_subreddit_data(self, subreddit: str, limit: int = 500) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
        """
        Fetch posts and comments from Qdrant for a specific subreddit
        
        Returns:
            embeddings: List of embedding vectors
            texts: List of text content
            metadata: List of metadata dictionaries
        """
        print(f"\nFetching data for r/{subreddit}...")
        
        embeddings = []
        texts = []
        metadata = []
        
        # Fetch from both collections
        for collection in ['reddit_posts', 'reddit_comments']:
            try:
                results = self.client.scroll(
                    collection_name=collection,
                    scroll_filter={
                        "must": [
                            {"key": "subreddit", "match": {"value": subreddit}}
                        ]
                    },
                    limit=limit // 2,  # Split limit between posts and comments
                    with_payload=True,
                    with_vectors=True
                )
                
                for point in results[0]:
                    # Extract embedding
                    embedding = self._extract_embedding(point)
                    if embedding is None:
                        continue
                    
                    # Extract text
                    text = self._extract_text(point, collection)
                    if len(text) < MIN_TEXT_LENGTH:
                        continue
                    
                    # Extract metadata
                    meta = self._extract_metadata(point, collection)
                    
                    embeddings.append(embedding)
                    texts.append(text)
                    metadata.append(meta)
                    
            except Exception as e:
                print(f"Error fetching from {collection}: {e}")
                continue
        
        print(f"Fetched {len(texts)} documents from r/{subreddit}")
        return embeddings, texts, metadata
    
    def _extract_embedding(self, point) -> np.ndarray:
        """Extract embedding from Qdrant point"""
        if hasattr(point.vector, 'tolist'):
            return np.array(point.vector)
        elif isinstance(point.vector, dict) and 'semantic' in point.vector:
            return np.array(point.vector['semantic'])
        elif isinstance(point.vector, list):
            return np.array(point.vector)
        else:
            return None
    
    def _extract_text(self, point, collection: str) -> str:
        """Extract text content from Qdrant point"""
        text = point.payload.get('content', '')
        
        # For posts, combine title and content
        if collection == 'reddit_posts' and 'title' in point.payload:
            title = point.payload['title']
            if title:
                text = f"{title} {text}"
        
        return text.strip()
    
    def _extract_metadata(self, point, collection: str) -> Dict:
        """Extract metadata from Qdrant point"""
        return {
            'id': point.id,
            'score': point.payload.get('score', 0),
            'type': 'post' if collection == 'reddit_posts' else 'comment',
            'created_at': point.payload.get('created_at', ''),
            'author': point.payload.get('author', ''),
            'subreddit': point.payload.get('subreddit', '')
        }
