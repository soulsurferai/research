#!/usr/bin/env python3
# emotion_embeddings.py - Provides embedding-based emotion analysis

import os
import json
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import requests
from sklearn.metrics.pairwise import cosine_similarity

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
            return os.environ.get("OPENAI_API_KEY") is not None

class EmotionEmbeddingAnalyzer:
    """Analyze emotions in text using OpenAI embeddings"""
    
    def __init__(self, cache_dir=None):
        """Initialize the emotion analyzer with rich emotion concepts"""
        
        # Load OpenAI API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Define emotion concepts with rich descriptions
        self.emotion_concepts = {
            'anger': "Text expressing anger, rage, frustration, outrage, irritation, or annoyance. The content shows displeasure, hostility, or antagonism toward someone or something. Example: 'This makes me so mad, I can't believe they would do that!'",
            
            'fear': "Text expressing fear, anxiety, worry, dread, or nervousness. The content shows concern about potential threats, danger, or negative outcomes. Example: 'I'm really worried about what might happen next, it's keeping me up at night.'",
            
            'joy': "Text expressing happiness, joy, delight, pleasure, or satisfaction. The content shows positive emotions, celebration, or gratitude. Example: 'I'm so happy about this news, it really made my day!'",
            
            'sadness': "Text expressing sadness, grief, sorrow, disappointment, or despair. The content shows negative emotions related to loss, failure, or unhappiness. Example: 'I feel so down about what happened, it's really heartbreaking.'",
            
            'disgust': "Text expressing disgust, revulsion, repulsion, or contempt. The content shows strong dislike, aversion, or disapproval. Example: 'That's absolutely revolting, I can't even look at it.'",
            
            'surprise': "Text expressing surprise, shock, astonishment, or amazement. The content shows reactions to unexpected events or information. Example: 'Wow, I never saw that coming! I'm completely shocked.'",
            
            'trust': "Text expressing trust, confidence, faith, or belief in someone or something. The content shows reliance, dependence, or positive expectations. Example: 'I really trust their judgment on this, they've never let me down.'",
            
            'anticipation': "Text expressing anticipation, expectation, looking forward to something, or excitement about future events. Example: 'I can't wait for the weekend, I'm so looking forward to it.'",
            
            'contentment': "Text expressing contentment, satisfaction, peace, or fulfillment. A calm, settled happiness rather than excitement. Example: 'I'm really content with how things are going, everything feels right.'",
            
            'empathy': "Text expressing understanding, compassion, or sharing in others' emotions. The content shows connection to others' experiences. Example: 'I understand exactly how you feel, and I'm here for you.'",
            
            'pride': "Text expressing pride, dignity, self-respect, or satisfaction with achievements. Example: 'I'm really proud of what we accomplished, it was a team effort.'",
            
            'guilt': "Text expressing guilt, remorse, regret, or responsibility for wrongdoing. Example: 'I feel terrible about what I said, I should have been more thoughtful.'",
            
            'shame': "Text expressing shame, embarrassment, humiliation, or dishonor. Example: 'I'm so embarrassed by my behavior, I don't know how to face them again.'",
            
            'confusion': "Text expressing confusion, bewilderment, uncertainty, or lack of understanding. Example: 'I'm completely lost here, I don't understand what's happening.'",
            
            'certainty': "Text expressing certainty, confidence, conviction, or strong belief. The content shows no doubt or hesitation. Example: 'I'm absolutely certain this is the right decision, there's no question about it.'"
        }
        
        # Set up embedding cache directory
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            # Default to 'embeddings_cache' directory at same level as Worldview
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            self.cache_dir = os.path.join(parent_dir, 'embeddings_cache')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"📁 Emotion embeddings will be cached in: {self.cache_dir}")
        
        # Store for emotion embeddings
        self.emotion_embeddings = {}
    
    def get_emotion_embeddings(self, force_refresh=False):
        """Get embeddings for all emotion concepts, using cache if available"""
        cache_file = os.path.join(self.cache_dir, "emotion_embeddings.json")
        
        # Check if cache exists and is not being forcibly refreshed
        if os.path.exists(cache_file) and not force_refresh:
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    print("✅ Loaded emotion embeddings from cache")
                    self.emotion_embeddings = cached_data
                    return cached_data
            except Exception as e:
                print(f"⚠️ Error loading from cache: {e}")
                # Continue to generate new embeddings
        
        # Generate new embeddings
        print("🔄 Generating emotion embeddings from OpenAI API...")
        embeddings = {}
        
        for emotion, description in self.emotion_concepts.items():
            print(f"  - Getting embedding for '{emotion}'...")
            embedding = self._get_openai_embedding(description)
            if embedding:
                embeddings[emotion] = embedding
            # Small delay to avoid rate limits
            time.sleep(0.5)
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(embeddings, f)
            print(f"✅ Saved emotion embeddings to cache: {cache_file}")
        except Exception as e:
            print(f"⚠️ Error saving to cache: {e}")
        
        self.emotion_embeddings = embeddings
        return embeddings
    
    def _get_openai_embedding(self, text):
        """Get embedding from OpenAI API"""
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-embedding-3-large",
            "input": text,
            "dimensions": 1536
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                embedding = result["data"][0]["embedding"]
                return embedding
            else:
                print(f"❌ API error ({response.status_code}): {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            return None
    
    def analyze_comment_emotions(self, comment_embeddings, comments=None):
        """Analyze emotions in comments using embeddings"""
        # Ensure we have emotion embeddings
        if not self.emotion_embeddings:
            self.get_emotion_embeddings()
        
        # Convert emotion embeddings to numpy array for efficient comparison
        emotion_vectors = {}
        for emotion, embedding in self.emotion_embeddings.items():
            emotion_vectors[emotion] = np.array(embedding)
        
        # Analyze each comment
        comment_emotions = []
        
        for i, comment_embedding in enumerate(comment_embeddings):
            # Calculate similarity to each emotion
            emotion_scores = {}
            for emotion, emotion_vector in emotion_vectors.items():
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    comment_embedding.reshape(1, -1), 
                    emotion_vector.reshape(1, -1)
                )[0][0]
                emotion_scores[emotion] = float(similarity)
            
            # Add to results
            comment_result = {
                "top_emotions": sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3],
                "all_emotions": emotion_scores
            }
            
            # Add original comment text if available
            if comments and i < len(comments):
                comment_result["text"] = comments[i]
            
            comment_emotions.append(comment_result)
        
        return comment_emotions
    
    def get_subreddit_emotion_profile(self, comment_embeddings):
        """Generate an emotion profile for a subreddit based on comment embeddings"""
        # Analyze individual comments
        comment_emotions = self.analyze_comment_emotions(comment_embeddings)
        
        # Aggregate emotions across all comments
        emotion_totals = {}
        emotion_counts = {}
        
        for comment in comment_emotions:
            for emotion, score in comment["all_emotions"].items():
                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + score
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate average score for each emotion
        emotion_profile = {}
        for emotion in emotion_totals:
            if emotion_counts[emotion] > 0:
                emotion_profile[emotion] = emotion_totals[emotion] / emotion_counts[emotion]
            else:
                emotion_profile[emotion] = 0
        
        # Calculate dominant emotions
        dominant_emotions = sorted(emotion_profile.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "profile": emotion_profile,
            "dominant_emotions": dominant_emotions,
            "comment_count": len(comment_emotions)
        }
    
    def compare_subreddit_emotions(self, embeddings_a, embeddings_b):
        """Compare emotional profiles between two subreddits"""
        # Get emotional profiles for both subreddits
        profile_a = self.get_subreddit_emotion_profile(embeddings_a)
        profile_b = self.get_subreddit_emotion_profile(embeddings_b)
        
        # Calculate differences in emotions
        emotion_differences = {}
        for emotion in profile_a["profile"]:
            score_a = profile_a["profile"][emotion]
            score_b = profile_b["profile"][emotion]
            emotion_differences[emotion] = score_a - score_b
        
        # Sort differences by magnitude
        sorted_differences = sorted(
            emotion_differences.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Identify emotional bridges (similar emotional responses)
        emotional_bridges = []
        for emotion, difference in sorted(emotion_differences.items(), key=lambda x: abs(x[1])):
            if abs(difference) < 0.05:  # Small difference threshold
                score_a = profile_a["profile"][emotion]
                score_b = profile_b["profile"][emotion]
                if score_a > 0.5 and score_b > 0.5:  # Both have significant presence
                    emotional_bridges.append({
                        "emotion": emotion,
                        "score_a": score_a,
                        "score_b": score_b,
                        "difference": difference
                    })
        
        return {
            "profile_a": profile_a,
            "profile_b": profile_b,
            "differences": sorted_differences,
            "emotional_bridges": sorted(emotional_bridges, key=lambda x: abs(x["difference"]))
        }


# Example usage (if run directly)
if __name__ == "__main__":
    if not load_environment():
        print("❌ Failed to load environment variables. Please check your .env file.")
        exit(1)
    
    # Simple test
    analyzer = EmotionEmbeddingAnalyzer()
    emotion_embeddings = analyzer.get_emotion_embeddings()
    print(f"Generated embeddings for {len(emotion_embeddings)} emotions")
    
    # Test with a sample comment
    sample_comment = "I'm really excited about this new technology, it's going to change everything!"
    sample_embedding = analyzer._get_openai_embedding(sample_comment)
    
    if sample_embedding:
        result = analyzer.analyze_comment_emotions([sample_embedding], [sample_comment])
        print("\nSample Comment Analysis:")
        print(f"Comment: {sample_comment}")
        print("Top emotions:")
        for emotion, score in result[0]["top_emotions"]:
            print(f"  - {emotion}: {score:.4f}")