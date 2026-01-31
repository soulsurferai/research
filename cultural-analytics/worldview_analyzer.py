#!/usr/bin/env python3
# worldview_analyzer.py - Analyzes semantic differences between two subreddits

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import json
import time
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

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

class SubredditWorldviewAnalyzer:
    """Analyze and compare the semantic 'worldview' of different subreddits"""
    
    def __init__(self):
        self._connect_to_qdrant()
        self.collection_name = "reddit_comments"
        self.output_dir = self._ensure_output_dir()
        
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
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _uuid_to_qdrant_format(self, uuid_str):
        """Convert UUID with dashes to Qdrant format without dashes"""
        if uuid_str and '-' in uuid_str:
            return uuid_str.replace('-', '')
        return uuid_str
    
    def compare_subreddits(self, subreddit_a, subreddit_b, sample_size=300, topic_filter=None):
        """Compare the semantic worldviews of two subreddits"""
        start_time = time.time()
        print(f"🔍 Analyzing worldviews: r/{subreddit_a} vs r/{subreddit_b}")
        
        # Create a timestamp and run ID for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{subreddit_a}_vs_{subreddit_b}_{timestamp}"
        
        # 1. Retrieve comment embeddings from both subreddits
        embeddings_a, comments_a = self._get_subreddit_embeddings(subreddit_a, sample_size, topic_filter)
        embeddings_b, comments_b = self._get_subreddit_embeddings(subreddit_b, sample_size, topic_filter)
        
        if len(embeddings_a) == 0 or len(embeddings_b) == 0:
            print("⚠️ Insufficient data for one or both subreddits")
            return None
            
        print(f"✅ Retrieved {len(embeddings_a)} embeddings from r/{subreddit_a}")
        print(f"✅ Retrieved {len(embeddings_b)} embeddings from r/{subreddit_b}")
        
        # 2. Run analyses
        results = {
            "run_metadata": {
                "run_id": run_id,
                "timestamp": timestamp,
                "subreddit_a": subreddit_a,
                "subreddit_b": subreddit_b,
                "sample_size_requested": sample_size,
                "sample_size_actual_a": len(embeddings_a),
                "sample_size_actual_b": len(embeddings_b),
                "topic_filter": topic_filter,
                "processing_time": None  # Will be filled at the end
            },
            "semantic_distance": self._calculate_semantic_distance(embeddings_a, embeddings_b),
            "visualization": self._visualize_semantic_space(
                embeddings_a, embeddings_b, subreddit_a, subreddit_b, run_id
            ),
            "key_concepts": self._extract_distinctive_concepts(
                comments_a, comments_b, subreddit_a, subreddit_b
            ),
            "emotional_profile": self._analyze_emotional_profiles(
                embeddings_a, embeddings_b, comments_a, comments_b
            ),
            "worldview_bridges": self._find_worldview_bridges(
                embeddings_a, embeddings_b, comments_a, comments_b
            ),
            "comment_samples": {
                f"r/{subreddit_a}_samples": comments_a[:10],  # First 10 comments as samples
                f"r/{subreddit_b}_samples": comments_b[:10]   # First 10 comments as samples
            }
        }
        
        # 3. Add embedding-based emotion analysis if available
        try:
            # Import the emotion analyzer
            from emotion_embeddings import EmotionEmbeddingAnalyzer
            
            print("\n🔍 Performing embedding-based emotion analysis...")
            emotion_analyzer = EmotionEmbeddingAnalyzer()
            emotion_comparison = emotion_analyzer.compare_subreddit_emotions(embeddings_a, embeddings_b)
            
            # Add to results
            results["embedding_emotional_analysis"] = emotion_comparison
            print("✅ Embedding-based emotion analysis complete")
            
        except Exception as e:
            print(f"⚠️ Embedding-based emotion analysis not available: {e}")
        
        # 4. Print results
        self._print_results(results, subreddit_a, subreddit_b)
        
        # 5. Save results to file
        output_file = os.path.join(self.output_dir, f"analysis_{run_id}.json")
        self._save_results(results, output_file)
        
        # 6. Update processing time
        results["run_metadata"]["processing_time"] = time.time() - start_time
        
        return results
    
    def _get_subreddit_embeddings(self, subreddit, sample_size=300, topic_filter=None):
        """Retrieve comment embeddings for a specific subreddit from Qdrant"""
        query_filter = {
            "must": [
                {"key": "subreddit", "match": {"value": subreddit}},
                {"key": "type", "match": {"value": "comment"}}  # Ensure we only get comments
            ]
        }
        
        # Add topic filter if provided
        if topic_filter:
            query_filter["must"].append(
                {"key": "content", "text_match": {"query": topic_filter}}
            )
        
        # Retrieve embeddings with scroll API for larger datasets
        embeddings = []
        comments = []
        offset = None
        
        try:
            print(f"Retrieving comments from r/{subreddit}...")
            
            # Use pagination to get more results if needed
            while len(embeddings) < sample_size:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=min(1000, sample_size - len(embeddings)),  # Increased batch size, capped at remaining needed
                    offset=offset,
                    with_payload=["content"],  # Get only the content field to reduce data transfer
                    with_vectors=["semantic"]  # Named vector field in Qdrant
                )
                
                points = scroll_result[0]
                if not points:
                    print(f"  - No more points found for r/{subreddit}")
                    break  # No more points to retrieve
                
                # Process this batch
                new_embeddings = 0
                for point in points:
                    if "semantic" in point.vector and point.payload.get("content"):
                        embeddings.append(point.vector["semantic"])
                        comments.append(point.payload.get("content", ""))
                        new_embeddings += 1
                
                # Update offset for next batch
                offset = scroll_result[1]
                print(f"  - Retrieved {len(embeddings)} comments so far (added {new_embeddings} in this batch)...")
                
                if offset is None or len(embeddings) >= sample_size:
                    break  # Either reached the end or got enough comments
                
            print(f"Retrieved {len(embeddings)} comments from r/{subreddit}")
            
        except Exception as e:
            print(f"❌ Error retrieving embeddings for r/{subreddit}: {e}")
        
        return np.array(embeddings), comments
    
    def _calculate_semantic_distance(self, embeddings_a, embeddings_b):
        """Calculate Jensen-Shannon divergence between embedding distributions"""
        # Calculate centroids (mean vectors) for each subreddit
        centroid_a = np.mean(embeddings_a, axis=0)
        centroid_b = np.mean(embeddings_b, axis=0)
        
        # Normalize centroids to create probability distributions
        centroid_a = np.abs(centroid_a)
        centroid_b = np.abs(centroid_b)
        
        centroid_a = centroid_a / np.sum(centroid_a)
        centroid_b = centroid_b / np.sum(centroid_b)
        
        # Calculate Jensen-Shannon divergence
        js_distance = jensenshannon(centroid_a, centroid_b)
        
        # Calculate additional distribution metrics
        variance_a = np.mean(np.var(embeddings_a, axis=0))
        variance_b = np.mean(np.var(embeddings_b, axis=0))
        
        # Calculate cosine similarity between centroids
        norm_a = np.linalg.norm(centroid_a)
        norm_b = np.linalg.norm(centroid_b)
        cosine_sim = np.dot(centroid_a, centroid_b) / (norm_a * norm_b) if norm_a * norm_b != 0 else 0
        
        return {
            "jensen_shannon_distance": float(js_distance),
            "cosine_similarity": float(cosine_sim),
            "internal_variance_a": float(variance_a),
            "internal_variance_b": float(variance_b),
            "variance_ratio": float(variance_a / variance_b) if variance_b != 0 else float('inf')
        }
    
    def _visualize_semantic_space(self, embeddings_a, embeddings_b, label_a, label_b, run_id):
        """Create a visualization of the two subreddits in semantic space"""
        # Combine embeddings for dimensionality reduction
        combined_embeddings = np.vstack([embeddings_a, embeddings_b])
        
        # Create labels for the plot
        labels = [label_a] * len(embeddings_a) + [label_b] * len(embeddings_b)
        
        # First reduce with PCA to 50 dimensions
        pca = PCA(n_components=min(50, combined_embeddings.shape[1]))
        reduced_embeddings = pca.fit_transform(combined_embeddings)
        
        # Then use t-SNE for final 2D visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings)-1))
        tsne_results = tsne.fit_transform(reduced_embeddings)
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'x': tsne_results[:, 0],
            'y': tsne_results[:, 1],
            'subreddit': labels
        })
        
        # Create plot
        plt.figure(figsize=(10, 8))
        ax = sns.scatterplot(x='x', y='y', hue='subreddit', data=df, alpha=0.7)
        plt.title(f'Semantic Space: r/{label_a} vs r/{label_b}')
        
        # Add contour to show density
        for subreddit, color in zip([label_a, label_b], ['blue', 'orange']):
            subset = df[df['subreddit'] == subreddit]
            sns.kdeplot(x=subset['x'], y=subset['y'], levels=5, color=color, alpha=0.3, ax=ax)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f"worldview_comparison_{run_id}.png")
        plt.savefig(plot_path)
        plt.close()
        
        return {
            'tsne_coordinates': tsne_results.tolist(),  # Convert to list for JSON serialization
            'labels': labels,
            'plot_path': plot_path,
            'dimensionality_reduction': {
                'pca_components': pca.n_components_,
                'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
                'tsne_perplexity': tsne.perplexity
            }
        }
    
    def _extract_distinctive_concepts(self, comments_a, comments_b, label_a, label_b):
        """Extract distinctive concepts that characterize each subreddit"""
        # Use TF-IDF to find distinctive terms
        # Adjust parameters to be more flexible (lower min_df, higher max_df)
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english', 
                              min_df=1, max_df=0.95)
        
        # Combine all comments for each subreddit
        text_a = " ".join(comments_a)
        text_b = " ".join(comments_b)
        
        # Fit TF-IDF
        tfidf_matrix = tfidf.fit_transform([text_a, text_b])
        feature_names = tfidf.get_feature_names_out()
        
        # Get top terms for each subreddit
        a_scores = tfidf_matrix[0].toarray()[0]
        b_scores = tfidf_matrix[1].toarray()[0]
        
        a_top_indices = a_scores.argsort()[-30:][::-1]
        b_top_indices = b_scores.argsort()[-30:][::-1]
        
        a_top_terms = [(feature_names[i], float(a_scores[i])) for i in a_top_indices]
        b_top_terms = [(feature_names[i], float(b_scores[i])) for i in b_top_indices]
        
        # Extract common terms
        common_terms = []
        for term in feature_names:
            a_score = a_scores[np.where(feature_names == term)[0][0]] if term in feature_names else 0
            b_score = b_scores[np.where(feature_names == term)[0][0]] if term in feature_names else 0
            
            # If term has significant weight in both subreddits
            if a_score > 0.01 and b_score > 0.01:
                common_terms.append((term, float(a_score), float(b_score)))
        
        # Sort common terms by combined score
        common_terms.sort(key=lambda x: x[1] + x[2], reverse=True)
        
        return {
            f"distinctive_concepts_{label_a}": a_top_terms,
            f"distinctive_concepts_{label_b}": b_top_terms,
            "common_concepts": common_terms[:20]  # Top 20 common terms
        }
    
    def _analyze_emotional_profiles(self, embeddings_a, embeddings_b, comments_a, comments_b):
        """Analyze emotional profiles of the two subreddits"""
        # Define emotional keywords for basic sentiment detection
        emotion_keywords = {
            'anger': ['angry', 'furious', 'outraged', 'mad', 'hate', 'rage', 'fury', 'pissed'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'anxious', 'fear', 'dread', 'panic'],
            'joy': ['happy', 'joy', 'delighted', 'excited', 'pleased', 'glad', 'thrilled', 'enjoyment'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'grief', 'loss', 'sorrow', 'despair'],
            'disgust': ['disgusting', 'gross', 'revolting', 'repulsive', 'vile', 'nasty', 'yuck', 'awful'],
            'surprise': ['surprised', 'shocked', 'astonished', 'amazed', 'unexpected', 'startled', 'stunned'],
            'trust': ['trust', 'believe', 'faith', 'confident', 'reliable', 'honest', 'dependable', 'integrity'],
            'anticipation': ['anticipate', 'expect', 'looking forward', 'hopeful', 'awaiting', 'eager'],
            'certainty': ['certainly', 'definitely', 'absolutely', 'surely', 'undoubtedly', 'must', 'always', 'never'],
            'uncertainty': ['maybe', 'perhaps', 'possibly', 'uncertain', 'unsure', 'doubt', 'might', 'could'],
            'moral': ['right', 'wrong', 'moral', 'ethical', 'justice', 'fair', 'unfair', 'should', 'ought'],
            'cognitive': ['think', 'know', 'believe', 'understand', 'consider', 'reason', 'logic', 'rational']
        }
        
        # Count emotion mentions in each subreddit
        emotion_counts_a = {emotion: 0 for emotion in emotion_keywords}
        emotion_counts_b = {emotion: 0 for emotion in emotion_keywords}
        
        for comment in comments_a:
            comment_lower = comment.lower()
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in comment_lower:
                        emotion_counts_a[emotion] += 1
                        break
        
        for comment in comments_b:
            comment_lower = comment.lower()
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in comment_lower:
                        emotion_counts_b[emotion] += 1
                        break
        
        # Normalize by comment count
        emotion_profile_a = {}
        emotion_profile_b = {}
        for emotion in emotion_counts_a:
            emotion_profile_a[emotion] = emotion_counts_a[emotion] / len(comments_a) if comments_a else 0
            emotion_profile_b[emotion] = emotion_counts_b[emotion] / len(comments_b) if comments_b else 0
        
        # Calculate emotion differences
        emotion_differences = {}
        for emotion in emotion_profile_a:
            emotion_differences[emotion] = emotion_profile_a[emotion] - emotion_profile_b[emotion]
        
        return {
            "emotion_profile_a": {k: float(v) for k, v in emotion_profile_a.items()},
            "emotion_profile_b": {k: float(v) for k, v in emotion_profile_b.items()},
            "emotion_differences": {k: float(v) for k, v in emotion_differences.items()},
            "raw_counts_a": emotion_counts_a,
            "raw_counts_b": emotion_counts_b
        }
    
    def _find_worldview_bridges(self, embeddings_a, embeddings_b, comments_a, comments_b):
        """Find potential bridges between the worldviews - comments with high similarity across communities"""
        # Calculate all pairwise similarities
        similarities = cosine_similarity(embeddings_a, embeddings_b)
        
        # Find top bridge candidates (highest cross-community similarity)
        top_bridges = []
        
        # Get top 5 bridges
        for _ in range(min(5, similarities.shape[0], similarities.shape[1])):
            # Find the maximum similarity
            max_i, max_j = np.unravel_index(np.argmax(similarities), similarities.shape)
            
            # Store the bridge
            top_bridges.append({
                "comment_a": comments_a[max_i],
                "comment_b": comments_b[max_j],
                "similarity": float(similarities[max_i, max_j])
            })
            
            # Set this pair's similarity to -1 so we don't select it again
            similarities[max_i, max_j] = -1
        
        # Find clusters of thematically similar comments across communities
        thematic_bridges = self._find_thematic_bridges(embeddings_a, embeddings_b, comments_a, comments_b)
        
        return {
            "bridge_candidates": top_bridges,
            "thematic_bridges": thematic_bridges
        }
    
    def _find_thematic_bridges(self, embeddings_a, embeddings_b, comments_a, comments_b):
        """Find clusters of thematically similar comments across communities"""
        # This is a simplified clustering approach
        # In a full implementation, you might use more sophisticated clustering methods
        
        # For demonstration, we'll find comments that share keywords between communities
        
        # Extract significant words from each comment
        def extract_significant_words(comment):
            # Remove punctuation and convert to lowercase
            clean_comment = re.sub(r'[^\w\s]', '', comment.lower())
            # Split into words
            words = clean_comment.split()
            # Remove common stop words (simplified list)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                          'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 
                          'by', 'about', 'like', 'through', 'over', 'before', 'after',
                          'between', 'under', 'above', 'of', 'from', 'up', 'down', 'that',
                          'this', 'these', 'those', 'it', 'they', 'them', 'their', 'he',
                          'she', 'his', 'her', 'i', 'we', 'you', 'me', 'us', 'my', 'your',
                          'our', 'what', 'when', 'where', 'why', 'how', 'which', 'who',
                          'whom', 'whose', 'if', 'then', 'else', 'so', 'as', 'than'}
            return [word for word in words if word not in stop_words and len(word) > 3]
        
        # Extract significant words for each comment
        words_a = [extract_significant_words(comment) for comment in comments_a]
        words_b = [extract_significant_words(comment) for comment in comments_b]
        
        # Find shared significant words
        all_words_a = set(word for comment_words in words_a for word in comment_words)
        all_words_b = set(word for comment_words in words_b for word in comment_words)
        shared_words = all_words_a.intersection(all_words_b)
        
        # Find the most common shared words
        word_counts_a = Counter(word for comment_words in words_a for word in comment_words if word in shared_words)
        word_counts_b = Counter(word for comment_words in words_b for word in comment_words if word in shared_words)
        
        # Find top shared themes
        shared_themes = []
        for word in shared_words:
            if word_counts_a[word] >= 3 and word_counts_b[word] >= 3:  # Minimum frequency threshold
                # Find example comments from each subreddit
                examples_a = [comments_a[i] for i, words in enumerate(words_a) if word in words][:2]
                examples_b = [comments_b[i] for i, words in enumerate(words_b) if word in words][:2]
                
                shared_themes.append({
                    "theme": word,
                    "frequency_a": word_counts_a[word],
                    "frequency_b": word_counts_b[word],
                    "examples_a": examples_a,
                    "examples_b": examples_b
                })
        
        # Sort by combined frequency
        shared_themes.sort(key=lambda x: x["frequency_a"] + x["frequency_b"], reverse=True)
        
        return shared_themes[:10]  # Return top 10 thematic bridges
    
    def _print_results(self, results, subreddit_a, subreddit_b):
        """Print the analysis results in a readable format"""
        print("\n" + "="*80)
        print(f"🌍 WORLDVIEW COMPARISON: r/{subreddit_a} vs r/{subreddit_b}")
        print("="*80)
        
        # Print semantic distance
        distance = results["semantic_distance"]
        print(f"\n📊 SEMANTIC DISTANCE: {distance['jensen_shannon_distance']:.4f}")
        print(f"   - Cosine similarity: {distance['cosine_similarity']:.4f}")
        print(f"   - r/{subreddit_a} internal variance: {distance['internal_variance_a']:.4f}")
        print(f"   - r/{subreddit_b} internal variance: {distance['internal_variance_b']:.4f}")
        print(f"   - Variance ratio: {distance['variance_ratio']:.4f}")
        
        # Print visualization info
        print(f"\n🖼️ VISUALIZATION: Saved to {results['visualization']['plot_path']}")
        
        # Print distinctive concepts
        concepts = results["key_concepts"]
        print(f"\n🔑 DISTINCTIVE CONCEPTS:")
        print(f"\n   r/{subreddit_a} top concepts:")
        for term, score in concepts[f"distinctive_concepts_{subreddit_a}"][:10]:
            print(f"      - {term}: {score:.4f}")
        
        print(f"\n   r/{subreddit_b} top concepts:")
        for term, score in concepts[f"distinctive_concepts_{subreddit_b}"][:10]:
            print(f"      - {term}: {score:.4f}")
        
        print(f"\n   Common concepts:")
        for term, score_a, score_b in concepts["common_concepts"][:5]:
            print(f"      - {term}: {score_a:.4f} / {score_b:.4f}")
        
        # Print emotional profiles (lexical)
        emotions = results["emotional_profile"]
        print(f"\n😢 LEXICAL EMOTIONAL PROFILES:")
        print(f"\n   r/{subreddit_a} emotions:")
        for emotion, score in sorted(emotions["emotion_profile_a"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      - {emotion}: {score:.4f}")
        
        print(f"\n   r/{subreddit_b} emotions:")
        for emotion, score in sorted(emotions["emotion_profile_b"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      - {emotion}: {score:.4f}")
        
        # Print embedding-based emotion analysis if available
        if "embedding_emotional_analysis" in results:
            embedding_emotions = results["embedding_emotional_analysis"]
            
            print(f"\n🧠 EMBEDDING-BASED EMOTIONAL PROFILES:")
            
            print(f"\n   r/{subreddit_a} dominant emotions:")
            for emotion, score in embedding_emotions["profile_a"]["dominant_emotions"][:5]:
                print(f"      - {emotion}: {score:.4f}")
            
            print(f"\n   r/{subreddit_b} dominant emotions:")
            for emotion, score in embedding_emotions["profile_b"]["dominant_emotions"][:5]:
                print(f"      - {emotion}: {score:.4f}")
            
            print(f"\n   Largest emotional differences:")
            for emotion, diff in embedding_emotions["differences"][:5]:
                print(f"      - {emotion}: {diff:.4f} ({'higher in r/'+subreddit_a if diff > 0 else 'higher in r/'+subreddit_b})")
            
            print(f"\n   Emotional bridges (shared emotions):")
            for bridge in embedding_emotions["emotional_bridges"][:3]:
                print(f"      - {bridge['emotion']}: {bridge['score_a']:.4f} vs {bridge['score_b']:.4f} (diff: {bridge['difference']:.4f})")
        
        # Print worldview bridges
        bridges = results["worldview_bridges"]["bridge_candidates"]
        print(f"\n🌉 WORLDVIEW BRIDGES:")
        for i, bridge in enumerate(bridges[:3], 1):
            print(f"\n   Bridge {i} (similarity: {bridge['similarity']:.4f}):")
            print(f"      r/{subreddit_a}: \"{bridge['comment_a'][:100]}...\"")
            print(f"      r/{subreddit_b}: \"{bridge['comment_b'][:100]}...\"")
        
        # Print output file location
        output_file = os.path.join(self.output_dir, f"analysis_{results['run_metadata']['run_id']}.json")
        print(f"\n💾 FULL RESULTS: Saved to {output_file}")
        print("\n" + "="*80)
    
    def _save_results(self, results, output_file):
        """Save the analysis results to a JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results successfully saved to {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

    def load_and_run_from_selection(self, selection_file="subreddit_selection.json"):
        """Load subreddit selection from file and run analysis"""
        try:
            with open(selection_file, 'r') as f:
                selection = json.load(f)
            
            print(f"📂 Loaded selection from {selection_file}")
            return self.compare_subreddits(
                selection["subreddit_a"],
                selection["subreddit_b"],
                selection.get("sample_size", 300),
                selection.get("topic_filter")
            )
        except FileNotFoundError:
            print(f"❌ Selection file not found: {selection_file}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in selection file: {selection_file}")
            return None
        except KeyError as e:
            print(f"❌ Missing required key in selection file: {e}")
            return None


if __name__ == "__main__":
    # Stand-alone usage
    if not load_environment():
        print("❌ Failed to load environment variables. Please check your .env file.")
        exit(1)
        
    analyzer = SubredditWorldviewAnalyzer()
    
    # Check if selection file exists
    selection_file = "subreddit_selection.json"
    if os.path.exists(selection_file):
        print(f"📂 Found selection file: {selection_file}")
        print("Do you want to use this selection? (y/n)")
        if input("> ").lower().startswith('y'):
            results = analyzer.load_and_run_from_selection(selection_file)
            exit(0)
    
    # If no selection file or user declined, ask for manual input
    print("\n🔍 Enter subreddits to compare:")
    subreddit_a = input("First subreddit: ").strip()
    subreddit_b = input("Second subreddit: ").strip()
    
    print("\n📊 Enter sample size (or press Enter for default 300):")
    sample_size_input = input("> ").strip()
    sample_size = int(sample_size_input) if sample_size_input.isdigit() else 300
    
    print("\n🔍 Enter topic filter (optional, press Enter to skip):")
    topic_filter = input("> ").strip() or None
    
    results = analyzer.compare_subreddits(subreddit_a, subreddit_b, sample_size, topic_filter)