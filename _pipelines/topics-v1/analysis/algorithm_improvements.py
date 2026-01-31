"""
algorithm_improvements.py - Enhanced algorithms for better topic discovery
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import spacy

class EnhancedTopicAnalyzer:
    """Advanced topic analysis with multiple improvements"""
    
    def __init__(self):
        # Load spaCy model for better NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Install spacy model: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    # 1. BETTER PREPROCESSING
    def extract_key_phrases(self, texts: List[str]) -> List[List[str]]:
        """Extract noun phrases and named entities instead of just words"""
        if not self.nlp:
            return []
        
        key_phrases = []
        for text in texts:
            doc = self.nlp(text[:1000])  # Limit for speed
            
            # Extract noun phrases
            phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                      if len(chunk.text.split()) <= 3 and len(chunk.text) > 3]
            
            # Extract named entities
            entities = [ent.text.lower() for ent in doc.ents 
                       if ent.label_ in ['ORG', 'PRODUCT', 'LAW', 'EVENT']]
            
            key_phrases.append(phrases + entities)
        
        return key_phrases
    
    # 2. TOPIC MODELING IMPROVEMENTS
    def run_lda_analysis(self, texts: List[str], n_topics: int = 10) -> Dict:
        """Latent Dirichlet Allocation for probabilistic topic modeling"""
        vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=50
        )
        
        lda.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-15:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topic_weight = topic[top_indices].tolist()
            
            topics[topic_idx] = {
                'words': top_words,
                'weights': topic_weight,
                'total_weight': topic.sum()
            }
        
        return topics
    
    # 3. HIERARCHICAL TOPIC MODELING
    def hierarchical_topics(self, embeddings: np.ndarray, texts: List[str], 
                          min_topics: int = 5, max_topics: int = 20) -> Dict:
        """Build topic hierarchy from coarse to fine-grained"""
        from sklearn.cluster import AgglomerativeClustering
        
        hierarchy = {}
        
        # Start with coarse clustering
        for n_clusters in [5, 10, 15, 20]:
            if n_clusters > len(texts) / 10:
                break
                
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = clustering.fit_predict(embeddings)
            
            hierarchy[f'level_{n_clusters}'] = {
                'labels': labels,
                'n_topics': n_clusters,
                'topic_sizes': np.bincount(labels).tolist()
            }
        
        return hierarchy
    
    # 4. DYNAMIC TOPIC NUMBER SELECTION
    def find_optimal_topics(self, embeddings: np.ndarray, 
                           min_topics: int = 3, max_topics: int = 30) -> int:
        """Use elbow method and silhouette score to find optimal topic count"""
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        scores = []
        
        for n_topics in range(min_topics, min(max_topics, len(embeddings) // 10)):
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                scores.append((n_topics, score))
        
        # Find elbow point
        if scores:
            best_n = max(scores, key=lambda x: x[1])[0]
            return best_n
        return 10
    
    # 5. ASPECT-BASED TOPIC EXTRACTION
    def extract_aspects(self, texts: List[str], aspects: List[str]) -> Dict:
        """Extract topics related to specific aspects"""
        aspect_topics = {}
        
        for aspect in aspects:
            # Find texts mentioning this aspect
            aspect_texts = [t for t in texts if aspect.lower() in t.lower()]
            
            if len(aspect_texts) > 10:
                # Run focused topic modeling
                vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
                dtm = vectorizer.fit_transform(aspect_texts)
                
                # Get top terms
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = dtm.sum(axis=0).A1
                top_indices = tfidf_scores.argsort()[-10:][::-1]
                
                aspect_topics[aspect] = {
                    'n_docs': len(aspect_texts),
                    'top_terms': [feature_names[i] for i in top_indices],
                    'scores': [tfidf_scores[i] for i in top_indices]
                }
        
        return aspect_topics
    
    # 6. TEMPORAL TOPIC EVOLUTION
    def analyze_topic_evolution(self, texts: List[str], 
                               timestamps: List[str], 
                               time_windows: int = 6) -> Dict:
        """Track how topics change over time"""
        import pandas as pd
        
        # Create time windows
        df = pd.DataFrame({'text': texts, 'time': pd.to_datetime(timestamps)})
        df['window'] = pd.qcut(df['time'], q=time_windows, labels=False)
        
        evolution = {}
        
        for window in range(time_windows):
            window_texts = df[df['window'] == window]['text'].tolist()
            
            if len(window_texts) > 20:
                # Extract topics for this window
                vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
                dtm = vectorizer.fit_transform(window_texts)
                
                # Get top terms
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = dtm.sum(axis=0).A1
                top_indices = tfidf_scores.argsort()[-10:][::-1]
                
                evolution[f'window_{window}'] = {
                    'time_range': (df[df['window'] == window]['time'].min(), 
                                  df[df['window'] == window]['time'].max()),
                    'n_docs': len(window_texts),
                    'top_terms': [feature_names[i] for i in top_indices]
                }
        
        return evolution


# 7. ADVANCED SIMILARITY METRICS
def calculate_topic_similarity(topic1_words: List[str], 
                              topic2_words: List[str],
                              word_embeddings: Dict[str, np.ndarray]) -> float:
    """Calculate semantic similarity between topics using word embeddings"""
    # Get embeddings for each topic's words
    emb1 = [word_embeddings.get(w, np.zeros(300)) for w in topic1_words if w in word_embeddings]
    emb2 = [word_embeddings.get(w, np.zeros(300)) for w in topic2_words if w in word_embeddings]
    
    if emb1 and emb2:
        # Calculate centroid of each topic
        centroid1 = np.mean(emb1, axis=0)
        centroid2 = np.mean(emb2, axis=0)
        
        # Cosine similarity
        similarity = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2))
        return similarity
    return 0.0


# 8. TOPIC QUALITY METRICS
def evaluate_topic_quality(topic_words: List[str], 
                         document_texts: List[str]) -> Dict[str, float]:
    """Multiple metrics for topic quality beyond coherence"""
    
    # 1. Uniqueness - how distinctive are the words
    all_words = ' '.join(document_texts).lower().split()
    word_freq = {w: all_words.count(w) for w in set(all_words)}
    
    uniqueness = np.mean([1.0 / (1 + np.log(word_freq.get(w, 1))) 
                         for w in topic_words])
    
    # 2. Coverage - what fraction of docs contain topic words
    coverage = sum(1 for doc in document_texts 
                  if any(w in doc.lower() for w in topic_words)) / len(document_texts)
    
    # 3. Exclusivity - how much these words appear together vs separately
    cooccurrence = sum(1 for doc in document_texts 
                      if all(w in doc.lower() for w in topic_words[:3]))
    individual = sum(sum(1 for doc in document_texts if w in doc.lower()) 
                    for w in topic_words[:3])
    exclusivity = cooccurrence / (individual / 3) if individual > 0 else 0
    
    return {
        'uniqueness': uniqueness,
        'coverage': coverage,
        'exclusivity': exclusivity,
        'combined_score': (uniqueness + coverage + exclusivity) / 3
    }
