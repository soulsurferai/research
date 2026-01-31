"""
topic_extraction.py - Extract meaningful topics from clusters
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import List, Dict, Tuple
import sys
sys.path.append('..')
from config import MAX_FEATURES, NGRAM_RANGE, TOP_WORDS_PER_TOPIC
from utils.text_utils import get_custom_stop_words, preprocess_for_tfidf


def extract_topics_tfidf(texts: List[str], labels: np.ndarray, 
                        n_words: int = TOP_WORDS_PER_TOPIC) -> Dict[int, Dict]:
    """
    Extract topics using TF-IDF for each cluster
    
    Args:
        texts: List of text documents
        labels: Cluster labels for each document
        n_words: Number of top words per topic
        
    Returns:
        Dictionary of cluster_id -> topic info
    """
    topics = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise cluster
    
    # Preprocess texts
    processed_texts = preprocess_for_tfidf(texts)
    
    # Get custom stopwords
    stop_words = get_custom_stop_words()
    
    for label in unique_labels:
        # Get texts in this cluster
        cluster_mask = labels == label
        cluster_texts = [processed_texts[i] for i, m in enumerate(cluster_mask) if m]
        
        if not cluster_texts:
            continue
        
        # Create TF-IDF vectorizer with better parameters
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            stop_words=stop_words,
            ngram_range=NGRAM_RANGE,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top words by TF-IDF score
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            top_indices = tfidf_scores.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [tfidf_scores[i] for i in top_indices]
            
            topics[label] = {
                'words': top_words,
                'scores': top_scores,
                'size': len(cluster_texts),
                'representative_docs': cluster_texts[:3]  # First 3 docs
            }
            
        except Exception as e:
            print(f"Error extracting topics for cluster {label}: {e}")
            continue
    
    return topics


def extract_topics_c_tfidf(texts: List[str], labels: np.ndarray, 
                          n_words: int = TOP_WORDS_PER_TOPIC) -> Dict[int, Dict]:
    """
    Extract topics using class-based TF-IDF (c-TF-IDF)
    More sophisticated than regular TF-IDF for topic modeling
    
    Args:
        texts: List of text documents  
        labels: Cluster labels for each document
        n_words: Number of top words per topic
        
    Returns:
        Dictionary of cluster_id -> topic info
    """
    topics = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    # Preprocess texts
    processed_texts = preprocess_for_tfidf(texts)
    
    # Create document per class
    docs_per_class = {}
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_texts = [processed_texts[i] for i, m in enumerate(cluster_mask) if m]
        docs_per_class[label] = ' '.join(cluster_texts)
    
    # Create count vectorizer
    vectorizer = CountVectorizer(
        stop_words=get_custom_stop_words(),
        ngram_range=NGRAM_RANGE,
        min_df=2
    )
    
    # Fit on all documents
    doc_term_matrix = vectorizer.fit_transform(list(docs_per_class.values()))
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate c-TF-IDF
    # Term frequency in each class
    tf = doc_term_matrix.toarray()
    
    # Document frequency (number of classes each term appears in)
    df = np.sum(tf > 0, axis=0)
    
    # IDF
    idf = np.log((len(docs_per_class) + 1) / (df + 1))
    
    # c-TF-IDF
    c_tf_idf = tf * idf
    
    # Extract top words for each topic
    for idx, label in enumerate(unique_labels):
        scores = c_tf_idf[idx]
        top_indices = scores.argsort()[-n_words:][::-1]
        
        topics[label] = {
            'words': [feature_names[i] for i in top_indices],
            'scores': [scores[i] for i in top_indices],
            'size': sum(labels == label),
            'representative_docs': [texts[i] for i, l in enumerate(labels) if l == label][:3]
        }
    
    return topics


def create_topic_summary(topics: Dict[int, Dict], labels: np.ndarray, 
                        subreddit: str) -> pd.DataFrame:
    """
    Create a summary DataFrame of topics
    
    Args:
        topics: Dictionary of topic information
        labels: Cluster labels
        subreddit: Subreddit name
        
    Returns:
        DataFrame with topic summaries
    """
    rows = []
    
    for topic_id, topic_info in topics.items():
        # Create readable topic name from top words
        topic_name = f"Topic_{topic_id}"
        
        # Get word representation
        word_repr = topic_info['words'][:5]  # Top 5 words
        
        # Create row
        row = {
            'Topic': topic_id,
            'Count': topic_info['size'],
            'Name': topic_name,
            'Representation': word_repr,
            'Representative_Docs': topic_info['representative_docs'],
            'subreddit': subreddit
        }
        rows.append(row)
    
    # Sort by count
    df = pd.DataFrame(rows).sort_values('Count', ascending=False)
    
    # Add noise statistics
    noise_count = sum(1 for l in labels if l == -1)
    if noise_count > 0:
        print(f"Note: {noise_count} documents classified as noise (not in any topic)")
    
    return df


def get_topic_diversity(topics: Dict[int, Dict], top_n: int = 20) -> float:
    """
    Calculate topic diversity score (how different topics are from each other)
    
    Args:
        topics: Dictionary of topic information
        top_n: Number of top words to consider per topic
        
    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    all_words = []
    
    for topic_info in topics.values():
        all_words.extend(topic_info['words'][:top_n])
    
    unique_words = set(all_words)
    total_words = len(all_words)
    
    diversity = len(unique_words) / total_words if total_words > 0 else 0
    
    return diversity
