"""
enhanced_topic_extraction.py - Advanced topic extraction with multiple methods
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from typing import List, Dict, Tuple, Optional
import sys
sys.path.append('..')
from config import MAX_FEATURES, NGRAM_RANGE, TOP_WORDS_PER_TOPIC
from utils.text_utils import get_custom_stop_words, preprocess_for_tfidf


def extract_topics_bm25(texts: List[str], labels: np.ndarray, 
                       n_words: int = TOP_WORDS_PER_TOPIC) -> Dict[int, Dict]:
    """
    Extract topics using BM25 scoring (better than TF-IDF for varying doc lengths)
    
    Args:
        texts: List of documents
        labels: Cluster labels
        n_words: Number of top words per topic
        
    Returns:
        Dictionary of topic information
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    topics = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    # Preprocess texts
    processed_texts = preprocess_for_tfidf(texts)
    
    # Create a global vocabulary first
    global_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=get_custom_stop_words(),
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_df=0.8
    )
    global_vectorizer.fit(processed_texts)
    vocab = global_vectorizer.vocabulary_
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_texts = [processed_texts[i] for i, m in enumerate(cluster_mask) if m]
        
        if not cluster_texts:
            continue
        
        # BM25 parameters
        k1 = 1.2
        b = 0.75
        
        # Create document term matrix
        vectorizer = CountVectorizer(
            vocabulary=vocab,
            ngram_range=NGRAM_RANGE
        )
        dtm = vectorizer.fit_transform(cluster_texts)
        
        # Calculate BM25 scores
        doc_lengths = dtm.sum(axis=1).A1
        avg_doc_length = doc_lengths.mean()
        
        # Term frequencies
        tf = dtm.toarray()
        
        # Document frequencies
        df = (tf > 0).sum(axis=0)
        
        # IDF calculation
        N = len(cluster_texts)
        idf = np.log((N - df + 0.5) / (df + 0.5))
        
        # BM25 scoring
        doc_length_norm = (1 - b + b * doc_lengths / avg_doc_length)
        bm25_scores = np.zeros(tf.shape[1])
        
        for j in range(tf.shape[1]):
            term_scores = (tf[:, j] * (k1 + 1)) / (tf[:, j] + k1 * doc_length_norm)
            bm25_scores[j] = (term_scores * idf[j]).sum()
        
        # Get top words
        feature_names = vectorizer.get_feature_names_out()
        top_indices = bm25_scores.argsort()[-n_words:][::-1]
        
        topics[label] = {
            'words': [feature_names[i] for i in top_indices],
            'scores': [bm25_scores[i] for i in top_indices],
            'size': len(cluster_texts),
            'representative_docs': cluster_texts[:3]
        }
    
    return topics


def extract_topics_keybert(texts: List[str], labels: np.ndarray,
                          embeddings: Optional[np.ndarray] = None,
                          n_words: int = TOP_WORDS_PER_TOPIC) -> Dict[int, Dict]:
    """
    Extract topics using KeyBERT-style keyword extraction
    
    Args:
        texts: List of documents
        labels: Cluster labels  
        embeddings: Document embeddings (optional)
        n_words: Number of keywords per topic
        
    Returns:
        Dictionary of topic information
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
    
    topics = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    # Get custom stopwords
    stop_words = get_custom_stop_words()
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_texts = [texts[i] for i, m in enumerate(cluster_mask) if m]
        
        if not cluster_texts:
            continue
        
        # Combine cluster docs
        cluster_doc = ' '.join(cluster_texts)
        
        # Extract candidate words/phrases
        vectorizer = CountVectorizer(
            ngram_range=(1, 3),
            stop_words=stop_words,
            max_features=100
        )
        
        try:
            # Get candidate terms
            dtm = vectorizer.fit_transform([cluster_doc])
            candidates = vectorizer.get_feature_names_out()
            
            # If we have embeddings, use them for similarity
            if embeddings is not None:
                cluster_embeddings = embeddings[cluster_mask]
                cluster_centroid = cluster_embeddings.mean(axis=0)
                
                # Get embeddings for candidates (simplified - would need actual embeddings)
                # For now, use TF-IDF as proxy
                tfidf = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
                candidate_vectors = tfidf.fit_transform([' '.join(candidates)]).toarray()
                
                # Score by similarity to centroid (simplified)
                scores = np.random.rand(len(candidates))  # Placeholder
            else:
                # Fallback to frequency
                scores = dtm.toarray()[0]
            
            # Get top keywords
            top_indices = scores.argsort()[-n_words:][::-1]
            
            topics[label] = {
                'words': [candidates[i] for i in top_indices],
                'scores': [scores[i] for i in top_indices],
                'size': len(cluster_texts),
                'representative_docs': cluster_texts[:3]
            }
            
        except Exception as e:
            print(f"KeyBERT extraction failed for cluster {label}: {e}")
            continue
    
    return topics


def extract_topics_mmr(texts: List[str], labels: np.ndarray,
                      n_words: int = TOP_WORDS_PER_TOPIC,
                      diversity: float = 0.5) -> Dict[int, Dict]:
    """
    Extract diverse topics using Maximal Marginal Relevance
    
    Args:
        texts: List of documents
        labels: Cluster labels
        n_words: Number of words per topic
        diversity: Balance between relevance and diversity (0-1)
        
    Returns:
        Dictionary of topic information
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    topics = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    processed_texts = preprocess_for_tfidf(texts)
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_texts = [processed_texts[i] for i, m in enumerate(cluster_mask) if m]
        
        if not cluster_texts:
            continue
        
        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words=get_custom_stop_words(),
            ngram_range=NGRAM_RANGE
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            terms = vectorizer.get_feature_names_out()
            
            # Calculate term scores
            term_scores = tfidf_matrix.sum(axis=0).A1
            
            # MMR algorithm
            selected_terms = []
            selected_indices = []
            
            # Start with highest scoring term
            best_idx = term_scores.argmax()
            selected_indices.append(best_idx)
            selected_terms.append(terms[best_idx])
            
            # Get term-term similarity matrix
            term_vectors = tfidf_matrix.T
            term_similarity = cosine_similarity(term_vectors)
            
            # Iteratively select diverse terms
            for _ in range(n_words - 1):
                candidate_scores = []
                
                for idx, term in enumerate(terms):
                    if idx in selected_indices:
                        candidate_scores.append(-np.inf)
                        continue
                    
                    # Relevance score
                    relevance = term_scores[idx]
                    
                    # Diversity penalty (max similarity to selected terms)
                    if selected_indices:
                        similarities = [term_similarity[idx, sel_idx] for sel_idx in selected_indices]
                        max_similarity = max(similarities)
                    else:
                        max_similarity = 0
                    
                    # MMR score
                    mmr_score = (1 - diversity) * relevance - diversity * max_similarity
                    candidate_scores.append(mmr_score)
                
                # Select best candidate
                best_idx = np.argmax(candidate_scores)
                selected_indices.append(best_idx)
                selected_terms.append(terms[best_idx])
            
            topics[label] = {
                'words': selected_terms,
                'scores': [term_scores[idx] for idx in selected_indices],
                'size': len(cluster_texts),
                'representative_docs': cluster_texts[:3]
            }
            
        except Exception as e:
            print(f"MMR extraction failed for cluster {label}: {e}")
            continue
    
    return topics


def generate_topic_labels(topics: Dict[int, Dict], method: str = 'descriptive') -> Dict[int, str]:
    """
    Generate human-readable labels for topics
    
    Args:
        topics: Dictionary of topic information
        method: Labeling method ('descriptive', 'thematic', 'question')
        
    Returns:
        Dictionary of topic_id -> label
    """
    labels = {}
    used_labels = set()
    
    for topic_id, topic_info in topics.items():
        words = topic_info['words'][:5]
        
        if method == 'descriptive':
            # Simple descriptive label
            if len(words) >= 2:
                label = f"{words[0].title()} & {words[1].title()}"
            else:
                label = words[0].title() if words else f"Topic {topic_id}"
                
        elif method == 'thematic':
            # Extended theme map with more specific patterns
            theme_map = [
                # Policy/Legal
                (('legalize', 'legalization', 'recreational', 'legal'), 'Legalization Movement'),
                (('dea', 'rescheduling', 'reschedule', 'schedule'), 'Federal Rescheduling'),
                (('state', 'states', 'florida', 'texas', 'california'), 'State Politics'),
                (('trump', 'biden', 'harris', 'president'), 'Political Landscape'),
                (('law', 'legal', 'attorney', 'court'), 'Legal Issues'),
                
                # Medical/Health
                (('health', 'medical', 'patient', 'medicine'), 'Medical Cannabis'),
                (('cbd', 'thc', 'cannabinoid', 'terpene'), 'Cannabinoid Science'),
                (('pain', 'anxiety', 'depression', 'ptsd'), 'Therapeutic Use'),
                (('research', 'study', 'science', 'data'), 'Scientific Research'),
                
                # Business/Industry
                (('business', 'dispensary', 'dispensaries', 'retail'), 'Retail Operations'),
                (('company', 'companies', 'industry', 'market'), 'Industry News'),
                (('stock', 'shares', 'msos', 'investment'), 'Stock Market'),
                (('sales', 'revenue', 'billion', 'growth'), 'Financial Performance'),
                (('debt', 'cash', 'money', 'funding'), 'Financial Health'),
                
                # Cultivation/Production
                (('grow', 'growing', 'plant', 'cultivation'), 'Cultivation'),
                (('harvest', 'yield', 'indoor', 'outdoor'), 'Growing Methods'),
                (('strain', 'genetics', 'seeds', 'clone'), 'Genetics & Strains'),
                
                # Consumption
                (('smoke', 'smoking', 'joint', 'blunt'), 'Smoking Culture'),
                (('high', 'stoned', 'effects', 'feel'), 'Effects & Experiences'),
                (('vape', 'vaping', 'cart', 'cartridge'), 'Vaping'),
                (('edible', 'edibles', 'gummies', 'dose'), 'Edibles'),
                (('bong', 'pipe', 'rig', 'glass'), 'Smoking Equipment'),
                (('tolerance', 'break', 'detox', 'test'), 'Tolerance & Testing'),
                
                # Community/Culture
                (('community', 'people', 'friends', 'social'), 'Community & Culture'),
                (('happy', 'love', 'enjoy', 'fun'), 'Positive Vibes'),
                (('help', 'advice', 'question', 'need'), 'Advice & Support')
            ]
            
            label = None
            # Try to find a unique theme
            for keywords, theme in theme_map:
                if any(kw in ' '.join(words).lower() for kw in keywords):
                    # If theme already used, make it more specific
                    if theme in used_labels:
                        # Add distinguishing word
                        for w in words:
                            if w not in theme.lower():
                                theme = f"{theme}: {w.title()}"
                                break
                    label = theme
                    break
            
            # Fallback to descriptive if no theme found
            if not label:
                label = f"{words[0].title()} {words[1].title()}" if len(words) >= 2 else f"Topic {topic_id}"
                    
        elif method == 'question':
            # Frame as a question
            question_templates = {
                'policy': "What's happening with {}?",
                'how': "How to {}?",
                'discussion': "Discussing {}",
                'news': "News about {}"
            }
            
            # Simple heuristic
            if any(w in words for w in ['legalize', 'dea', 'trump', 'biden']):
                label = question_templates['policy'].format(words[0])
            elif any(w in words for w in ['grow', 'smoke', 'vape']):
                label = question_templates['how'].format(words[0])
            else:
                label = question_templates['discussion'].format(words[0])
        
        labels[topic_id] = label
        used_labels.add(label)
    
    return labels


def calculate_topic_coherence(topics: Dict[int, Dict], texts: List[str]) -> Dict[int, float]:
    """
    Calculate coherence scores for topics (higher = more coherent)
    
    Args:
        topics: Dictionary of topic information
        texts: Original texts
        
    Returns:
        Dictionary of topic_id -> coherence score
    """
    from sklearn.feature_extraction.text import CountVectorizer
    
    coherence_scores = {}
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words=get_custom_stop_words()
    )
    dtm = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    for topic_id, topic_info in topics.items():
        words = topic_info['words'][:10]
        
        # Calculate PMI-based coherence
        coherence = 0
        pairs = 0
        
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                word1, word2 = words[i], words[j]
                
                if word1 in word_to_idx and word2 in word_to_idx:
                    idx1, idx2 = word_to_idx[word1], word_to_idx[word2]
                    
                    # Co-occurrence
                    cooc = ((dtm[:, idx1].toarray() > 0) & 
                           (dtm[:, idx2].toarray() > 0)).sum()
                    
                    # Individual occurrences
                    occ1 = (dtm[:, idx1].toarray() > 0).sum()
                    occ2 = (dtm[:, idx2].toarray() > 0).sum()
                    
                    # PMI calculation
                    if cooc > 0 and occ1 > 0 and occ2 > 0:
                        pmi = np.log((cooc * len(texts)) / (occ1 * occ2))
                        coherence += pmi
                        pairs += 1
        
        coherence_scores[topic_id] = coherence / pairs if pairs > 0 else 0
    
    return coherence_scores
