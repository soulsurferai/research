"""
text_utils.py - Text preprocessing utilities
"""

import re
from typing import List, Tuple
import sys
sys.path.append('..')
from config import CANNABIS_STOPWORDS, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH


def clean_reddit_text(text: str) -> str:
    """Clean Reddit-specific formatting and noise from text"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove Reddit formatting
    text = re.sub(r'/u/\S+', '', text)  # User mentions
    text = re.sub(r'/r/\S+', '', text)  # Subreddit mentions
    text = re.sub(r'\[deleted\]', '', text)
    text = re.sub(r'\[removed\]', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*+', '', text)  # Bold/italic
    text = re.sub(r'#+\s*', '', text)  # Headers
    text = re.sub(r'&gt;', '', text)  # Quotes
    text = re.sub(r'&amp;', '&', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def filter_boilerplate(texts: List[str], metadata: List[dict]) -> Tuple[List[str], List[dict]]:
    """Filter out boilerplate content like daily discussion threads"""
    filtered_texts = []
    filtered_metadata = []
    
    boilerplate_patterns = [
        r'Welcome to the .* Daily Discussion Thread',
        r'New to Reddit\? \[Read This\]',
        r'This thread is intended for the community to talk',
        r'Unrelated discussion will always be removed',
        r'Please remember proper \[reddiquette\]'
    ]
    
    for text, meta in zip(texts, metadata):
        # Check if text matches boilerplate patterns
        is_boilerplate = any(re.search(pattern, text) for pattern in boilerplate_patterns)
        
        # Also check if it's too similar to daily thread format
        if 'daily discussion thread' in text.lower() and text.count('*') > 10:
            is_boilerplate = True
        
        if not is_boilerplate and MIN_TEXT_LENGTH <= len(text) <= MAX_TEXT_LENGTH:
            filtered_texts.append(text)
            filtered_metadata.append(meta)
    
    return filtered_texts, filtered_metadata


def preprocess_for_tfidf(texts: List[str]) -> List[str]:
    """Preprocess texts for TF-IDF analysis"""
    processed = []
    
    for text in texts:
        # Clean the text
        text = clean_reddit_text(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers (optional - might want to keep some)
        text = re.sub(r'\d+', '', text)
        
        # Remove single characters
        text = re.sub(r'\b\w\b', '', text)
        
        # Remove excessive spaces
        text = re.sub(r'\s+', ' ', text)
        
        processed.append(text.strip())
    
    return processed


def get_custom_stop_words():
    """Get combined list of stopwords for cannabis topic analysis"""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    # Combine sklearn's English stopwords with our custom ones
    stop_words = list(ENGLISH_STOP_WORDS) + CANNABIS_STOPWORDS
    
    # Add single letters
    stop_words.extend(list('abcdefghijklmnopqrstuvwxyz'))
    
    return list(set(stop_words))
