"""
config.py - Configuration constants for topic analysis
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env', '.env'))

# Qdrant Configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# Analysis Configuration
OUTPUT_DIR = 'results'
DEFAULT_SAMPLE_SIZE = 500
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 3
PCA_COMPONENTS = 50

# Text Processing Configuration
MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 5000

# Topic Extraction Configuration
MAX_FEATURES = 100
NGRAM_RANGE = (1, 2)
TOP_WORDS_PER_TOPIC = 10

# Cannabis-specific stopwords
CANNABIS_STOPWORDS = [
    'deleted', 'removed', 'bot', 'moderator', 'automod',
    'lol', 'lmao', 'edit', 'update', 'reddit', 'subreddit',
    'http', 'https', 'www', 'com', 'r/', 'u/',
    # Add more Reddit-specific terms
    'upvote', 'downvote', 'karma', 'post', 'comment',
    'thread', 'ama', 'iama', 'til', 'eli5',
    # Daily thread boilerplate
    'daily', 'discussion', 'thread', 'welcome', 'new to reddit',
    'read this', 'want to start', 'search bar', 'rule',
    'reddiquette', 'participating', 'conversation',
    # Common filler words
    'just', 'like', 'really', 'think', 'know', 'well',
    'good', 'got', 'get', 'getting', 'going', 'gonna',
    'want', 'wanted', 'wanting', 'need', 'needed',
    'looking', 'look', 'looks', 'looked',
    'make', 'made', 'making', 'makes',
    'people', 'person', 'guy', 'guys',
    'time', 'times', 'day', 'days', 'year', 'years',
    'youtube', 'video', 'watch', 'link',
    've', 'll', 'm', 's', 't', 'd'
]

# Target subreddits
CANNABIS_SUBREDDITS = [
    'cannabis',
    'weed', 
    'trees',
    'Marijuana',  # Capital M - as stored in Qdrant
    'weedstocks',
    'weedbiz'
]
