#!/usr/bin/env python3
"""
refactoring_plan.py - Plan for modularizing large scripts
"""

# Current large files and their proposed breakup:

REFACTORING_PLAN = {
    'enhanced_topic_extraction.py': {
        'current_size': '~400 lines',
        'issues': [
            'Multiple extraction methods in one file',
            'Topic labeling mixed with extraction',
            'Quality metrics mixed with algorithms'
        ],
        'proposed_split': {
            'extraction/bm25.py': 'extract_topics_bm25() and helpers',
            'extraction/mmr.py': 'extract_topics_mmr() and MMR algorithm',
            'extraction/keybert.py': 'extract_topics_keybert()',
            'labeling/topic_labeler.py': 'generate_topic_labels() and variations',
            'metrics/coherence.py': 'calculate_topic_coherence() and quality metrics'
        }
    },
    
    'advanced_nlp_tools.py': {
        'current_size': '~350 lines',
        'issues': [
            'Too many different NLP techniques in one file',
            'Mixing analysis types (argument, narrative, metaphor)'
        ],
        'proposed_split': {
            'nlp/argument_mining.py': 'ArgumentMiner class',
            'nlp/narrative_analysis.py': 'NarrativeAnalyzer class', 
            'nlp/metaphor_detection.py': 'MetaphorDetector class',
            'nlp/network_analysis.py': 'DiscussionNetworkAnalyzer class',
            'nlp/complexity_metrics.py': 'LinguisticComplexityAnalyzer class'
        }
    },
    
    'sentiment_analysis.py': {
        'current_size': '~300 lines',
        'issues': [
            'Multiple sentiment approaches in one file',
            'Mixing emotion, stance, and toxicity detection'
        ],
        'proposed_split': {
            'sentiment/aspect_sentiment.py': 'AspectSentimentAnalyzer class',
            'sentiment/emotion_detector.py': 'EmotionDetector class',
            'sentiment/stance_detector.py': 'StanceDetector class',
            'sentiment/toxicity_analyzer.py': 'ToxicityAnalyzer class',
            'sentiment/discourse_quality.py': 'DiscourseQualityAnalyzer class'
        }
    }
}

# Benefits of refactoring:
BENEFITS = {
    'modularity': 'Each file has single responsibility',
    'testability': 'Can unit test individual components',
    'reusability': 'Easy to import just what you need',
    'maintainability': 'Easier to find and fix bugs',
    'collaboration': 'Multiple people can work on different modules',
    'performance': 'Load only necessary components'
}

# Implementation priority:
PRIORITY = [
    '1. Create directory structure',
    '2. Split enhanced_topic_extraction.py (most used)',
    '3. Split sentiment_analysis.py (clear separation)',
    '4. Split advanced_nlp_tools.py (most complex)',
    '5. Update imports in main scripts',
    '6. Add comprehensive __init__.py files'
]

def generate_refactoring_script(module_name: str, splits: dict) -> str:
    """Generate script to split a module"""
    script = f"""#!/bin/bash
# Refactor {module_name}

# Create directories
mkdir -p analysis/{{extraction,labeling,metrics,nlp,sentiment}}

# Split the file
"""
    for new_file, content in splits.items():
        script += f"# Extract {content} to {new_file}\n"
    
    return script

# Example refactoring for one function:
EXAMPLE_REFACTOR = '''
# Before (in enhanced_topic_extraction.py):
def extract_topics_bm25(texts, labels, n_words=10):
    """300 lines of BM25 implementation"""
    ...

# After (in analysis/extraction/bm25.py):
"""BM25 topic extraction algorithm"""
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class BM25Extractor:
    """BM25-based topic extraction"""
    
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
    
    def extract_topics(self, texts: List[str], labels: np.ndarray, 
                      n_words: int = 10) -> Dict[int, Dict]:
        """Extract topics using BM25 scoring"""
        # Implementation here (100 lines)
        pass
'''

if __name__ == "__main__":
    print("REFACTORING PLAN FOR TOPICS_V1")
    print("="*60)
    
    for file, info in REFACTORING_PLAN.items():
        print(f"\n{file} ({info['current_size']})")
        print("Issues:")
        for issue in info['issues']:
            print(f"  - {issue}")
        print("Proposed split:")
        for new_file, content in info['proposed_split'].items():
            print(f"  {new_file:30} → {content}")
