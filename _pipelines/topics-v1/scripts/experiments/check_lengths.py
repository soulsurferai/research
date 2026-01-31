#!/usr/bin/env python3
"""Check script lengths in the project"""

import os

files = {
    'topics_simple.py': 'topics_simple.py',
    'topics_modular.py': 'topics_modular.py', 
    'topics_advanced.py': 'topics_advanced.py',
    'enhanced_topic_extraction.py': 'analysis/enhanced_topic_extraction.py',
    'enhanced_clustering.py': 'analysis/enhanced_clustering.py',
    'sentiment_analysis.py': 'analysis/sentiment_analysis.py',
    'advanced_nlp_tools.py': 'analysis/advanced_nlp_tools.py',
    'algorithm_improvements.py': 'analysis/algorithm_improvements.py'
}

for name, path in files.items():
    try:
        with open(path, 'r') as f:
            lines = len(f.readlines())
        print(f"{name:35} : {lines:4} lines")
    except:
        print(f"{name:35} : NOT FOUND")
