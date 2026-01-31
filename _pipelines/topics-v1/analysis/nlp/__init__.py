"""
NLP Analysis Module for Cannabis Community Analysis

This module provides advanced Natural Language Processing tools for analyzing
cannabis-related discussions across different communities.

Modules:
    metaphor: Detect and analyze conceptual metaphors in cannabis discourse

Future modules (to be extracted from advanced_nlp_tools.py):
    argument_mining: Extract claims, evidence, and reasoning patterns
    narrative: Analyze narrative structures and story patterns
    framing: Identify how cannabis is framed in discussions
    complexity: Measure linguistic complexity and sophistication
    network: Analyze discussion networks and influence patterns
"""

from .metaphor import (
    MetaphorDetector,
    analyze_community_metaphors
)

__all__ = [
    # Metaphor detection
    'MetaphorDetector',
    'analyze_community_metaphors',
]

# Version info
__version__ = '0.1.0'
__author__ = 'Noetheca Topics Analysis'
