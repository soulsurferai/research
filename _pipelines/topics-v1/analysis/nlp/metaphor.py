"""
metaphor.py - Detect and analyze conceptual metaphors in cannabis discourse

This module identifies how cannabis is conceptualized through metaphorical language,
revealing underlying cognitive frameworks used by different communities.
"""

import re
from typing import List, Dict, Tuple
from collections import defaultdict


class MetaphorDetector:
    """Detect conceptual metaphors used in cannabis discussions"""
    
    def __init__(self):
        """Initialize with cannabis-specific metaphor patterns"""
        self.metaphor_patterns = {
            'journey': {
                'keywords': ['journey', 'path', 'road', 'destination', 'milestone', 
                            'trip', 'voyage', 'route', 'way', 'travel'],
                'description': 'Cannabis use as a journey or travel experience'
            },
            'war': {
                'keywords': ['battle', 'fight', 'weapon', 'victory', 'defeat', 
                            'combat', 'struggle', 'war', 'enemy', 'conquer'],
                'description': 'Cannabis policy/addiction as warfare'
            },
            'medicine': {
                'keywords': ['healing', 'cure', 'treatment', 'prescription', 'dose',
                            'remedy', 'therapy', 'patient', 'symptom', 'diagnose'],
                'description': 'Cannabis as medicine or therapeutic agent'
            },
            'nature': {
                'keywords': ['plant', 'grow', 'flower', 'roots', 'bloom', 
                            'seed', 'harvest', 'cultivate', 'organic', 'natural'],
                'description': 'Cannabis in natural/agricultural terms'
            },
            'freedom': {
                'keywords': ['liberation', 'chains', 'prison', 'escape', 'free',
                            'locked', 'release', 'captive', 'bound', 'liberty'],
                'description': 'Cannabis legalization as freedom/captivity'
            },
            'enlightenment': {
                'keywords': ['awakening', 'consciousness', 'clarity', 'vision', 'light',
                            'enlighten', 'aware', 'mindful', 'spiritual', 'transcend'],
                'description': 'Cannabis as spiritual/consciousness tool'
            },
            'relationship': {
                'keywords': ['love', 'romance', 'marry', 'divorce', 'affair',
                            'relationship', 'partner', 'faithful', 'cheat', 'bond'],
                'description': 'Cannabis use as romantic relationship'
            },
            'food': {
                'keywords': ['appetite', 'hunger', 'feast', 'taste', 'flavor',
                            'recipe', 'ingredient', 'cook', 'chef', 'gourmet'],
                'description': 'Cannabis as culinary experience'
            }
        }
    
    def detect_metaphors(self, texts: List[str]) -> Dict[str, Dict]:
        """
        Detect metaphor usage across a corpus of texts
        
        Args:
            texts: List of text documents
            
        Returns:
            Dictionary with metaphor statistics and examples
        """
        metaphor_counts = defaultdict(int)
        metaphor_examples = defaultdict(list)
        metaphor_contexts = defaultdict(list)
        
        for text in texts:
            text_lower = text.lower()
            
            # Check each metaphor type
            for metaphor_type, pattern_info in self.metaphor_patterns.items():
                keywords = pattern_info['keywords']
                
                for keyword in keywords:
                    if keyword in text_lower:
                        metaphor_counts[metaphor_type] += 1
                        
                        # Extract context if we don't have enough examples
                        if len(metaphor_examples[metaphor_type]) < 5:
                            context = self._extract_context(text, keyword)
                            if context:
                                metaphor_examples[metaphor_type].append({
                                    'keyword': keyword,
                                    'context': context
                                })
        
        # Calculate statistics
        total_metaphors = sum(metaphor_counts.values())
        
        results = {
            'metaphor_counts': dict(metaphor_counts),
            'total_metaphors': total_metaphors,
            'metaphor_distribution': {
                m_type: count / total_metaphors if total_metaphors > 0 else 0
                for m_type, count in metaphor_counts.items()
            },
            'dominant_metaphors': self._get_dominant_metaphors(metaphor_counts, n=3),
            'examples': dict(metaphor_examples),
            'descriptions': {
                m_type: info['description'] 
                for m_type, info in self.metaphor_patterns.items()
            }
        }
        
        return results
    
    def analyze_metaphor_cooccurrence(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze which metaphors tend to appear together
        
        Args:
            texts: List of text documents
            
        Returns:
            Co-occurrence matrix of metaphor types
        """
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for text in texts:
            text_lower = text.lower()
            present_metaphors = set()
            
            # Find all metaphors present in this text
            for metaphor_type, pattern_info in self.metaphor_patterns.items():
                if any(keyword in text_lower for keyword in pattern_info['keywords']):
                    present_metaphors.add(metaphor_type)
            
            # Count co-occurrences
            for m1 in present_metaphors:
                for m2 in present_metaphors:
                    if m1 != m2:
                        cooccurrence[m1][m2] += 1
        
        return {
            f"{m1}-{m2}": count 
            for m1, counts in cooccurrence.items() 
            for m2, count in counts.items()
        }
    
    def compare_metaphor_usage(self, 
                              texts_a: List[str], 
                              texts_b: List[str],
                              label_a: str = "Group A",
                              label_b: str = "Group B") -> Dict[str, Dict]:
        """
        Compare metaphor usage between two groups of texts
        
        Args:
            texts_a: First group of texts
            texts_b: Second group of texts
            label_a: Label for first group
            label_b: Label for second group
            
        Returns:
            Comparison statistics and differences
        """
        results_a = self.detect_metaphors(texts_a)
        results_b = self.detect_metaphors(texts_b)
        
        comparison = {
            label_a: results_a['metaphor_distribution'],
            label_b: results_b['metaphor_distribution'],
            'differences': {}
        }
        
        # Calculate differences
        all_metaphors = set(results_a['metaphor_distribution'].keys()) | \
                       set(results_b['metaphor_distribution'].keys())
        
        for metaphor in all_metaphors:
            freq_a = results_a['metaphor_distribution'].get(metaphor, 0)
            freq_b = results_b['metaphor_distribution'].get(metaphor, 0)
            comparison['differences'][metaphor] = freq_a - freq_b
        
        # Sort by biggest differences
        comparison['biggest_differences'] = sorted(
            comparison['differences'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        return comparison
    
    def _extract_context(self, text: str, keyword: str, window: int = 50) -> str:
        """Extract context around a keyword"""
        text_lower = text.lower()
        idx = text_lower.find(keyword.lower())
        
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        
        context = text[start:end].strip()
        
        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
            
        return context
    
    def _get_dominant_metaphors(self, 
                               metaphor_counts: Dict[str, int], 
                               n: int = 3) -> List[Tuple[str, int]]:
        """Get the n most common metaphor types"""
        return sorted(
            metaphor_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]


def analyze_community_metaphors(texts: List[str], 
                               community: str) -> Dict[str, any]:
    """
    Convenience function to analyze metaphors for a specific community
    
    Args:
        texts: List of texts from the community
        community: Name of the community
        
    Returns:
        Analysis results with community context
    """
    detector = MetaphorDetector()
    results = detector.detect_metaphors(texts)
    
    # Add community-specific interpretation
    results['community'] = community
    results['interpretation'] = _interpret_metaphor_profile(
        results['dominant_metaphors'],
        community
    )
    
    return results


def _interpret_metaphor_profile(dominant_metaphors: List[Tuple[str, int]], 
                               community: str) -> str:
    """Interpret what dominant metaphors say about a community"""
    
    if not dominant_metaphors:
        return "No clear metaphorical patterns detected"
    
    top_metaphor = dominant_metaphors[0][0]
    
    interpretations = {
        'journey': f"{community} views cannabis use as a personal journey or experience",
        'war': f"{community} frames cannabis issues in terms of conflict and struggle",
        'medicine': f"{community} conceptualizes cannabis primarily as medicine",
        'nature': f"{community} emphasizes the natural/plant aspects of cannabis",
        'freedom': f"{community} sees cannabis through the lens of personal liberty",
        'enlightenment': f"{community} associates cannabis with consciousness expansion",
        'relationship': f"{community} treats cannabis use like a personal relationship",
        'food': f"{community} approaches cannabis with culinary/gourmet perspective"
    }
    
    return interpretations.get(top_metaphor, 
                               f"{community} shows unique metaphorical patterns")
