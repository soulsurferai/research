#!/usr/bin/env python3
"""
discover_emotion_markers.py - Discover actual emotional language in your data

This helps fine-tune emotion definitions by finding:
1. What emotional words actually appear
2. Context around emotional expressions
3. Community-specific emotional language
"""

import json
import re
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Set
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import OUTPUT_DIR
from utils import QdrantFetcher


class EmotionMarkerDiscovery:
    """Discover emotional language patterns in actual data"""
    
    def __init__(self):
        self.fetcher = QdrantFetcher()
        
        # Seed words for each emotion pole - minimal starting point
        self.seed_emotions = {
            'hope': ['hope', 'hopeful', 'optimistic'],
            'despair': ['hopeless', 'desperate', 'despair'],
            'trust': ['trust', 'believe', 'faith'],
            'suspicion': ['suspicious', 'doubt', 'skeptical'],
            'excitement': ['excited', 'thrilled', 'amazing'],
            'anxiety': ['anxious', 'worried', 'scared'],
            'pride': ['proud', 'accomplished', 'success'],
            'shame': ['ashamed', 'embarrassed', 'guilty'],
            'empowerment': ['empowered', 'control', 'strong'],
            'helplessness': ['helpless', 'powerless', 'weak'],
            'community': ['together', 'community', 'support'],
            'isolation': ['alone', 'isolated', 'lonely'],
            'clarity': ['clear', 'understand', 'obvious'],
            'confusion': ['confused', 'unclear', 'lost']
        }
        
        # Common patterns that indicate emotional states
        self.emotion_patterns = {
            'hope': [
                r"can't wait (for|to|until)",
                r"looking forward to",
                r"excited about the future",
                r"things are getting better",
                r"finally happening"
            ],
            'despair': [
                r"no point in",
                r"given up on",
                r"never going to",
                r"what's the point",
                r"lost all hope"
            ],
            'anxiety': [
                r"freaking out about",
                r"can't stop worrying",
                r"keeps me up at night",
                r"terrified of",
                r"panic about"
            ],
            'pride': [
                r"proud of (my|our|the)",
                r"accomplished (something|my|our)",
                r"finally did it",
                r"look how far",
                r"achievement unlocked"
            ],
            'community': [
                r"we're all in this together",
                r"love this community",
                r"you guys are",
                r"thanks? (everyone|guys|all)",
                r"one of us"
            ]
        }
    
    def find_emotion_contexts(self, texts: List[str], emotion: str, 
                            window_size: int = 50) -> List[Dict]:
        """Find contexts where emotion words appear"""
        contexts = []
        seed_words = self.seed_emotions.get(emotion, [])
        patterns = self.emotion_patterns.get(emotion, [])
        
        # Create regex for seed words
        seed_pattern = r'\b(' + '|'.join(re.escape(word) for word in seed_words) + r')\b'
        seed_regex = re.compile(seed_pattern, re.IGNORECASE)
        
        for text in texts:
            # Find seed word matches
            for match in seed_regex.finditer(text):
                start = max(0, match.start() - window_size)
                end = min(len(text), match.end() + window_size)
                context = text[start:end]
                
                contexts.append({
                    'emotion': emotion,
                    'matched_word': match.group(),
                    'context': context,
                    'full_text': text[:200]  # First 200 chars for reference
                })
            
            # Find pattern matches
            for pattern_str in patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                for match in pattern.finditer(text):
                    start = max(0, match.start() - window_size)
                    end = min(len(text), match.end() + window_size)
                    context = text[start:end]
                    
                    contexts.append({
                        'emotion': emotion,
                        'matched_pattern': match.group(),
                        'context': context,
                        'full_text': text[:200]
                    })
        
        return contexts
    
    def extract_cooccurring_words(self, contexts: List[Dict], 
                                 emotion: str) -> Counter:
        """Find words that co-occur with emotional expressions"""
        cooccurring = Counter()
        
        # Words to exclude (common words)
        exclude = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                  'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 
                  'was', 'were', 'been', 'be', 'have', 'has', 'had', 
                  'do', 'does', 'did', 'will', 'would', 'could', 'should',
                  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that'}
        
        for ctx in contexts:
            # Extract words from context
            words = re.findall(r'\b\w+\b', ctx['context'].lower())
            
            # Count words that aren't in exclude list or seed words
            seed_words = set(self.seed_emotions.get(emotion, []))
            for word in words:
                if (len(word) > 3 and 
                    word not in exclude and 
                    word not in seed_words):
                    cooccurring[word] += 1
        
        return cooccurring
    
    def discover_subreddit_emotions(self, subreddit: str, 
                                  n_samples: int = 1000) -> Dict:
        """Discover emotional language patterns in a subreddit"""
        print(f"\n🔍 Discovering emotional language in r/{subreddit}...")
        
        # Fetch data
        _, texts, _ = self.fetcher.fetch_subreddit_data(subreddit, n_samples)
        
        discoveries = {}
        
        # Analyze each emotion
        for emotion in self.seed_emotions:
            print(f"  Analyzing {emotion}...")
            
            # Find contexts
            contexts = self.find_emotion_contexts(texts, emotion)
            
            if contexts:
                # Extract co-occurring words
                cooccurring = self.extract_cooccurring_words(contexts, emotion)
                
                # Get top co-occurring words
                top_words = cooccurring.most_common(20)
                
                discoveries[emotion] = {
                    'count': len(contexts),
                    'top_cooccurring': top_words,
                    'example_contexts': contexts[:5],  # Save a few examples
                    'discovered_phrases': self.extract_phrases(contexts, emotion)
                }
            else:
                discoveries[emotion] = {
                    'count': 0,
                    'top_cooccurring': [],
                    'example_contexts': [],
                    'discovered_phrases': []
                }
        
        return discoveries
    
    def extract_phrases(self, contexts: List[Dict], emotion: str) -> List[str]:
        """Extract common phrases associated with an emotion"""
        phrases = Counter()
        
        # Look for 2-4 word phrases around emotion words
        for ctx in contexts:
            text = ctx['context'].lower()
            
            # Find 2-4 word sequences
            words = text.split()
            for i in range(len(words) - 1):
                for length in [2, 3, 4]:
                    if i + length <= len(words):
                        phrase = ' '.join(words[i:i+length])
                        
                        # Check if phrase contains emotion-related content
                        if any(seed in phrase for seed in self.seed_emotions.get(emotion, [])):
                            phrases[phrase] += 1
        
        # Return most common phrases
        return [phrase for phrase, count in phrases.most_common(10) if count > 2]
    
    def compare_community_emotions(self, subreddits: List[str], 
                                 n_samples: int = 500) -> Dict:
        """Compare emotional language across communities"""
        all_discoveries = {}
        
        for subreddit in subreddits:
            discoveries = self.discover_subreddit_emotions(subreddit, n_samples)
            all_discoveries[subreddit] = discoveries
        
        # Find community-specific vs universal patterns
        comparisons = {}
        
        for emotion in self.seed_emotions:
            emotion_comparison = {
                'universal_words': [],
                'community_specific': {},
                'phrase_variations': {}
            }
            
            # Collect all words for this emotion across communities
            all_words = defaultdict(list)
            all_phrases = defaultdict(list)
            
            for sub, disc in all_discoveries.items():
                if emotion in disc:
                    # Collect words
                    for word, count in disc[emotion]['top_cooccurring']:
                        all_words[word].append(sub)
                    
                    # Collect phrases
                    for phrase in disc[emotion]['discovered_phrases']:
                        all_phrases[phrase].append(sub)
            
            # Find universal words (appear in multiple communities)
            for word, subs in all_words.items():
                if len(subs) >= len(subreddits) / 2:
                    emotion_comparison['universal_words'].append(word)
                else:
                    # Community-specific
                    for sub in subs:
                        if sub not in emotion_comparison['community_specific']:
                            emotion_comparison['community_specific'][sub] = []
                        emotion_comparison['community_specific'][sub].append(word)
            
            # Store phrase variations
            emotion_comparison['phrase_variations'] = dict(all_phrases)
            
            comparisons[emotion] = emotion_comparison
        
        return {
            'discoveries': all_discoveries,
            'comparisons': comparisons
        }
    
    def generate_refined_markers(self, discoveries: Dict) -> Dict:
        """Generate refined emotion markers based on discoveries"""
        refined_markers = {}
        
        for emotion, data in discoveries['comparisons'].items():
            # Start with seed words
            markers = self.seed_emotions[emotion].copy()
            
            # Add universal words that appear frequently
            markers.extend(data['universal_words'][:10])
            
            # Add common phrases (converted to individual words)
            for phrase, communities in data['phrase_variations'].items():
                if len(communities) >= 2:  # Appears in multiple communities
                    # Extract meaningful words from phrase
                    words = phrase.split()
                    for word in words:
                        if (len(word) > 3 and 
                            word not in markers and 
                            word not in ['the', 'and', 'for', 'with']):
                            markers.append(word)
            
            # Remove duplicates while preserving order
            seen = set()
            refined_markers[emotion] = [
                x for x in markers if not (x in seen or seen.add(x))
            ]
        
        return refined_markers
    
    def save_discoveries(self, discoveries: Dict, output_prefix: str):
        """Save discovered patterns and refined markers"""
        # Save full discoveries
        discovery_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_emotion_discoveries.json')
        with open(discovery_path, 'w') as f:
            json.dump(discoveries, f, indent=2)
        
        # Generate and save refined markers
        refined = self.generate_refined_markers(discoveries)
        refined_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_refined_markers.json')
        with open(refined_path, 'w') as f:
            json.dump(refined, f, indent=2)
        
        # Generate human-readable report
        report = self.generate_discovery_report(discoveries, refined)
        report_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_emotion_discovery_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        return discovery_path, refined_path, report_path
    
    def generate_discovery_report(self, discoveries: Dict, refined: Dict) -> str:
        """Generate human-readable report of discoveries"""
        report = ["# Emotion Marker Discovery Report\n"]
        
        # Summary
        report.append("## Summary\n")
        report.append(f"Analyzed {len(discoveries['discoveries'])} communities\n")
        
        # Universal patterns
        report.append("\n## Universal Emotional Patterns\n")
        for emotion, data in discoveries['comparisons'].items():
            if data['universal_words']:
                report.append(f"\n### {emotion.title()}")
                report.append(f"Universal markers: {', '.join(data['universal_words'][:10])}")
        
        # Community-specific patterns
        report.append("\n## Community-Specific Patterns\n")
        for sub, disc in discoveries['discoveries'].items():
            report.append(f"\n### r/{sub}")
            
            # Find this community's unique expressions
            for emotion, data in disc.items():
                if data['count'] > 0:
                    report.append(f"\n**{emotion.title()}** ({data['count']} instances)")
                    
                    # Top co-occurring words
                    if data['top_cooccurring']:
                        top_5 = [word for word, _ in data['top_cooccurring'][:5]]
                        report.append(f"- Top words: {', '.join(top_5)}")
                    
                    # Example context
                    if data['example_contexts']:
                        ctx = data['example_contexts'][0]
                        snippet = ctx['context'][:100] + '...'
                        report.append(f"- Example: \"{snippet}\"")
        
        # Refined markers
        report.append("\n## Refined Emotion Markers\n")
        report.append("Based on the analysis, here are refined marker sets:\n")
        
        for emotion, markers in refined.items():
            report.append(f"\n**{emotion.title()}**: {', '.join(markers)}")
        
        return '\n'.join(report)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Discover emotional language patterns in Reddit data'
    )
    
    parser.add_argument(
        '--subreddits',
        nargs='+',
        default=['cannabis', 'anxiety', 'depression', 'happy'],
        help='Subreddits to analyze'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=500,
        help='Samples per subreddit'
    )
    
    parser.add_argument(
        '--output-prefix',
        default='emotion_discovery',
        help='Prefix for output files'
    )
    
    args = parser.parse_args()
    
    print("🔍 Discovering Emotional Language Patterns\n")
    
    discoverer = EmotionMarkerDiscovery()
    
    # Run discovery
    discoveries = discoverer.compare_community_emotions(
        args.subreddits, 
        args.samples
    )
    
    # Save results
    paths = discoverer.save_discoveries(discoveries, args.output_prefix)
    
    print(f"\n✅ Discovery complete!")
    print(f"📊 Full discoveries: {paths[0]}")
    print(f"🎯 Refined markers: {paths[1]}")
    print(f"📄 Report: {paths[2]}")
    
    # Print quick summary
    print("\n🎯 Quick Summary:")
    refined = discoverer.generate_refined_markers(discoveries)
    for emotion, markers in list(refined.items())[:3]:
        print(f"{emotion}: {', '.join(markers[:5])}...")


if __name__ == "__main__":
    main()
