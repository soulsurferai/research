#!/usr/bin/env python3
"""
test_metaphor_detection.py - Test the metaphor detection module
"""

import json
from analysis.nlp import MetaphorDetector, analyze_community_metaphors

# Sample texts for testing
sample_texts = {
    'medical_focused': [
        "Cannabis has been my medicine for years, treating my chronic pain better than any prescription.",
        "The healing properties of this plant are incredible. It's nature's remedy.",
        "My doctor prescribed medical marijuana and it's been a real cure for my symptoms.",
        "This treatment has given me my life back. The therapeutic effects are undeniable."
    ],
    'journey_focused': [
        "My cannabis journey started five years ago, and what a trip it's been.",
        "I'm still finding my path with different strains, exploring new destinations.",
        "Each session is a voyage into relaxation, a journey away from stress.",
        "The road to understanding cannabis has been long but rewarding."
    ],
    'freedom_focused': [
        "Legalization means freedom from unjust imprisonment over a plant.",
        "We need to break the chains of prohibition and liberate cannabis users.",
        "Too many people are locked up for something that should be a personal liberty.",
        "It's time to escape the prison of outdated drug laws."
    ]
}

def test_basic_detection():
    """Test basic metaphor detection"""
    print("TESTING BASIC METAPHOR DETECTION")
    print("="*60)
    
    detector = MetaphorDetector()
    
    # Combine all sample texts
    all_texts = []
    for texts in sample_texts.values():
        all_texts.extend(texts)
    
    results = detector.detect_metaphors(all_texts)
    
    print(f"\nTotal metaphors found: {results['total_metaphors']}")
    print(f"\nMetaphor counts:")
    for m_type, count in sorted(results['metaphor_counts'].items(), 
                                key=lambda x: x[1], reverse=True):
        print(f"  {m_type:15} : {count:3} ({results['metaphor_distribution'][m_type]:.1%})")
    
    print(f"\nDominant metaphors: {results['dominant_metaphors']}")
    
    print(f"\nExample contexts:")
    for m_type, examples in results['examples'].items():
        print(f"\n{m_type.upper()}:")
        for ex in examples[:2]:
            print(f"  '{ex['keyword']}': {ex['context']}")


def test_comparison():
    """Test metaphor comparison between groups"""
    print("\n\nTESTING METAPHOR COMPARISON")
    print("="*60)
    
    detector = MetaphorDetector()
    
    # Compare medical vs freedom focused texts
    comparison = detector.compare_metaphor_usage(
        sample_texts['medical_focused'],
        sample_texts['freedom_focused'],
        'Medical Focus',
        'Freedom Focus'
    )
    
    print("\nMetaphor usage comparison:")
    print(f"{'Metaphor':15} {'Medical':>10} {'Freedom':>10} {'Difference':>12}")
    print("-"*50)
    
    for metaphor in ['medicine', 'freedom', 'nature', 'journey']:
        med_freq = comparison['Medical Focus'].get(metaphor, 0)
        free_freq = comparison['Freedom Focus'].get(metaphor, 0)
        diff = comparison['differences'].get(metaphor, 0)
        print(f"{metaphor:15} {med_freq:10.1%} {free_freq:10.1%} {diff:+12.1%}")
    
    print(f"\nBiggest differences:")
    for metaphor, diff in comparison['biggest_differences']:
        print(f"  {metaphor}: {diff:+.1%}")


def test_cooccurrence():
    """Test metaphor co-occurrence analysis"""
    print("\n\nTESTING METAPHOR CO-OCCURRENCE")
    print("="*60)
    
    detector = MetaphorDetector()
    
    # Create texts with multiple metaphors
    mixed_texts = [
        "My healing journey with medical cannabis has been liberating.",
        "This natural medicine has freed me from the chains of prescription drugs.",
        "The path to wellness through this plant medicine is a journey of freedom.",
        "Nature's cure has been my escape from pharmaceutical prison."
    ]
    
    cooccurrence = detector.analyze_metaphor_cooccurrence(mixed_texts)
    
    print("\nMetaphors that appear together:")
    for pair, count in sorted(cooccurrence.items(), 
                             key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {pair}: {count} times")


def test_community_analysis():
    """Test the convenience function"""
    print("\n\nTESTING COMMUNITY ANALYSIS")
    print("="*60)
    
    # Analyze journey-focused community
    results = analyze_community_metaphors(
        sample_texts['journey_focused'],
        'r/psychonauts'  # Hypothetical community
    )
    
    print(f"\nCommunity: {results['community']}")
    print(f"Interpretation: {results['interpretation']}")
    print(f"Dominant metaphors: {results['dominant_metaphors']}")


if __name__ == "__main__":
    test_basic_detection()
    test_comparison()
    test_cooccurrence()
    test_community_analysis()
    
    print("\n\nAll tests completed! ✓")
