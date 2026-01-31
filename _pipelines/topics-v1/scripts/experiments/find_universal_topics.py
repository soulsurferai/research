#!/usr/bin/env python3
"""
find_universal_topics.py - Identify topics that bridge all cannabis communities
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import os

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CANNABIS_SUBREDDITS, OUTPUT_DIR
from sklearn.metrics.pairwise import cosine_similarity


def load_analysis_data():
    """Load the analysis summary data"""
    with open(os.path.join(OUTPUT_DIR, 'analysis_summary_advanced.json'), 'r') as f:
        return json.load(f)


def extract_topic_words(data: Dict) -> Dict[str, List[Tuple[str, List[str]]]]:
    """Extract topic words for each subreddit"""
    topic_words = {}
    
    for subreddit, analysis in data.items():
        if 'topics' in analysis:
            subreddit_topics = []
            for topic_id, topic_data in analysis['topics'].items():
                # Get the top words for this topic
                words = topic_data.get('words', [])
                label = topic_data.get('label', f'Topic {topic_id}')
                subreddit_topics.append((label, words))
            topic_words[subreddit] = subreddit_topics
    
    return topic_words


def find_common_themes(topic_words: Dict[str, List[Tuple[str, List[str]]]]) -> Dict[str, List[str]]:
    """Find themes that appear across multiple subreddits"""
    # Count word occurrences across all topics
    word_to_subreddits = defaultdict(set)
    theme_to_subreddits = defaultdict(set)
    
    for subreddit, topics in topic_words.items():
        for label, words in topics:
            # Track which subreddits have each word
            for word in words:
                word_to_subreddits[word].add(subreddit)
            
            # Track theme labels
            theme_to_subreddits[label].add(subreddit)
    
    # Find universal words (appear in 4+ subreddits)
    universal_words = {
        word: list(subreddits) 
        for word, subreddits in word_to_subreddits.items() 
        if len(subreddits) >= 4
    }
    
    # Find common themes
    common_themes = {
        theme: list(subreddits)
        for theme, subreddits in theme_to_subreddits.items()
        if len(subreddits) >= 2
    }
    
    return {
        'universal_words': universal_words,
        'common_themes': common_themes,
        'word_distribution': {
            word: len(subs) for word, subs in word_to_subreddits.items()
        }
    }


def analyze_topic_similarity(data: Dict) -> Dict[str, float]:
    """Analyze semantic similarity between topics across communities"""
    similarities = []
    topic_pairs = []
    
    # Compare topics between different subreddits
    subreddits = list(data.keys())
    for i, sub1 in enumerate(subreddits):
        for j, sub2 in enumerate(subreddits[i+1:], i+1):
            if 'topics' not in data[sub1] or 'topics' not in data[sub2]:
                continue
                
            # Compare each topic from sub1 with each from sub2
            for t1_id, t1_data in data[sub1]['topics'].items():
                for t2_id, t2_data in data[sub2]['topics'].items():
                    # Simple word overlap similarity
                    words1 = set(t1_data.get('words', []))
                    words2 = set(t2_data.get('words', []))
                    
                    if words1 and words2:
                        overlap = len(words1 & words2)
                        similarity = overlap / min(len(words1), len(words2))
                        
                        if similarity > 0.3:  # Significant overlap
                            similarities.append(similarity)
                            topic_pairs.append({
                                'sub1': sub1,
                                'sub2': sub2,
                                'topic1': t1_data.get('label', t1_id),
                                'topic2': t2_data.get('label', t2_id),
                                'similarity': similarity,
                                'common_words': list(words1 & words2)
                            })
    
    # Sort by similarity
    topic_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        'average_similarity': np.mean(similarities) if similarities else 0,
        'max_similarity': max(similarities) if similarities else 0,
        'similar_topic_pairs': topic_pairs[:20]  # Top 20 most similar
    }


def identify_bridge_topics(common_themes: Dict, topic_similarity: Dict) -> List[Dict]:
    """Identify topics that could serve as bridges between communities"""
    bridges = []
    
    # Universal words that could serve as bridges
    universal_words = common_themes['universal_words']
    for word, subreddits in universal_words.items():
        if len(subreddits) >= 5:  # Appears in most communities
            bridges.append({
                'type': 'universal_word',
                'content': word,
                'communities': subreddits,
                'strength': len(subreddits) / 6.0  # Normalized by total communities
            })
    
    # Similar topics between different communities
    for pair in topic_similarity['similar_topic_pairs'][:10]:
        if pair['similarity'] > 0.5:
            bridges.append({
                'type': 'similar_topics',
                'content': f"{pair['topic1']} ↔ {pair['topic2']}",
                'communities': [pair['sub1'], pair['sub2']],
                'strength': pair['similarity'],
                'common_words': pair['common_words']
            })
    
    return bridges


def generate_insights_report(results: Dict):
    """Generate a report of universal topics and bridges"""
    report = ["# Universal Topics & Bridge Analysis\n"]
    
    # Universal words section
    report.append("## 🌍 Universal Words (Found in 4+ Communities)\n")
    universal = results['common_themes']['universal_words']
    for word, subs in sorted(universal.items(), key=lambda x: len(x[1]), reverse=True):
        report.append(f"- **{word}**: {', '.join(subs)} ({len(subs)}/6 communities)")
    
    # Common themes section
    report.append("\n## 🔗 Common Theme Labels (Found in 2+ Communities)\n")
    themes = results['common_themes']['common_themes']
    theme_groups = defaultdict(list)
    for theme, subs in themes.items():
        theme_groups[len(subs)].append((theme, subs))
    
    for count in sorted(theme_groups.keys(), reverse=True):
        report.append(f"\n### Themes in {count} communities:")
        for theme, subs in theme_groups[count]:
            report.append(f"- **{theme}**: {', '.join(subs)}")
    
    # Bridge topics section
    report.append("\n## 🌉 Bridge Topics (High Potential for Cross-Community Connection)\n")
    for bridge in results['bridges'][:10]:
        if bridge['type'] == 'universal_word':
            report.append(f"\n### Universal Concept: '{bridge['content']}'")
            report.append(f"- Appears in: {', '.join(bridge['communities'])}")
            report.append(f"- Bridge strength: {bridge['strength']:.2%}")
        else:
            report.append(f"\n### Similar Topics: {bridge['content']}")
            report.append(f"- Communities: {', '.join(bridge['communities'])}")
            report.append(f"- Similarity: {bridge['strength']:.2%}")
            report.append(f"- Common words: {', '.join(bridge['common_words'][:5])}")
    
    # Insights section
    report.append("\n## 💡 Key Insights\n")
    
    # Calculate some statistics
    total_universal = len(results['common_themes']['universal_words'])
    avg_similarity = results['topic_similarity']['average_similarity']
    
    report.append(f"1. **{total_universal} universal words** connect 4+ cannabis communities")
    report.append(f"2. **Average topic similarity** between communities: {avg_similarity:.1%}")
    report.append(f"3. **Strongest bridges** are around: {', '.join(list(universal.keys())[:5])}")
    
    # Recommendations
    report.append("\n## 🎯 Recommendations for Community Building\n")
    report.append("1. **Focus on universal concepts** like 'legalization', 'medical', and 'federal' that resonate across all communities")
    report.append("2. **Build content around bridge topics** that naturally connect different community interests")
    report.append("3. **Translate community-specific language** - same concepts, different vocabularies")
    report.append("4. **Create cross-community initiatives** around shared concerns like policy and health")
    
    return '\n'.join(report)


def main():
    print("🔍 Finding Universal Topics Across Cannabis Communities...\n")
    
    # Load analysis data
    data = load_analysis_data()
    
    # Extract topic words
    print("📊 Extracting topic words from each community...")
    topic_words = extract_topic_words(data)
    
    # Find common themes
    print("🔗 Identifying common themes and universal words...")
    common_themes = find_common_themes(topic_words)
    
    # Analyze topic similarity
    print("📈 Calculating topic similarities between communities...")
    topic_similarity = analyze_topic_similarity(data)
    
    # Identify bridge topics
    print("🌉 Identifying potential bridge topics...")
    bridges = identify_bridge_topics(common_themes, topic_similarity)
    
    # Compile results
    results = {
        'common_themes': common_themes,
        'topic_similarity': topic_similarity,
        'bridges': bridges
    }
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'universal_topics_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved detailed results to {output_path}")
    
    # Generate report
    report = generate_insights_report(results)
    report_path = os.path.join(OUTPUT_DIR, 'universal_topics_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📝 Generated report at {report_path}")
    
    # Print summary
    print("\n🎯 Quick Summary:")
    print(f"- Found {len(common_themes['universal_words'])} universal words")
    print(f"- Identified {len(bridges)} potential bridge topics")
    print(f"- Average cross-community topic similarity: {topic_similarity['average_similarity']:.1%}")
    
    # Print top universal words
    print("\n🌟 Top Universal Words:")
    for word, subs in list(common_themes['universal_words'].items())[:10]:
        print(f"  - '{word}' appears in {len(subs)} communities")


if __name__ == "__main__":
    main()
