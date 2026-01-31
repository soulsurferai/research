#!/usr/bin/env python3
"""
emotion_landscape_analysis.py - Nuanced emotional analysis across cannabis communities

Goes beyond positive/negative to analyze complex emotional states:
- Hope vs Despair
- Trust vs Suspicion  
- Excitement vs Anxiety
- Pride vs Shame
- Empowerment vs Helplessness
- Community vs Isolation
- Clarity vs Confusion
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import sys
from typing import Dict, List, Tuple
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CANNABIS_SUBREDDITS, OUTPUT_DIR
from utils import QdrantFetcher


class EmotionLandscapeAnalyzer:
    """Analyze nuanced emotional landscapes in cannabis discourse"""
    
    def __init__(self):
        self.fetcher = QdrantFetcher()
        
        # Define emotional dimensions with associated markers
        self.emotion_dimensions = {
            'hope_despair': {
                'hope': ['hope', 'hopeful', 'optimistic', 'future', 'promising', 'excited for', 
                        'looking forward', 'better days', 'progress', 'breakthrough', 'finally'],
                'despair': ['hopeless', 'desperate', 'giving up', 'no point', 'doomed', 'fucked',
                           'never going to', 'lost cause', 'despair', 'pointless', 'why bother']
            },
            'trust_suspicion': {
                'trust': ['trust', 'reliable', 'legitimate', 'verified', 'proven', 'confident',
                         'faith in', 'count on', 'dependable', 'authentic', 'genuine'],
                'suspicion': ['suspicious', 'sketchy', 'shady', 'doubt', 'questionable', 'scam',
                             "don't trust", 'fake', 'lying', 'rigged', 'conspiracy']
            },
            'excitement_anxiety': {
                'excitement': ['excited', 'pumped', 'hyped', 'thrilled', 'stoked', "can't wait",
                              'amazing', 'incredible', 'best thing', 'game changer', 'revolutionary'],
                'anxiety': ['anxious', 'worried', 'nervous', 'paranoid', 'freaking out', 'scared',
                           'panic', 'overwhelming', 'stressed', 'fear', 'terrified']
            },
            'pride_shame': {
                'pride': ['proud', 'accomplished', 'success', 'achievement', 'milestone', 'victory',
                         'winning', 'conquered', 'overcame', 'badge of honor', 'dignity'],
                'shame': ['ashamed', 'embarrassed', 'humiliated', 'guilty', 'stigma', 'hide',
                         'secret', 'closeted', 'judged', 'disgrace', 'failure']
            },
            'empowerment_helplessness': {
                'empowerment': ['empowered', 'control', 'agency', 'choice', 'freedom', 'independent',
                               'taking charge', 'my decision', 'liberated', 'self-determined', 'sovereign'],
                'helplessness': ['helpless', 'powerless', 'trapped', 'stuck', 'no choice', 'forced',
                                'victim', 'at mercy of', 'dependent', 'enslaved', 'oppressed']
            },
            'community_isolation': {
                'community': ['community', 'together', 'family', 'tribe', 'belong', 'support',
                             'solidarity', 'united', 'friends', 'brotherhood', 'sisterhood'],
                'isolation': ['alone', 'isolated', 'lonely', 'outcast', 'excluded', 'alienated',
                             'nobody understands', 'by myself', 'disconnected', 'solitary', 'rejected']
            },
            'clarity_confusion': {
                'clarity': ['clear', 'understand', 'makes sense', 'obvious', 'enlightened', 'realize',
                           'crystal clear', 'figured out', 'coherent', 'lucid', 'illuminated'],
                'confusion': ['confused', 'unclear', "don't understand", 'mixed signals', 'contradictory',
                             'makes no sense', 'lost', 'bewildered', 'puzzled', 'ambiguous']
            }
        }
        
        # Compile regex patterns for efficiency
        self.emotion_patterns = {}
        for dimension, poles in self.emotion_dimensions.items():
            self.emotion_patterns[dimension] = {}
            for pole, markers in poles.items():
                # Create regex pattern with word boundaries
                pattern = r'\b(' + '|'.join(re.escape(marker) for marker in markers) + r')\b'
                self.emotion_patterns[dimension][pole] = re.compile(pattern, re.IGNORECASE)
    
    def analyze_text_emotions(self, text: str) -> Dict[str, Dict[str, float]]:
        """Analyze emotional dimensions in a single text"""
        emotions = {}
        text_lower = text.lower()
        
        for dimension, poles in self.emotion_patterns.items():
            dimension_scores = {}
            
            for pole, pattern in poles.items():
                # Count occurrences of emotional markers
                matches = pattern.findall(text_lower)
                dimension_scores[pole] = len(matches)
            
            # Normalize to get relative strength
            total = sum(dimension_scores.values())
            if total > 0:
                emotions[dimension] = {
                    pole: score / total for pole, score in dimension_scores.items()
                }
            else:
                emotions[dimension] = {pole: 0.0 for pole in dimension_scores}
        
        return emotions
    
    def analyze_subreddit_emotions(self, subreddit: str, n_samples: int = 1000) -> Dict:
        """Analyze emotional landscape of a subreddit"""
        print(f"\n💭 Analyzing emotional landscape of r/{subreddit}...")
        
        # Fetch data
        embeddings, texts, metadata = self.fetcher.fetch_subreddit_data(subreddit, n_samples)
        
        # Convert to docs format expected by the rest of the method
        docs = [{'content': text, 'metadata': meta} for text, meta in zip(texts, metadata)]
        
        if not docs:
            print(f"  ❌ No data found for r/{subreddit}")
            return {}
        
        # Analyze each document
        emotion_scores = defaultdict(lambda: defaultdict(list))
        emotion_examples = defaultdict(lambda: defaultdict(list))
        
        for doc in docs:
            text = doc.get('content', '')
            if len(text) < 50:  # Skip very short texts
                continue
            
            # Get emotion scores for this text
            text_emotions = self.analyze_text_emotions(text)
            
            # Aggregate scores and collect examples
            for dimension, poles in text_emotions.items():
                for pole, score in poles.items():
                    if score > 0:
                        emotion_scores[dimension][pole].append(score)
                        
                        # Collect examples of high-scoring texts
                        if score > 0.7 and len(emotion_examples[dimension][pole]) < 3:
                            # Extract relevant snippet
                            snippet = self._extract_snippet(text, self.emotion_dimensions[dimension][pole])
                            if snippet:
                                emotion_examples[dimension][pole].append(snippet)
        
        # Calculate aggregate statistics
        results = {
            'subreddit': subreddit,
            'num_docs': len(docs),
            'dimensions': {}
        }
        
        for dimension in self.emotion_dimensions:
            dim_results = {}
            
            for pole in self.emotion_patterns[dimension]:
                scores = emotion_scores[dimension][pole]
                if scores:
                    dim_results[pole] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'count': len(scores),
                        'examples': emotion_examples[dimension][pole][:2]  # Top 2 examples
                    }
                else:
                    dim_results[pole] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'count': 0,
                        'examples': []
                    }
            
            # Calculate dimension balance (-1 to 1, where -1 is all negative pole, 1 is all positive)
            pos_pole, neg_pole = list(self.emotion_patterns[dimension].keys())
            if dim_results[pos_pole]['count'] + dim_results[neg_pole]['count'] > 0:
                balance = (dim_results[pos_pole]['count'] - dim_results[neg_pole]['count']) / \
                         (dim_results[pos_pole]['count'] + dim_results[neg_pole]['count'])
            else:
                balance = 0.0
            
            results['dimensions'][dimension] = {
                'poles': dim_results,
                'balance': balance,
                'dominant': pos_pole if balance > 0 else neg_pole
            }
        
        return results
    
    def _extract_snippet(self, text: str, markers: List[str], context_words: int = 30) -> str:
        """Extract a snippet around emotional markers"""
        words = text.split()
        text_lower = text.lower()
        
        for marker in markers:
            if marker.lower() in text_lower:
                # Find the position of the marker
                for i, word in enumerate(words):
                    if marker.lower() in word.lower():
                        # Extract context around the marker
                        start = max(0, i - context_words)
                        end = min(len(words), i + context_words)
                        snippet = ' '.join(words[start:end])
                        
                        # Clean up and add ellipsis if truncated
                        if start > 0:
                            snippet = '...' + snippet
                        if end < len(words):
                            snippet = snippet + '...'
                        
                        return snippet[:300]  # Limit length
        
        return ""
    
    def create_emotion_radar_data(self, all_results: Dict) -> Dict:
        """Create data for radar chart visualization"""
        radar_data = {
            'dimensions': list(self.emotion_dimensions.keys()),
            'subreddits': {}
        }
        
        for subreddit, results in all_results.items():
            if 'dimensions' not in results:
                continue
                
            # Get balance scores for each dimension
            balances = []
            for dimension in radar_data['dimensions']:
                if dimension in results['dimensions']:
                    # Convert balance from -1,1 to 0,1 scale for radar chart
                    balance = results['dimensions'][dimension]['balance']
                    normalized_balance = (balance + 1) / 2
                    balances.append(normalized_balance)
                else:
                    balances.append(0.5)  # Neutral
            
            radar_data['subreddits'][subreddit] = balances
        
        return radar_data
    
    def create_emotion_comparison_matrix(self, all_results: Dict) -> pd.DataFrame:
        """Create comparison matrix of emotional dimensions across subreddits"""
        matrix_data = []
        
        for subreddit, results in all_results.items():
            if 'dimensions' not in results:
                continue
                
            row = {'subreddit': subreddit}
            
            for dimension in self.emotion_dimensions:
                if dimension in results['dimensions']:
                    balance = results['dimensions'][dimension]['balance']
                    dominant = results['dimensions'][dimension]['dominant']
                    
                    # Create meaningful labels
                    if abs(balance) < 0.1:
                        label = 'Balanced'
                    elif balance > 0.5:
                        label = f'Strong {dominant}'
                    elif balance > 0.2:
                        label = f'Moderate {dominant}'
                    elif balance < -0.5:
                        label = f'Strong {dominant}'
                    elif balance < -0.2:
                        label = f'Moderate {dominant}'
                    else:
                        label = f'Slight {dominant}'
                    
                    row[dimension] = balance
                    row[f'{dimension}_label'] = label
                else:
                    row[dimension] = 0
                    row[f'{dimension}_label'] = 'No data'
            
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data)
    
    def generate_insights_report(self, all_results: Dict, radar_data: Dict, comparison_matrix: pd.DataFrame):
        """Generate insights report on emotional landscapes"""
        report = ["# Cannabis Community Emotional Landscapes Analysis\n"]
        
        report.append("## 🎭 Overview\n")
        report.append("This analysis goes beyond simple positive/negative sentiment to explore complex emotional dimensions:")
        report.append("- **Hope ↔ Despair**: Optimism about the future vs feeling defeated")
        report.append("- **Trust ↔ Suspicion**: Faith in systems/products vs skepticism")
        report.append("- **Excitement ↔ Anxiety**: Enthusiasm vs worry and fear")
        report.append("- **Pride ↔ Shame**: Dignity and accomplishment vs stigma")
        report.append("- **Empowerment ↔ Helplessness**: Agency and control vs powerlessness")
        report.append("- **Community ↔ Isolation**: Belonging vs feeling alone")
        report.append("- **Clarity ↔ Confusion**: Understanding vs bewilderment\n")
        
        # Community profiles
        report.append("## 🌈 Community Emotional Profiles\n")
        
        for subreddit, results in all_results.items():
            if 'dimensions' not in results:
                continue
                
            report.append(f"### r/{subreddit}")
            report.append(f"*Analyzed {results['num_docs']} posts*\n")
            
            # Find strongest emotions
            strong_emotions = []
            for dimension, data in results['dimensions'].items():
                if abs(data['balance']) > 0.3:
                    strong_emotions.append((dimension, data['dominant'], data['balance']))
            
            if strong_emotions:
                report.append("**Dominant emotional themes:**")
                for dim, pole, balance in sorted(strong_emotions, key=lambda x: abs(x[2]), reverse=True):
                    strength = 'Strong' if abs(balance) > 0.5 else 'Moderate'
                    report.append(f"- {strength} {pole.title()} ({dim.replace('_', ' vs ')})")
            
            # Add example quotes
            report.append("\n**Example expressions:**")
            example_count = 0
            for dimension, data in results['dimensions'].items():
                for pole, pole_data in data['poles'].items():
                    if pole_data['examples'] and example_count < 2:
                        report.append(f"- {pole.title()}: \"{pole_data['examples'][0]}\"")
                        example_count += 1
            
            report.append("")
        
        # Cross-community insights
        report.append("## 🔍 Cross-Community Insights\n")
        
        # Find most hopeful/despairing communities
        hope_scores = [(sub, res['dimensions'].get('hope_despair', {}).get('balance', 0)) 
                      for sub, res in all_results.items() if 'dimensions' in res]
        hope_scores.sort(key=lambda x: x[1], reverse=True)
        
        if hope_scores:
            report.append("**Most hopeful communities:**")
            for sub, score in hope_scores[:3]:
                if score > 0.2:
                    report.append(f"- r/{sub} (hope balance: {score:.2f})")
        
        # Find most anxious communities
        anxiety_scores = [(sub, -res['dimensions'].get('excitement_anxiety', {}).get('balance', 0)) 
                         for sub, res in all_results.items() if 'dimensions' in res]
        anxiety_scores.sort(key=lambda x: x[1], reverse=True)
        
        if anxiety_scores:
            report.append("\n**Most anxious communities:**")
            for sub, score in anxiety_scores[:3]:
                if score > 0.2:
                    report.append(f"- r/{sub} (anxiety level: {score:.2f})")
        
        # Find most empowered communities
        empower_scores = [(sub, res['dimensions'].get('empowerment_helplessness', {}).get('balance', 0)) 
                         for sub, res in all_results.items() if 'dimensions' in res]
        empower_scores.sort(key=lambda x: x[1], reverse=True)
        
        if empower_scores:
            report.append("\n**Most empowered communities:**")
            for sub, score in empower_scores[:3]:
                if score > 0.2:
                    report.append(f"- r/{sub} (empowerment balance: {score:.2f})")
        
        # Strategic insights
        report.append("\n## 💡 Strategic Insights\n")
        report.append("1. **Hope as a bridge**: Communities with high hope scores are ideal for positive messaging")
        report.append("2. **Address anxiety**: High-anxiety communities need reassurance and clear information")
        report.append("3. **Leverage pride**: Communities with pride can be ambassadors for normalization")
        report.append("4. **Combat isolation**: Communities feeling isolated need connection initiatives")
        report.append("5. **Clarity gaps**: Confused communities represent education opportunities")
        
        return '\n'.join(report)


def create_visualization_config(radar_data: Dict, comparison_matrix: pd.DataFrame) -> Dict:
    """Create configuration for visualizations"""
    
    # Radar chart config
    radar_config = {
        'type': 'radar',
        'data': radar_data,
        'title': 'Emotional Landscape by Community',
        'description': 'Each axis represents an emotional dimension. Outer edge = positive pole, center = negative pole',
        'colors': {
            'cannabis': '#2E7D32',      # Deep green
            'weed': '#66BB6A',          # Medium green  
            'trees': '#81C784',         # Light green
            'Marijuana': '#1B5E20',     # Dark green
            'weedstocks': '#FFB300',    # Amber
            'weedbiz': '#F57C00'        # Orange
        }
    }
    
    # Heatmap config
    heatmap_config = {
        'type': 'heatmap',
        'data': comparison_matrix.to_dict('records'),
        'title': 'Emotional Balance Heatmap',
        'description': 'Red = negative pole dominant, Green = positive pole dominant, Yellow = balanced',
        'dimensions': list(radar_data['dimensions'])
    }
    
    # Emotion flow diagram config
    flow_config = {
        'type': 'sankey',
        'title': 'Emotional State Flows',
        'description': 'How different emotional states connect within communities',
        'note': 'Width represents frequency of co-occurrence'
    }
    
    return {
        'radar': radar_config,
        'heatmap': heatmap_config,
        'flow': flow_config,
        'export_formats': ['png', 'svg', 'interactive_html']
    }


def main():
    print("🎭 Starting Nuanced Emotion Analysis Across Cannabis Communities\n")
    
    analyzer = EmotionLandscapeAnalyzer()
    all_results = {}
    
    # Analyze each subreddit
    for subreddit in CANNABIS_SUBREDDITS:
        results = analyzer.analyze_subreddit_emotions(subreddit, n_samples=500)
        if results:
            all_results[subreddit] = results
            print(f"✅ Completed analysis for r/{subreddit}")
    
    # Create visualization data
    print("\n📊 Creating visualization data...")
    radar_data = analyzer.create_emotion_radar_data(all_results)
    comparison_matrix = analyzer.create_emotion_comparison_matrix(all_results)
    
    # Create visualization config
    viz_config = create_visualization_config(radar_data, comparison_matrix)
    
    # Generate insights report
    print("📝 Generating insights report...")
    report = analyzer.generate_insights_report(all_results, radar_data, comparison_matrix)
    
    # Save all results
    output_data = {
        'analysis_results': all_results,
        'radar_data': radar_data,
        'comparison_matrix': comparison_matrix.to_dict('records'),
        'visualization_config': viz_config
    }
    
    # Save JSON data
    json_path = os.path.join(OUTPUT_DIR, 'emotion_landscape_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Saved analysis data to {json_path}")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'emotion_landscape_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Saved report to {report_path}")
    
    # Save visualization data separately for easy plotting
    radar_path = os.path.join(OUTPUT_DIR, 'emotion_radar_data.json')
    with open(radar_path, 'w') as f:
        json.dump(radar_data, f, indent=2)
    print(f"📊 Saved radar chart data to {radar_path}")
    
    # Print summary
    print("\n🎯 Analysis Complete!")
    print(f"Analyzed {len(all_results)} communities across 7 emotional dimensions")
    print("\nKey findings:")
    
    # Find most emotionally distinct communities
    max_balance = 0
    most_distinct = None
    for sub, results in all_results.items():
        if 'dimensions' in results:
            avg_balance = np.mean([abs(d['balance']) for d in results['dimensions'].values()])
            if avg_balance > max_balance:
                max_balance = avg_balance
                most_distinct = sub
    
    if most_distinct:
        print(f"- Most emotionally distinct: r/{most_distinct}")
    
    print("\n🎨 Next step: Run the visualization script to create charts!")
    print("python scripts/experiments/create_emotion_visualizations.py")


if __name__ == "__main__":
    main()
