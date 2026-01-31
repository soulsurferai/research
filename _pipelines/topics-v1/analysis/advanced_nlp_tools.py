"""
advanced_nlp_tools.py - Additional NLP techniques for cannabis community analysis
"""

import networkx as nx
from typing import List, Dict, Set, Tuple
import numpy as np
from collections import Counter, defaultdict

class AdvancedNLPAnalyzer:
    """Advanced NLP tools for deeper semantic analysis"""
    
    # 1. ARGUMENT MINING
    def extract_arguments(self, texts: List[str]) -> Dict:
        """Extract claims, evidence, and reasoning patterns"""
        
        argument_indicators = {
            'claims': ['believe', 'think', 'argue', 'claim', 'position is'],
            'evidence': ['study shows', 'research indicates', 'data suggests', 'according to'],
            'reasoning': ['because', 'therefore', 'thus', 'consequently', 'as a result'],
            'counter': ['however', 'but', 'on the other hand', 'alternatively']
        }
        
        arguments = {
            'claims': [],
            'evidence': [],
            'reasoning': [],
            'counter_arguments': []
        }
        
        for text in texts:
            sentences = text.split('.')
            for sent in sentences:
                sent_lower = sent.lower()
                
                for arg_type, indicators in argument_indicators.items():
                    if any(ind in sent_lower for ind in indicators):
                        arguments[arg_type].append(sent.strip())
        
        return arguments
    
    # 2. NARRATIVE ANALYSIS
    def analyze_narratives(self, texts: List[str]) -> Dict:
        """Identify common narrative structures and story patterns"""
        
        narrative_elements = {
            'personal_story': ['i was', 'i had', 'my experience', 'happened to me'],
            'transformation': ['used to', 'changed my', 'before and after', 'transformed'],
            'conflict': ['struggle', 'fight', 'battle', 'against'],
            'resolution': ['finally', 'solved', 'overcame', 'succeeded'],
            'cautionary': ['warning', 'be careful', 'watch out', 'danger']
        }
        
        narratives = defaultdict(list)
        
        for text in texts:
            text_lower = text.lower()
            
            # Check for narrative elements
            found_elements = []
            for element, markers in narrative_elements.items():
                if any(marker in text_lower for marker in markers):
                    found_elements.append(element)
            
            if len(found_elements) >= 2:  # Multi-element narrative
                narrative_type = '_'.join(sorted(found_elements))
                narratives[narrative_type].append({
                    'text': text[:200] + '...',
                    'elements': found_elements
                })
        
        # Identify dominant narrative types
        narrative_counts = {k: len(v) for k, v in narratives.items()}
        
        return {
            'narrative_types': narrative_counts,
            'dominant_narratives': sorted(narrative_counts.items(), 
                                        key=lambda x: x[1], reverse=True)[:5],
            'personal_story_ratio': sum(1 for t in texts if 'my ' in t.lower()) / len(texts)
        }
    
    # 3. METAPHOR DETECTION
    def detect_metaphors(self, texts: List[str]) -> Dict:
        """Identify conceptual metaphors used in cannabis discourse"""
        
        metaphor_patterns = {
            'journey': ['journey', 'path', 'road', 'destination', 'milestone'],
            'war': ['battle', 'fight', 'weapon', 'victory', 'defeat'],
            'medicine': ['healing', 'cure', 'treatment', 'prescription', 'dose'],
            'nature': ['plant', 'grow', 'flower', 'roots', 'bloom'],
            'freedom': ['liberation', 'chains', 'prison', 'escape', 'free'],
            'enlightenment': ['awakening', 'consciousness', 'clarity', 'vision', 'light']
        }
        
        metaphor_usage = defaultdict(int)
        metaphor_examples = defaultdict(list)
        
        for text in texts:
            text_lower = text.lower()
            
            for metaphor_type, words in metaphor_patterns.items():
                for word in words:
                    if word in text_lower:
                        metaphor_usage[metaphor_type] += 1
                        if len(metaphor_examples[metaphor_type]) < 3:
                            # Extract surrounding context
                            idx = text_lower.find(word)
                            context = text[max(0, idx-50):min(len(text), idx+50)]
                            metaphor_examples[metaphor_type].append(context)
        
        return {
            'metaphor_frequencies': dict(metaphor_usage),
            'dominant_metaphors': sorted(metaphor_usage.items(), 
                                       key=lambda x: x[1], reverse=True)[:3],
            'examples': dict(metaphor_examples)
        }
    
    # 4. SOCIAL NETWORK ANALYSIS
    def analyze_discussion_networks(self, texts: List[str], 
                                   authors: List[str]) -> Dict:
        """Analyze interaction patterns and influence networks"""
        
        # Build interaction graph
        G = nx.Graph()
        
        # Add nodes for authors
        unique_authors = list(set(authors))
        G.add_nodes_from(unique_authors)
        
        # Add edges based on mentions/replies
        for i, (text, author) in enumerate(zip(texts, authors)):
            # Simple heuristic: if text mentions another user
            for other_author in unique_authors:
                if other_author != author and other_author in text:
                    G.add_edge(author, other_author)
        
        # Calculate network metrics
        metrics = {}
        
        if len(G.nodes()) > 0:
            metrics['n_users'] = len(G.nodes())
            metrics['n_interactions'] = len(G.edges())
            metrics['density'] = nx.density(G)
            
            # Find influential users
            if len(G.edges()) > 0:
                centrality = nx.eigenvector_centrality_numpy(G)
                top_influencers = sorted(centrality.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]
                metrics['top_influencers'] = top_influencers
            
            # Find communities
            if len(G.edges()) > 10:
                communities = nx.community.greedy_modularity_communities(G)
                metrics['n_communities'] = len(communities)
                metrics['largest_community_size'] = max(len(c) for c in communities)
        
        return metrics
    
    # 5. LINGUISTIC COMPLEXITY ANALYSIS
    def analyze_linguistic_complexity(self, texts: List[str]) -> Dict:
        """Measure linguistic sophistication of discussions"""
        
        complexity_metrics = []
        
        for text in texts:
            words = text.split()
            sentences = text.split('.')
            
            if len(words) > 0 and len(sentences) > 0:
                # Average sentence length
                avg_sent_length = len(words) / len(sentences)
                
                # Vocabulary diversity (Type-Token Ratio)
                unique_words = set(w.lower() for w in words)
                ttr = len(unique_words) / len(words)
                
                # Word length distribution
                avg_word_length = sum(len(w) for w in words) / len(words)
                
                # Syllable estimation (simple heuristic)
                syllables = sum(max(1, len([v for v in w if v in 'aeiouAEIOU'])) 
                               for w in words)
                avg_syllables = syllables / len(words)
                
                # Flesch Reading Ease approximation
                flesch = 206.835 - 1.015 * avg_sent_length - 84.6 * avg_syllables
                
                complexity_metrics.append({
                    'avg_sentence_length': avg_sent_length,
                    'vocabulary_diversity': ttr,
                    'avg_word_length': avg_word_length,
                    'reading_ease': max(0, min(100, flesch))
                })
        
        if complexity_metrics:
            avg_metrics = {
                key: np.mean([m[key] for m in complexity_metrics])
                for key in complexity_metrics[0].keys()
            }
            
            # Categorize reading level
            reading_ease = avg_metrics['reading_ease']
            if reading_ease >= 90:
                level = 'Very Easy (5th grade)'
            elif reading_ease >= 80:
                level = 'Easy (6th grade)'
            elif reading_ease >= 70:
                level = 'Fairly Easy (7th grade)'
            elif reading_ease >= 60:
                level = 'Standard (8th-9th grade)'
            elif reading_ease >= 50:
                level = 'Fairly Difficult (10th-12th grade)'
            elif reading_ease >= 30:
                level = 'Difficult (College)'
            else:
                level = 'Very Difficult (Graduate)'
            
            avg_metrics['reading_level'] = level
            
            return avg_metrics
        
        return {}
    
    # 6. FRAME ANALYSIS
    def analyze_framing(self, texts: List[str]) -> Dict:
        """Identify how cannabis is framed in discussions"""
        
        frames = {
            'medical_frame': {
                'keywords': ['patient', 'treatment', 'medicine', 'therapy', 'symptoms'],
                'count': 0,
                'examples': []
            },
            'criminal_frame': {
                'keywords': ['illegal', 'arrest', 'crime', 'jail', 'conviction'],
                'count': 0,
                'examples': []
            },
            'economic_frame': {
                'keywords': ['tax', 'revenue', 'business', 'market', 'profit'],
                'count': 0,
                'examples': []
            },
            'moral_frame': {
                'keywords': ['right', 'wrong', 'moral', 'ethical', 'values'],
                'count': 0,
                'examples': []
            },
            'freedom_frame': {
                'keywords': ['freedom', 'liberty', 'choice', 'rights', 'personal'],
                'count': 0,
                'examples': []
            },
            'harm_frame': {
                'keywords': ['danger', 'risk', 'harm', 'damage', 'negative'],
                'count': 0,
                'examples': []
            }
        }
        
        for text in texts:
            text_lower = text.lower()
            
            for frame_name, frame_data in frames.items():
                if any(kw in text_lower for kw in frame_data['keywords']):
                    frame_data['count'] += 1
                    if len(frame_data['examples']) < 3:
                        frame_data['examples'].append(text[:150] + '...')
        
        # Calculate frame distribution
        total_frames = sum(f['count'] for f in frames.values())
        
        frame_distribution = {
            name: {
                'count': data['count'],
                'percentage': data['count'] / total_frames if total_frames > 0 else 0,
                'examples': data['examples']
            }
            for name, data in frames.items()
        }
        
        return {
            'frame_distribution': frame_distribution,
            'dominant_frame': max(frames.items(), key=lambda x: x[1]['count'])[0]
        }


# 7. SEMANTIC CHANGE DETECTION
def detect_semantic_drift(texts: List[str], 
                         timestamps: List[str],
                         target_words: List[str]) -> Dict:
    """Detect how word meanings change over time"""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create time windows
    df = pd.DataFrame({'text': texts, 'time': pd.to_datetime(timestamps)})
    df['year'] = df['time'].dt.year
    
    semantic_changes = {}
    
    for word in target_words:
        # Find contexts where word appears
        word_contexts = defaultdict(list)
        
        for _, row in df.iterrows():
            if word.lower() in row['text'].lower():
                # Extract surrounding words
                words = row['text'].lower().split()
                if word in words:
                    idx = words.index(word)
                    context = words[max(0, idx-5):idx] + words[idx+1:min(len(words), idx+6)]
                    word_contexts[row['year']].extend(context)
        
        # Analyze context changes over time
        if len(word_contexts) >= 2:
            years = sorted(word_contexts.keys())
            
            # Compare early vs late contexts
            early_context = ' '.join(word_contexts[years[0]])
            late_context = ' '.join(word_contexts[years[-1]])
            
            # Simple similarity using TF-IDF
            vectorizer = TfidfVectorizer()
            try:
                tfidf = vectorizer.fit_transform([early_context, late_context])
                similarity = (tfidf * tfidf.T).toarray()[0, 1]
                
                semantic_changes[word] = {
                    'similarity': similarity,
                    'drift': 1 - similarity,
                    'early_year': years[0],
                    'late_year': years[-1],
                    'meaning_stable': similarity > 0.7
                }
            except:
                pass
    
    return semantic_changes
