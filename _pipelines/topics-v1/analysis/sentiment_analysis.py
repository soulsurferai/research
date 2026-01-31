"""
sentiment_analysis.py - Multiple sentiment analysis approaches for cannabis communities
"""

import numpy as np
from typing import List, Dict, Tuple
from transformers import pipeline
import re

class CannabisSimientAnalyzer:
    """Sentiment analysis tailored for cannabis discussions"""
    
    def __init__(self):
        # Initialize transformer model for nuanced sentiment
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
        except:
            print("Install transformers: pip install transformers")
            self.sentiment_pipeline = None
    
    # 1. ASPECT-BASED SENTIMENT
    def analyze_aspect_sentiment(self, texts: List[str]) -> Dict:
        """Analyze sentiment toward specific aspects of cannabis"""
        
        aspects = {
            'legalization': ['legalize', 'legal', 'law', 'decriminalize'],
            'medical': ['medical', 'patient', 'treatment', 'therapy'],
            'recreational': ['recreational', 'fun', 'party', 'social'],
            'business': ['business', 'profit', 'investment', 'market'],
            'health': ['health', 'wellness', 'benefit', 'harm'],
            'enforcement': ['police', 'arrest', 'criminal', 'jail']
        }
        
        aspect_sentiments = {}
        
        for aspect, keywords in aspects.items():
            aspect_texts = []
            
            # Find sentences mentioning this aspect
            for text in texts:
                sentences = text.split('.')
                for sent in sentences:
                    if any(kw in sent.lower() for kw in keywords):
                        aspect_texts.append(sent)
            
            if aspect_texts and self.sentiment_pipeline:
                # Analyze sentiment for this aspect
                sentiments = self.sentiment_pipeline(aspect_texts[:100])  # Limit for speed
                
                # Aggregate results
                pos = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
                neg = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
                neu = sum(1 for s in sentiments if s['label'] == 'NEUTRAL')
                
                aspect_sentiments[aspect] = {
                    'positive_ratio': pos / len(sentiments) if sentiments else 0,
                    'negative_ratio': neg / len(sentiments) if sentiments else 0,
                    'neutral_ratio': neu / len(sentiments) if sentiments else 0,
                    'n_mentions': len(aspect_texts)
                }
        
        return aspect_sentiments
    
    # 2. EMOTION DETECTION
    def detect_emotions(self, texts: List[str]) -> Dict:
        """Detect specific emotions beyond positive/negative"""
        
        emotion_patterns = {
            'joy': r'\b(happy|excited|thrilled|love|amazing|wonderful|fantastic)\b',
            'fear': r'\b(afraid|scared|worried|anxious|nervous|paranoid)\b',
            'anger': r'\b(angry|mad|furious|pissed|hate|disgusted)\b',
            'sadness': r'\b(sad|depressed|disappointed|upset|crying|miserable)\b',
            'surprise': r'\b(surprised|shocked|amazed|unexpected|wow|unbelievable)\b',
            'anticipation': r'\b(excited|looking forward|can\'t wait|hoping|expecting)\b',
            'trust': r'\b(trust|believe|reliable|honest|genuine|authentic)\b',
            'disgust': r'\b(disgusting|gross|nasty|horrible|awful|terrible)\b'
        }
        
        emotion_counts = {emotion: 0 for emotion in emotion_patterns}
        total_emotions = 0
        
        for text in texts:
            text_lower = text.lower()
            for emotion, pattern in emotion_patterns.items():
                matches = re.findall(pattern, text_lower)
                emotion_counts[emotion] += len(matches)
                total_emotions += len(matches)
        
        # Normalize to percentages
        if total_emotions > 0:
            emotion_dist = {emotion: count/total_emotions 
                           for emotion, count in emotion_counts.items()}
        else:
            emotion_dist = emotion_counts
        
        return {
            'emotion_distribution': emotion_dist,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get),
            'total_emotional_expressions': total_emotions
        }
    
    # 3. STANCE DETECTION
    def detect_stance(self, texts: List[str]) -> Dict:
        """Detect stance on key cannabis issues"""
        
        stances = {
            'pro_legalization': {
                'indicators': ['should be legal', 'legalize it', 'end prohibition', 
                              'tax and regulate', 'personal freedom'],
                'count': 0
            },
            'anti_legalization': {
                'indicators': ['should stay illegal', 'dangerous drug', 'gateway drug',
                              'protect children', 'public safety'],
                'count': 0
            },
            'pro_medical': {
                'indicators': ['medical benefits', 'helps patients', 'natural medicine',
                              'better than pills', 'therapeutic'],
                'count': 0
            },
            'business_positive': {
                'indicators': ['great investment', 'growing market', 'profit opportunity',
                              'industry growth', 'bullish'],
                'count': 0
            },
            'business_negative': {
                'indicators': ['overvalued', 'bubble', 'losing money', 'bad investment',
                              'bearish'],
                'count': 0
            }
        }
        
        for text in texts:
            text_lower = text.lower()
            for stance, data in stances.items():
                for indicator in data['indicators']:
                    if indicator in text_lower:
                        data['count'] += 1
        
        total_stances = sum(data['count'] for data in stances.values())
        
        return {
            'stance_counts': {stance: data['count'] for stance, data in stances.items()},
            'stance_distribution': {stance: data['count']/total_stances if total_stances > 0 else 0
                                   for stance, data in stances.items()},
            'dominant_stance': max(stances, key=lambda x: stances[x]['count'])
        }
    
    # 4. TOXICITY DETECTION
    def detect_toxicity(self, texts: List[str]) -> Dict:
        """Detect toxic language and hostile discourse"""
        
        toxic_patterns = {
            'personal_attacks': r'\b(idiot|stupid|moron|loser|dumb)\b',
            'profanity': r'\b(fuck|shit|damn|hell|ass)\b',
            'threats': r'\b(kill|hurt|destroy|attack|fight)\b',
            'discrimination': r'\b(racist|sexist|homophobic|bigot)\b'
        }
        
        toxicity_scores = []
        toxic_examples = []
        
        for text in texts[:1000]:  # Sample for speed
            text_lower = text.lower()
            toxic_count = 0
            
            for category, pattern in toxic_patterns.items():
                matches = re.findall(pattern, text_lower)
                toxic_count += len(matches)
            
            toxicity_score = min(toxic_count / len(text.split()), 1.0)
            toxicity_scores.append(toxicity_score)
            
            if toxicity_score > 0.1:
                toxic_examples.append({
                    'text': text[:100] + '...',
                    'score': toxicity_score
                })
        
        return {
            'avg_toxicity': np.mean(toxicity_scores),
            'toxic_ratio': sum(1 for s in toxicity_scores if s > 0) / len(toxicity_scores),
            'high_toxicity_ratio': sum(1 for s in toxicity_scores if s > 0.1) / len(toxicity_scores),
            'examples': sorted(toxic_examples, key=lambda x: x['score'], reverse=True)[:5]
        }
    
    # 5. DISCOURSE QUALITY
    def analyze_discourse_quality(self, texts: List[str]) -> Dict:
        """Analyze the quality of discourse"""
        
        quality_indicators = {
            'evidence_based': r'\b(study|research|data|evidence|source|link)\b',
            'questioning': r'\?',
            'respectful': r'\b(thanks|appreciate|respect|agree|understand)\b',
            'constructive': r'\b(suggest|recommend|idea|solution|help)\b'
        }
        
        quality_scores = []
        
        for text in texts:
            text_lower = text.lower()
            score = 0
            
            # Check quality indicators
            for indicator, pattern in quality_indicators.items():
                if re.search(pattern, text_lower):
                    score += 1
            
            # Penalize for all caps (shouting)
            if text.isupper() and len(text) > 10:
                score -= 1
            
            # Bonus for longer, thoughtful posts
            if len(text.split()) > 50:
                score += 1
            
            quality_scores.append(score / 4)  # Normalize
        
        return {
            'avg_quality_score': np.mean(quality_scores),
            'high_quality_ratio': sum(1 for s in quality_scores if s > 0.5) / len(quality_scores),
            'evidence_based_ratio': sum(1 for t in texts if 'study' in t.lower() or 'research' in t.lower()) / len(texts),
            'question_ratio': sum(1 for t in texts if '?' in t) / len(texts)
        }


# 6. COMMUNITY-SPECIFIC SENTIMENT TRACKING
def track_community_sentiment_evolution(texts: List[str], 
                                      timestamps: List[str],
                                      community: str) -> Dict:
    """Track how sentiment evolves over time in a community"""
    import pandas as pd
    
    # Create dataframe
    df = pd.DataFrame({
        'text': texts,
        'timestamp': pd.to_datetime(timestamps),
        'community': community
    })
    
    # Sort by time
    df = df.sort_values('timestamp')
    
    # Create time windows (weekly)
    df['week'] = df['timestamp'].dt.to_period('W')
    
    # Simple sentiment scoring (can be replaced with transformer model)
    positive_words = set(['good', 'great', 'love', 'awesome', 'happy', 'best'])
    negative_words = set(['bad', 'hate', 'terrible', 'worst', 'awful', 'sucks'])
    
    def simple_sentiment(text):
        words = text.lower().split()
        pos = sum(1 for w in words if w in positive_words)
        neg = sum(1 for w in words if w in negative_words)
        return (pos - neg) / (len(words) + 1)
    
    df['sentiment'] = df['text'].apply(simple_sentiment)
    
    # Aggregate by week
    weekly_sentiment = df.groupby('week').agg({
        'sentiment': ['mean', 'std'],
        'text': 'count'
    }).reset_index()
    
    return {
        'weekly_sentiment': weekly_sentiment.to_dict('records'),
        'overall_trend': 'improving' if df['sentiment'].iloc[-100:].mean() > df['sentiment'].iloc[:100].mean() else 'declining',
        'volatility': df['sentiment'].std()
    }
