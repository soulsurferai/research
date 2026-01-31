# FIXED VERSION - Module 4: Preference Phrase Extraction
# Replace the cell 10 code in feature_extraction.ipynb with this

# Download nltk sentence tokenizer - FIXED VERSION
import nltk
from nltk.tokenize import sent_tokenize
import re

print("Ensuring NLTK resources are available...")
try:
    # Try to find punkt
    nltk.data.find('tokenizers/punkt')
    print("✅ NLTK punkt found")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=False)
    # Also download punkt_tab for newer NLTK versions
    try:
        nltk.download('punkt_tab', quiet=False)
    except:
        pass
    print("✅ NLTK punkt downloaded")

# Test that sent_tokenize works
test_sent = "Hello world. This is a test."
try:
    test_result = sent_tokenize(test_sent)
    print(f"✅ sent_tokenize working: {len(test_result)} sentences from test")
except Exception as e:
    print(f"❌ ERROR: sent_tokenize failed: {e}")
    raise

# Enhanced preference patterns with semantic equivalents

# LOVE/POSITIVE - Sentiment-bearing verbs with first-person subjects
LOVE_PATTERNS = [
    # Explicit love
    r"I love", r"I loved", r"I adore", r"I adored",
    r"I absolutely love", r"I really love", r"I totally love",
    
    # Evaluative predicates (semantic equivalents)
    r"I'm glad", r"I am glad", r"I'm happy", r"I am happy", 
    r"I'm thrilled", r"I am thrilled",
    r"I'm delighted", r"I am delighted", r"I'm pleased", r"I am pleased",
    r"I enjoy", r"I enjoyed", r"I appreciate", r"I appreciated",
    r"I hope",  # ADDED - very common in reviews
    
    # Epistemic modality - SIMPLIFIED
    r"I think it's great", r"I think it is great",
    r"I think it's amazing", r"I think it is amazing",
    r"I find it great", r"I find this great",
    
    # Positive feeling states
    r"I felt great", r"I was impressed", r"I was amazed",
    r"really good", r"very good", r"so good",  # ADDED
]

# HATE/NEGATIVE - Negative sentiment constructions
HATE_PATTERNS = [
    # Explicit hate
    r"I hate", r"I hated", r"I despise", r"I despised",
    r"I really hate", r"I absolutely hate", 
    r"I can't stand", r"I cannot stand", r"I could not stand",
    
    # Evaluative predicates (negative)
    r"I dislike", r"I disliked", 
    r"I'm disappointed", r"I am disappointed",
    r"I'm frustrated", r"I am frustrated",
    r"I'm annoyed", r"I am annoyed",
    
    # Epistemic modality - SIMPLIFIED  
    r"I think it's terrible", r"I think it is terrible",
    r"I think it's awful", r"I find it terrible",
    
    # Negative feeling states
    r"I felt terrible", r"I was disappointed", r"I was bored",
    r"really bad", r"very bad", r"so bad",  # ADDED
]

# WISH/REGRET - Counterfactual and regret constructions
WISH_PATTERNS = [
    # Explicit wish
    r"I wish", r"I wished", r"if only", r"If only",
    r"I hope",  # Can be wish/regret depending on context
    
    # Counterfactual modality - SIMPLIFIED
    r"I would have preferred", r"I would have liked", r"I would have wanted",
    r"I would rather", r"I'd rather",
    r"should have been better", r"could have been better",
    
    # Regret markers
    r"I'm sorry", r"I am sorry",
    r"unfortunately", r"sadly", r"regrettably",
    r"I regret", r"I regretted",
    
    # Preference statements (negative)
    r"it would be better if",
]

def extract_preference_phrases_enhanced(text):
    """
    Enhanced preference extraction with semantic pattern matching.
    Returns full sentences containing patterns.
    
    FIXED VERSION with better error handling and logging.
    """
    try:
        text_str = str(text)
        
        # Tokenize into sentences
        sentences = sent_tokenize(text_str)
        
        # Debug: Check if we got sentences
        if len(sentences) == 0:
            return {
                'love_statements': None,
                'love_count': 0,
                'hate_statements': None,
                'hate_count': 0,
                'wish_statements': None,
                'wish_count': 0,
                'questions': None,
                'question_count': 0
            }
        
        # Find love/positive statements
        love_sents = []
        for sent in sentences:
            for pattern in LOVE_PATTERNS:
                if re.search(pattern, sent, re.IGNORECASE):
                    love_sents.append(sent.strip())
                    break  # Only count once per sentence
        
        # Find hate/negative statements
        hate_sents = []
        for sent in sentences:
            for pattern in HATE_PATTERNS:
                if re.search(pattern, sent, re.IGNORECASE):
                    hate_sents.append(sent.strip())
                    break
        
        # Find wish/regret statements  
        wish_sents = []
        for sent in sentences:
            for pattern in WISH_PATTERNS:
                if re.search(pattern, sent, re.IGNORECASE):
                    wish_sents.append(sent.strip())
                    break
        
        # Find questions
        question_sents = [sent.strip() for sent in sentences if sent.strip().endswith('?')]
        
        return {
            'love_statements': ' ||| '.join(love_sents) if love_sents else None,
            'love_count': len(love_sents),
            'hate_statements': ' ||| '.join(hate_sents) if hate_sents else None,
            'hate_count': len(hate_sents),
            'wish_statements': ' ||| '.join(wish_sents) if wish_sents else None,
            'wish_count': len(wish_sents),
            'questions': ' ||| '.join(question_sents) if question_sents else None,
            'question_count': len(question_sents)
        }
    except Exception as e:
        # DON'T silently return zeros - print the error!
        print(f"ERROR in extract_preference_phrases_enhanced: {e}")
        import traceback
        traceback.print_exc()
        return {
            'love_statements': None,
            'love_count': 0,
            'hate_statements': None,
            'hate_count': 0,
            'wish_statements': None,
            'wish_count': 0,
            'questions': None,
            'question_count': 0
        }

print("Extracting preference phrases with FIXED semantic patterns...")
pref_results = df['Review_Text'].progress_apply(extract_preference_phrases_enhanced)
pref_df = pd.DataFrame(pref_results.tolist())
df = pd.concat([df, pref_df], axis=1)

# Stats
print(f"✅ Preference phrase extraction complete")
print(f"   Reviews with love statements: {(df['love_count'] > 0).sum()} ({((df['love_count'] > 0).sum()/len(df)*100):.1f}%)")
print(f"   Reviews with hate statements: {(df['hate_count'] > 0).sum()} ({((df['hate_count'] > 0).sum()/len(df)*100):.1f}%)")
print(f"   Reviews with wish statements: {(df['wish_count'] > 0).sum()} ({((df['wish_count'] > 0).sum()/len(df)*100):.1f}%)")
print(f"   Reviews with questions: {(df['question_count'] > 0).sum()} ({((df['question_count'] > 0).sum()/len(df)*100):.1f}%)")
print(f"\n   Average counts per review:")
print(f"   - Love: {df['love_count'].mean():.2f}")
print(f"   - Hate: {df['hate_count'].mean():.2f}")
print(f"   - Wish: {df['wish_count'].mean():.2f}")
print(f"   - Questions: {df['question_count'].mean():.2f}")

# Show example extractions
if (df['love_count'] > 0).sum() > 0:
    sample_love = df[df['love_count'] > 0].iloc[0]
    print(f"\n   Example love statement:")
    print(f"   '{sample_love['love_statements'].split(' ||| ')[0]}'")

if (df['hate_count'] > 0).sum() > 0:
    sample_hate = df[df['hate_count'] > 0].iloc[0]
    print(f"\n   Example hate statement:")
    print(f"   '{sample_hate['hate_statements'].split(' ||| ')[0]}'")
