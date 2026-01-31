"""
Feature Extraction - Incremental Improvements
IMDb Review Analysis - Phase 2.5

Purpose: Add/improve specific features without reprocessing entire pipeline
Input: reviews_enhanced.csv (existing enhanced dataset)
Output: reviews_enhanced.csv (updated with new features)
Processing Time: ~3-5 minutes

Improvements:
1. Gender Detection v2: Improved from 8.1% → 30-40% coverage
2. Emotion Detection: 8 new columns using NRCLex
"""

# Standard libraries
import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Progress bars
from tqdm import tqdm

# NLP - Emotion detection
from nrclex import NRCLex

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

print("✅ Imports complete\n")

# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path('/Users/USER/Desktop/JAMES/Noetheca/Reviews/Data')
INPUT_FILE = DATA_DIR / 'reviews_enhanced.csv'
OUTPUT_FILE = DATA_DIR / 'reviews_enhanced.csv'  # Overwrite same file

print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"\n⚠️  Note: This will update the existing file\n")

# ==============================================================================
# Load Existing Enhanced Data
# ==============================================================================

print("Loading existing enhanced dataset...")
df = pd.read_csv(INPUT_FILE, encoding='utf-8')

print(f"✅ Loaded {len(df):,} reviews")
print(f"Current columns: {len(df.columns)}\n")

# ==============================================================================
# Module 2.5: Improved Gender Detection
# ==============================================================================

print("="*60)
print("MODULE 2.5: IMPROVED GENDER DETECTION")
print("="*60)
print("\nGoal: Increase gender detection from 8.1% → 30-40%\n")

# Top 200 most common names (lightweight subset for performance)
COMMON_MALE_NAMES = {
    'james', 'john', 'robert', 'michael', 'william', 'david', 'richard', 'joseph',
    'thomas', 'charles', 'daniel', 'matthew', 'anthony', 'mark', 'donald', 'steven',
    'paul', 'andrew', 'joshua', 'kenneth', 'kevin', 'brian', 'george', 'edward',
    'ronald', 'timothy', 'jason', 'jeffrey', 'ryan', 'jacob', 'gary', 'nicholas',
    'eric', 'jonathan', 'stephen', 'larry', 'justin', 'scott', 'brandon', 'benjamin',
    'samuel', 'frank', 'gregory', 'raymond', 'alexander', 'patrick', 'jack', 'dennis',
    'jerry', 'tyler', 'aaron', 'jose', 'henry', 'adam', 'douglas', 'nathan',
    'peter', 'zachary', 'kyle', 'walter', 'harold', 'jeremy', 'ethan', 'carl',
    'keith', 'roger', 'gerald', 'christian', 'terry', 'sean', 'arthur', 'austin',
    'noah', 'lawrence', 'jesse', 'joe', 'bryan', 'billy', 'jordan', 'albert',
    'dylan', 'bruce', 'willie', 'gabriel', 'logan', 'alan', 'juan', 'ralph',
    'roy', 'eugene', 'randy', 'vincent', 'russell', 'louis', 'philip', 'bobby',
    'johnny', 'bradley', 'howard', 'fred', 'ernest', 'martin', 'craig', 'todd',
    'bob', 'mike', 'steve', 'tony', 'chris', 'dave', 'dan', 'matt', 'josh',
    'jim', 'bill', 'rob', 'rick', 'sam', 'max', 'ben', 'alex', 'nick', 'tom'
}

COMMON_FEMALE_NAMES = {
    'mary', 'patricia', 'jennifer', 'linda', 'barbara', 'elizabeth', 'susan', 'jessica',
    'sarah', 'karen', 'nancy', 'margaret', 'lisa', 'betty', 'dorothy', 'sandra',
    'ashley', 'kimberly', 'donna', 'emily', 'michelle', 'carol', 'amanda', 'melissa',
    'deborah', 'stephanie', 'rebecca', 'laura', 'sharon', 'cynthia', 'kathleen', 'amy',
    'shirley', 'angela', 'helen', 'anna', 'brenda', 'pamela', 'nicole', 'emma',
    'samantha', 'katherine', 'christine', 'debra', 'rachel', 'catherine', 'carolyn', 'janet',
    'ruth', 'maria', 'heather', 'diane', 'virginia', 'julie', 'joyce', 'victoria',
    'olivia', 'kelly', 'christina', 'lauren', 'joan', 'evelyn', 'judith', 'megan',
    'cheryl', 'andrea', 'hannah', 'jacqueline', 'martha', 'gloria', 'teresa', 'ann',
    'sara', 'madison', 'frances', 'kathryn', 'janice', 'jean', 'abigail', 'alice',
    'judy', 'sophia', 'grace', 'denise', 'amber', 'doris', 'marilyn', 'danielle',
    'beverly', 'isabella', 'theresa', 'diana', 'natalie', 'brittany', 'charlotte', 'marie',
    'kayla', 'alexis', 'lori', 'jane', 'julia', 'rose', 'kate', 'lily', 'lucy',
    'sophie', 'chloe', 'ella', 'katie', 'beth', 'claire', 'jenny', 'kim', 'sue',
    'liz', 'jess', 'annie', 'molly', 'megan', 'holly', 'abby', 'zoe', 'leah'
}

MALE_HONORIFICS = [
    'mr', 'mister', 'sir', 'lord', 'king', 'prince', 'duke', 'baron',
    'pastor', 'father', 'brother', 'monk', 'reverend', 'rabbi',
    'captain', 'general', 'admiral', 'colonel'
]

FEMALE_HONORIFICS = [
    'mrs', 'miss', 'ms', 'lady', 'queen', 'princess', 'duchess', 'baroness',
    'sister', 'nun', 'mother', 'madam', 'dame',
    'her-excellency', 'her-majesty', 'her-highness'
]

MALE_KEYWORDS = [
    'guy', 'dude', 'bro', 'man', 'boy', 'lad', 'male', 'husband', 'dad', 'father'
]

FEMALE_KEYWORDS = [
    'girl', 'gal', 'lady', 'woman', 'female', 'wife', 'mom', 'mother', 'chick', 'sis'
]

def split_username_intelligent(username):
    """
    Split username into component parts for name extraction.
    
    Examples:
    - "JohnSmith1985" → ["John", "Smith"]
    - "mary_reviews" → ["mary", "reviews"]
    - "bobafett1138" → ["bobafett"]
    """
    # Replace separators with spaces
    username = re.sub(r'[_\-.]', ' ', username)
    
    # Split on capital letters (CamelCase)
    username = re.sub(r'([a-z])([A-Z])', r'\1 \2', username)
    
    # Split on numbers
    username = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', username)
    username = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', username)
    
    # Split and clean
    parts = username.lower().split()
    
    # Filter out very short parts and numbers
    parts = [p for p in parts if len(p) >= 3 and not p.isdigit()]
    
    return parts

def analyze_username_improved(username):
    """
    Improved gender detection with smart username parsing.
    
    Detection hierarchy:
    1. Honorifics (100% confidence)
    2. Semantic keywords (95% confidence)
    3. Common name list (80% confidence)
    4. Unknown
    """
    if pd.isna(username):
        return 'unknown'
    
    username_str = str(username)
    username_lower = username_str.lower()
    
    # TIER 1: Check honorifics
    for honorific in MALE_HONORIFICS:
        if honorific in username_lower:
            return 'male'
    
    for honorific in FEMALE_HONORIFICS:
        if honorific in username_lower:
            return 'female'
    
    # TIER 2: Check semantic keywords
    for keyword in MALE_KEYWORDS:
        if keyword in username_lower:
            return 'male'
    
    for keyword in FEMALE_KEYWORDS:
        if keyword in username_lower:
            return 'female'
    
    # TIER 3: Split username and check against common names
    parts = split_username_intelligent(username_str)
    
    for part in parts:
        if part in COMMON_MALE_NAMES:
            return 'male'
        if part in COMMON_FEMALE_NAMES:
            return 'female'
    
    return 'unknown'

# Test the function
print("Testing improved gender detection:")
test_cases = ['JohnSmith1985', 'mary_reviews', 'Boba_Fett1138', 'pastorjames', 'sarahloveshorror']
for username in test_cases:
    parts = split_username_intelligent(username)
    gender = analyze_username_improved(username)
    parts_str = str(parts)
    print(f"  {username:20} → Parts: {parts_str:40} → {gender}")

print("\nApplying to all reviewers...\n")

# Store old values for comparison
old_gender = df['username_gender_hint'].copy()

# Apply improved detection
print("Processing...")
df['username_gender_hint'] = df['Reviewer'].apply(lambda x: analyze_username_improved(x))

# Stats
print("\n" + "="*60)
print("GENDER DETECTION IMPROVEMENT")
print("="*60)

old_identified = (old_gender != 'unknown').sum()
new_identified = (df['username_gender_hint'] != 'unknown').sum()

print(f"\nBefore (v1): {old_identified} identified ({old_identified/len(df)*100:.1f}%)")
print(f"After (v2):  {new_identified} identified ({new_identified/len(df)*100:.1f}%)")

improvement = new_identified - old_identified
print(f"\n✅ Improvement: +{improvement} reviewers (+{improvement/len(df)*100:.1f}%)\n")

# ==============================================================================
# Module 5: Emotion Detection (NEW)
# ==============================================================================

print("="*60)
print("MODULE 5: EMOTION DETECTION (NEW)")
print("="*60)
print("\nGoal: Add 8 emotion columns using NRCLex\n")

emotion_cols = ['emotion_joy', 'emotion_trust', 'emotion_fear', 'emotion_surprise',
                'emotion_sadness', 'emotion_disgust', 'emotion_anger', 'emotion_anticipation']

if all(col in df.columns for col in emotion_cols):
    print("✅ Emotion columns already exist, skipping...\n")
    SKIP_EMOTIONS = True
else:
    print("Adding emotion detection...\n")
    SKIP_EMOTIONS = False

if not SKIP_EMOTIONS:
    def extract_emotions(text):
        """Extract emotion scores using NRCLex."""
        try:
            emotion_obj = NRCLex(str(text))
            emotions = emotion_obj.affect_frequencies
            
            return {
                'emotion_joy': emotions.get('joy', 0.0),
                'emotion_trust': emotions.get('trust', 0.0),
                'emotion_fear': emotions.get('fear', 0.0),
                'emotion_surprise': emotions.get('surprise', 0.0),
                'emotion_sadness': emotions.get('sadness', 0.0),
                'emotion_disgust': emotions.get('disgust', 0.0),
                'emotion_anger': emotions.get('anger', 0.0),
                'emotion_anticipation': emotions.get('anticipation', 0.0)
            }
        except Exception as e:
            return {col: 0.0 for col in emotion_cols}
    
    print("Processing emotions (this may take 2-3 minutes)...\n")
    
    # Use tqdm for progress
    tqdm.pandas(desc="Extracting emotions")
    emotion_results = df['Review_Text'].progress_apply(extract_emotions)
    emotion_df = pd.DataFrame(emotion_results.tolist())
    
    # Add to main dataframe
    df = pd.concat([df, emotion_df], axis=1)
    
    print("\n✅ Emotion detection complete")
    print(f"\nAverage emotion scores:")
    for col in emotion_cols:
        print(f"  {col:25}: {df[col].mean():.3f}")
    print()

# ==============================================================================
# Export Updated Dataset
# ==============================================================================

print("="*60)
print("EXPORTING UPDATED DATASET")
print("="*60)

df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"\n✅ Saved: {OUTPUT_FILE}")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
print(f"   File size: {file_size:.2f} MB")

print("\n" + "="*60)
print("✅ INCREMENTAL UPDATES COMPLETE!")
print("="*60)
print("\nReady for analysis phase (movie_insights.ipynb)\n")
