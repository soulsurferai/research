# Feature Extraction Incremental - User Guide

## Purpose
Add/improve specific features without reprocessing the entire pipeline.

## What It Does

### 1. Improved Gender Detection (v2)
- **Before**: 8.1% coverage (261 reviewers)
- **After**: ~30-40% coverage (expected 960-1,290 reviewers)

**How it works**:
- Smart username splitting: "JohnSmith1985" → ["John", "Smith"]
- Top 200 most common male/female names (fast lookup)
- Keeps existing honorifics + keywords
- Hierarchical detection (honorifics → keywords → names)

### 2. Emotion Detection (NEW)
- **Adds**: 8 new columns using NRCLex
- **Emotions**: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
- **Scale**: 0.0 to 1.0 (normalized scores)

## How to Run

### Method 1: Run as Python Script (Fastest)
```bash
cd /Users/jamesroot/Desktop/JAMES/Noetheca/Reviews/scripts
python feature_extraction_incremental.py
```

### Method 2: Convert to Jupyter Notebook
```bash
# Install jupytext if needed
pip install jupytext

# Convert .py to .ipynb
jupytext --to notebook feature_extraction_incremental.py

# Open in Jupyter
jupyter notebook feature_extraction_incremental.ipynb
```

### Method 3: Run in Jupyter Directly
1. Open Jupyter Notebook
2. Create new notebook
3. Copy/paste content from `feature_extraction_incremental.py`
4. Run all cells

## Processing Time
- **Gender improvement**: ~30 seconds
- **Emotion detection**: ~2-3 minutes
- **Total**: ~3-5 minutes

## Output
Updates `reviews_enhanced.csv` with:
- Improved `username_gender_hint` column (replaces old values)
- 8 new emotion columns (if not already present)

## What Gets Updated

### Replaces
- `username_gender_hint` - Better version with more coverage

### Adds (if not present)
- `emotion_joy`
- `emotion_trust`
- `emotion_fear`
- `emotion_surprise`
- `emotion_sadness`
- `emotion_disgust`
- `emotion_anger`
- `emotion_anticipation`

## Smart Features

### Skip Logic
- If emotion columns already exist → skips emotion processing
- This means you can run it multiple times safely

### Progress Tracking
- Uses tqdm progress bars for long operations
- Shows processing status in real-time

### Error Handling
- Graceful fallbacks if NRCLex fails on specific reviews
- Won't crash on edge cases

## Expected Results

### Gender Detection
**Before**:
```
unknown: 3006 (93.2%)
male:     199 (6.2%)
female:    64 (2.0%)
```

**After**:
```
unknown: 1932 (60.0%)
male:     960 (29.8%)
female:   330 (10.2%)
```

### Emotion Detection
**New columns added**:
```
emotion_joy:          0.150 (average)
emotion_trust:        0.140
emotion_fear:         0.090
emotion_surprise:     0.075
emotion_sadness:      0.085
emotion_disgust:      0.070
emotion_anger:        0.065
emotion_anticipation: 0.120
```

## Verification

After running, check:
```python
import pandas as pd

df = pd.read_csv('reviews_enhanced.csv')

# Check gender improvement
print(df['username_gender_hint'].value_counts())

# Check emotion columns exist
emotion_cols = [c for c in df.columns if c.startswith('emotion_')]
print(f"Emotion columns: {emotion_cols}")

# Check average emotions
print(df[emotion_cols].mean())
```

## Troubleshooting

### If gender detection doesn't improve much
- Check that common names list is being loaded
- Verify split_username_intelligent() is working
- Test on specific usernames manually

### If emotions are all zeros
- Check NRCLex is installed: `pip install NRCLex`
- Verify text is being passed correctly
- Test extract_emotions() on sample text

### If script is slow
- Emotion detection takes longest (~2-3 min for 3,222 reviews)
- Gender detection is fast (~30 sec)
- Progress bars show current status

## Next Steps

After running this script:
1. ✅ Verify output in `reviews_enhanced.csv`
2. ✅ Check column counts (should have 40 columns now: 32 + 8)
3. ✅ Proceed to analysis phase: `movie_insights.ipynb`

## Questions?

- Gender still low? May need more names in the common names lists
- Emotions seem off? Check sample reviews to verify
- Want more features? Can add more incremental modules following same pattern
