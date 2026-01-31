# Test script to debug preference phrase extraction
import nltk
import re
from nltk.tokenize import sent_tokenize

# Download BOTH punkt versions for compatibility
print("Downloading NLTK resources...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("✅ NLTK resources downloaded")
except Exception as e:
    print(f"Error downloading NLTK: {e}")

# Test text
test_text = """I'm here because Mark Kermode told me to watch this. I hope he's right, the poster looks a bit racy. Rogers plays this well. She's creepy and convincing. It takes a little time to really dig in and you need an open mind, but this is really good."""

# Enhanced preference patterns
LOVE_PATTERNS = [
    r"I love", r"I loved", r"I adore",
    r"I'm glad", r"I am glad",
    r"I enjoy", r"I enjoyed",
    r"I hope",  # ADD THIS
    r"really good",  # ADD THIS
]

WISH_PATTERNS = [
    r"I wish", r"I hoped"
]

print("\n" + "="*60)
print("TESTING PREFERENCE PHRASE EXTRACTION")
print("="*60)

print("\nTest text:")
print(test_text[:200] + "...")

# Test tokenization
print("\n\nStep 1: Testing sent_tokenize...")
try:
    sentences = sent_tokenize(test_text)
    print(f"✅ Tokenized into {len(sentences)} sentences")
    for i, sent in enumerate(sentences):
        print(f"  {i+1}. {sent[:80]}...")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sentences = []

# Test pattern matching
print("\n\nStep 2: Testing LOVE pattern matching...")
love_matches = []
for sent in sentences:
    for pattern in LOVE_PATTERNS:
        if re.search(pattern, sent, re.IGNORECASE):
            love_matches.append((pattern, sent[:100]))
            print(f"  ✅ '{pattern}' matched in: '{sent[:80]}...'")
            break

print(f"\nTotal LOVE matches: {len(love_matches)}")

print("\n\nStep 3: Testing WISH pattern matching...")
wish_matches = []
for sent in sentences:
    for pattern in WISH_PATTERNS:
        if re.search(pattern, sent, re.IGNORECASE):
            wish_matches.append((pattern, sent[:100]))
            print(f"  ✅ '{pattern}' matched in: '{sent[:80]}...'")
            break

print(f"\nTotal WISH matches: {len(wish_matches)}")

print("\n" + "="*60)
if len(love_matches) > 0 or len(wish_matches) > 0:
    print("✅ PATTERNS ARE WORKING!")
    print("The issue in the notebook is likely:")
    print("1. NLTK punkt not downloaded in the Jupyter environment")
    print("2. OR the try/except is catching errors silently")
else:
    print("❌ PATTERNS STILL NOT MATCHING")
    print("Need to investigate further...")
print("="*60)
