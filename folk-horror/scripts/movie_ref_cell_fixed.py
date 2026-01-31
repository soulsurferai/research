# Stopwords to exclude (common false positives)
MOVIE_STOPWORDS = {
    'the', 'it', 'this', 'that', 'these', 'those', 'they', 'them',
    'an', 'a', 'to', 'of', 'in', 'on', 'at', 'for', 'with',
    'movie', 'film', 'most', 'more', 'some', 'any', 'all',
    # Add common garbage phrases
    'i was', 'i said', 'i had', 'a lot', 'a good', 'a bad', 'a horror', 
    'a bunch', 'a child', 'a dark'
}

# Enhanced comparison patterns (15 templates)
COMPARISON_PATTERNS = [
    # Direct comparisons
    r'better than ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'worse than ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'superior to ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'inferior to ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    
    # Similarity comparisons
    r'like ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'similar to ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'reminds me of ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'reminded of ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'echoes ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    
    # Quality comparisons
    r'compared to ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'as good as ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'not as good as ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'pales in comparison to ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    
    # Relative phrases
    r'more .{1,20} than ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
    r'less .{1,20} than ["\']?([A-Z][\w\s:&\'\-]+?)["\']?[\.,;\s]',
]

def extract_movie_references_enhanced(text, nlp_model, film_db):
    """
    Enhanced movie reference extraction - STRICT validation only.
    Only trusts quoted titles to avoid garbage.
    """
    try:
        text_str = str(text)
        movies = set()
        comparisons = []
        
        # ONLY use quoted titles (most reliable)
        quoted_titles = re.findall(r'["\']([A-Z][A-Za-z\s:&\'-]{3,50})["\']', text_str)
        for title in quoted_titles:
            title = title.strip()
            words = title.split()
            
            # Must be 2-6 words, not a stopword, not a garbage phrase
            title_lower = title.lower()
            if (2 <= len(words) <= 6 and 
                title_lower not in MOVIE_STOPWORDS and
                not any(garbage in title_lower for garbage in ['i was', 'i said', 'i had', 'a lot', 'a good', 'a bad', 'a bunch', 'a child', 'a dark', 'a horror'])):
                movies.add(title)
        
        # Extract comparison contexts (but don't trust the movie titles from them)
        for pattern in COMPARISON_PATTERNS:
            if re.search(pattern, text_str, re.IGNORECASE):
                # Just mark that a comparison exists
                comparisons.append("comparison_found")
                break
        
        return {
            'movies_mentioned': ','.join(sorted(movies)) if movies else None,
            'movie_mention_count': len(movies),
            'has_comparisons': len(comparisons) > 0,
            'comparison_context': None  # Skip storing contexts to avoid garbage
        }
    except Exception as e:
        return {
            'movies_mentioned': None,
            'movie_mention_count': 0,
            'has_comparisons': False,
            'comparison_context': None
        }

print("Extracting movie references with enhanced methods...")
movie_results = df['Review_Text'].progress_apply(
    lambda x: extract_movie_references_enhanced(x, nlp, movie_titles)
)
movie_df = pd.DataFrame(movie_results.tolist())
df = pd.concat([df, movie_df], axis=1)

# Stats
print(f"✅ Movie reference extraction complete")
print(f"   Reviews with movie mentions: {df['movies_mentioned'].notna().sum()} ({(df['movies_mentioned'].notna().sum()/len(df)*100):.1f}%)")
print(f"   Reviews with comparisons: {df['has_comparisons'].sum()} ({(df['has_comparisons'].sum()/len(df)*100):.1f}%)")
print(f"   Average movies per review: {df['movie_mention_count'].mean():.2f}")

# Most mentioned movies
all_mentioned = []
for movies in df['movies_mentioned'].dropna():
    all_mentioned.extend(movies.split(','))
if all_mentioned:
    print(f"\n   Top 10 most mentioned movies:")
    for movie, count in Counter(all_mentioned).most_common(10):
        print(f"   - {movie}: {count}")
