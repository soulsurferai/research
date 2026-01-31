#!/usr/bin/env python3
"""
IMDb Review Extraction Script  
Extracts reviews from copy-pasted text files and outputs to CSV
"""

import re
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set


def clean_text(text: str) -> str:
    """Clean encoding issues and normalize whitespace"""
    if not text:
        return ""
    
    # Fix common encoding issues
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '—',
        'â€"': '–',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        '…': '...',
        'â€¦': '...',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã ': 'à',
        'Ã´': 'ô',
        'Ã§': 'ç',
        'â€¢': '•',
    }
    
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    
    # Normalize whitespace  
    text = ' '.join(text.split())
    
    return text.strip()


def parse_date(date_str: str) -> Optional[str]:
    """Parse date string to ISO format YYYY-MM-DD"""
    if not date_str:
        return None
    
    try:
        # Format: "Oct 3, 2011" or "May 28, 2021"
        dt = datetime.strptime(date_str.strip(), "%b %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except:
        return None


def parse_metadata_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the metadata line that contains username, date, and 'Permalink'
    Format: "    cookiela2001Nov 8, 2004Permalink"
    Returns: (username, date_iso)
    """
    if not line:
        return None, None
    
    # Strip leading/trailing whitespace and remove "Permalink"
    cleaned = line.strip().replace("Permalink", "").strip()
    
    if not cleaned:
        return None, None
    
    # Month abbreviations to search for
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Find where the date starts by looking for month abbreviation
    date_start = -1
    for month in months:
        if month in cleaned:
            date_start = cleaned.index(month)
            break
    
    if date_start == -1:
        # No date found, entire string is probably the username
        return cleaned, None
    
    # Split into username and date
    username = cleaned[:date_start].strip()
    date_str = cleaned[date_start:].strip()
    
    # Parse the date
    date_iso = parse_date(date_str)
    
    return username, date_iso


def extract_movie_title(filename: str) -> str:
    """Extract movie title from filename"""
    # Remove .txt extension and replace underscores/hyphens with spaces
    title = Path(filename).stem.replace('_', ' ').replace('-', ' ')
    # Title case each word
    return title.title()


def count_words(text: str) -> int:
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())


def generate_review_id(movie: str, reviewer: str, date: str) -> str:
    """
    Generate unique, deterministic review ID
    Format: imdb_moviename_username_YYYYMMDD
    """
    # Clean movie name for ID
    movie_clean = movie.lower().replace(' ', '_').replace("'", "")
    
    # Clean reviewer name for ID
    reviewer_clean = reviewer.lower().replace(' ', '_').replace('-', '_')
    
    # Clean date (remove hyphens)
    date_clean = date.replace('-', '') if date and date != 'unknown' else 'nodate'
    
    return f"imdb_{movie_clean}_{reviewer_clean}_{date_clean}"


def get_existing_review_ids(csv_file: Path) -> Set[str]:
    """Get set of existing review IDs from CSV to prevent duplicates"""
    if not csv_file.exists():
        return set()
    
    existing_ids = set()
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Review_ID' in row:
                    existing_ids.add(row['Review_ID'])
    except Exception as e:
        print(f"Warning: Could not read existing CSV: {e}")
        return set()
    
    return existing_ids


def extract_reviews(text_content: str, movie_title: str) -> List[Dict]:
    """Extract all reviews from text content"""
    reviews = []
    skipped_count = 0
    
    lines = text_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for rating pattern (e.g., "7/10" or "10/10")
        rating_match = re.match(r'^(\d+)/10$', line)
        
        if rating_match:
            # Start of a new review
            review = {
                'Movie_Title': movie_title,
                'Source': 'IMDb',
                'Rating': int(rating_match.group(1)),
                'Spoiler_Flag': False,
                'Review_Title': None,
                'Review_Text': None,
                'Reviewer': None,
                'Review_Date': None,
                'Helpful_Votes_Up': None,
                'Helpful_Votes_Down': None,
            }
            
            i += 1
            
            # Skip duplicate rating lines (sometimes rating appears twice)
            while i < len(lines) and re.match(r'^(\d+)/10$', lines[i].strip()):
                i += 1
            
            # Collect all content until we hit "Helpful•" pattern
            content_lines = []
            while i < len(lines):
                current = lines[i].strip()
                
                # Stop when we hit "Helpful•" pattern
                if re.match(r'^Helpful\s*[•â€¢]\s*\d+', current):
                    # Extract helpful votes (upvotes)
                    helpful_match = re.search(r'Helpful\s*[•â€¢]\s*(\d+)', current)
                    if helpful_match:
                        review['Helpful_Votes_Up'] = int(helpful_match.group(1))
                    i += 1
                    break
                
                # Collect non-empty lines
                if current:
                    content_lines.append(current)
                
                i += 1
            
            # After "Helpful•X", the next line should be downvotes
            if i < len(lines):
                current = lines[i].strip()
                if re.match(r'^\d+$', current):
                    review['Helpful_Votes_Down'] = int(current)
                    i += 1
            
            # Skip any blank lines before metadata
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            
            # Next should be the metadata line
            if i < len(lines):
                metadata_line = lines[i]
                if metadata_line.startswith(' ') and 'Permalink' in metadata_line:
                    username, date_iso = parse_metadata_line(metadata_line)
                    review['Reviewer'] = username
                    review['Review_Date'] = date_iso
                    i += 1
            
            # Process content_lines
            if content_lines:
                first_line = content_lines[0] if content_lines else ""
                remaining_lines = content_lines[1:] if len(content_lines) > 1 else []
                
                # Check if second line is "Spoiler"
                if remaining_lines and remaining_lines[0].lower() == 'spoiler':
                    review['Spoiler_Flag'] = True
                    remaining_lines = remaining_lines[1:]
                
                # Determine title vs text
                if len(first_line) < 100 and remaining_lines:
                    review['Review_Title'] = clean_text(first_line)
                    review['Review_Text'] = clean_text(' '.join(remaining_lines))
                else:
                    review['Review_Title'] = None
                    review['Review_Text'] = clean_text(' '.join([first_line] + remaining_lines))
            
            review['Review_Length'] = count_words(review['Review_Text'])
            review['Review_ID'] = generate_review_id(
                review['Movie_Title'],
                review['Reviewer'] or 'unknown',
                review['Review_Date'] or 'unknown'
            )
            
            if review['Review_Text'] and review['Reviewer']:
                reviews.append(review)
            else:
                skipped_count += 1
        else:
            i += 1
    
    if skipped_count > 0:
        print(f"  ⚠ Skipped {skipped_count} review(s) with missing fields")
    
    return reviews


def save_to_csv(reviews: List[Dict], output_file: Path, append: bool = True):
    """Save reviews to CSV file, avoiding duplicates"""
    if not reviews:
        print("No reviews to save")
        return
    
    # Get existing review IDs
    existing_ids = get_existing_review_ids(output_file)
    
    # Filter out duplicates
    new_reviews = [r for r in reviews if r['Review_ID'] not in existing_ids]
    
    if not new_reviews:
        print(f"✓ No new reviews - all {len(reviews)} already exist in CSV")
        return
    
    duplicate_count = len(reviews) - len(new_reviews)
    if duplicate_count > 0:
        print(f"  ℹ Skipping {duplicate_count} duplicate(s) already in CSV")
    
    fieldnames = [
        'Review_ID', 'Movie_Title', 'Source', 'Reviewer', 'Review_Date',
        'Rating', 'Review_Title', 'Review_Text', 'Review_Length',
        'Helpful_Votes_Up', 'Helpful_Votes_Down', 'Spoiler_Flag'
    ]
    
    file_exists = output_file.exists() and append
    
    with open(output_file, 'a' if append else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        
        if not file_exists:
            writer.writeheader()
        
        for review in new_reviews:
            row = {k: (v if v is not None else 'NULL') for k, v in review.items()}
            writer.writerow(row)
    
    print(f"✓ Saved {len(new_reviews)} new review(s) to {output_file}")


def main(input_file: str):
    """Main extraction function"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return
    
    print(f"\nReading {input_path.name}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    movie_title = extract_movie_title(input_path.name)
    print(f"Movie: {movie_title}")
    
    print("Extracting reviews...")
    reviews = extract_reviews(content, movie_title)
    
    print(f"\nFound {len(reviews)} valid reviews")
    spoiler_count = sum(1 for r in reviews if r['Spoiler_Flag'])
    print(f"  - {spoiler_count} with spoiler flag")
    print(f"  - {len(reviews) - spoiler_count} without spoiler flag")
    
    if reviews:
        print(f"\nSample review ID: {reviews[0]['Review_ID']}")
    
    output_file = input_path.parent / 'Data' / 'all_reviews.csv'
    output_file.parent.mkdir(exist_ok=True)
    
    save_to_csv(reviews, output_file, append=True)
    
    print(f"\n✓ Extraction complete!")
    print(f"  Movie: {movie_title}")
    print(f"  Reviews extracted: {len(reviews)}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python extract.py <path_to_review_file.txt>")
        print("Example: python extract.py ../the_rapture.txt")
        sys.exit(1)
    
    main(sys.argv[1])
