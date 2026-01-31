#!/usr/bin/env python3
"""
IGDB Video Game Entity Extractor v2
====================================
Extracts video game data from IGDB API for entity detection in Reddit comments.

v2 Changes:
- Tighter alias generation (no overly generic terms)
- Minimum alias length filter (5+ chars)
- Blocklist for problematic aliases
- ASCII-only aliases (drop non-Latin scripts)
- Better priority tier logic
- Deduplication of games by canonical name

Requirements:
    pip install requests python-dotenv

Usage:
    python igdb_game_extractor_v2.py
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


# =============================================================================
# CONFIGURATION
# =============================================================================

ENV_PATH = Path(__file__).parent / ".env"
if not ENV_PATH.exists():
    ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

TWITCH_AUTH_URL = "https://id.twitch.tv/oauth2/token"
IGDB_API_URL = "https://api.igdb.com/v4"

REQUEST_DELAY = 0.26  # seconds between requests (4 req/sec limit)

OUTPUT_FILE = Path(__file__).parent / "video_game_entities.json"


# =============================================================================
# ALIAS CONFIGURATION
# =============================================================================

# Minimum length for an alias to be included
MIN_ALIAS_LENGTH = 5

# Aliases that are too generic and will cause false positives
# These get matched against lowercase versions
ALIAS_BLOCKLIST = {
    # Generic franchise names that appear everywhere
    "the lord of the rings",
    "lord of the rings",
    "star wars",
    "final fantasy",
    "world of warcraft",
    "dragon age",
    "the witcher",
    "the elder scrolls",
    "elder scrolls",
    "dark souls",
    "total war",
    "middle-earth",
    "middle earth",
    
    # Generic subtitles
    "origins",
    "shadowlands",
    "remastered",
    "definitive edition",
    "game of the year edition",
    "goty",
    "special edition",
    "legendary edition",
    "ultimate edition",
    "complete edition",
    "anniversary edition",
    "enhanced edition",
    "director's cut",
    "gold edition",
    
    # Single common words
    "online",
    "mobile",
    "legends",
    "heroes",
    "warriors",
    "chronicles",
    "saga",
    "quest",
    "adventure",
    "battle",
    "war",
    "shadow",
    "ring",
    "rings",
    "tower",
    "towers",
    "king",
    "return",
    "fellowship",
    
    # Problematic short forms
    "lotr",  # Too common in LOTR discussions, not game-specific
    "wow",
    "ff",
    "tw",
    "da",
    "ds",
    "es",
}

# Regex pattern for non-ASCII characters
NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7F]')


# =============================================================================
# IGDB GENRE MAPPING
# =============================================================================

GENRES = {
    2: "Point-and-click",
    4: "Fighting",
    5: "Shooter",
    7: "Music",
    8: "Platform",
    9: "Puzzle",
    10: "Racing",
    11: "Real Time Strategy (RTS)",
    12: "Role-playing (RPG)",
    13: "Simulator",
    14: "Sport",
    15: "Strategy",
    16: "Turn-based strategy (TBS)",
    24: "Tactical",
    25: "Hack and slash/Beat 'em up",
    26: "Quiz/Trivia",
    30: "Pinball",
    31: "Adventure",
    32: "Indie",
    33: "Arcade",
    34: "Visual Novel",
    35: "Card & Board Game",
    36: "MOBA",
}

RELEVANT_GENRE_IDS = [12, 31, 15, 16, 24, 25, 11]


# =============================================================================
# AUTHENTICATION
# =============================================================================

def get_access_token() -> str:
    """Authenticate with Twitch to get IGDB API access token."""
    if not TWITCH_CLIENT_ID or not TWITCH_CLIENT_SECRET:
        raise ValueError(
            "Missing Twitch credentials. Ensure TWITCH_CLIENT_ID and "
            "TWITCH_CLIENT_SECRET are set in your .env file."
        )
    
    response = requests.post(
        TWITCH_AUTH_URL,
        params={
            "client_id": TWITCH_CLIENT_ID,
            "client_secret": TWITCH_CLIENT_SECRET,
            "grant_type": "client_credentials"
        }
    )
    response.raise_for_status()
    data = response.json()
    
    print(f"✓ Authenticated. Token expires in {data['expires_in']//3600} hours.")
    return data["access_token"]


# =============================================================================
# API FUNCTIONS
# =============================================================================

def igdb_query(endpoint: str, query: str, access_token: str) -> list:
    """Execute a query against the IGDB API."""
    url = f"{IGDB_API_URL}/{endpoint}"
    headers = {
        "Client-ID": TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    
    response = requests.post(url, headers=headers, data=query)
    
    if response.status_code == 429:
        print("  Rate limited, waiting...")
        time.sleep(1)
        return igdb_query(endpoint, query, access_token)
    
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    
    return response.json()


def fetch_games_by_genre(access_token: str, genre_ids: list, min_ratings: int = 20, limit: int = 500) -> list:
    """Fetch popular games from specified genres."""
    genres_str = ",".join(map(str, genre_ids))
    
    query = f"""
    fields id, name, slug, genres, total_rating, total_rating_count, 
           first_release_date, category;
    where genres = ({genres_str}) 
      & total_rating_count >= {min_ratings}
      & category = 0;
    sort total_rating_count desc;
    limit {limit};
    """
    
    print(f"  Fetching from genres {genre_ids[:3]}...")
    results = igdb_query("games", query, access_token)
    print(f"  → {len(results)} games")
    return results


def fetch_games_by_search(access_token: str, search_terms: list, limit: int = 100) -> list:
    """Search for games by name/keyword."""
    all_results = []
    
    for term in search_terms:
        query = f"""
        fields id, name, slug, genres, total_rating, total_rating_count,
               first_release_date, category;
        search "{term}";
        limit {limit};
        """
        
        print(f"  Searching '{term}'...")
        results = igdb_query("games", query, access_token)
        print(f"  → {len(results)} results")
        all_results.extend(results)
    
    return all_results


def fetch_alternative_names(access_token: str, game_ids: list) -> dict:
    """Fetch alternative names for games, filtering to ASCII-only."""
    if not game_ids:
        return {}
    
    batch_size = 50
    all_alt_names = {}
    
    for i in range(0, len(game_ids), batch_size):
        batch = game_ids[i:i + batch_size]
        ids_str = ",".join(map(str, batch))
        
        query = f"""
        fields game, name, comment;
        where game = ({ids_str});
        limit 500;
        """
        
        results = igdb_query("alternative_names", query, access_token)
        
        for item in results:
            game_id = item.get("game")
            name = item.get("name", "")
            
            # Skip non-ASCII names (Japanese, Chinese, Russian, etc.)
            if NON_ASCII_PATTERN.search(name):
                continue
            
            if game_id and name:
                if game_id not in all_alt_names:
                    all_alt_names[game_id] = []
                all_alt_names[game_id].append(name)
    
    return all_alt_names


def fetch_top_games_overall(access_token: str, limit: int = 300) -> list:
    """Fetch most popular games across all genres."""
    query = f"""
    fields id, name, slug, genres, total_rating, total_rating_count,
           first_release_date, category;
    where total_rating_count >= 100 & category = 0;
    sort total_rating_count desc;
    limit {limit};
    """
    
    print(f"  Fetching top {limit} games overall...")
    results = igdb_query("games", query, access_token)
    print(f"  → {len(results)} games")
    return results


# =============================================================================
# ALIAS GENERATION (TIGHTENED)
# =============================================================================

def is_valid_alias(alias: str, canonical_name: str) -> bool:
    """
    Check if an alias is valid (specific enough to use).
    
    Args:
        alias: The proposed alias
        canonical_name: The game's canonical name
    
    Returns:
        True if alias is safe to use
    """
    alias_lower = alias.lower().strip()
    
    # Too short
    if len(alias_lower) < MIN_ALIAS_LENGTH:
        return False
    
    # In blocklist
    if alias_lower in ALIAS_BLOCKLIST:
        return False
    
    # Contains non-ASCII
    if NON_ASCII_PATTERN.search(alias):
        return False
    
    # Is just the canonical name
    if alias_lower == canonical_name.lower():
        return False
    
    # Is just a number
    if alias_lower.isdigit():
        return False
    
    # Is just "Game N" pattern where N is a number
    if re.match(r'^.+\s+\d+$', alias_lower) and len(alias_lower.split()) == 2:
        # Allow things like "Witcher 3" but not just "3"
        pass
    
    return True


def generate_aliases(name: str) -> list:
    """
    Generate reasonable aliases from a game name.
    Much tighter than v1 - only generates high-confidence aliases.
    """
    aliases = []
    
    # Remove "The " prefix variant
    if name.lower().startswith("the "):
        variant = name[4:]
        if is_valid_alias(variant, name):
            aliases.append(variant)
    
    # Handle "Name: Subtitle" - but only keep FULL subtitle, not parts
    if ": " in name:
        parts = name.split(": ", 1)
        main_title = parts[0].strip()
        subtitle = parts[1].strip()
        
        # "Main Title: Subtitle" -> "Main Title Subtitle" (no colon)
        no_colon = f"{main_title} {subtitle}"
        if is_valid_alias(no_colon, name):
            aliases.append(no_colon)
    
    # Handle "Name - Subtitle" similarly
    if " - " in name:
        no_dash = name.replace(" - ", " ")
        if is_valid_alias(no_dash, name):
            aliases.append(no_dash)
    
    # Common abbreviations for well-known games (manually curated)
    abbreviations = generate_abbreviation(name)
    for abbrev in abbreviations:
        if is_valid_alias(abbrev, name):
            aliases.append(abbrev)
    
    # Remove apostrophes variant (Baldur's -> Baldurs)
    if "'" in name or "'" in name:
        no_apostrophe = name.replace("'", "").replace("'", "")
        if is_valid_alias(no_apostrophe, name) and no_apostrophe != name:
            aliases.append(no_apostrophe)
    
    return list(set(aliases))


def generate_abbreviation(name: str) -> list:
    """
    Generate common abbreviations for a game name.
    Only for specific patterns that are widely used.
    """
    abbrevs = []
    
    # Pattern: "Word Word N" -> "WWN" (e.g., "Baldur's Gate 3" -> "BG3")
    match = re.match(r"^(.+?)\s+(\d+)$", name)
    if match:
        title_part = match.group(1)
        number = match.group(2)
        
        # Get capital letters from title
        capitals = re.findall(r'\b[A-Z]', title_part)
        if len(capitals) >= 2:
            abbrev = ''.join(capitals) + number
            if len(abbrev) >= 3:  # At least 3 chars
                abbrevs.append(abbrev)
    
    # Pattern: "Word Word: Subtitle N" -> "WWN"
    match = re.match(r"^(.+?):\s*(.+?)\s+(\d+)$", name)
    if match:
        title_part = match.group(1)
        number = match.group(3)
        capitals = re.findall(r'\b[A-Z]', title_part)
        if len(capitals) >= 2:
            abbrev = ''.join(capitals) + number
            if len(abbrev) >= 3:
                abbrevs.append(abbrev)
    
    # Roman numeral conversions
    roman_map = {'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6', 
                 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10'}
    for roman, arabic in roman_map.items():
        if f" {roman}" in name and not name.endswith(roman):
            # Don't convert if it's part of a word
            converted = re.sub(rf'\b{roman}\b', arabic, name)
            if converted != name and is_valid_alias(converted, name):
                abbrevs.append(converted)
    
    return abbrevs


def generate_regex_pattern(canonical: str, aliases: list) -> str:
    """Generate regex pattern for matching game in text."""
    all_names = [canonical] + aliases
    
    escaped = []
    for name in all_names:
        pattern = re.escape(name)
        # Allow flexible whitespace
        pattern = pattern.replace(r"\ ", r"\s+")
        pattern = pattern.replace(r"\-", r"[\s-]*")
        pattern = pattern.replace(r"\:", r":?\s*")
        escaped.append(pattern)
    
    combined = "|".join(escaped)
    return f"(?i)\\b({combined})\\b"


# =============================================================================
# MANUAL ADDITIONS (MODS AND IMPORTANT GAMES)
# =============================================================================

MANUAL_ADDITIONS = [
    {
        "canonical_name": "Third Age: Total War",
        "slug": "third-age-total-war-mod",
        "aliases": ["Third Age Total War", "TATW"],
        "is_mod": True,
        "base_game": "Medieval II: Total War",
        "lotr_related": True,
        "priority_tier": 1,
    },
    {
        "canonical_name": "Divide and Conquer",
        "slug": "divide-and-conquer-mod",
        "aliases": ["DaC submod"],
        "is_mod": True,
        "base_game": "Third Age: Total War",
        "lotr_related": True,
        "priority_tier": 1,
    },
    {
        "canonical_name": "Realms in Exile",
        "slug": "realms-in-exile-mod",
        "aliases": ["CK3 LOTR mod", "Realms in Exile CK3"],
        "is_mod": True,
        "base_game": "Crusader Kings III",
        "lotr_related": True,
        "priority_tier": 1,
    },
    {
        "canonical_name": "The Last Days of the Third Age",
        "slug": "last-days-third-age-mod",
        "aliases": ["TLD mod", "Last Days mod"],
        "is_mod": True,
        "base_game": "Mount & Blade: Warband",
        "lotr_related": True,
        "priority_tier": 1,
    },
    {
        "canonical_name": "Age of the Ring",
        "slug": "age-of-the-ring-mod",
        "aliases": ["AotR", "Age of the Ring mod"],
        "is_mod": True,
        "base_game": "Battle for Middle-earth II",
        "lotr_related": True,
        "priority_tier": 1,
    },
    {
        "canonical_name": "Edain Mod",
        "slug": "edain-mod",
        "aliases": ["Edain"],
        "is_mod": True,
        "base_game": "Battle for Middle-earth II",
        "lotr_related": True,
        "priority_tier": 1,
    },
]

# Search terms for LOTR games
LOTR_SEARCH_TERMS = [
    "Shadow of Mordor",
    "Shadow of War", 
    "Battle for Middle Earth",
    "War in the North",
    "Return to Moria",
    "Gollum game",
    "LOTRO",
    "Lord of the Rings Online",
    "Aragorn's Quest",
    "Third Age",
    "Two Towers game",
    "Return of the King game",
]

# High-priority fantasy games
PRIORITY_FANTASY_GAMES = [
    "Baldur's Gate 3",
    "Elden Ring",
    "Witcher 3",
    "Skyrim",
    "Dark Souls",
    "Dragon Age Inquisition",
    "Divinity Original Sin 2",
    "Pillars of Eternity",
    "Pathfinder Wrath",
    "Diablo 4",
    "World of Warcraft",
    "Total War Warhammer",
]


# =============================================================================
# LOTR DETECTION
# =============================================================================

LOTR_KEYWORDS = [
    "lord of the rings", "middle-earth", "middle earth", "mordor", 
    "tolkien", "hobbit", "gollum", "moria", "gondor", "rohan",
    "sauron", "aragorn", "frodo", "gandalf", "isengard", "helm's deep",
    "minas tirith", "shire", "rivendell", "lotro"
]

def is_lotr_related(name: str, slug: str) -> bool:
    """Check if a game is LOTR-related."""
    text = f"{name} {slug}".lower()
    return any(kw in text for kw in LOTR_KEYWORDS)


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_video_game_entities():
    """Main extraction pipeline."""
    print("=" * 60)
    print("IGDB Video Game Entity Extractor v2")
    print("=" * 60)
    print()
    
    # Authenticate
    print("Step 1: Authenticating...")
    access_token = get_access_token()
    print()
    
    # Collect games
    all_games = {}
    
    print("Step 2: Fetching top games by popularity...")
    for game in fetch_top_games_overall(access_token, limit=300):
        all_games[game["id"]] = game
    print()
    
    print("Step 3: Fetching from relevant genres (RPG, Strategy, Adventure)...")
    for game in fetch_games_by_genre(access_token, RELEVANT_GENRE_IDS, min_ratings=15, limit=500):
        all_games[game["id"]] = game
    print()
    
    print("Step 4: Searching LOTR-specific titles...")
    for game in fetch_games_by_search(access_token, LOTR_SEARCH_TERMS, limit=50):
        all_games[game["id"]] = game
    print()
    
    print("Step 5: Searching priority fantasy titles...")
    for game in fetch_games_by_search(access_token, PRIORITY_FANTASY_GAMES, limit=30):
        all_games[game["id"]] = game
    print()
    
    print(f"Total unique games from IGDB: {len(all_games)}")
    print()
    
    # Fetch alternative names
    print("Step 6: Fetching alternative names (ASCII only)...")
    game_ids = list(all_games.keys())
    alt_names = fetch_alternative_names(access_token, game_ids)
    print(f"  → Found alt names for {len(alt_names)} games")
    print()
    
    # Build entity list
    print("Step 7: Building entity list with filtered aliases...")
    entities = []
    seen_names = set()  # Deduplicate by canonical name
    
    for game_id, game in all_games.items():
        name = game.get("name", "")
        slug = game.get("slug", "")
        
        # Skip if we've already seen this name
        name_lower = name.lower()
        if name_lower in seen_names:
            continue
        seen_names.add(name_lower)
        
        # Get and filter aliases
        official_aliases = alt_names.get(game_id, [])
        generated_aliases = generate_aliases(name)
        
        # Filter all aliases
        all_aliases = []
        for alias in official_aliases + generated_aliases:
            if is_valid_alias(alias, name):
                all_aliases.append(alias)
        
        # Deduplicate aliases
        all_aliases = list(set(all_aliases))
        
        # Determine LOTR relation
        lotr = is_lotr_related(name, slug)
        
        # Determine priority tier based on rating count
        rating_count = game.get("total_rating_count") or 0
        if lotr:
            priority = 1  # All LOTR games are priority 1
        elif rating_count >= 500:
            priority = 1
        elif rating_count >= 100:
            priority = 2
        else:
            priority = 3
        
        # Parse release year
        release_year = None
        if game.get("first_release_date"):
            try:
                release_year = datetime.fromtimestamp(game["first_release_date"]).year
            except:
                pass
        
        entity = {
            "igdb_id": game_id,
            "canonical_name": name,
            "slug": slug,
            "aliases": all_aliases,
            "regex_pattern": generate_regex_pattern(name, all_aliases),
            "genres": [GENRES.get(g, f"Unknown({g})") for g in game.get("genres", [])],
            "release_year": release_year,
            "total_rating": game.get("total_rating"),
            "rating_count": rating_count,
            "lotr_related": lotr,
            "priority_tier": priority,
            "is_mod": False,
        }
        
        entities.append(entity)
    
    # Add manual entries
    print("Step 8: Adding manual entries (mods)...")
    for manual in MANUAL_ADDITIONS:
        # Filter manual aliases too
        filtered_aliases = [a for a in manual.get("aliases", []) 
                          if is_valid_alias(a, manual["canonical_name"])]
        
        entity = {
            "igdb_id": None,
            "canonical_name": manual["canonical_name"],
            "slug": manual["slug"],
            "aliases": filtered_aliases,
            "regex_pattern": generate_regex_pattern(
                manual["canonical_name"], 
                filtered_aliases
            ),
            "genres": [],
            "release_year": None,
            "total_rating": None,
            "rating_count": None,
            "lotr_related": manual.get("lotr_related", False),
            "priority_tier": manual.get("priority_tier", 2),
            "is_mod": True,
            "base_game": manual.get("base_game"),
        }
        entities.append(entity)
    
    print(f"  → Added {len(MANUAL_ADDITIONS)} mods")
    print()
    
    # Sort: LOTR first, then by rating count
    entities.sort(key=lambda x: (
        0 if x["lotr_related"] else 1,
        x["priority_tier"],
        -(x["rating_count"] or 0)
    ))
    
    # Build output
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "2.0",
            "total_entities": len(entities),
            "lotr_related_count": sum(1 for e in entities if e["lotr_related"]),
            "mods_count": sum(1 for e in entities if e["is_mod"]),
            "tier_1_count": sum(1 for e in entities if e["priority_tier"] == 1),
            "tier_2_count": sum(1 for e in entities if e["priority_tier"] == 2),
            "tier_3_count": sum(1 for e in entities if e["priority_tier"] == 3),
            "alias_rules": {
                "min_length": MIN_ALIAS_LENGTH,
                "ascii_only": True,
                "blocklist_applied": True
            }
        },
        "games": entities
    }
    
    # Write output
    print(f"Step 9: Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total games: {len(entities)}")
    print(f"LOTR-related: {output['metadata']['lotr_related_count']}")
    print(f"Mods: {output['metadata']['mods_count']}")
    print(f"Tier 1: {output['metadata']['tier_1_count']}")
    print(f"Tier 2: {output['metadata']['tier_2_count']}")
    print(f"Tier 3: {output['metadata']['tier_3_count']}")
    print()
    
    # Show LOTR games
    print("LOTR-related games found:")
    lotr_games = [e for e in entities if e["lotr_related"]]
    for e in lotr_games[:15]:
        alias_preview = f" (aliases: {', '.join(e['aliases'][:2])})" if e['aliases'] else ""
        mod_tag = " [MOD]" if e["is_mod"] else ""
        print(f"  • {e['canonical_name']}{mod_tag}{alias_preview}")
    if len(lotr_games) > 15:
        print(f"  ... and {len(lotr_games) - 15} more")
    
    return output


if __name__ == "__main__":
    try:
        extract_video_game_entities()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        raise