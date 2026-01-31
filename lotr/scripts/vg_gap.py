import json
import re

# 1. LOAD THE DATA
input_file = 'vg_entities_curated.json'
output_file = 'vg_entities_curated_v2.json'

try:
    with open(input_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['games'])} games from {input_file}")
except FileNotFoundError:
    print("Error: Input file not found.")
    exit()

# 2. DEFINE MISSING GAMES
# We assign placeholder IDs (90000+) to distinguish from IGDB imports
missing_games = [
    {
        "igdb_id": 90001, 
        "canonical_name": "Tales of the Shire: A The Lord of the Rings Game",
        "slug": "tales-of-the-shire",
        "aliases": ["Tales of the Shire", "Shire game"],
        "regex_pattern": r"(?i)\b(Tales\s+of\s+the\s+Shire)\b",
        "genres": ["Simulation", "Adventure", "Cozy"],
        "release_year": 2024,
        "total_rating": None, # Not yet rated/released fully
        "rating_count": 0,
        "lotr_related": True,
        "priority_tier": 1,
        "is_mod": False
    },
    {
        "igdb_id": 90002,
        "canonical_name": "The Lord of the Rings: Gollum",
        "slug": "the-lord-of-the-rings-gollum",
        "aliases": ["Gollum game", "LOTR Gollum"],
        "regex_pattern": r"(?i)\b(The\s+Lord\s+of\s+the\s+Rings:\s+Gollum|Gollum\s+game)\b", 
        # Note: avoiding just "Gollum" to prevent clashing with the character entity
        "genres": ["Adventure", "Stealth"],
        "release_year": 2023,
        "total_rating": 35.0, # Known poor rating
        "rating_count": 100,
        "lotr_related": True,
        "priority_tier": 1,
        "is_mod": False
    }
]

# 3. APPLY PATCHES
games_list = data['games']
changes_log = []

# A. Add Missing Games
for game in missing_games:
    # Check for duplicates by slug just in case
    if not any(g['slug'] == game['slug'] for g in games_list):
        games_list.append(game)
        changes_log.append(f"Added: {game['canonical_name']}")

# B. Modify BFME Entries
for game in games_list:
    # Patch BFME 1
    if "Battle for Middle-earth" in game['canonical_name'] and "II" not in game['canonical_name']:
        if "BFME" not in game['aliases']:
            game['aliases'].append("BFME")
            # Update Regex to include standalone BFME
            # Pattern logic: look for existing pattern OR standalone BFME
            # We insert |BFME inside the capture group
            if "|BFME)" not in game['regex_pattern']: 
                # Simple replace to inject it into the OR group
                game['regex_pattern'] = game['regex_pattern'].replace(")", "|BFME)")
            changes_log.append("Updated BFME 1: Added alias 'BFME' and regex")

    # Patch BFME 2
    if "Battle for Middle-earth II" in game['canonical_name']:
        # Update regex to handle "BFME 2" (space) and "BFME2" (no space)
        # Current regex likely looks like ...|BFME2)...
        # We want ...|BFME\s*2)...
        if "BFME2" in game['regex_pattern']:
            game['regex_pattern'] = game['regex_pattern'].replace("BFME2", r"BFME\s*2")
            changes_log.append("Updated BFME 2: Regex now matches 'BFME 2' and 'BFME2'")

# 4. UPDATE METADATA
data['metadata']['total_entities'] = len(games_list)
data['metadata']['curation_version'] = "2.1-patched-gaps"

# 5. SAVE
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print("\n--- PATCH REPORT ---")
for log in changes_log:
    print(f"- {log}")
print(f"\nSaved updated list to: {output_file}")