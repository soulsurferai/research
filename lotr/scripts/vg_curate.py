#!/usr/bin/env python3
"""
Video Game Entity Curation - Research Grade
=============================================
Produces a tight, high-precision entity list for business research.

Design principles:
- Precision over recall (false positives damage analysis more than missing niche games)
- One entry per base game (no edition variants)
- Minimum engagement threshold (if not discussed on IGDB, not discussed on Reddit)
- All LOTR content preserved (core research target)

Input:  video_game_entities.json (raw IGDB extraction)
Output: vg_entities_curated.json (research-grade working file)

Expected output: ~150-250 high-signal entries
"""

import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path(__file__).parent / "video_game_entities.json"
OUTPUT_FILE = Path(__file__).parent / "vg_entities_curated.json"

# Engagement thresholds
MIN_RATING_COUNT_GENERAL = 100  # Non-LOTR games need 100+ ratings
MIN_RATING_COUNT_DLC = 50       # DLC/expansions need 50+ to be worth keeping
MIN_RATING_COUNT_LOTR = 0       # Keep all LOTR content regardless

# =============================================================================
# EDITION/DLC PATTERNS
# =============================================================================

# These get STRIPPED from names before grouping
EDITION_SUFFIXES = [
    # Standard editions
    r":\s*Collector'?s?\s*Edition$",
    r":\s*Deluxe\s*Edition$",
    r":\s*Ultimate\s*Edition$",
    r":\s*Gold\s*Edition$",
    r":\s*Premium\s*Edition$",
    r":\s*Special\s*Edition$",
    r":\s*Limited\s*Edition$",
    r":\s*Legendary\s*Edition$",
    r":\s*Anniversary\s*Edition$",
    r":\s*Definitive\s*Edition$",
    r":\s*Complete\s*Edition$",
    r":\s*Enhanced\s*Edition$",
    r":\s*Game\s*of\s*the\s*Year\s*Edition$",
    r":\s*GOTY\s*Edition$",
    r":\s*GOTY$",
    r":\s*Digital\s*Deluxe\s*Edition$",
    r":\s*Launch\s*Edition$",
    r":\s*Day\s*One\s*Edition$",
    r":\s*Standard\s*Edition$",
    r":\s*Royal\s*Edition$",
    r":\s*Imperial\s*Edition$",
    r":\s*Champion\s*Edition$",
    r":\s*Hero\s*Edition$",
    r":\s*Starter\s*Edition$",
    r":\s*Essential\s*Edition$",
    r":\s*Tarnished\s*Edition$",
    
    # Dash variants
    r"\s+-\s+Collector'?s?\s*Edition$",
    r"\s+-\s+Deluxe\s*Edition$",
    r"\s+-\s+Ultimate\s*Edition$",
    r"\s+-\s+Gold\s*Edition$",
    r"\s+-\s+Premium\s*Edition$",
    r"\s+-\s+Special\s*Edition$",
    r"\s+-\s+Limited\s*Edition$",
    r"\s+-\s+Legendary\s*Edition$",
    r"\s+-\s+Anniversary\s*Edition$",
    r"\s+-\s+Definitive\s*Edition$",
    r"\s+-\s+Complete\s*Edition$",
    r"\s+-\s+Enhanced\s*Edition$",
    r"\s+-\s+Game\s*of\s*the\s*Year\s*Edition$",
    r"\s+-\s+GOTY$",
    
    # Anniversary patterns
    r":\s*\d+th\s+Anniversary\s*Edition$",
    r"\s+-\s+\d+th\s+Anniversary\s*Edition$",
    
    # Bundle patterns  
    r":\s*Premium\s*Bundle$",
    r"\s+-\s+Premium\s*Bundle$",
    r":\s*Expansion\s*Bundle$",
    r"\s+-\s+Expansion\s*Bundle$",
]

# DLC indicators - entries matching these need higher engagement threshold
DLC_INDICATORS = [
    r"\s+-\s+DLC",
    r":\s*DLC",
    r"Season\s*Pass",
    r"Expansion\s*Pass",
    r"Skin\s*Pack",
    r"Character\s*Pack",
    r"Weapon\s*Pack",
    r"Map\s*Pack",
    r"New\s*Quest",
    r"Bonus\s*Content",
    r"Digital\s*Extras",
    r"Soundtrack",
    r"Art\s*Book",
    r"Cosmetic",
]

# Noise patterns - always remove
NOISE_PATTERNS = [
    r"\.exe$",           # Executable filenames as aliases
    r"Quad\s*Pack$",     # Bundle packs
    r"Triple\s*Pack$",
    r"Double\s*Pack$",
    r"\+.*\+",           # Multi-game bundles like "A + B + C"
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def strip_edition_suffix(name: str) -> str:
    """Remove edition suffixes to get base game name."""
    result = name
    for pattern in EDITION_SUFFIXES:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result.strip()


def is_dlc(name: str) -> bool:
    """Check if entry appears to be DLC/expansion."""
    for pattern in DLC_INDICATORS:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    return False


def is_noise(name: str) -> bool:
    """Check if entry is noise that should be removed."""
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    return False


def clean_aliases(aliases: list) -> list:
    """Remove problematic aliases like executable filenames."""
    cleaned = []
    for alias in aliases:
        # Skip executable filenames
        if re.search(r'\.(exe|dll)$', alias, re.IGNORECASE):
            continue
        # Skip very short aliases (likely abbreviations that cause false positives)
        if len(alias) < 4:
            continue
        cleaned.append(alias)
    return cleaned


def normalize_for_grouping(name: str) -> str:
    """
    Create normalized key for grouping related games.
    Strips editions, normalizes numerals, lowercases.
    """
    # First strip edition suffixes
    key = strip_edition_suffix(name).lower()
    
    # Remove "The " prefix
    key = re.sub(r"^the\s+", "", key)
    
    # Normalize roman numerals to arabic
    roman_map = [
        (r"\bviii\b", "8"), (r"\bvii\b", "7"), (r"\bvi\b", "6"),
        (r"\biv\b", "4"), (r"\bv\b", "5"), (r"\biii\b", "3"),
        (r"\bii\b", "2"), (r"\bi\b", "1"),
    ]
    for roman, arabic in roman_map:
        key = re.sub(roman, arabic, key)
    
    # Remove punctuation and normalize whitespace
    key = re.sub(r"[^\w\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    
    return key


def get_priority_tier(game: dict) -> int:
    """Assign priority tier based on engagement and LOTR relevance."""
    rating_count = game.get("rating_count") or 0
    is_lotr = game.get("lotr_related", False)
    
    if is_lotr:
        return 1  # All LOTR content is tier 1
    elif rating_count >= 500:
        return 1
    elif rating_count >= 100:
        return 2
    else:
        return 3


# =============================================================================
# MAIN CURATION PIPELINE
# =============================================================================

def curate_entities():
    """Main curation pipeline."""
    print("=" * 60)
    print("Video Game Entity Curation - Research Grade")
    print("=" * 60)
    print()
    
    # Load raw data
    print(f"Loading {INPUT_FILE.name}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    raw_games = raw_data.get("games", [])
    print(f"  Input: {len(raw_games)} raw entries")
    print()
    
    # ===========================================
    # Step 1: Separate LOTR from others
    # ===========================================
    print("Step 1: Separating LOTR content...")
    lotr_games = [g for g in raw_games if g.get("lotr_related")]
    other_games = [g for g in raw_games if not g.get("lotr_related")]
    print(f"  LOTR games: {len(lotr_games)}")
    print(f"  Other games: {len(other_games)}")
    print()
    
    # ===========================================
    # Step 2: Filter noise from other games
    # ===========================================
    print("Step 2: Removing noise entries...")
    filtered = []
    removed_noise = 0
    
    for game in other_games:
        name = game.get("canonical_name", "")
        if is_noise(name):
            removed_noise += 1
            continue
        filtered.append(game)
    
    print(f"  Removed: {removed_noise} noise entries")
    print()
    
    # ===========================================
    # Step 3: Apply engagement thresholds
    # ===========================================
    print("Step 3: Applying engagement thresholds...")
    engaged = []
    removed_low_engagement = 0
    
    for game in filtered:
        name = game.get("canonical_name", "")
        rating_count = game.get("rating_count") or 0
        
        # DLC needs higher threshold
        if is_dlc(name):
            if rating_count >= MIN_RATING_COUNT_DLC:
                engaged.append(game)
            else:
                removed_low_engagement += 1
        # Regular games need standard threshold
        else:
            if rating_count >= MIN_RATING_COUNT_GENERAL:
                engaged.append(game)
            else:
                removed_low_engagement += 1
    
    print(f"  Removed: {removed_low_engagement} low-engagement entries")
    print(f"  Remaining: {len(engaged)} games")
    print()
    
    # ===========================================
    # Step 4: Collapse edition variants
    # ===========================================
    print("Step 4: Collapsing edition variants...")
    
    # Group by normalized base name
    groups = defaultdict(list)
    for game in engaged:
        name = game.get("canonical_name", "")
        key = normalize_for_grouping(name)
        groups[key].append(game)
    
    # Keep best entry per group
    collapsed = []
    editions_collapsed = 0
    
    for key, group in groups.items():
        if len(group) == 1:
            collapsed.append(group[0])
        else:
            # Sort by: rating_count desc, then prefer shorter names
            group.sort(key=lambda g: (
                -(g.get("rating_count") or 0),
                len(g.get("canonical_name", ""))
            ))
            collapsed.append(group[0])
            editions_collapsed += len(group) - 1
    
    print(f"  Collapsed: {editions_collapsed} edition variants")
    print(f"  Unique base games: {len(collapsed)}")
    print()
    
    # ===========================================
    # Step 5: Process LOTR games (keep all, but collapse editions)
    # ===========================================
    print("Step 5: Processing LOTR games...")
    
    lotr_groups = defaultdict(list)
    for game in lotr_games:
        name = game.get("canonical_name", "")
        key = normalize_for_grouping(name)
        lotr_groups[key].append(game)
    
    lotr_collapsed = []
    for key, group in lotr_groups.items():
        group.sort(key=lambda g: (
            -(g.get("rating_count") or 0),
            len(g.get("canonical_name", ""))
        ))
        lotr_collapsed.append(group[0])
    
    print(f"  LOTR entries after collapsing: {len(lotr_collapsed)}")
    print()
    
    # ===========================================
    # Step 6: Combine and deduplicate
    # ===========================================
    print("Step 6: Combining and finalizing...")
    
    all_curated = lotr_collapsed + collapsed
    
    # Dedupe by igdb_id
    seen_ids = set()
    final_games = []
    for game in all_curated:
        gid = game.get("igdb_id")
        if gid is None or gid not in seen_ids:
            # Clean up the entry
            game["aliases"] = clean_aliases(game.get("aliases", []))
            game["priority_tier"] = get_priority_tier(game)
            
            # Regenerate regex with cleaned aliases
            game["regex_pattern"] = generate_regex_pattern(
                game["canonical_name"], 
                game["aliases"]
            )
            
            final_games.append(game)
            if gid:
                seen_ids.add(gid)
    
    # Sort: LOTR first, then by tier, then by engagement
    final_games.sort(key=lambda x: (
        0 if x.get("lotr_related") else 1,
        x.get("priority_tier", 3),
        -(x.get("rating_count") or 0)
    ))
    
    print(f"  Final count: {len(final_games)} entries")
    print()
    
    # ===========================================
    # Step 7: Build output
    # ===========================================
    lotr_count = sum(1 for g in final_games if g.get("lotr_related"))
    mod_count = sum(1 for g in final_games if g.get("is_mod"))
    tier_counts = {1: 0, 2: 0, 3: 0}
    for g in final_games:
        tier_counts[g.get("priority_tier", 3)] += 1
    
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source_file": INPUT_FILE.name,
            "curation_version": "2.0-research-grade",
            "total_entities": len(final_games),
            "lotr_related_count": lotr_count,
            "mods_count": mod_count,
            "tier_1_count": tier_counts[1],
            "tier_2_count": tier_counts[2],
            "tier_3_count": tier_counts[3],
            "curation_rules": {
                "min_rating_general": MIN_RATING_COUNT_GENERAL,
                "min_rating_dlc": MIN_RATING_COUNT_DLC,
                "editions_collapsed": True,
                "noise_filtered": True,
                "lotr_preserved": True
            }
        },
        "games": final_games
    }
    
    # Write output
    print(f"Writing to {OUTPUT_FILE.name}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # ===========================================
    # Summary
    # ===========================================
    print()
    print("=" * 60)
    print("CURATION COMPLETE")
    print("=" * 60)
    print(f"Input:  {len(raw_games)} raw entries")
    print(f"Output: {len(final_games)} curated entries")
    print(f"Reduction: {100*(len(raw_games)-len(final_games))/len(raw_games):.0f}%")
    print()
    print(f"LOTR games:  {lotr_count}")
    print(f"Mods:        {mod_count}")
    print(f"Tier 1:      {tier_counts[1]}")
    print(f"Tier 2:      {tier_counts[2]}")
    print(f"Tier 3:      {tier_counts[3]}")
    print()
    
    # List all LOTR entries
    print("LOTR content in curated list:")
    for g in final_games:
        if g.get("lotr_related"):
            mod_tag = " [MOD]" if g.get("is_mod") else ""
            rating = g.get("rating_count") or 0
            print(f"  • {g['canonical_name']}{mod_tag} ({rating} ratings)")
    print()
    
    # List top non-LOTR games
    print("Top 15 non-LOTR games by engagement:")
    non_lotr = [g for g in final_games if not g.get("lotr_related")]
    for g in non_lotr[:15]:
        print(f"  • {g['canonical_name']} ({g.get('rating_count', 0)} ratings)")
    
    return output


def generate_regex_pattern(canonical: str, aliases: list) -> str:
    """Generate regex pattern for matching game in text."""
    all_names = [canonical] + aliases
    
    escaped = []
    for name in all_names:
        pattern = re.escape(name)
        pattern = pattern.replace(r"\ ", r"\s+")
        pattern = pattern.replace(r"\-", r"[\s-]*")
        pattern = pattern.replace(r"\:", r":?\s*")
        escaped.append(pattern)
    
    combined = "|".join(escaped)
    return f"(?i)\\b({combined})\\b"


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        print("Ensure video_game_entities.json exists in the same directory.")
        exit(1)
    
    curate_entities()