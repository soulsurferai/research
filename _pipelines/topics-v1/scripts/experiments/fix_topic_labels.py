"""
fix_topic_labels.py - Fix the duplicate labeling issue
"""

import json
import pandas as pd
from collections import defaultdict

def create_improved_theme_map():
    """Create a more comprehensive and specific theme map"""
    
    theme_map = {
        # Policy/Legal (keep existing)
        'legalization': ['legalize', 'legalization', 'recreational', 'legal', 'decriminalize'],
        'rescheduling': ['dea', 'rescheduling', 'reschedule', 'schedule', 'federal'],
        'politics': ['trump', 'biden', 'harris', 'president', 'election'],
        'state_policy': ['state', 'states', 'florida', 'texas', 'california'],
        
        # Medical/Health
        'medical_use': ['medical', 'patient', 'treatment', 'therapy', 'medicine'],
        'cannabinoids': ['cbd', 'thc', 'cannabinoid', 'terpene', 'compound'],
        'health_conditions': ['pain', 'anxiety', 'depression', 'ptsd', 'cancer'],
        'health_issues': ['chs', 'heart', 'paranoid', 'panic', 'sick'],
        
        # Business/Finance
        'investment': ['stock', 'shares', 'msos', 'investment', 'market'],
        'business_ops': ['business', 'dispensary', 'retail', 'company', 'industry'],
        'financial': ['revenue', 'profit', 'debt', 'cash', 'money'],
        
        # Consumption/Effects
        'effects': ['high', 'stoned', 'felt', 'feeling', 'effects'],
        'consumption_method': ['smoke', 'smoking', 'vape', 'edible', 'dab'],
        'equipment': ['bong', 'pipe', 'joint', 'blunt', 'rig'],
        'strains': ['indica', 'sativa', 'strain', 'hybrid', 'genetics'],
        'products': ['flower', 'bud', 'wax', 'oil', 'concentrate'],
        
        # Lifestyle/Culture
        'tolerance': ['tolerance', 'break', 'quit', 'stopping', 'detox'],
        'social': ['friends', 'party', 'social', 'share', 'community'],
        'growing': ['grow', 'plant', 'harvest', 'cultivation', 'seeds'],
        'daily_use': ['daily', 'wake', 'routine', 'morning', 'night'],
        
        # Other
        'advice': ['help', 'advice', 'question', 'need', 'should'],
        'personal_story': ['story', 'experience', 'happened', 'tried', 'first'],
        'deals': ['price', 'cheap', 'expensive', 'deal', 'worth'],
        'testing': ['test', 'drug test', 'clean', 'detox', 'pass']
    }
    
    return theme_map


def generate_specific_labels(topics_data):
    """Generate more specific, non-duplicate labels"""
    
    theme_map = create_improved_theme_map()
    
    # Track used labels per subreddit to avoid duplication
    used_labels = defaultdict(set)
    
    new_labels = []
    
    for item in topics_data:
        subreddit = item['subreddit']
        words = item['words']
        word_string = ' '.join(words).lower()
        
        # Find best matching theme
        best_theme = None
        best_score = 0
        
        for theme, keywords in theme_map.items():
            # Count matching keywords
            matches = sum(1 for kw in keywords if kw in word_string)
            # Weight by keyword position
            for i, word in enumerate(words[:5]):
                if word in keywords:
                    matches += (5-i) / 5  # Higher weight for earlier words
            
            if matches > best_score:
                best_score = matches
                best_theme = theme
        
        # Generate label based on theme and specific words
        if best_theme:
            base_label = {
                'legalization': 'Legalization Discussion',
                'rescheduling': 'Federal Policy',
                'politics': 'Political Landscape',
                'state_policy': 'State Regulations',
                'medical_use': 'Medical Cannabis',
                'cannabinoids': 'Cannabinoid Science',
                'health_conditions': 'Therapeutic Applications',
                'health_issues': 'Health Concerns',
                'investment': 'Stock Market',
                'business_ops': 'Industry Operations',
                'financial': 'Financial Analysis',
                'effects': 'Effects & Experiences',
                'consumption_method': 'Consumption Methods',
                'equipment': 'Equipment & Tools',
                'strains': 'Strain Discussion',
                'products': 'Product Types',
                'tolerance': 'Tolerance & Breaks',
                'social': 'Social Aspects',
                'growing': 'Cultivation',
                'daily_use': 'Usage Patterns',
                'advice': 'Community Support',
                'personal_story': 'Personal Experiences',
                'deals': 'Pricing & Deals',
                'testing': 'Drug Testing'
            }.get(best_theme, 'General Discussion')
            
            # Make unique if already used
            if base_label in used_labels[subreddit]:
                # Add distinguishing feature
                distinguisher = None
                
                # Try to find a unique word not in the base label
                for word in words[:5]:
                    if word not in base_label.lower() and len(word) > 3:
                        distinguisher = word.title()
                        break
                
                if distinguisher:
                    label = f"{base_label}: {distinguisher}"
                else:
                    # Use index if no good distinguisher
                    count = sum(1 for l in used_labels[subreddit] if l.startswith(base_label))
                    label = f"{base_label} {count + 1}"
            else:
                label = base_label
                
            used_labels[subreddit].add(label)
            
        else:
            # Fallback to descriptive label
            label = f"{words[0].title()} {words[1].title()}"
            
        new_labels.append({
            'subreddit': subreddit,
            'original_label': item['label'],
            'new_label': label,
            'words': words,
            'size': item['size']
        })
    
    return new_labels


def main():
    """Process the analysis results and suggest better labels"""
    
    # Load the full results
    with open('quick_results/analysis_summary_advanced.json', 'r') as f:
        results = json.load(f)
    
    # Extract all topics that need relabeling
    topics_to_fix = []
    
    for sub, data in results.items():
        if 'topics' in data:
            for topic in data['topics']:
                topics_to_fix.append({
                    'subreddit': sub,
                    'label': topic['label'],
                    'words': topic['words'],
                    'size': topic['size'],
                    'coherence': topic.get('coherence', 0)
                })
    
    # Generate new labels
    new_labels = generate_specific_labels(topics_to_fix)
    
    # Show comparison
    print("LABEL IMPROVEMENTS")
    print("="*80)
    
    current_subreddit = None
    for item in new_labels:
        if item['subreddit'] != current_subreddit:
            current_subreddit = item['subreddit']
            print(f"\n{current_subreddit}:")
            print("-"*40)
        
        if item['original_label'] != item['new_label']:
            print(f"  {item['original_label']:35} → {item['new_label']}")
            print(f"    ({item['size']} docs, words: {', '.join(item['words'][:3])})")
    
    # Save improved labels
    output_file = 'quick_results/improved_topic_labels.json'
    with open(output_file, 'w') as f:
        json.dump(new_labels, f, indent=2)
    
    print(f"\n\nSaved improved labels to {output_file}")
    
    # Show label distribution
    print("\n\nLABEL DISTRIBUTION")
    print("="*80)
    
    label_counts = defaultdict(lambda: defaultdict(int))
    for item in new_labels:
        base_label = item['new_label'].split(':')[0].strip()
        label_counts[item['subreddit']][base_label] += 1
    
    for sub in ['cannabis', 'weed', 'trees', 'Marijuana', 'weedstocks', 'weedbiz']:
        if sub in label_counts:
            print(f"\n{sub}:")
            for label, count in sorted(label_counts[sub].items(), 
                                      key=lambda x: x[1], reverse=True):
                print(f"  {label:30} : {count}")


if __name__ == "__main__":
    main()
