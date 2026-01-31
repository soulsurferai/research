"""
analyze_label_issues.py - Quick analysis of labeling problems
"""

# From the analyze_other_topics.py output, here's what we learned:

LABEL_PROBLEMS = {
    'r/weed': {
        'total_topics': 12,
        'smoking_culture_labels': 11,
        'problem': 'Almost all topics labeled "Smoking Culture" with minor variants',
        'examples': [
            'Smoking Culture: High (138 docs)',
            'Smoking Culture: Weed (91 docs)', 
            'Smoking Culture: Smoke (62 docs)',
            'Smoking Culture: New (62 docs)',
            'Smoking Culture: Joint (23 docs)',
            'Smoking Culture: Bud (17 docs)'
        ]
    },
    
    'r/trees': {
        'total_topics': 6,
        'smoking_culture_labels': 4,
        'problem': '4 of 6 topics are "Smoking Culture"',
        'examples': [
            'Smoking Culture: Weed (205 docs)',
            'Smoking Culture: Weed (166 docs)',
            'Smoking Culture: Weed (122 docs)',
            'Smoking Culture (171 docs)'
        ]
    },
    
    'r/weedbiz': {
        'total_topics': 6,
        'retail_operations_labels': 4,
        'problem': '4 of 6 topics are "Retail Operations"',
        'examples': [
            'Retail Operations (273 docs)',
            'Retail Operations: Cannabis (265 docs)',
            'Retail Operations: Weed (211 docs)',
            'Retail Operations: Work (199 docs)'
        ]
    },
    
    'r/Marijuana': {
        'odd_topics': ['Room Plastic (18 docs)', 'Graft Gum Graft (13 docs)'],
        'problem': 'Contains very specific/odd topics that are probably noise'
    }
}

SUGGESTED_IMPROVEMENTS = {
    'r/weed': {
        'Smoking Culture: High': 'Effects & Experiences',
        'Smoking Culture: Weed': 'General Discussion',
        'Smoking Culture: Smoke': 'Consumption Methods',
        'Smoking Culture: New': 'First Time Users',
        'Smoking Culture: Joint': 'Rolling & Joints',
        'Smoking Culture: Bud': 'Flower Quality'
    },
    
    'r/trees': {
        'Smoking Culture: Weed': 'Community Discussion',
        'Smoking Culture': 'Equipment & Methods',
        'Cannabinoid Science: Weed': 'Science & Effects'
    },
    
    'r/weedbiz': {
        'Retail Operations': 'General Business',
        'Retail Operations: Cannabis': 'Cannabis Industry',
        'Retail Operations: Weed': 'Product Business',
        'Retail Operations: Work': 'Employment & Careers'
    }
}

# New categories to add to theme map
NEW_CATEGORIES = [
    'Health Issues (CHS, heart problems, panic)',
    'Strain Types (indica, sativa, hybrid)',
    'Tolerance & Quitting (breaks, cold turkey)',
    'Drug Testing (clean, detox, pass)',
    'First Time Users (new, tried, first)',
    'Equipment Types (bong, joint, vape)',
    'Price & Deals (cheap, expensive, worth)',
    'Employment (job, work, career, industry)'
]

print("LABELING ISSUES SUMMARY")
print("="*80)

for sub, issues in LABEL_PROBLEMS.items():
    print(f"\n{sub}:")
    print(f"  Problem: {issues['problem']}")
    if 'examples' in issues:
        print("  Examples:")
        for ex in issues['examples'][:3]:
            print(f"    - {ex}")

print("\n\nSUGGESTED LABEL IMPROVEMENTS")
print("="*80)

for sub, improvements in SUGGESTED_IMPROVEMENTS.items():
    print(f"\n{sub}:")
    for old, new in improvements.items():
        print(f"  {old:30} → {new}")

print("\n\nNEW CATEGORIES NEEDED")
print("="*80)
for cat in NEW_CATEGORIES:
    print(f"  - {cat}")
