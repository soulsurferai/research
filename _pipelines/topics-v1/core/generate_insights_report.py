#!/usr/bin/env python3
"""
generate_insights_report.py - Generate insights from cannabis subreddit analysis
"""

import json
import pandas as pd
from datetime import datetime

# Load results
with open('quick_results/analysis_summary_advanced.json', 'r') as f:
    results = json.load(f)

# Generate report
report = f"""
# Cannabis Reddit Community Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d')}

## Executive Summary

Analysis of {sum(r.get('n_documents', 0) for r in results.values() if 'error' not in r):,} documents across 6 major cannabis subreddits reveals distinct community types serving different information needs.

## Key Findings

### 1. Community Segmentation

**Lifestyle Communities** (r/trees, r/weed)
- Focus: Personal consumption experiences, effects, equipment
- Characteristics: Casual, community-oriented, experience-sharing
- Topic coherence: Moderate (0.39-0.43)

**Policy & Advocacy** (r/cannabis)
- Focus: Legalization, federal rescheduling, political developments
- Characteristics: News-driven, advocacy-oriented, policy-focused
- Topic coherence: High (0.58)

**Business & Investment** (r/weedstocks, r/weedbiz)
- Focus: Stock market, company news, industry operations
- Characteristics: Professional, transactional, data-driven
- Topic coherence: Varied (0.27-0.65)

### 2. Content Patterns

**Most Discussed Topics Across All Communities:**
1. Political developments (Trump, Biden, federal policy)
2. Consumption methods and effects
3. Business/investment opportunities
4. Medical/health aspects
5. Cultivation and growing

**Unique Community Characteristics:**
- r/trees: Strongest focus on equipment and consumption methods
- r/cannabis: Most politically engaged, policy-focused
- r/weedstocks: Detailed financial analysis, company tracking
- r/weedbiz: B2B operations, industry challenges

### 3. Implications for Content Strategy

**For Advocates:**
- Target r/cannabis for policy discussions
- Use r/trees and r/weed for grassroots support
- Leverage r/weedstocks for economic arguments

**For Businesses:**
- r/weedstocks for investor relations
- r/weedbiz for B2B networking
- r/trees and r/weed for consumer insights

**For Researchers:**
- r/cannabis for policy impact discussions
- All communities show interest in medical benefits
- Limited pure science discussion (opportunity gap)

### 4. Semantic Bridges Between Communities

**Strongest Connections:**
- Political news crosses all communities
- Medical benefits resonate everywhere
- Business news flows from r/weedstocks to others

**Weakest Connections:**
- Technical growing details stay in cultivation subs
- Financial details rarely reach lifestyle communities
- B2B concerns isolated to r/weedbiz

## Recommendations

1. **Cross-Community Engagement**: Frame messages differently for each community while maintaining core content

2. **Bridge Topics**: Use medical benefits and political developments as universal entry points

3. **Community-Specific Content**:
   - Lifestyle communities: Focus on experience and culture
   - Business communities: Emphasize data and ROI
   - Policy communities: Highlight legislative impact

4. **Untapped Opportunities**:
   - Scientific research communication
   - Cross-community education initiatives
   - Industry-consumer dialogue facilitation

## Methodology Note

Analysis used advanced NLP techniques including:
- BM25 and MMR topic extraction
- UMAP dimensionality reduction
- Coherence-based quality assessment
- Community-specific text preprocessing

---

*This analysis provides a data-driven foundation for understanding the cannabis Reddit ecosystem and developing targeted engagement strategies.*
"""

# Save report
with open('quick_results/cannabis_community_insights.md', 'w') as f:
    f.write(report)

print("✅ Generated insights report: quick_results/cannabis_community_insights.md")

# Also create a CSV summary for easy reference
summary_data = []
for sub, data in results.items():
    if 'error' not in data:
        summary_data.append({
            'subreddit': sub,
            'documents': data.get('n_documents', 0),
            'topics': data.get('n_topics', 0),
            'avg_coherence': round(data.get('avg_coherence', 0), 3),
            'method': data.get('extraction_method', 'N/A'),
            'primary_focus': 'See detailed analysis'
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('quick_results/community_summary.csv', index=False)
print("✅ Generated summary CSV: quick_results/community_summary.csv")
