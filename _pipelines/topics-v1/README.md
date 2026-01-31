# Topics_V1 - Cannabis Subreddit Topic Analysis ✅

## Overview
This project performs advanced topic modeling on 6 cannabis-related subreddits to understand their distinct discussion themes and community characteristics. Using BERTopic, UMAP, and smart clustering with pre-computed embeddings from Qdrant, we efficiently identify and compare topics across communities.

## Key Results
- **4,244 documents analyzed** across 6 subreddits
- **Successfully mapped** community positioning along knowledge and engagement dimensions
- **Discovered** r/Marijuana has the highest topic coherence (0.700) with science focus
- **Identified** clear spectrum from lifestyle communities to professional/business forums

## Purpose
- ✅ Identify main discussion topics in each cannabis subreddit
- ✅ Compare topic distributions across communities  
- ✅ Create positioning analysis showing community characteristics
- ✅ Understand how different cannabis communities serve different information needs

## Subreddits Analyzed
- `cannabis` - Policy and advocacy focused (402 docs, 6 topics)
- `weed` - General cannabis discussion (761 docs, 12 topics)
- `trees` - Lifestyle and culture (796 docs, 6 topics)
- `Marijuana` - Science and medical focus (664 docs, 12 topics) 
- `weedstocks` - Investment and business (538 docs, 18 topics)
- `weedbiz` - Industry and entrepreneurship (1,083 docs, 6 topics)

## Technical Approach
1. **Data Source**: Qdrant vector database with pre-computed embeddings
2. **Modular Architecture**: Refactored into clean, reusable components
3. **Advanced Algorithms**: UMAP + smart HDBSCAN with subclustering
4. **Quality Metrics**: Topic coherence scoring with BM25/MMR selection
5. **Enhanced Preprocessing**: Reddit-specific noise filtering

## Project Structure
```
Topics_V1/
├── config.py              # Centralized configuration
├── utils/                 # Reusable utilities
│   ├── json_utils.py     # JSON encoding with numpy support
│   ├── qdrant_utils.py   # Qdrant data fetching
│   └── text_utils.py     # Reddit text preprocessing
├── analysis/             # Core algorithms
│   ├── clustering.py     # UMAP, HDBSCAN, subclustering
│   ├── enhanced_clustering.py    # Advanced clustering methods
│   └── enhanced_topic_extraction.py  # BM25, MMR, coherence scoring
├── topics_advanced.py    # Main analysis pipeline
├── create_gartner_matrix.py      # Positioning visualization
├── create_topic_comparison.py    # Topic distribution charts
├── generate_insights_report.py   # Analysis summary
└── quick_results/        # Output directory with all results

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run advanced analysis
python topics_advanced.py

# Create visualizations
python create_gartner_matrix.py
python create_topic_comparison.py
python generate_insights_report.py
```

## Key Findings

### Community Positioning
- **Knowledge Spectrum**: Personal (trees/weed) → Scientific (Marijuana) → Policy (cannabis) → Business (weedstocks/weedbiz)
- **Engagement Styles**: Community-focused (trees) → Mixed (weed/Marijuana) → Civic (cannabis) → Transactional (business subs)

### Topic Quality
- **Highest coherence**: r/Marijuana (0.700) - well-defined scientific discussions
- **Lowest coherence**: r/weedbiz (0.272) - broad business topics
- **Most topics**: r/weedstocks (18) - diverse financial discussions

### Semantic Bridges
Universal topics crossing communities:
- Federal policy developments
- Medical/health benefits
- Legalization progress
- Product safety

## Output Files
- `analysis_summary_advanced.json` - Complete analysis results
- `subreddit_comparison.csv` - Summary statistics
- `[subreddit]_topics_advanced.csv` - Detailed topics per community
- `cannabis_gartner_matrix.png/pdf` - Positioning visualization
- `topic_theme_comparison.png` - Topic distribution charts
- `cannabis_community_insights.md` - Strategic insights report

## Technical Notes
- Fixed r/Marijuana case sensitivity issue (stored as "Marijuana" not "marijuana")
- Implemented smart subclustering to handle mega-clusters
- Used coherence-based selection between BM25 and MMR extraction
- Extensive stopword list for Reddit-specific content

## Applications
This analysis provides foundation for:
- **Content Strategy**: Target content to appropriate communities
- **Advocacy**: Frame messages for different audiences
- **Business Intelligence**: Understand customer segments
- **Research**: Identify knowledge gaps and opportunities

## Next Steps
- Temporal analysis of topic evolution
- Cross-community semantic bridge identification
- Integration with Concord project for ideological bridging
- Expansion to other substance-related communities

---

*Part of the Noetheca project - demonstrating semantic analysis capabilities for understanding diverse community perspectives*
